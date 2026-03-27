"""
save_pseudo_lidar_nuscenes_v2.py  —  Pseudo-LiDAR v2

Архитектурные изменения по сравнению с v1:

  1. Многовидовой инференс DA3
     Все 6 камер одного sample'а передаются в одном вызове DA3.
     Модель видит перекрытия между видами и строит геометрически
     согласованную depth-карту по каждой камере.

  2. RGB-интенсивность
     Столбец intensity (4-й) заполняется яркостью (luminance) пикселя
     из RGB-изображения вместо нуля. Это соответствует поведению
     реального LiDAR и может улучшить LiDAR-backbone BEVDet.

Вывод:   ./output/pseudo_lidar_nuscenes_v2/
BEVDet:  configs/dal/dal-base-pseudo-v2.py
         data/nuscenes_pseudo_v2 → output/pseudo_lidar_nuscenes_v2/

Запуск:
    conda run -n depth2lidar python scripts/v2/save_pseudo_lidar_nuscenes_v2.py \\
        --dataroot /home/max/Phd/Phase1/Sparse4D/data/nuscenes \\
        --version v1.0-mini \\
        --output_dir ./output/pseudo_lidar_nuscenes_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PIL_Image

# ── пути к общему модулю ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth2lidar.back_projection import BackProjector
from depth2lidar.depth_estimator import DepthEstimator
from depth2lidar.nuscenes_loader import NuScenesLoader
from depth2lidar.pointcloud_filter import RangeFilter, PointCloudFilter, StatisticalOutlierFilter
from depth2lidar.sky_masker import SemanticMasker

# ── константы ─────────────────────────────────────────────────────────────────
_NUSCENES_HW = (900, 1600)

X_RANGE    = (-51.2, 51.2)
Y_RANGE    = (-51.2, 51.2)
Z_RANGE    = (-2.0,  1.5)
EGO_RADIUS = 2.5

_DUMMY_POINT = np.array([[50.0, 50.0, 0.0, 0.0]], dtype=np.float32)  # XYZI


# ── вспомогательные функции ───────────────────────────────────────────────────

def scale_intrinsic(K: np.ndarray, src_hw: tuple, dst_hw: tuple) -> np.ndarray:
    """Анизотропный scale intrinsics (sx ≠ sy), без паддинга — соответствует
    upper_bound_resize в DA3 InputProcessor."""
    sy = dst_hw[0] / src_hw[0]
    sx = dst_hw[1] / src_hw[1]
    K = K.copy()
    K[0, 0] *= sx; K[1, 1] *= sy
    K[0, 2] *= sx; K[1, 2] *= sy
    return K


def _rgb_intensity(image: np.ndarray, valid_mask: np.ndarray,
                   depth_hw: tuple) -> np.ndarray:
    """Luminance (BT.601) для валидных пикселей depth map.

    Args:
        image:      (900, 1600, 3) uint8 RGB.
        valid_mask: (dH, dW) bool — пиксели с валидной глубиной.
        depth_hw:   (dH, dW) — размер depth map.

    Returns:
        intensity:  (N,) float32 ∈ [0, 1].
    """
    pil = PIL_Image.fromarray(image)
    img_small = pil.resize((depth_hw[1], depth_hw[0]), PIL_Image.BILINEAR)
    rgb = np.array(img_small, dtype=np.float32) / 255.0   # (dH, dW, 3)
    rgb_valid = rgb[valid_mask]                             # (N, 3)
    intensity = (0.299 * rgb_valid[:, 0]
                 + 0.587 * rgb_valid[:, 1]
                 + 0.114 * rgb_valid[:, 2])
    return intensity.astype(np.float32)


# ── основные функции пайплайна ────────────────────────────────────────────────

def build_pseudo_lidar_global_v2(
    cam_data_dict: dict,
    estimator: DepthEstimator,
    back_proj: BackProjector,
    range_filter: RangeFilter,
    sky_masker: SemanticMasker | None = None,
) -> np.ndarray:
    """Строит pseudo-LiDAR (XYZI, float64) в global frame.

    Новшества v2:
      • Все 6 камер — один вызов DA3 (многовидовой батч).
      • Intensity = luminance из RGB.

    Returns:
        (N, 4) float64 — [X, Y, Z, intensity] в global frame.
    """
    cameras = list(cam_data_dict.values())
    images  = [cam.image for cam in cameras]

    # Intrinsics в оригинальном разрешении (6, 3, 3) для DA3
    ixts_orig = np.array([cam.intrinsic for cam in cameras], dtype=np.float32)

    # ── Многовидовой инференс ──────────────────────────────────────────────
    # Передаём только intrinsics — без extrinsics, чтобы не запускать
    # Umeyama-rescaling, который может исказить метрическую глубину.
    # DA3 всё равно делает cross-view attention внутри батча.
    with torch.no_grad():
        prediction = estimator._model.inference(
            images,
            intrinsics=ixts_orig,
            extrinsics=None,           # без rescaling
        )
    # prediction.depth: (6, H, W) tensor или numpy

    all_points: list[np.ndarray] = []

    for i, cam in enumerate(cameras):
        # ── depth map камеры i ─────────────────────────────────────────────
        depth_map = prediction.depth[i]
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.float().cpu().numpy()
        if depth_map.ndim == 3:          # (1, H, W) → (H, W)
            depth_map = depth_map[0]
        depth_map = np.clip(depth_map, 0.0, estimator.max_depth).astype(np.float32)

        depth_hw = depth_map.shape[:2]

        # ── sky/vegetation маска ───────────────────────────────────────────
        if sky_masker is not None:
            depth_map, _ = sky_masker.apply_to_depth(depth_map, cam.image)

        # ── intrinsics для back-projection ─────────────────────────────────
        intrinsic = scale_intrinsic(cam.intrinsic.copy(), _NUSCENES_HW, depth_hw)

        # ── back-projection в camera frame ─────────────────────────────────
        points_cam, valid_mask = back_proj.depth_to_points_cam(depth_map, intrinsic)

        if len(points_cam) == 0:
            continue

        # ── RGB intensity ──────────────────────────────────────────────────
        intensity = _rgb_intensity(cam.image, valid_mask, depth_hw)   # (N,)

        # ── cam → ego ──────────────────────────────────────────────────────
        points_ego = back_proj.cam_to_ego(points_cam, cam.cam2ego)     # (N, 3)
        # Формируем (N, 4): [X, Y, Z, I]
        points_ego_xyzi = np.concatenate(
            [points_ego, intensity[:, None]], axis=1
        )

        # ── pre-filter (в ego frame; range_filter работает на 3+ столбцах) ─
        points_ego_xyzi = range_filter.filter(points_ego_xyzi)

        if len(points_ego_xyzi) == 0:
            continue

        # ── ego → global (через ego_pose камеры, с её timestamp'ом) ────────
        points_global_xyz = back_proj.ego_to_global(
            points_ego_xyzi[:, :3], cam.ego2global
        )                                                                # (K, 3)
        points_global_xyzi = np.concatenate(
            [points_global_xyz, points_ego_xyzi[:, 3:]], axis=1
        )                                                                # (K, 4)
        all_points.append(points_global_xyzi)

    total = sum(len(p) for p in all_points)
    if total == 0:
        return np.empty((0, 4), dtype=np.float64)

    # float64 в global frame — без cast в float32 (потеря 1-3см точности)
    return np.concatenate(all_points, axis=0)


def global_to_lidar_sensor_xyzi(
    points_global_xyzi: np.ndarray,
    nusc,
    sample_token: str,
) -> np.ndarray:
    """global (XYZI) → LiDAR sensor frame (XYZI).

    Трансформация применяется только к XYZ; интенсивность переносится без изменений.
    """
    if len(points_global_xyzi) == 0:
        return np.empty((0, 4), dtype=np.float32)

    sample   = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

    ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    T_ego2global = BackProjector.build_transform(
        np.array(ego_pose["rotation"],    dtype=np.float64),
        np.array(ego_pose["translation"], dtype=np.float64),
    )
    T_global2ego = np.linalg.inv(T_ego2global)

    cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    T_sensor2ego = BackProjector.build_transform(
        np.array(cs["rotation"],    dtype=np.float64),
        np.array(cs["translation"], dtype=np.float64),
    )
    T_ego2sensor    = np.linalg.inv(T_sensor2ego)
    T_global2sensor = T_ego2sensor @ T_global2ego

    N      = len(points_global_xyzi)
    xyz    = points_global_xyzi[:, :3].astype(np.float64)
    ones   = np.ones((N, 1), dtype=np.float64)
    pts_h  = np.concatenate([xyz, ones], axis=1)
    xyz_s  = (T_global2sensor @ pts_h.T).T[:, :3].astype(np.float32)

    intensity = points_global_xyzi[:, 3:].astype(np.float32)
    return np.concatenate([xyz_s, intensity], axis=1)   # (N, 4) float32


def _range_filter_sensor_xyzi(
    points_sensor_xyzi: np.ndarray,
    nusc,
    sample_token: str,
    range_filter: RangeFilter,
) -> np.ndarray:
    """Range-filter в ego frame; возвращает точки в sensor frame с интенсивностью."""
    if len(points_sensor_xyzi) == 0:
        return points_sensor_xyzi

    sample   = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cs       = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    T_s2e    = BackProjector.build_transform(
        np.array(cs["rotation"],    dtype=np.float64),
        np.array(cs["translation"], dtype=np.float64),
    )

    N     = len(points_sensor_xyzi)
    xyz   = points_sensor_xyzi[:, :3].astype(np.float64)
    pts_h = np.concatenate([xyz, np.ones((N, 1))], axis=1)
    pts_ego = (T_s2e @ pts_h.T).T[:, :3].astype(np.float32)

    mask = (
        (pts_ego[:, 0] >= range_filter.x_range[0]) & (pts_ego[:, 0] <= range_filter.x_range[1]) &
        (pts_ego[:, 1] >= range_filter.y_range[0]) & (pts_ego[:, 1] <= range_filter.y_range[1]) &
        (pts_ego[:, 2] >= range_filter.z_range[0]) & (pts_ego[:, 2] <= range_filter.z_range[1]) &
        (pts_ego[:, 0] ** 2 + pts_ego[:, 1] ** 2 > range_filter.ego_radius ** 2)
    )
    return points_sensor_xyzi[mask]


def get_lidar_filename(nusc, sample_token: str) -> str:
    sample   = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    return lidar_sd["filename"]


def save_bin_v2(points_sensor_xyzi: np.ndarray, out_path: Path) -> None:
    """Сохраняет (N, 5) float32: X, Y, Z, intensity (из RGB), ring=0.

    Если точек нет — добавляет одну dummy-точку (не крашит spconv).
    """
    if len(points_sensor_xyzi) == 0:
        points_sensor_xyzi = _DUMMY_POINT

    N    = len(points_sensor_xyzi)
    ring = np.zeros((N, 1), dtype=np.float32)
    # Столбцы: X Y Z I R
    xyzir = np.concatenate(
        [points_sensor_xyzi[:, :4].astype(np.float32), ring], axis=1
    )                                                                  # (N, 5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xyzir.tofile(str(out_path))


# ── argparse ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pseudo-LiDAR v2: multi-view DA3 + RGB intensity")
    p.add_argument("--dataroot",   default="/home/max/Phd/Phase1/Sparse4D/data/nuscenes")
    p.add_argument("--version",    default="v1.0-mini")
    p.add_argument("--output_dir", default="./output/pseudo_lidar_nuscenes_v2")
    p.add_argument("--max_depth",      type=float, default=60.0)
    p.add_argument("--no_sky_mask",    action="store_true")
    p.add_argument("--no_veg_mask",    action="store_true")
    p.add_argument("--no_sor",         action="store_true")
    p.add_argument("--sor_neighbors",  type=int,   default=20)
    p.add_argument("--sor_std_ratio",  type=float, default=2.0)
    p.add_argument("--voxel_size",     type=float, default=0.1)
    p.add_argument("--max_points",     type=int,   default=None)
    p.add_argument("--max_samples",    type=int,   default=None)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Загрузка nuScenes ...")
    loader = NuScenesLoader(dataroot=args.dataroot, version=args.version)
    nusc   = loader.nusc

    print("Загрузка модели глубины ...")
    estimator    = DepthEstimator(max_depth=args.max_depth)
    back_proj    = BackProjector(min_depth=0.5, max_depth=args.max_depth)
    range_filter = RangeFilter(
        x_range=X_RANGE, y_range=Y_RANGE, z_range=Z_RANGE, ego_radius=EGO_RADIUS
    )

    sky_masker: SemanticMasker | None = None
    if not args.no_sky_mask or not args.no_veg_mask:
        sky_masker = SemanticMasker(
            mask_sky=not args.no_sky_mask,
            mask_vegetation=not args.no_veg_mask,
        )

    sor_filter: StatisticalOutlierFilter | None = None
    if not args.no_sor:
        sor_filter = StatisticalOutlierFilter(
            nb_neighbors=args.sor_neighbors,
            std_ratio=args.sor_std_ratio,
        )

    metadata: dict[str, str] = {}
    total_scenes = len(nusc.scene)
    processed    = 0

    for scene_idx, scene in enumerate(nusc.scene):
        print(f"\n{'='*60}")
        print(f"Сцена {scene_idx + 1}/{total_scenes}: {scene['name']}")

        sample     = nusc.get("sample", scene["first_sample_token"])
        sample_idx = 0

        while True:
            token = sample["token"]
            orig_filename = get_lidar_filename(nusc, token)
            out_path      = outdir / orig_filename

            cam_data_dict = loader.load_sample(token)

            # ── 1. Pseudo-LiDAR в global frame (все 6 камер одним батчем) ──
            points_global_xyzi = build_pseudo_lidar_global_v2(
                cam_data_dict, estimator, back_proj, range_filter, sky_masker,
            )

            # ── 2. global → LiDAR sensor frame ────────────────────────────
            points_sensor_xyzi = global_to_lidar_sensor_xyzi(
                points_global_xyzi, nusc, token
            )

            # ── 3. Voxel downsampling в sensor frame (на XYZ+I) ───────────
            if len(points_sensor_xyzi) > 0:
                points_sensor_xyzi = PointCloudFilter.voxel_downsample(
                    points_sensor_xyzi, voxel_size=args.voxel_size
                )

            # ── 4. SOR ─────────────────────────────────────────────────────
            if sor_filter is not None and len(points_sensor_xyzi) > 0:
                points_sensor_xyzi = sor_filter.filter(points_sensor_xyzi)

            # ── 5. Range filter в ego frame ───────────────────────────────
            if len(points_sensor_xyzi) > 0:
                points_sensor_xyzi = _range_filter_sensor_xyzi(
                    points_sensor_xyzi, nusc, token, range_filter
                )

            # ── 6. Ограничение числа точек ────────────────────────────────
            if args.max_points is not None and len(points_sensor_xyzi) > args.max_points:
                points_sensor_xyzi = PointCloudFilter.random_subsample(
                    points_sensor_xyzi, args.max_points
                )

            # ── 7. Сохранение ─────────────────────────────────────────────
            save_bin_v2(points_sensor_xyzi, out_path)
            metadata[token] = str(out_path.relative_to(outdir))

            processed += 1
            print(f"  [{sample_idx:03d}] {token[:12]}...  "
                  f"{len(points_sensor_xyzi):,} pts → {out_path.name}")

            if sample["next"] == "":
                break
            if args.max_samples and sample_idx + 1 >= args.max_samples:
                break
            sample     = nusc.get("sample", sample["next"])
            sample_idx += 1

    # ── metadata.json ─────────────────────────────────────────────────────────
    meta_path = outdir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Готово. Обработано sample'ов: {processed}")
    print(f"Файлы: {outdir}/samples/LIDAR_TOP/")
    print(f"Маппинг: {meta_path}")
    print()
    print("Следующий шаг — создать symlink и pkl:")
    print(f"  ln -sfn {outdir.resolve()} /home/max/Phd/Phase1/BEVDet/data/nuscenes_pseudo_v2")
    print(f"  python scripts/v2/make_pseudo_pkl_v2.py \\")
    print(f"    --src_pkl {outdir}/bevdetv5-nuscenes_infos_val_pseudo.pkl \\")
    print(f"    --output_dir {outdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
