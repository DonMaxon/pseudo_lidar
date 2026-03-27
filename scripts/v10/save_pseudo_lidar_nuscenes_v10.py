"""
save_pseudo_lidar_nuscenes_v10.py

v10 = v7 + три исправления:

  Fix 3. _range_filter_sensor: теперь фильтрует прямо в sensor (LiDAR) frame
          с BEVDet-точными bounds (x/y ±54.0, z -3..5), а не конвертирует в ego
          и применяет ego-frame bounds.

  Fix 4. voxel_downsample: в каждом вокселе теперь оставляется точка,
          ближайшая к сенсору (min r = sqrt(x²+y²+z²)), а не произвольная
          первая после lexsort. Ближайшая точка = наиболее надёжная.

  Fix 5. SOR std_ratio: 2.0 → 3.0. Менее агрессивное удаление —
          меньше ложных удалений плотных, но "шумящих" структур.

  + SegFormer-B5 (ADE20K, 640×640) вместо B0 — выше качество сегментации
    неба и растительности, меньше ложных масок.

Запуск:
    python scripts/v10/save_pseudo_lidar_nuscenes_v10.py \\
        --dataroot /home/max/Phd/Phase1/Sparse4D/data/nuscenes \\
        --version v1.0-mini \\
        --output_dir ./output/pseudo_lidar_nuscenes_v10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth2lidar.back_projection import BackProjector
from depth2lidar.depth_estimator import DepthEstimator
from depth2lidar.nuscenes_loader import NuScenesLoader
from depth2lidar.pointcloud_filter import RangeFilter, PointCloudFilter, StatisticalOutlierFilter
from depth2lidar.sky_masker import SemanticMasker

_NUSCENES_HW = (900, 1600)

# Pre-filter в ego frame (как в v7 — не изменяем в этой версии)
X_RANGE = (-55.0, 55.0)
Y_RANGE = (-55.0, 55.0)
Z_RANGE = (-2.0, 7.0)
EGO_RADIUS = 2.5

# BEVDet DAL point_cloud_range в LiDAR sensor frame: [-54,-54,-3, 54,54,5]
_SENSOR_X_RANGE = (-54.0, 54.0)
_SENSOR_Y_RANGE = (-54.0, 54.0)
_SENSOR_Z_RANGE = (-3.0, 5.0)

# Fix: SegFormer-B5 вместо B0
_SEGFORMER_MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"

_DUMMY_POINT = np.array([[50.0, 50.0, 0.0]], dtype=np.float32)


def scale_intrinsic(K: np.ndarray, src_hw: tuple, dst_hw: tuple) -> np.ndarray:
    sy = dst_hw[0] / src_hw[0]
    sx = dst_hw[1] / src_hw[1]
    K = K.copy()
    K[0, 0] *= sx; K[1, 1] *= sy
    K[0, 2] *= sx; K[1, 2] *= sy
    return K


def voxel_downsample_closest(points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
    """
    Voxel downsampling, оставляющий в каждом вокселе ближайшую к сенсору точку.

    Fix 4: вместо произвольной первой после lexsort — выбирается точка с
    минимальным r = sqrt(x²+y²+z²). Ближайшие точки более надёжны
    (меньше проецирование ошибок глубины на расстоянии).
    """
    if len(points) == 0:
        return np.empty((0, points.shape[1] if points.ndim > 1 else 3), dtype=np.float32)

    # Сортируем по расстоянию от сенсора: ближайшие — первые
    r2 = points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2
    dist_order = np.argsort(r2, kind="stable")
    sorted_pts = points[dist_order]

    # Назначаем воксельные координаты
    coords = np.floor(sorted_pts[:, :3] / voxel_size).astype(np.int32)

    # np.unique возвращает индекс первого вхождения каждого уникального вокселя.
    # После сортировки по r — первое вхождение = ближайшая точка.
    _, first_idx = np.unique(coords, axis=0, return_index=True)
    return sorted_pts[first_idx].astype(np.float32)


def build_pseudo_lidar_global(
    cam_data_dict: dict,
    estimator: DepthEstimator,
    back_proj: BackProjector,
    range_filter: RangeFilter,
    sky_masker: SemanticMasker | None = None,
) -> np.ndarray:
    """Строит pseudo-LiDAR в global frame из всех 6 камер одного sample."""
    all_points = []

    for cam_name, cam in cam_data_dict.items():
        depth_map, _ = estimator.predict_with_intrinsics(cam.image, cam.intrinsic)

        if sky_masker is not None:
            depth_map, _ = sky_masker.apply_to_depth(depth_map, cam.image)

        intrinsic = cam.intrinsic.copy()
        depth_hw = depth_map.shape[:2]
        if depth_hw != _NUSCENES_HW:
            intrinsic = scale_intrinsic(intrinsic, _NUSCENES_HW, depth_hw)

        points_cam, _ = back_proj.depth_to_points_cam(depth_map, intrinsic)
        points_ego = back_proj.cam_to_ego(points_cam, cam.cam2ego)
        points_ego = range_filter.filter(points_ego)

        points_global = back_proj.ego_to_global(points_ego, cam.ego2global)
        all_points.append(points_global)

    total = sum(len(p) for p in all_points)
    if total == 0:
        return np.empty((0, 3), dtype=np.float32)

    return np.concatenate(all_points, axis=0)


def global_to_lidar_sensor(points_global: np.ndarray, nusc, sample_token: str) -> np.ndarray:
    """Трансформирует точки из global frame в систему LiDAR-сенсора."""
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sd = nusc.get("sample_data", lidar_token)

    ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    T_ego2global = BackProjector.build_transform(
        rotation=np.array(ego_pose["rotation"], dtype=np.float64),
        translation=np.array(ego_pose["translation"], dtype=np.float64),
    )
    T_global2ego = np.linalg.inv(T_ego2global)

    cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    T_sensor2ego = BackProjector.build_transform(
        rotation=np.array(cs["rotation"], dtype=np.float64),
        translation=np.array(cs["translation"], dtype=np.float64),
    )
    T_ego2sensor = np.linalg.inv(T_sensor2ego)

    T_global2sensor = T_ego2sensor @ T_global2ego

    N = len(points_global)
    pts_h = np.concatenate([points_global.astype(np.float64), np.ones((N, 1))], axis=1)
    pts_sensor = (T_global2sensor @ pts_h.T).T[:, :3]
    return pts_sensor.astype(np.float32)


def _range_filter_sensor(points_sensor: np.ndarray, ego_radius: float) -> np.ndarray:
    """
    Fix 3: фильтрует в sensor (LiDAR) frame с BEVDet-точными bounds.

    v7 конвертировал в ego и применял ego-frame bounds (z_range=(-2,1.5)),
    что почти дублировало pre-filter. Теперь применяем BEVDet's
    point_cloud_range напрямую в sensor frame.
    """
    mask = (
        (points_sensor[:, 0] >= _SENSOR_X_RANGE[0]) & (points_sensor[:, 0] <= _SENSOR_X_RANGE[1]) &
        (points_sensor[:, 1] >= _SENSOR_Y_RANGE[0]) & (points_sensor[:, 1] <= _SENSOR_Y_RANGE[1]) &
        (points_sensor[:, 2] >= _SENSOR_Z_RANGE[0]) & (points_sensor[:, 2] <= _SENSOR_Z_RANGE[1]) &
        (points_sensor[:, 0] ** 2 + points_sensor[:, 1] ** 2 > ego_radius ** 2)
    )
    return points_sensor[mask]


def get_lidar_filename(nusc, sample_token: str) -> str:
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sd = nusc.get("sample_data", lidar_token)
    return lidar_sd["filename"]


def save_bin(points_sensor: np.ndarray, out_path: Path) -> None:
    """Сохраняет облако точек: float32 XYZIR (5 каналов, intensity=0, ring=0)."""
    if len(points_sensor) == 0:
        points_sensor = _DUMMY_POINT
    N = len(points_sensor)
    zeros = np.zeros((N, 1), dtype=np.float32)
    xyzir = np.concatenate([points_sensor, zeros, zeros], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xyzir.tofile(str(out_path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v10: fix sensor-frame filter + closest voxel + SOR3 + SegFormer-B5")
    p.add_argument("--dataroot", default="/home/max/Phd/Phase1/Sparse4D/data/nuscenes")
    p.add_argument("--version",  default="v1.0-mini")
    p.add_argument("--output_dir", default="./output/pseudo_lidar_nuscenes_v10")
    p.add_argument("--max_depth", type=float, default=60.0)
    p.add_argument("--no_sky_mask", action="store_true")
    p.add_argument("--no_veg_mask", action="store_true")
    p.add_argument("--no_sor", action="store_true")
    p.add_argument("--sor_neighbors", type=int, default=20)
    p.add_argument("--sor_std_ratio", type=float, default=3.0,  # Fix 5: было 2.0
                   help="SOR std ratio (default 3.0, v7 was 2.0)")
    p.add_argument("--voxel_size", type=float, default=0.1)
    p.add_argument("--max_points", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Ограничить число sample'ов (для быстрой проверки)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Загрузка nuScenes ...")
    loader = NuScenesLoader(dataroot=args.dataroot, version=args.version)
    nusc = loader.nusc

    print("Загрузка модели глубины ...")
    estimator    = DepthEstimator(max_depth=args.max_depth)
    back_proj    = BackProjector(min_depth=0.5, max_depth=args.max_depth)
    range_filter = RangeFilter(x_range=X_RANGE, y_range=Y_RANGE, z_range=Z_RANGE,
                               ego_radius=EGO_RADIUS)

    sky_masker: SemanticMasker | None = None
    if not args.no_sky_mask or not args.no_veg_mask:
        sky_masker = SemanticMasker(
            mask_sky=not args.no_sky_mask,
            mask_vegetation=not args.no_veg_mask,
            model_id=_SEGFORMER_MODEL,  # Fix: B5 вместо B0
        )

    sor_filter: StatisticalOutlierFilter | None = None
    if not args.no_sor:
        sor_filter = StatisticalOutlierFilter(
            nb_neighbors=args.sor_neighbors,
            std_ratio=args.sor_std_ratio,  # Fix 5: default 3.0
        )

    metadata: dict[str, str] = {}
    total_scenes = len(nusc.scene)
    processed    = 0

    for scene_idx, scene in enumerate(nusc.scene):
        scene_name = scene["name"]
        print(f"\n{'='*60}")
        print(f"Сцена {scene_idx + 1}/{total_scenes}: {scene_name}")

        sample = nusc.get("sample", scene["first_sample_token"])
        sample_idx = 0

        while True:
            token = sample["token"]
            orig_filename = get_lidar_filename(nusc, token)
            out_path = outdir / orig_filename

            cam_data_dict = loader.load_sample(token)

            points_global = build_pseudo_lidar_global(
                cam_data_dict, estimator, back_proj, range_filter, sky_masker,
            )

            points_sensor = global_to_lidar_sensor(points_global, nusc, token)

            # Fix 4: closest-to-sensor voxel downsampling
            if len(points_sensor) > 0:
                points_sensor = voxel_downsample_closest(points_sensor, voxel_size=args.voxel_size)

            # Fix 5: SOR с std_ratio=3.0
            if sor_filter is not None and len(points_sensor) > 0:
                points_sensor = sor_filter.filter(points_sensor)

            # Fix 3: финальный фильтр в sensor frame с BEVDet-точными bounds
            if len(points_sensor) > 0:
                points_sensor = _range_filter_sensor(points_sensor, EGO_RADIUS)

            if args.max_points is not None and len(points_sensor) > args.max_points:
                points_sensor = PointCloudFilter.random_subsample(points_sensor, args.max_points)

            save_bin(points_sensor, out_path)
            metadata[token] = str(out_path.relative_to(outdir))

            processed += 1
            print(f"  [{sample_idx:03d}] {token[:12]}...  {len(points_sensor):,} pts → {out_path.name}")

            if sample["next"] == "":
                break
            if args.max_samples and sample_idx + 1 >= args.max_samples:
                break

            sample = nusc.get("sample", sample["next"])
            sample_idx += 1

    meta_path = outdir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Готово. Обработано sample'ов: {processed}")
    print(f"Файлы: {outdir}/samples/LIDAR_TOP/")
    print(f"Маппинг: {meta_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
