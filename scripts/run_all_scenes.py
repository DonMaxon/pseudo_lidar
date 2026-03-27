"""
run_all_scenes.py

Проходит по всем сценам nuScenes, для каждого sample:
  1. Загружает изображения всех 6 камер
  2. Предсказывает карту глубины (DA3METRIC-LARGE)
  3. Строит псевдо-LiDAR облако точек (back-projection + фильтр)
  4. Загружает реальный LiDAR из nuScenes
  5. Сохраняет BEV-сравнение: псевдо-LiDAR (синий) vs GT LiDAR (оранжевый)

Структура вывода:
    output_dir/
        scene-0001/
            0000_<token>_bev.jpg
            0001_<token>_bev.jpg
            ...
        scene-0002/
            ...

Запуск:
    python scripts/run_all_scenes.py \\
        --dataroot /home/max/Phd/Phase1/Sparse4D/data \\
        --version v1.0-mini \\
        --output_dir ./output/all_scenes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth2lidar.back_projection import BackProjector
from depth2lidar.depth_estimator import DepthEstimator
from depth2lidar.lidar_comparison import LidarComparator
from depth2lidar.nuscenes_loader import NuScenesLoader, NUSCENES_CAMERAS
from depth2lidar.pointcloud_filter import RangeFilter, PointCloudFilter, StatisticalOutlierFilter
from depth2lidar.sky_masker import SemanticMasker

# Нативное разрешение nuScenes (для масштабирования intrinsics)
_NUSCENES_HW = (900, 1600)

X_RANGE = (-51.2, 51.2)
Y_RANGE = (-51.2, 51.2)
Z_RANGE = (-2.0, 1.5)
EGO_RADIUS = 2.5      # м — маска ego-vehicle в XY-плоскости
RESOLUTION = 0.1  # м/пиксель


# ---------------------------------------------------------------------------
# Intrinsics scaling
# ---------------------------------------------------------------------------

def scale_intrinsic(K: np.ndarray, src_hw: tuple, dst_hw: tuple) -> np.ndarray:
    sy = dst_hw[0] / src_hw[0]
    sx = dst_hw[1] / src_hw[1]
    K = K.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K


# ---------------------------------------------------------------------------
# BEV rendering (стандартная ориентация: вперёд = вверх, влево = влево)
# ---------------------------------------------------------------------------

def render_bev(points: np.ndarray, color_rgb: tuple) -> np.ndarray:
    """
    Рендерит облако точек в BEV-изображение (H, W, 3).

    Ориентация:
        вперёд (X+) → верхний край
        влево  (Y+) → левый край
    """
    W = int((Y_RANGE[1] - Y_RANGE[0]) / RESOLUTION)
    H = int((X_RANGE[1] - X_RANGE[0]) / RESOLUTION)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    if len(points) == 0:
        return canvas

    px = ((Y_RANGE[1] - points[:, 1]) / RESOLUTION).astype(np.int32)
    py = ((X_RANGE[1] - points[:, 0]) / RESOLUTION).astype(np.int32)

    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    canvas[py[valid], px[valid]] = color_rgb
    return canvas


def make_bev_comparison(
    pseudo_points: np.ndarray,
    gt_points: np.ndarray,
    scene_name: str,
    sample_idx: int,
    sample_token: str,
) -> np.ndarray:
    """
    Создаёт изображение с тремя BEV-панелями:
        [псевдо-LiDAR]  [GT LiDAR]  [наложение]

    Ось X (вперёд) = вверх, ось Y (влево) = влево.
    Нос машины смотрит вверх, ego-позиция в центре.
    """
    import cv2

    bev_pseudo  = render_bev(pseudo_points, (60, 140, 230))   # синий
    bev_gt      = render_bev(gt_points,     (230, 100, 30))   # оранжевый
    bev_overlay = render_bev(gt_points,     (230, 100, 30))   # оранжевый базовый
    # поверх — синий псевдо
    if len(pseudo_points) > 0:
        px = ((Y_RANGE[1] - pseudo_points[:, 1]) / RESOLUTION).astype(np.int32)
        py = ((X_RANGE[1] - pseudo_points[:, 0]) / RESOLUTION).astype(np.int32)
        W = bev_overlay.shape[1]
        H = bev_overlay.shape[0]
        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        bev_overlay[py[valid], px[valid]] = (60, 140, 230)

    H_bev, W_bev = bev_pseudo.shape[:2]
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1

    # Метаинфо
    info_lines = [
        f"Scene: {scene_name}  Sample: {sample_idx:03d}",
        f"Token: {sample_token[:12]}...",
        f"Pseudo pts: {len(pseudo_points):,}",
        f"GT pts:     {len(gt_points):,}",
    ]

    def _add_labels(img: np.ndarray, title: str, color: tuple) -> np.ndarray:
        out = img.copy()
        # Рамка направлений
        cx, cy_center = W_bev // 2, H_bev // 2
        cv2.line(out, (cx, cy_center - 40), (cx, 10), (80, 80, 80), 1)
        cv2.putText(out, "вперёд", (cx - 28, 22), font, 0.38, (160, 160, 160), 1)
        cv2.putText(out, "влево",  (6, cy_center + 4), font, 0.38, (160, 160, 160), 1)
        cv2.putText(out, "вправо", (W_bev - 55, cy_center + 4), font, 0.38, (160, 160, 160), 1)
        # Крест ego-позиции
        cv2.drawMarker(out, (cx, cy_center), (255, 255, 0),
                       cv2.MARKER_CROSS, 12, 1)
        # Заголовок панели
        cv2.putText(out, title, (6, H_bev - 8), font, font_scale, color, thickness)
        return out

    panel_pseudo  = _add_labels(bev_pseudo,  "Pseudo-LiDAR", (60, 140, 230))
    panel_gt      = _add_labels(bev_gt,      "GT LiDAR",     (230, 100, 30))
    panel_overlay = _add_labels(bev_overlay, "Overlay",      (200, 200, 200))

    # Метаданные — отдельная полоска снизу
    info_h = 22 * len(info_lines) + 10
    info_bar = np.zeros((info_h, W_bev * 3, 3), dtype=np.uint8)
    for i, line in enumerate(info_lines):
        cv2.putText(info_bar, line, (8, 20 + i * 22),
                    font, 0.52, (200, 200, 200), 1)

    grid = np.concatenate([panel_pseudo, panel_gt, panel_overlay], axis=1)
    return np.concatenate([grid, info_bar], axis=0)


# ---------------------------------------------------------------------------
# Pseudo-LiDAR pipeline для одного sample
# ---------------------------------------------------------------------------

def build_pseudo_lidar(
    cam_data_dict: dict,
    estimator: DepthEstimator,
    back_proj: BackProjector,
    range_filter: RangeFilter,
    sky_masker: SemanticMasker | None = None,
    sor_filter: StatisticalOutlierFilter | None = None,
    voxel_size: float = 0.1,
    max_points: int | None = None,
) -> np.ndarray:
    all_points = []

    for cam_name, cam in cam_data_dict.items():
        depth_map, _ = estimator.predict_with_intrinsics(cam.image)

        # Маскирование неба
        if sky_masker is not None:
            depth_map, _ = sky_masker.apply_to_depth(depth_map, cam.image)

        # Масштабируем intrinsics под реальный размер depth map
        intrinsic = cam.intrinsic.copy()
        depth_hw  = depth_map.shape[:2]
        if depth_hw != _NUSCENES_HW:
            intrinsic = scale_intrinsic(intrinsic, _NUSCENES_HW, depth_hw)

        points_cam, _ = back_proj.depth_to_points_cam(depth_map, intrinsic)
        points_ego     = back_proj.cam_to_ego(points_cam, cam.cam2ego)
        points_filtered = range_filter.filter(points_ego)
        all_points.append(points_filtered)

    if not all_points:
        return np.empty((0, 3), dtype=np.float32)

    merged = np.concatenate(all_points, axis=0)
    downsampled = PointCloudFilter.voxel_downsample(merged, voxel_size=voxel_size)

    if sor_filter is not None:
        downsampled = sor_filter.filter(downsampled)

    if max_points is not None:
        downsampled = PointCloudFilter.random_subsample(downsampled, max_points)

    return downsampled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BEV-сравнение псевдо-LiDAR и GT по всем сценам")
    p.add_argument("--dataroot", default="/home/max/Phd/Phase1/Sparse4D/data")
    p.add_argument("--version",  default="v1.0-mini")
    p.add_argument("--output_dir", default="./output/all_scenes")
    p.add_argument("--max_depth", type=float, default=60.0)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Ограничить число sample'ов на сцену (для быстрой проверки)")
    p.add_argument("--no_sky_mask", action="store_true",
                   help="Отключить маскирование неба (ADE20K class 2)")
    p.add_argument("--no_veg_mask", action="store_true",
                   help="Отключить маскирование растительности (ADE20K: tree/grass/plant)")
    p.add_argument("--no_sor", action="store_true",
                   help="Отключить Statistical Outlier Removal")
    p.add_argument("--sor_neighbors", type=int, default=20,
                   help="SOR: число ближайших соседей (default: 20)")
    p.add_argument("--sor_std_ratio", type=float, default=2.0,
                   help="SOR: порог в единицах std (default: 2.0, меньше = агрессивнее)")
    p.add_argument("--voxel_size", type=float, default=0.1,
                   help="Размер вокселя для voxel downsampling в метрах (default: 0.1)")
    p.add_argument("--max_points", type=int, default=None,
                   help="Жёсткий лимит точек после всех фильтров (random subsampling)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import cv2

    args   = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Инициализация ---
    print("Загрузка nuScenes ...")
    loader = NuScenesLoader(dataroot=args.dataroot, version=args.version)
    nusc   = loader.nusc

    print("Загрузка модели глубины ...")
    estimator    = DepthEstimator(max_depth=args.max_depth)
    back_proj    = BackProjector(min_depth=0.5, max_depth=args.max_depth)
    range_filter = RangeFilter(x_range=X_RANGE, y_range=Y_RANGE, z_range=Z_RANGE, ego_radius=EGO_RADIUS)
    comparator   = LidarComparator(x_range=X_RANGE, y_range=Y_RANGE, z_range=Z_RANGE)

    sky_masker: SemanticMasker | None = None
    mask_sky = not args.no_sky_mask
    mask_veg = not args.no_veg_mask
    if mask_sky or mask_veg:
        sky_masker = SemanticMasker(mask_sky=mask_sky, mask_vegetation=mask_veg)

    sor_filter: StatisticalOutlierFilter | None = None
    if not args.no_sor:
        sor_filter = StatisticalOutlierFilter(
            nb_neighbors=args.sor_neighbors,
            std_ratio=args.sor_std_ratio,
        )
        print(f"SOR: nb_neighbors={args.sor_neighbors}, std_ratio={args.sor_std_ratio}")

    total_scenes  = len(nusc.scene)
    total_samples = 0

    # --- Проход по сценам ---
    for scene_idx, scene in enumerate(nusc.scene):
        scene_name = scene["name"]
        scene_dir  = outdir / scene_name
        scene_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Сцена {scene_idx + 1}/{total_scenes}: {scene_name}")
        print(f"{'='*60}")

        sample = nusc.get("sample", scene["first_sample_token"])
        sample_idx = 0

        while True:
            token = sample["token"]
            print(f"  [{sample_idx:03d}] {token[:16]}...", end=" ")

            # 1. Загрузка изображений всех камер
            cam_data_dict = loader.load_sample(token)

            # 2. Псевдо-LiDAR
            pseudo_points = build_pseudo_lidar(
                cam_data_dict, estimator, back_proj, range_filter, sky_masker, sor_filter,
                voxel_size=args.voxel_size,
                max_points=args.max_points,
            )

            # 3. GT LiDAR
            gt_points = comparator.load_gt_lidar_ego(nusc, token)
            gt_points = comparator._range_filter(gt_points)

            print(f"pseudo={len(pseudo_points):,}  gt={len(gt_points):,}")

            # 4. BEV-визуализация
            bev_img = make_bev_comparison(
                pseudo_points, gt_points, scene_name, sample_idx, token
            )
            sample_stem = f"{sample_idx:04d}_{token[:8]}"
            out_path = scene_dir / f"{sample_stem}_bev.jpg"
            cv2.imwrite(
                str(out_path),
                cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

            # 5. Сохранение исходных изображений камер
            cam_dir = scene_dir / sample_stem
            cam_dir.mkdir(exist_ok=True)
            for cam_name, cam in cam_data_dict.items():
                cam_path = cam_dir / f"{cam_name}.jpg"
                cv2.imwrite(
                    str(cam_path),
                    cv2.cvtColor(cam.image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )

            total_samples += 1

            if sample["next"] == "":
                break
            if args.max_samples and sample_idx + 1 >= args.max_samples:
                break

            sample     = nusc.get("sample", sample["next"])
            sample_idx += 1

    print(f"\n{'='*60}")
    print(f"Готово. Обработано сцен: {total_scenes},  sample'ов: {total_samples}")
    print(f"Результаты: {outdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
