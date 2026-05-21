"""
save_pseudo_lidar_nuscenes_v11.py

v11 = v7 + Edge Filtering (Sobel) + Sparsification (32-beam LiDAR pattern)

Две новые техники поверх v7:

1. Edge Filtering (Sobel):
   На depth map применяется оператор Собеля для обнаружения резких перепадов глубины.
   Пиксели на границах объектов (источник "длинных хвостов") исключаются из back-projection.
   Маска расширяется дилатацией для захвата соседних артефактных пикселей.

2. Sparsification (имитация 32-лучевого лидара):
   Псевдолидар проецируется на range image в сферических координатах.
   Из каждой ячейки (elevation_beam × azimuth_bin) оставляется ближайшая точка.
   Имитирует Velodyne HDL-32E: 32 луча, -30.67° до +10.67°, ~0.2° по азимуту.

Запуск:
    python scripts/v11/save_pseudo_lidar_nuscenes_v11.py \\
        --dataroot /home/max/Phd/Phase1/Sparse4D/data/nuscenes \\
        --version v1.0-mini \\
        --output_dir ./output/pseudo_lidar_nuscenes_v11
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth2lidar.back_projection import BackProjector
from depth2lidar.depth_estimator import DepthEstimator
from depth2lidar.nuscenes_loader import NuScenesLoader
from depth2lidar.pointcloud_filter import RangeFilter, PointCloudFilter, StatisticalOutlierFilter
from depth2lidar.sky_masker import SemanticMasker

_NUSCENES_HW = (900, 1600)

X_RANGE = (-51.2, 51.2)
Y_RANGE = (-51.2, 51.2)
Z_RANGE = (-2.0, 1.5)
EGO_RADIUS = 2.5

_DUMMY_POINT = np.array([[50.0, 50.0, 0.0]], dtype=np.float32)

# Velodyne HDL-32E параметры
_N_BEAMS = 32
_ELEV_MIN_DEG = -30.67
_ELEV_MAX_DEG = 10.67
_N_AZIMUTH = 1800  # 0.2° на bin


def apply_sobel_edge_filter(depth_map: np.ndarray,
                             grad_threshold_m: float = 2.0,
                             dilation_size: int = 3) -> np.ndarray:
    """
    Обнуляет пиксели на резких перепадах глубины (границы объектов).

    Алгоритм:
      1. Вычисляет градиент глубины (Собель по X и Y)
      2. Строит бинарную маску где |grad| > threshold
      3. Расширяет маску дилатацией
      4. Обнуляет depth_map в этих пикселях (BackProjector пропустит их при min_depth=0.5)

    Args:
        depth_map: (H, W) float32
        grad_threshold_m: порог градиента в метрах на пиксель
        dilation_size: размер структурного элемента для дилатации
    """
    # Градиент по X и Y через конечные разности (быстрее scipy.sobel, нет зависимости)
    grad_x = np.abs(np.gradient(depth_map, axis=1))
    grad_y = np.abs(np.gradient(depth_map, axis=0))
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    edge_mask = grad_mag > grad_threshold_m

    if dilation_size > 0:
        struct = np.ones((dilation_size, dilation_size), dtype=bool)
        edge_mask = binary_dilation(edge_mask, structure=struct)

    depth_filtered = depth_map.copy()
    depth_filtered[edge_mask] = 0.0
    return depth_filtered


def sparsify_to_lidar_pattern(points: np.ndarray,
                               n_beams: int = _N_BEAMS,
                               n_azimuth: int = _N_AZIMUTH,
                               elev_min_deg: float = _ELEV_MIN_DEG,
                               elev_max_deg: float = _ELEV_MAX_DEG) -> np.ndarray:
    """
    Прореживает облако точек до паттерна реального лидара.

    Алгоритм:
      1. XYZ → сферические координаты (r, azimuth, elevation)
      2. Каждой точке назначается ближайший elevation beam (из n_beams равномерных)
      3. Каждой точке назначается azimuth bin (из n_azimuth равномерных по 360°)
      4. В каждой ячейке (beam, azimuth) оставляется точка с минимальным r (ближайшая)

    Args:
        points: (N, 3) float32 в sensor frame
        n_beams: число elevation-лучей (32 для HDL-32E, 64 для HDL-64E)
        n_azimuth: число azimuth-бинов (1800 = 0.2° на бин)
        elev_min_deg / elev_max_deg: диапазон elevation для HDL-32E

    Returns:
        points (M, 3) — подмножество входных точек, M << N
    """
    if len(points) == 0:
        return points

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r_xy = np.sqrt(x ** 2 + y ** 2)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    azimuth_deg = np.degrees(np.arctan2(y, x))          # -180 .. 180
    elevation_deg = np.degrees(np.arctan2(z, r_xy))     # -90 .. 90

    # Назначаем beam_idx: ближайший из n_beams равномерных elevation-лучей
    beam_elevations = np.linspace(elev_min_deg, elev_max_deg, n_beams)
    elev_diffs = np.abs(elevation_deg[:, None] - beam_elevations[None, :])  # (N, n_beams)
    beam_idx = np.argmin(elev_diffs, axis=1)  # (N,)

    # Назначаем azimuth_idx: равномерная сетка [-180, 180)
    azimuth_idx = ((azimuth_deg + 180.0) / 360.0 * n_azimuth).astype(np.int32)
    azimuth_idx = np.clip(azimuth_idx, 0, n_azimuth - 1)

    # Для каждой ячейки (beam, azimuth) оставляем ближайшую точку
    # Кодируем ключ как одно int64 число для быстрого поиска
    cell_key = beam_idx.astype(np.int64) * n_azimuth + azimuth_idx.astype(np.int64)

    # Сортируем по r (возрастание), затем по cell_key — берём первое вхождение каждого cell
    sort_order = np.lexsort((r, cell_key))
    sorted_keys = cell_key[sort_order]

    # unique: первое вхождение каждого уникального ключа (= ближайшая точка)
    _, first_idx = np.unique(sorted_keys, return_index=True)
    selected = sort_order[first_idx]

    return points[selected]


def scale_intrinsic(K: np.ndarray, src_hw: tuple, dst_hw: tuple) -> np.ndarray:
    sy = dst_hw[0] / src_hw[0]
    sx = dst_hw[1] / src_hw[1]
    K = K.copy()
    K[0, 0] *= sx; K[1, 1] *= sy
    K[0, 2] *= sx; K[1, 2] *= sy
    return K


def build_pseudo_lidar_global(
    cam_data_dict: dict,
    estimator: DepthEstimator,
    back_proj: BackProjector,
    range_filter: RangeFilter,
    sky_masker: SemanticMasker | None = None,
    sobel_threshold: float = 2.0,
    sobel_dilation: int = 3,
) -> np.ndarray:
    all_points = []

    for cam_name, cam in cam_data_dict.items():
        depth_map, _ = estimator.predict_with_intrinsics(cam.image, cam.intrinsic)

        if sky_masker is not None:
            depth_map, _ = sky_masker.apply_to_depth(depth_map, cam.image)

        # Edge filtering: обнуляем пиксели на резких границах глубины
        depth_map = apply_sobel_edge_filter(
            depth_map,
            grad_threshold_m=sobel_threshold,
            dilation_size=sobel_dilation,
        )

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


def _range_filter_sensor(points_sensor: np.ndarray, nusc, sample_token: str,
                          range_filter: RangeFilter) -> np.ndarray:
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    T_s2e = BackProjector.build_transform(
        np.array(cs["rotation"], dtype=np.float64),
        np.array(cs["translation"], dtype=np.float64),
    )
    N = len(points_sensor)
    pts_h = np.concatenate([points_sensor.astype(np.float64), np.ones((N, 1))], axis=1)
    pts_ego = (T_s2e @ pts_h.T).T[:, :3].astype(np.float32)
    mask = (
        (pts_ego[:, 0] >= range_filter.x_range[0]) & (pts_ego[:, 0] <= range_filter.x_range[1]) &
        (pts_ego[:, 1] >= range_filter.y_range[0]) & (pts_ego[:, 1] <= range_filter.y_range[1]) &
        (pts_ego[:, 2] >= range_filter.z_range[0]) & (pts_ego[:, 2] <= range_filter.z_range[1]) &
        (pts_ego[:, 0] ** 2 + pts_ego[:, 1] ** 2 > range_filter.ego_radius ** 2)
    )
    return points_sensor[mask]


def get_lidar_filename(nusc, sample_token: str) -> str:
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sd = nusc.get("sample_data", lidar_token)
    return lidar_sd["filename"]


def save_bin(points_sensor: np.ndarray, out_path: Path) -> None:
    if len(points_sensor) == 0:
        points_sensor = _DUMMY_POINT
    N = len(points_sensor)
    zeros = np.zeros((N, 1), dtype=np.float32)
    xyzir = np.concatenate([points_sensor, zeros, zeros], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xyzir.tofile(str(out_path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v11: v7 + Sobel edge filter + Sparsification")
    p.add_argument("--dataroot",   default="/home/max/Phd/Phase1/Sparse4D/data/nuscenes")
    p.add_argument("--version",    default="v1.0-mini")
    p.add_argument("--output_dir", default="./output/pseudo_lidar_nuscenes_v11")
    p.add_argument("--max_depth",  type=float, default=60.0)
    p.add_argument("--no_sky_mask",  action="store_true")
    p.add_argument("--no_veg_mask",  action="store_true")
    p.add_argument("--no_sor",       action="store_true")
    p.add_argument("--sor_neighbors", type=int,   default=20)
    p.add_argument("--sor_std_ratio", type=float, default=2.0)
    p.add_argument("--voxel_size",    type=float, default=0.1)
    # Edge filter params
    p.add_argument("--no_edge_filter",    action="store_true", help="Отключить Sobel edge filter")
    p.add_argument("--sobel_threshold",   type=float, default=2.0,
                   help="Порог градиента глубины в метрах/пиксель (default: 2.0)")
    p.add_argument("--sobel_dilation",    type=int,   default=3,
                   help="Размер ядра дилатации маски рёбер (default: 3)")
    # Sparsification params
    p.add_argument("--no_sparsify",    action="store_true", help="Отключить sparsification")
    p.add_argument("--n_beams",        type=int,   default=32,
                   help="Число elevation-лучей для имитации лидара (default: 32)")
    p.add_argument("--n_azimuth",      type=int,   default=1800,
                   help="Число azimuth-бинов (default: 1800 = 0.2°/бин)")
    p.add_argument("--max_samples",    type=int,   default=None)
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
        )

    sor_filter: StatisticalOutlierFilter | None = None
    if not args.no_sor:
        sor_filter = StatisticalOutlierFilter(
            nb_neighbors=args.sor_neighbors,
            std_ratio=args.sor_std_ratio,
        )

    print(f"Edge filter: {'OFF' if args.no_edge_filter else f'ON (threshold={args.sobel_threshold}m, dilation={args.sobel_dilation})'}")
    print(f"Sparsification: {'OFF' if args.no_sparsify else f'ON ({args.n_beams} beams, {args.n_azimuth} azimuth bins)'}")

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
                sobel_threshold=args.sobel_threshold if not args.no_edge_filter else 1e9,
                sobel_dilation=args.sobel_dilation if not args.no_edge_filter else 0,
            )

            points_sensor = global_to_lidar_sensor(points_global, nusc, token)

            if len(points_sensor) > 0:
                points_sensor = PointCloudFilter.voxel_downsample(
                    points_sensor, voxel_size=args.voxel_size
                )

            if sor_filter is not None and len(points_sensor) > 0:
                points_sensor = sor_filter.filter(points_sensor)

            if len(points_sensor) > 0:
                points_sensor = _range_filter_sensor(points_sensor, nusc, token, range_filter)

            # Sparsification: имитируем 32-лучевой лидар
            if not args.no_sparsify and len(points_sensor) > 0:
                pts_before = len(points_sensor)
                points_sensor = sparsify_to_lidar_pattern(
                    points_sensor,
                    n_beams=args.n_beams,
                    n_azimuth=args.n_azimuth,
                )
                print(f"  [{sample_idx:03d}] sparsify: {pts_before:,} → {len(points_sensor):,} pts")

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
