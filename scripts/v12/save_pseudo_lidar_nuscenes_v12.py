"""
save_pseudo_lidar_nuscenes_v12.py

v12 = v11 (Sobel edge filter + Sparsification 32-beam)
     + RANSAC Ground Plane Fitting
     + Jump Point Filtering

Новые техники поверх v11:

3. RANSAC Ground Plane Fitting:
   Отбирает нижние точки облака (кандидаты на землю) и вписывает плоскость
   через RANSAC. Если плоскость наклонена (pitch/roll ошибка depth-модели),
   все точки облака корректируются по Z чтобы земля была горизонтальной.
   Исправляет вертикальный дрейф глубины на больших дистанциях.

4. Jump Point Filtering:
   Строит range image (n_beams × n_azimuth) из sparsified облака.
   Для каждой точки проверяет разницу дальности r с 8 соседями в range image.
   Если min(|r - r_neighbor|) > threshold — точка "прыгает" между объектом
   и фоном и удаляется. Работает только после sparsification.

Запуск:
    python scripts/v12/save_pseudo_lidar_nuscenes_v12.py \\
        --dataroot /home/max/Phd/Phase1/Sparse4D/data/nuscenes \\
        --version v1.0-mini \\
        --output_dir ./output/pseudo_lidar_nuscenes_v12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.linear_model import RANSACRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth2lidar.back_projection import BackProjector
from depth2lidar.depth_estimator import DepthEstimator
from depth2lidar.nuscenes_loader import NuScenesLoader
from depth2lidar.pointcloud_filter import RangeFilter, PointCloudFilter, StatisticalOutlierFilter
from depth2lidar.sky_masker import SemanticMasker

_NUSCENES_HW = (900, 1600)

X_RANGE    = (-51.2, 51.2)
Y_RANGE    = (-51.2, 51.2)
Z_RANGE    = (-2.0, 1.5)
EGO_RADIUS = 2.5

_DUMMY_POINT = np.array([[50.0, 50.0, 0.0]], dtype=np.float32)

# Velodyne HDL-32E
_N_BEAMS    = 32
_ELEV_MIN   = -30.67
_ELEV_MAX   =  10.67
_N_AZIMUTH  = 1800


# ─── Sobel edge filter ────────────────────────────────────────────────────────

def apply_sobel_edge_filter(depth_map: np.ndarray,
                             grad_threshold_m: float = 2.0,
                             dilation_size: int = 3) -> np.ndarray:
    grad_x = np.abs(np.gradient(depth_map, axis=1))
    grad_y = np.abs(np.gradient(depth_map, axis=0))
    edge_mask = np.sqrt(grad_x ** 2 + grad_y ** 2) > grad_threshold_m
    if dilation_size > 0:
        edge_mask = binary_dilation(edge_mask,
                                    structure=np.ones((dilation_size, dilation_size), dtype=bool))
    out = depth_map.copy()
    out[edge_mask] = 0.0
    return out


# ─── RANSAC ground plane correction ─────────────────────────────────────────

def ransac_ground_plane_correction(points: np.ndarray,
                                    z_min: float = -2.5,
                                    z_max: float = -0.8,
                                    residual_threshold: float = 0.1,
                                    min_ground_pts: int = 100) -> np.ndarray:
    """
    Корректирует Z-координаты всех точек по наклону плоскости земли.

    Алгоритм:
      1. Отбирает точки с Z ∈ [z_min, z_max] как кандидаты на землю.
      2. Вписывает плоскость z = a*x + b*y + c через RANSAC.
      3. Ожидаемый Z земли = медиана Z у inlier-точек (z_ref).
      4. Для каждой точки: z_corrected = z + (z_ref - predicted_z_ground(x,y)).
         Это «выравнивает» наклонённую плоскость земли в горизонт.

    Args:
        points: (N, 3) float32 в LiDAR sensor frame
        z_min / z_max: диапазон Z для отбора кандидатов земли
        residual_threshold: допуск RANSAC (м)
        min_ground_pts: минимум точек для запуска RANSAC
    """
    if len(points) == 0:
        return points

    mask_ground = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    gpts = points[mask_ground]

    if len(gpts) < min_ground_pts:
        return points

    try:
        ransac = RANSACRegressor(
            residual_threshold=residual_threshold,
            max_trials=200,
            random_state=42,
        )
        ransac.fit(gpts[:, :2], gpts[:, 2])

        inlier_z = gpts[ransac.inlier_mask_, 2]
        z_ref = float(np.median(inlier_z))

        z_pred = ransac.predict(points[:, :2]).astype(np.float32)
        corrected = points.copy()
        corrected[:, 2] += (z_ref - z_pred)
        return corrected

    except Exception:
        return points


# ─── Sparsification ───────────────────────────────────────────────────────────

def sparsify_to_lidar_pattern(points: np.ndarray,
                               n_beams: int = _N_BEAMS,
                               n_azimuth: int = _N_AZIMUTH,
                               elev_min_deg: float = _ELEV_MIN,
                               elev_max_deg: float = _ELEV_MAX) -> np.ndarray:
    if len(points) == 0:
        return points

    x, y, z   = points[:, 0], points[:, 1], points[:, 2]
    r_xy      = np.sqrt(x ** 2 + y ** 2)
    r         = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    az_deg    = np.degrees(np.arctan2(y, x))
    elev_deg  = np.degrees(np.arctan2(z, r_xy))

    beams      = np.linspace(elev_min_deg, elev_max_deg, n_beams)
    beam_idx   = np.argmin(np.abs(elev_deg[:, None] - beams[None, :]), axis=1)
    az_idx     = np.clip(((az_deg + 180.0) / 360.0 * n_azimuth).astype(np.int32),
                         0, n_azimuth - 1)

    cell_key   = beam_idx.astype(np.int64) * n_azimuth + az_idx.astype(np.int64)
    order      = np.lexsort((r, cell_key))
    _, first   = np.unique(cell_key[order], return_index=True)
    return points[order[first]]


# ─── Jump Point Filter ────────────────────────────────────────────────────────

def jump_point_filter(points: np.ndarray,
                      n_beams: int = _N_BEAMS,
                      n_azimuth: int = _N_AZIMUTH,
                      elev_min_deg: float = _ELEV_MIN,
                      elev_max_deg: float = _ELEV_MAX,
                      jump_threshold_m: float = 3.0) -> np.ndarray:
    """
    Удаляет точки с резким скачком дальности r относительно соседей в range image.

    Алгоритм:
      1. Строит range image (n_beams × n_azimuth) из облака точек.
      2. Для каждой ячейки вычисляет минимальный |r - r_neighbor| среди 8 соседей.
      3. Точки с min_diff > jump_threshold_m помечаются как шум и удаляются.

    Полностью векторизован через numpy roll — O(n_beams × n_azimuth).
    """
    if len(points) == 0:
        return points

    x, y, z  = points[:, 0], points[:, 1], points[:, 2]
    r_xy     = np.sqrt(x ** 2 + y ** 2)
    r        = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    az_deg   = np.degrees(np.arctan2(y, x))
    elev_deg = np.degrees(np.arctan2(z, r_xy))

    beams     = np.linspace(elev_min_deg, elev_max_deg, n_beams)
    beam_idx  = np.argmin(np.abs(elev_deg[:, None] - beams[None, :]), axis=1)
    az_idx    = np.clip(((az_deg + 180.0) / 360.0 * n_azimuth).astype(np.int32),
                        0, n_azimuth - 1)

    # Range image: inf = пусто
    rimg = np.full((n_beams, n_azimuth), np.inf, dtype=np.float32)
    rimg[beam_idx, az_idx] = r

    # Минимальная дальность среди 8 соседей (vectorized)
    min_neigh = np.full_like(rimg, np.inf)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = np.roll(np.roll(rimg, di, axis=0), dj, axis=1)
            min_neigh = np.minimum(min_neigh, shifted)

    # Разница дальности между точкой и ближайшим соседом
    diff = np.abs(rimg - min_neigh)

    # Jump: большой скачок И сосед существует (не inf)
    jump = (diff > jump_threshold_m) & (min_neigh < np.inf)

    keep = ~jump[beam_idx, az_idx]
    return points[keep]


# ─── Вспомогательные функции (из v11) ────────────────────────────────────────

def scale_intrinsic(K: np.ndarray, src_hw: tuple, dst_hw: tuple) -> np.ndarray:
    sy = dst_hw[0] / src_hw[0];  sx = dst_hw[1] / src_hw[1]
    K = K.copy()
    K[0, 0] *= sx;  K[1, 1] *= sy
    K[0, 2] *= sx;  K[1, 2] *= sy
    return K


def build_pseudo_lidar_global(cam_data_dict, estimator, back_proj, range_filter,
                               sky_masker=None,
                               sobel_threshold=2.0, sobel_dilation=3) -> np.ndarray:
    all_points = []
    for cam_name, cam in cam_data_dict.items():
        depth_map, _ = estimator.predict_with_intrinsics(cam.image, cam.intrinsic)
        if sky_masker is not None:
            depth_map, _ = sky_masker.apply_to_depth(depth_map, cam.image)
        depth_map = apply_sobel_edge_filter(depth_map, sobel_threshold, sobel_dilation)
        intrinsic = cam.intrinsic.copy()
        depth_hw  = depth_map.shape[:2]
        if depth_hw != _NUSCENES_HW:
            intrinsic = scale_intrinsic(intrinsic, _NUSCENES_HW, depth_hw)
        pts_cam = back_proj.depth_to_points_cam(depth_map, intrinsic)[0]
        pts_ego = back_proj.cam_to_ego(pts_cam, cam.cam2ego)
        pts_ego = range_filter.filter(pts_ego)
        all_points.append(back_proj.ego_to_global(pts_ego, cam.ego2global))
    total = sum(len(p) for p in all_points)
    if total == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(all_points, axis=0)


def global_to_lidar_sensor(points_global: np.ndarray, nusc, sample_token: str) -> np.ndarray:
    sample   = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    T_e2g    = BackProjector.build_transform(
        np.array(ego_pose["rotation"],  dtype=np.float64),
        np.array(ego_pose["translation"], dtype=np.float64))
    cs       = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    T_s2e    = BackProjector.build_transform(
        np.array(cs["rotation"],    dtype=np.float64),
        np.array(cs["translation"], dtype=np.float64))
    T_g2s    = np.linalg.inv(T_s2e) @ np.linalg.inv(T_e2g)
    N        = len(points_global)
    pts_h    = np.concatenate([points_global.astype(np.float64), np.ones((N, 1))], axis=1)
    return (T_g2s @ pts_h.T).T[:, :3].astype(np.float32)


def _range_filter_sensor(points_sensor, nusc, sample_token, range_filter):
    sample   = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cs       = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    T_s2e    = BackProjector.build_transform(
        np.array(cs["rotation"],    dtype=np.float64),
        np.array(cs["translation"], dtype=np.float64))
    N        = len(points_sensor)
    pts_h    = np.concatenate([points_sensor.astype(np.float64), np.ones((N, 1))], axis=1)
    pts_ego  = (T_s2e @ pts_h.T).T[:, :3].astype(np.float32)
    mask = (
        (pts_ego[:, 0] >= range_filter.x_range[0]) & (pts_ego[:, 0] <= range_filter.x_range[1]) &
        (pts_ego[:, 1] >= range_filter.y_range[0]) & (pts_ego[:, 1] <= range_filter.y_range[1]) &
        (pts_ego[:, 2] >= range_filter.z_range[0]) & (pts_ego[:, 2] <= range_filter.z_range[1]) &
        (pts_ego[:, 0] ** 2 + pts_ego[:, 1] ** 2 > range_filter.ego_radius ** 2)
    )
    return points_sensor[mask]


def get_lidar_filename(nusc, sample_token):
    sample   = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    return lidar_sd["filename"]


def save_bin(points_sensor, out_path):
    if len(points_sensor) == 0:
        points_sensor = _DUMMY_POINT
    N    = len(points_sensor)
    z    = np.zeros((N, 1), dtype=np.float32)
    xyzir = np.concatenate([points_sensor, z, z], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xyzir.tofile(str(out_path))


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="v12: v11 + RANSAC ground plane + jump point filter")
    p.add_argument("--dataroot",   default="/home/max/Phd/Phase1/Sparse4D/data/nuscenes")
    p.add_argument("--version",    default="v1.0-mini")
    p.add_argument("--output_dir", default="./output/pseudo_lidar_nuscenes_v12")
    p.add_argument("--max_depth",  type=float, default=60.0)
    p.add_argument("--no_sky_mask",  action="store_true")
    p.add_argument("--no_veg_mask",  action="store_true")
    p.add_argument("--no_sor",       action="store_true")
    p.add_argument("--sor_neighbors", type=int,   default=20)
    p.add_argument("--sor_std_ratio", type=float, default=2.0)
    p.add_argument("--voxel_size",    type=float, default=0.1)
    # Sobel
    p.add_argument("--no_edge_filter",  action="store_true")
    p.add_argument("--sobel_threshold", type=float, default=2.0)
    p.add_argument("--sobel_dilation",  type=int,   default=3)
    # Sparsification
    p.add_argument("--no_sparsify", action="store_true")
    p.add_argument("--n_beams",     type=int, default=32)
    p.add_argument("--n_azimuth",   type=int, default=1800)
    # RANSAC
    p.add_argument("--no_ransac",             action="store_true")
    p.add_argument("--ransac_z_min",          type=float, default=-2.5)
    p.add_argument("--ransac_z_max",          type=float, default=-0.8)
    p.add_argument("--ransac_threshold",      type=float, default=0.1,
                   help="Допуск RANSAC (м)")
    p.add_argument("--ransac_min_pts",        type=int,   default=100)
    # Jump filter
    p.add_argument("--no_jump_filter",        action="store_true")
    p.add_argument("--jump_threshold",        type=float, default=3.0,
                   help="Порог скачка дальности (м)")
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Загрузка nuScenes ...")
    loader = NuScenesLoader(dataroot=args.dataroot, version=args.version)
    nusc   = loader.nusc

    print("Загрузка модели глубины ...")
    estimator    = DepthEstimator(max_depth=args.max_depth)
    back_proj    = BackProjector(min_depth=0.5, max_depth=args.max_depth)
    range_filter = RangeFilter(x_range=X_RANGE, y_range=Y_RANGE, z_range=Z_RANGE,
                               ego_radius=EGO_RADIUS)

    sky_masker = None
    if not args.no_sky_mask or not args.no_veg_mask:
        sky_masker = SemanticMasker(
            mask_sky=not args.no_sky_mask,
            mask_vegetation=not args.no_veg_mask,
        )

    sor_filter = None
    if not args.no_sor:
        sor_filter = StatisticalOutlierFilter(
            nb_neighbors=args.sor_neighbors,
            std_ratio=args.sor_std_ratio,
        )

    print(f"Edge filter : {'OFF' if args.no_edge_filter else f'ON (thr={args.sobel_threshold}m, dil={args.sobel_dilation})'}")
    print(f"RANSAC      : {'OFF' if args.no_ransac     else f'ON (z=[{args.ransac_z_min},{args.ransac_z_max}], thr={args.ransac_threshold}m)'}")
    print(f"Sparsify    : {'OFF' if args.no_sparsify   else f'ON ({args.n_beams} beams, {args.n_azimuth} az-bins)'}")
    print(f"Jump filter : {'OFF' if args.no_jump_filter else f'ON (thr={args.jump_threshold}m)'}")

    metadata     = {}
    total_scenes = len(nusc.scene)
    processed    = 0

    for scene_idx, scene in enumerate(nusc.scene):
        print(f"\n{'='*60}\nСцена {scene_idx+1}/{total_scenes}: {scene['name']}")
        sample     = nusc.get("sample", scene["first_sample_token"])
        sample_idx = 0

        while True:
            token         = sample["token"]
            orig_filename = get_lidar_filename(nusc, token)
            out_path      = outdir / orig_filename

            cam_data_dict = loader.load_sample(token)

            # 1. Depth → global
            points_global = build_pseudo_lidar_global(
                cam_data_dict, estimator, back_proj, range_filter, sky_masker,
                sobel_threshold=args.sobel_threshold if not args.no_edge_filter else 1e9,
                sobel_dilation=args.sobel_dilation   if not args.no_edge_filter else 0,
            )

            # 2. Global → sensor
            points_sensor = global_to_lidar_sensor(points_global, nusc, token)

            # 3. RANSAC ground plane correction (до voxel, в полном облаке)
            if not args.no_ransac and len(points_sensor) > 0:
                points_sensor = ransac_ground_plane_correction(
                    points_sensor,
                    z_min=args.ransac_z_min,
                    z_max=args.ransac_z_max,
                    residual_threshold=args.ransac_threshold,
                    min_ground_pts=args.ransac_min_pts,
                )

            # 4. Voxel downsampling
            if len(points_sensor) > 0:
                points_sensor = PointCloudFilter.voxel_downsample(
                    points_sensor, voxel_size=args.voxel_size)

            # 5. SOR
            if sor_filter is not None and len(points_sensor) > 0:
                points_sensor = sor_filter.filter(points_sensor)

            # 6. Range filter
            if len(points_sensor) > 0:
                points_sensor = _range_filter_sensor(points_sensor, nusc, token, range_filter)

            # 7. Sparsification
            if not args.no_sparsify and len(points_sensor) > 0:
                points_sensor = sparsify_to_lidar_pattern(
                    points_sensor,
                    n_beams=args.n_beams,
                    n_azimuth=args.n_azimuth,
                )

            # 8. Jump point filter (после sparsification — работает с range image)
            if not args.no_jump_filter and len(points_sensor) > 0:
                before = len(points_sensor)
                points_sensor = jump_point_filter(
                    points_sensor,
                    n_beams=args.n_beams,
                    n_azimuth=args.n_azimuth,
                    jump_threshold_m=args.jump_threshold,
                )
                print(f"  [{sample_idx:03d}] jump filter: {before:,} → {len(points_sensor):,} pts")

            save_bin(points_sensor, out_path)
            metadata[token] = str(out_path.relative_to(outdir))
            processed += 1
            print(f"  [{sample_idx:03d}] {token[:12]}...  {len(points_sensor):,} pts → {out_path.name}")

            if sample["next"] == "":
                break
            if args.max_samples and sample_idx + 1 >= args.max_samples:
                break
            sample     = nusc.get("sample", sample["next"])
            sample_idx += 1

    meta_path = outdir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Готово. Обработано sample'ов: {processed}")
    print(f"Файлы: {outdir}/samples/LIDAR_TOP/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
