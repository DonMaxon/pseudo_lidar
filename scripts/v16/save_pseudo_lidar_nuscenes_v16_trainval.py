"""
save_pseudo_lidar_nuscenes_v16_trainval.py

v16_trainval = v16 + два исправления для генерации на полном trainval:

  Исправление 1 — обработка отсутствующих файлов:
    Оригинальный v16 падал с FileNotFoundError на scene-0103, потому что
    v1.0-trainval скачан неполностью (часть камерных изображений отсутствует).
    Теперь сэмплы с отсутствующими файлами пропускаются с предупреждением.

  Исправление 2 — более мягкий фильтр Собеля:
    Значения по умолчанию изменены: sobel_threshold 2.0 → 0.5 м,
    sobel_dilation 3 → 1 пиксель.
    Причина: агрессивный фильтр Собеля убивал большинство точек на дистанции
    10–30 м (граница у каждого объекта на средней дальности), оставляя
    облако почти пустым в этом диапазоне → провал NDS на trainval val (0.107).
    Мягкий порог сохраняет больше точек на средних дистанциях.

Запуск:
    python scripts/v16/save_pseudo_lidar_nuscenes_v16_trainval.py \\
        --dataroot /home/max/Phd/Phase1/dataset/v1.0-trainval \\
        --version v1.0-trainval \\
        --output_dir /home/max/Phd/Phase1/dataset/pseudo_lidar_v16_trainval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
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


# ─── Bilateral filter ────────────────────────────────────────────────────────

def apply_bilateral_filter(depth_map: np.ndarray,
                            d: int = 9,
                            sigma_depth: float = 1.0,
                            sigma_space: float = 9.0) -> np.ndarray:
    return cv2.bilateralFilter(
        depth_map.astype(np.float32),
        d=d,
        sigmaColor=sigma_depth,
        sigmaSpace=sigma_space,
    )


# ─── Sobel edge filter ────────────────────────────────────────────────────────

def apply_sobel_edge_filter(depth_map: np.ndarray,
                             grad_threshold_m: float = 0.5,
                             dilation_size: int = 1) -> np.ndarray:
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

    rimg = np.full((n_beams, n_azimuth), np.inf, dtype=np.float32)
    rimg[beam_idx, az_idx] = r

    min_neigh = np.full_like(rimg, np.inf)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = np.roll(np.roll(rimg, di, axis=0), dj, axis=1)
            min_neigh = np.minimum(min_neigh, shifted)

    diff = np.abs(rimg - min_neigh)
    jump = (diff > jump_threshold_m) & (min_neigh < np.inf)
    keep = ~jump[beam_idx, az_idx]
    return points[keep]


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def scale_intrinsic(K: np.ndarray, src_hw: tuple, dst_hw: tuple) -> np.ndarray:
    sy = dst_hw[0] / src_hw[0];  sx = dst_hw[1] / src_hw[1]
    K = K.copy()
    K[0, 0] *= sx;  K[1, 1] *= sy
    K[0, 2] *= sx;  K[1, 2] *= sy
    return K


# ─── Grid Penetration Filter ──────────────────────────────────────────────────

def grid_penetration_filter(points_global: np.ndarray,
                             cam_infos: list,
                             penetration_threshold: float = 1.0) -> np.ndarray:
    if len(points_global) == 0:
        return points_global

    N = len(points_global)
    pts_h = np.concatenate(
        [points_global.astype(np.float64), np.ones((N, 1))], axis=1
    )

    penetrates = np.zeros(N, dtype=bool)

    for cam in cam_infos:
        depth_map = cam['depth_map']
        K         = cam['K']
        T_g2c     = cam['T_g2c']
        H, W = depth_map.shape

        pts_cam = (T_g2c @ pts_h.T).T[:, :3]

        z = pts_cam[:, 2]
        in_front = z > 0.1

        u = K[0, 0] * pts_cam[:, 0] / np.where(in_front, z, 1.0) + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / np.where(in_front, z, 1.0) + K[1, 2]

        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)

        in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        valid = in_front & in_bounds

        if not np.any(valid):
            continue

        idx = np.where(valid)[0]
        d_point = z[idx]
        d_ref   = depth_map[vi[idx], ui[idx]]

        pen = idx[d_point > d_ref + penetration_threshold]
        penetrates[pen] = True

    return points_global[~penetrates]


def build_pseudo_lidar_global(cam_data_dict, estimator, back_proj, range_filter,
                               sky_masker=None,
                               sobel_threshold=0.5, sobel_dilation=1,
                               bilateral_d=9, bilateral_sigma_depth=1.0,
                               bilateral_sigma_space=9.0, no_bilateral=False):
    all_points = []
    cam_infos  = []

    for cam_name, cam in cam_data_dict.items():
        depth_map_raw, _ = estimator.predict_with_intrinsics(cam.image, cam.intrinsic)
        if sky_masker is not None:
            depth_map_raw, _ = sky_masker.apply_to_depth(depth_map_raw, cam.image)

        intrinsic = cam.intrinsic.copy()
        depth_hw  = depth_map_raw.shape[:2]
        if depth_hw != _NUSCENES_HW:
            intrinsic = scale_intrinsic(intrinsic, _NUSCENES_HW, depth_hw)

        depth_for_sobel = depth_map_raw
        if not no_bilateral:
            depth_for_sobel = apply_bilateral_filter(
                depth_map_raw, d=bilateral_d,
                sigma_depth=bilateral_sigma_depth,
                sigma_space=bilateral_sigma_space,
            )
        depth_map_filt = apply_sobel_edge_filter(depth_for_sobel, sobel_threshold, sobel_dilation)

        pts_cam = back_proj.depth_to_points_cam(depth_map_filt, intrinsic)[0]
        pts_ego = back_proj.cam_to_ego(pts_cam, cam.cam2ego)
        pts_ego = range_filter.filter(pts_ego)
        all_points.append(back_proj.ego_to_global(pts_ego, cam.ego2global))

        T_cam2global = cam.ego2global @ cam.cam2ego
        cam_infos.append({
            'depth_map': depth_map_raw.astype(np.float32),
            'K':         intrinsic,
            'T_g2c':     np.linalg.inv(T_cam2global),
        })

    total = sum(len(p) for p in all_points)
    if total == 0:
        return np.empty((0, 3), dtype=np.float32), cam_infos
    return np.concatenate(all_points, axis=0), cam_infos


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
    p = argparse.ArgumentParser(
        description="v16_trainval: v16 + обработка отсутствующих файлов + мягкий фильтр Собеля"
    )
    p.add_argument("--dataroot",   default="/home/max/Phd/Phase1/dataset/v1.0-trainval")
    p.add_argument("--version",    default="v1.0-trainval")
    p.add_argument("--output_dir", default="./output/pseudo_lidar_nuscenes_v16_trainval")
    p.add_argument("--max_depth",  type=float, default=60.0)
    p.add_argument("--no_sky_mask",  action="store_true")
    p.add_argument("--no_veg_mask",  action="store_true")
    p.add_argument("--no_sor",       action="store_true")
    p.add_argument("--sor_neighbors", type=int,   default=20)
    p.add_argument("--sor_std_ratio", type=float, default=2.0)
    p.add_argument("--voxel_size",    type=float, default=0.1)
    # Sobel — смягчённые значения по умолчанию (исправление 2)
    p.add_argument("--no_edge_filter",  action="store_true")
    p.add_argument("--sobel_threshold", type=float, default=0.5,
                   help="Порог градиента глубины (м). По умолчанию 0.5 вместо 2.0 "
                        "для сохранения точек на средних дистанциях 10-30 м")
    p.add_argument("--sobel_dilation",  type=int,   default=1,
                   help="Расширение маски краёв (пикс). По умолчанию 1 вместо 3")
    # Sparsification
    p.add_argument("--no_sparsify", action="store_true")
    p.add_argument("--n_beams",     type=int, default=32)
    p.add_argument("--n_azimuth",   type=int, default=1800)
    # RANSAC
    p.add_argument("--no_ransac",             action="store_true")
    p.add_argument("--ransac_z_min",          type=float, default=-2.5)
    p.add_argument("--ransac_z_max",          type=float, default=-0.8)
    p.add_argument("--ransac_threshold",      type=float, default=0.1)
    p.add_argument("--ransac_min_pts",        type=int,   default=100)
    # Jump filter
    p.add_argument("--no_jump_filter",        action="store_true")
    p.add_argument("--jump_threshold",        type=float, default=3.0)
    p.add_argument("--max_samples", type=int, default=None)
    # Bilateral filter
    p.add_argument("--no_bilateral",           action="store_true")
    p.add_argument("--bilateral_d",            type=int,   default=9)
    p.add_argument("--bilateral_sigma_depth",  type=float, default=1.0)
    p.add_argument("--bilateral_sigma_space",  type=float, default=9.0)
    # Grid Penetration Filter
    p.add_argument("--no_penetration",         action="store_true")
    p.add_argument("--penetration_threshold",  type=float, default=1.0)
    p.add_argument("--verbose", action="store_true")
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

    print(f"Bilateral   : {'OFF' if args.no_bilateral else f'ON (d={args.bilateral_d}, sig_depth={args.bilateral_sigma_depth}m, sig_space={args.bilateral_sigma_space}px)'}")
    print(f"Edge filter : {'OFF' if args.no_edge_filter else f'ON (thr={args.sobel_threshold}m, dil={args.sobel_dilation})'}")
    print(f"Penetration : {'OFF' if args.no_penetration else f'ON (thr={args.penetration_threshold}m)'}")
    print(f"RANSAC      : {'OFF' if args.no_ransac     else f'ON (z=[{args.ransac_z_min},{args.ransac_z_max}], thr={args.ransac_threshold}m)'}")
    print(f"Sparsify    : {'OFF' if args.no_sparsify   else f'ON ({args.n_beams} beams, {args.n_azimuth} az-bins)'}")
    print(f"Jump filter : {'OFF' if args.no_jump_filter else f'ON (thr={args.jump_threshold}m)'}")

    metadata     = {}
    total_scenes = len(nusc.scene)
    processed    = 0
    skipped      = 0

    for scene_idx, scene in enumerate(nusc.scene):
        print(f"\n{'='*60}\nСцена {scene_idx+1}/{total_scenes}: {scene['name']}")
        sample     = nusc.get("sample", scene["first_sample_token"])
        sample_idx = 0

        while True:
            token         = sample["token"]
            orig_filename = get_lidar_filename(nusc, token)
            out_path      = outdir / orig_filename

            # Исправление 1: пропускаем сэмплы с отсутствующими файлами
            try:
                cam_data_dict = loader.load_sample(token)
            except FileNotFoundError as e:
                print(f"  [{sample_idx:03d}] ПРОПУСК — файл не найден: {e}")
                skipped += 1
                if sample["next"] == "":
                    break
                if args.max_samples and sample_idx + 1 >= args.max_samples:
                    break
                sample     = nusc.get("sample", sample["next"])
                sample_idx += 1
                continue

            # 1. Depth → global
            points_global, cam_infos = build_pseudo_lidar_global(
                cam_data_dict, estimator, back_proj, range_filter, sky_masker,
                sobel_threshold=args.sobel_threshold if not args.no_edge_filter else 1e9,
                sobel_dilation=args.sobel_dilation   if not args.no_edge_filter else 0,
                bilateral_d=args.bilateral_d,
                bilateral_sigma_depth=args.bilateral_sigma_depth,
                bilateral_sigma_space=args.bilateral_sigma_space,
                no_bilateral=args.no_bilateral,
            )

            # 1b. Grid Penetration Filter (в глобальных координатах)
            if not args.no_penetration and len(points_global) > 0:
                before = len(points_global)
                points_global = grid_penetration_filter(
                    points_global,
                    cam_infos,
                    penetration_threshold=args.penetration_threshold,
                )
                if args.verbose:
                    print(f"  [{sample_idx:03d}] penetration filter: {before:,} → {len(points_global):,} pts")

            # 2. Global → sensor
            points_sensor = global_to_lidar_sensor(points_global, nusc, token)

            # 3. RANSAC ground plane correction
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

            # 8. Jump point filter
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
    print(f"Готово. Обработано: {processed}, пропущено: {skipped}")
    print(f"Файлы: {outdir}/samples/LIDAR_TOP/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
