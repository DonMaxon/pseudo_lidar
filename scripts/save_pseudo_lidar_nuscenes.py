"""
save_pseudo_lidar_nuscenes.py

Генерирует pseudo-LiDAR облака точек для всех sample'ов nuScenes mini
и сохраняет их в формате, совместимом с nuScenes LIDAR_TOP:
  - float32 XYZIR (intensity = 0, ring = 0)
  - координаты в системе LiDAR-сенсора (как оригинальные .pcd.bin)
  - имена файлов совпадают с оригинальными

Структура вывода:
    output_dir/
        samples/
            LIDAR_TOP/
                n008-2018-...__LIDAR_TOP__....pcd.bin   ← замена оригиналу
        metadata.json   ← token → filename маппинг

Чтобы подменить реальный LiDAR:
    Укажите output_dir вместо оригинального dataroot,
    остальные папки (v1.0-mini/, maps/, sweeps/, CAM_*) оставьте симлинками
    или скопируйте — loader подхватит pseudo-LiDAR автоматически.

Запуск:
    python scripts/save_pseudo_lidar_nuscenes.py \\
        --dataroot /home/max/Phd/Phase1/Sparse4D/data/nuscenes \\
        --version v1.0-mini \\
        --output_dir ./output/pseudo_lidar_nuscenes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth2lidar.back_projection import BackProjector
from depth2lidar.depth_estimator import DepthEstimator
from depth2lidar.nuscenes_loader import NuScenesLoader
from depth2lidar.pointcloud_filter import RangeFilter, PointCloudFilter, StatisticalOutlierFilter
from depth2lidar.sky_masker import SemanticMasker

# Нативное разрешение nuScenes
_NUSCENES_HW = (900, 1600)

X_RANGE = (-51.2, 51.2)
Y_RANGE = (-51.2, 51.2)
Z_RANGE = (-2.0, 1.5)  # узкий диапазон: отсекает шумные точки зданий/деревьев сверху
EGO_RADIUS = 2.5

# Одна dummy-точка на случай пустого облака (в пределах диапазона, но далеко от объектов)
_DUMMY_POINT = np.array([[50.0, 50.0, 0.0]], dtype=np.float32)


def scale_intrinsic(K: np.ndarray, src_hw: tuple, dst_hw: tuple) -> np.ndarray:
    """
    Масштабирует intrinsics при анизотропном ресайзе (разные sx и sy).

    DA3 (upper_bound_resize) обрабатывает (900,1600) → (280,504) в два шага:
      1) _resize_longest_side: scale=504/1600=0.315, h=round(900*0.315)=284, w=504
         → K[0] *= 0.315, K[1] *= 284/900
      2) _make_divisible_by_resize: nearest 14-multiple(284)=280
         → K[0] *= 1.0, K[1] *= 280/284
      Итог: fx,cx *= 504/1600 = 0.315; fy,cy *= 280/900 = 0.3111

    Это эквивалентно простому анизотропному ресайзу: sx=dst_w/src_w, sy=dst_h/src_h.
    Паддинга нет — cx и cy масштабируются без смещения.
    """
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
) -> np.ndarray:
    """Строит pseudo-LiDAR в global frame из всех 6 камер одного sample.

    Каждая камера трансформируется через свой ego pose (своя временна́я метка),
    что устраняет временну́ю несогласованность между камерами и LiDAR.

    Returns:
        points_global (N, 3) float32 — сырые точки без downsampling/SOR.
        Downsampling и SOR выполняются позже, в sensor frame.
    """
    all_points = []

    for cam_name, cam in cam_data_dict.items():
        depth_map, _ = estimator.predict_with_intrinsics(cam.image, cam.intrinsic)

        if sky_masker is not None:
            depth_map, _ = sky_masker.apply_to_depth(depth_map, cam.image)

        intrinsic = cam.intrinsic.copy()
        depth_hw = depth_map.shape[:2]
        if depth_hw != _NUSCENES_HW:
            # DA3 использует анизотропный ресайз (upper_bound_resize): sx≠sy, без паддинга
            intrinsic = scale_intrinsic(intrinsic, _NUSCENES_HW, depth_hw)

        points_cam, _ = back_proj.depth_to_points_cam(depth_map, intrinsic)
        points_ego = back_proj.cam_to_ego(points_cam, cam.cam2ego)
        # Лёгкий pre-filter в ego frame чтобы не тащить заведомо далёкие точки в global
        points_ego = range_filter.filter(points_ego)

        # ego (timestamp камеры) → global
        points_global = back_proj.ego_to_global(points_ego, cam.ego2global)
        all_points.append(points_global)

    # Проверяем суммарное число точек (список всегда непустой при 6 камерах)
    total = sum(len(p) for p in all_points)
    if total == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Оставляем float64 — в global frame float32 теряет ~1-3см точности (~400км от нуля).
    # Приведение к float32 происходит позже в global_to_lidar_sensor (в sensor frame ±60м — OK).
    return np.concatenate(all_points, axis=0)


def global_to_lidar_sensor(points_global: np.ndarray, nusc, sample_token: str) -> np.ndarray:
    """
    Трансформирует точки из global frame в систему LiDAR-сенсора.

    Правильная цепочка (с учётом временно́й метки LiDAR):
        global → inv(ego2global_lidar) → ego_lidar → inv(sensor2ego_lidar) → sensor

    Это устраняет смещение ~0.1–0.4м из-за разницы временны́х меток
    камер и LiDAR (~10–44ms).
    """
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sd = nusc.get("sample_data", lidar_token)

    # ego2global на момент захвата LiDAR
    ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    T_ego2global = BackProjector.build_transform(
        rotation=np.array(ego_pose["rotation"], dtype=np.float64),
        translation=np.array(ego_pose["translation"], dtype=np.float64),
    )
    T_global2ego = np.linalg.inv(T_ego2global)

    # sensor2ego LiDAR
    cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    T_sensor2ego = BackProjector.build_transform(
        rotation=np.array(cs["rotation"], dtype=np.float64),
        translation=np.array(cs["translation"], dtype=np.float64),
    )
    T_ego2sensor = np.linalg.inv(T_sensor2ego)

    # global → ego_lidar → sensor_lidar
    T_global2sensor = T_ego2sensor @ T_global2ego

    N = len(points_global)
    pts_h = np.concatenate([points_global.astype(np.float64), np.ones((N, 1))], axis=1)
    pts_sensor = (T_global2sensor @ pts_h.T).T[:, :3]
    return pts_sensor.astype(np.float32)


def _range_filter_sensor(points_sensor: np.ndarray, nusc, sample_token: str,
                          range_filter: RangeFilter) -> np.ndarray:
    """Применяет range filter в ego frame: конвертирует sensor→ego, фильтрует, возвращает sensor."""
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
    """Возвращает оригинальный путь к файлу LIDAR_TOP (например samples/LIDAR_TOP/n008-...).pcd.bin)."""
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sd = nusc.get("sample_data", lidar_token)
    return lidar_sd["filename"]  # например "samples/LIDAR_TOP/n008-....pcd.bin"


def save_bin(points_sensor: np.ndarray, out_path: Path) -> None:
    """Сохраняет облако точек в формате nuScenes LIDAR_TOP: float32 XYZIR.
    5 каналов: X, Y, Z, intensity=0, ring=0  (load_dim=5 в BEVDet/mmdet3d).
    Если точек нет — добавляет одну dummy-точку, чтобы не крашить spconv.
    """
    if len(points_sensor) == 0:
        points_sensor = _DUMMY_POINT
    N = len(points_sensor)
    zeros = np.zeros((N, 1), dtype=np.float32)
    xyzir = np.concatenate([points_sensor, zeros, zeros], axis=1)  # (N, 5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xyzir.tofile(str(out_path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Сохранение pseudo-LiDAR в формате nuScenes LIDAR_TOP")
    p.add_argument("--dataroot", default="/home/max/Phd/Phase1/Sparse4D/data/nuscenes")
    p.add_argument("--version",  default="v1.0-mini")
    p.add_argument("--output_dir", default="./output/pseudo_lidar_nuscenes")
    p.add_argument("--max_depth", type=float, default=60.0)
    p.add_argument("--no_sky_mask", action="store_true")
    p.add_argument("--no_veg_mask", action="store_true")
    p.add_argument("--no_sor", action="store_true")
    p.add_argument("--sor_neighbors", type=int, default=20)
    p.add_argument("--sor_std_ratio", type=float, default=2.0)
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
    range_filter = RangeFilter(x_range=X_RANGE, y_range=Y_RANGE, z_range=Z_RANGE, ego_radius=EGO_RADIUS)

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

    # Маппинг sample_token → сохранённый файл (для metadata.json)
    metadata: dict[str, str] = {}

    total_scenes  = len(nusc.scene)
    processed     = 0

    for scene_idx, scene in enumerate(nusc.scene):
        scene_name = scene["name"]
        print(f"\n{'='*60}")
        print(f"Сцена {scene_idx + 1}/{total_scenes}: {scene_name}")

        sample = nusc.get("sample", scene["first_sample_token"])
        sample_idx = 0

        while True:
            token = sample["token"]

            # 1. Оригинальное имя файла (сохраняем структуру каталогов)
            orig_filename = get_lidar_filename(nusc, token)
            out_path = outdir / orig_filename  # e.g. output_dir/samples/LIDAR_TOP/n008-....pcd.bin

            # 2. Изображения камер
            cam_data_dict = loader.load_sample(token)

            # 3. Pseudo-LiDAR в global frame (каждая камера через свой ego2global)
            #    Downsampling НЕ делается здесь — выполняется в sensor frame (лучшая точность)
            points_global = build_pseudo_lidar_global(
                cam_data_dict, estimator, back_proj, range_filter,
                sky_masker,
            )

            # 4. global → LiDAR sensor frame (через ego LiDAR на его временно́й метке)
            points_sensor = global_to_lidar_sensor(points_global, nusc, token)

            # 5. Voxel downsampling в sensor frame
            #    (в sensor frame координаты ±60м → float32 точность ~7мкм, нет потерь)
            if len(points_sensor) > 0:
                points_sensor = PointCloudFilter.voxel_downsample(
                    points_sensor, voxel_size=args.voxel_size
                )

            # 6. SOR в sensor frame
            if sor_filter is not None and len(points_sensor) > 0:
                points_sensor = sor_filter.filter(points_sensor)

            # 7. Range filter в ego frame (убираем точки вне BEVDet range)
            if len(points_sensor) > 0:
                points_sensor = _range_filter_sensor(points_sensor, nusc, token, range_filter)

            # 8. Опциональное ограничение числа точек
            if args.max_points is not None and len(points_sensor) > args.max_points:
                points_sensor = PointCloudFilter.random_subsample(points_sensor, args.max_points)

            # 9. Сохранение (save_bin добавит dummy-точку если облако пустое)
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

    # Сохраняем маппинг
    meta_path = outdir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Готово. Обработано sample'ов: {processed}")
    print(f"Файлы: {outdir}/samples/LIDAR_TOP/")
    print(f"Маппинг: {meta_path}")
    print()
    print("Как использовать (drop-in замена):")
    print(f"  Укажите dataroot={outdir} в nuScenes-loader")
    print(f"  Или создайте симлинки на остальные папки:")
    print(f"    ln -s {args.dataroot}/{{v1.0-mini,maps,sweeps}} {outdir}/")
    print(f"    ln -s {args.dataroot}/samples/CAM_* {outdir}/samples/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
