"""
image_to_lidar.py

Построение псевдо-LiDAR облака точек из одного изображения nuScenes
без загрузки датасета.

Глубина предсказывается моделью DA3METRIC-LARGE (Depth Anything 3).
Модель также предсказывает intrinsics камеры — они используются
по умолчанию. Флаг --use_default_intrinsics переключает на типовые
значения nuScenes v1.0 (если доверие к предсказанным intrinsics низкое).
cam2ego берётся из захардкоженных монтажных позиций nuScenes.

Пайплайн:
    1. Загрузить изображение из файла
    2. Предсказать метрическую карту глубины + intrinsics (DA3METRIC-LARGE)
    3. Back-projection: depth → 3D точки в системе камеры
    4. cam2ego: точки в системе ego-vehicle
    5. Фильтр по диапазону (BEV range + высота)
    6. Сохранить .bin (XYZI float32)
    7. Сохранить визуализацию: оригинал | карта глубины | точки на фото | BEV

Запуск:
    python scripts/image_to_lidar.py \\
        --image /path/to/image.jpg \\
        --camera CAM_FRONT \\
        --output_dir ./output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth2lidar.back_projection import BackProjector
from depth2lidar.depth_estimator import DepthEstimator
from depth2lidar.pointcloud_filter import RangeFilter
from depth2lidar.visualize import colorize_depth, project_points_on_image, points_to_bev_image


# ---------------------------------------------------------------------------
# Типовые калибровки nuScenes v1.0  (усреднены по сценам trainval/mini)
# ---------------------------------------------------------------------------

# intrinsic: fx, fy, cx, cy  (изображения 1600 x 900)
_INTRINSICS: dict[str, tuple[float, float, float, float]] = {
    "CAM_FRONT":       (1266.417, 1266.417,  816.267, 491.507),
    "CAM_FRONT_LEFT":  (1272.594, 1272.594,  826.511, 479.675),
    "CAM_FRONT_RIGHT": (1260.846, 1260.846,  807.967, 495.319),
    "CAM_BACK":        ( 809.200,  809.200,  829.200, 481.800),
    "CAM_BACK_LEFT":   (1256.744, 1256.744,  792.113, 492.756),
    "CAM_BACK_RIGHT":  (1259.570, 1259.570,  807.614, 501.435),
}

# cam2ego: translation [x, y, z] (м) + quaternion [w, x, y, z]
_CAM2EGO: dict[str, dict] = {
    "CAM_FRONT": {
        "translation": [1.72200568,  0.00475453, 1.49491292],
        "rotation":    [0.49999516, -0.49999516, 0.50000484, -0.50000484],
    },
    "CAM_FRONT_LEFT": {
        "translation": [1.39411561,  0.49500757, 1.52995796],
        "rotation":    [0.64612171, -0.62757697, 0.38471008, -0.38485972],
    },
    "CAM_FRONT_RIGHT": {
        "translation": [1.39213955, -0.46720318, 1.52995796],
        "rotation":    [0.53850882, -0.53031298, -0.46102496, 0.46427563],
    },
    "CAM_BACK": {
        "translation": [0.02594499,  0.00942673, 1.56702391],
        "rotation":    [0.49833042, -0.50167023, -0.49833042,  0.50167023],
    },
    "CAM_BACK_LEFT": {
        "translation": [0.06839815,  0.48148542, 1.56386762],
        "rotation":    [0.36439558, -0.36349091, -0.60699594,  0.60648736],
    },
    "CAM_BACK_RIGHT": {
        "translation": [0.06319978, -0.47104974, 1.55993487],
        "rotation":    [0.58000418, -0.57827227,  0.41553114, -0.41327237],
    },
}


# Нативное разрешение nuScenes, под которое захардкожены intrinsics
_NUSCENES_HW = (900, 1600)


def build_intrinsic(camera: str) -> np.ndarray:
    fx, fy, cx, cy = _INTRINSICS[camera]
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)


def scale_intrinsic(K: np.ndarray, src_hw: tuple, dst_hw: tuple) -> np.ndarray:
    """Масштабировать intrinsic матрицу при изменении разрешения изображения."""
    sy = dst_hw[0] / src_hw[0]
    sx = dst_hw[1] / src_hw[1]
    K = K.copy()
    K[0, 0] *= sx   # fx
    K[1, 1] *= sy   # fy
    K[0, 2] *= sx   # cx
    K[1, 2] *= sy   # cy
    return K


def build_cam2ego(camera: str) -> np.ndarray:
    entry = _CAM2EGO[camera]
    return BackProjector.build_transform(
        rotation=np.array(entry["rotation"], dtype=np.float64),
        translation=np.array(entry["translation"], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Одно изображение nuScenes → псевдо-LiDAR (без загрузки датасета)"
    )
    p.add_argument("--image", required=True,
                   help="Путь к изображению (.jpg / .png)")
    p.add_argument("--camera", required=True,
                   choices=list(_INTRINSICS.keys()),
                   help="Название камеры nuScenes")
    p.add_argument("--output_dir", default="./output",
                   help="Папка для сохранения результатов")
    p.add_argument("--use_default_intrinsics", action="store_true",
                   help="Использовать типовые nuScenes intrinsics вместо "
                        "предсказанных моделью")
    p.add_argument("--min_depth", type=float, default=0.5,
                   help="Минимальная глубина (м)")
    p.add_argument("--max_depth", type=float, default=60.0,
                   help="Максимальная глубина (м)")
    p.add_argument("--x_range", type=float, nargs=2, default=[-51.2, 51.2],
                   metavar=("X_MIN", "X_MAX"))
    p.add_argument("--y_range", type=float, nargs=2, default=[-51.2, 51.2],
                   metavar=("Y_MIN", "Y_MAX"))
    p.add_argument("--z_range", type=float, nargs=2, default=[-5.0, 3.0],
                   metavar=("Z_MIN", "Z_MAX"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_bin(points: np.ndarray, path: Path) -> None:
    intensity = np.zeros((len(points), 1), dtype=np.float32)
    xyzi = np.concatenate([points.astype(np.float32), intensity], axis=1)
    xyzi.tofile(str(path))
    print(f"    → .bin: {path}  ({len(points):,} точек, {xyzi.nbytes / 1024:.1f} KB)")


def save_visualization(
    image: np.ndarray,
    depth: np.ndarray,
    points_cam: np.ndarray,
    points_ego: np.ndarray,
    intrinsic: np.ndarray,
    output_dir: Path,
    stem: str,
) -> None:
    try:
        import cv2
    except ImportError:
        print("    opencv не установлен — визуализация пропущена")
        return

    panel_orig  = image.copy()
    panel_depth = colorize_depth(depth, min_depth=0.5, max_depth=60.0, cmap="plasma")
    panel_proj  = project_points_on_image(
        image=image,
        points_cam=points_cam[::10],
        intrinsic=intrinsic,
        point_size=1,
        cmap="plasma",
        max_depth=60.0,
    )

    bev_gray = points_to_bev_image(points_ego, x_range=(-51.2, 51.2), y_range=(-51.2, 51.2))
    bev_rgb  = np.stack([bev_gray] * 3, axis=-1)
    h_img    = panel_orig.shape[0]
    bev_rgb  = cv2.resize(bev_rgb, (int(bev_rgb.shape[1] * h_img / bev_rgb.shape[0]), h_img))

    target_h, target_w = panel_orig.shape[:2]
    panels = [
        cv2.resize(panel_orig,  (target_w, target_h)),
        cv2.resize(panel_depth, (target_w, target_h)),
        cv2.resize(panel_proj,  (target_w, target_h)),
        cv2.resize(bev_rgb,     (target_w, target_h)),
    ]

    labels = ["Original", "Depth map", "Points on image", "BEV (top-down)"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (8, 28), font, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

    grid     = np.concatenate(panels, axis=1)
    vis_path = output_dir / f"{stem}_visualization.jpg"
    cv2.imwrite(str(vis_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"    → визуализация: {vis_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 60
    stem = f"{Path(args.image).stem}_{args.camera}"

    # ------------------------------------------------------------------
    # 1. Загрузка изображения
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("Шаг 1/5: Загрузка изображения")
    print(sep)

    image = np.array(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    print(f"  Файл    : {args.image}")
    print(f"  Камера  : {args.camera}")
    print(f"  Размер  : {image.shape[1]}x{image.shape[0]} px")

    intrinsic = build_intrinsic(args.camera)
    cam2ego   = build_cam2ego(args.camera)
    print(f"  Intrinsic (типовые nuScenes):\n{intrinsic}")

    # ------------------------------------------------------------------
    # 2. Карта глубины + intrinsics от модели
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("Шаг 2/5: Предсказание метрической карты глубины (DA3METRIC-LARGE)")
    print(sep)

    estimator = DepthEstimator(max_depth=args.max_depth)
    depth_map, intrinsic_pred = estimator.predict_with_intrinsics(image)

    print(f"  Размер depth map : {depth_map.shape}")
    print(f"  min              : {depth_map.min():.3f} м")
    print(f"  max              : {depth_map.max():.3f} м")
    print(f"  mean             : {depth_map.mean():.3f} м")
    print(f"  median           : {np.median(depth_map):.3f} м")
    for thr in (0.5, 1.0, 5.0, 10.0, 30.0, 60.0):
        pct = 100.0 * (depth_map < thr).sum() / depth_map.size
        print(f"  пикселей < {thr:5.1f} м : {pct:.1f}%")

    if args.use_default_intrinsics or intrinsic_pred is None:
        if intrinsic_pred is None:
            print(f"  Intrinsics       : модель не вернула — используем типовые nuScenes")
        else:
            print(f"  Intrinsics       : типовые nuScenes (--use_default_intrinsics)")
    else:
        intrinsic = intrinsic_pred.astype(np.float64)
        print(f"  Intrinsics       : предсказаны моделью")
        print(f"  {intrinsic}")

    # Масштабируем intrinsics если depth map имеет другое разрешение
    # (DA3 может изменять размер при инференсе)
    depth_hw = depth_map.shape[:2]
    if depth_hw != _NUSCENES_HW and (args.use_default_intrinsics or intrinsic_pred is None):
        intrinsic = scale_intrinsic(intrinsic, _NUSCENES_HW, depth_hw)
        print(f"  Intrinsics масштабированы: {_NUSCENES_HW} → {depth_hw}  "
              f"(fx={intrinsic[0,0]:.1f}, fy={intrinsic[1,1]:.1f}, "
              f"cx={intrinsic[0,2]:.1f}, cy={intrinsic[1,2]:.1f})")

    # ------------------------------------------------------------------
    # 3. Back-projection
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("Шаг 3/5: Back-projection → camera frame")
    print(sep)

    projector = BackProjector(min_depth=args.min_depth, max_depth=args.max_depth)
    points_cam, valid_mask = projector.depth_to_points_cam(depth_map, intrinsic)

    total   = depth_map.size
    n_valid = valid_mask.sum()
    print(f"  Валидных пикселей : {n_valid:,} / {total:,}  ({100*n_valid/total:.1f}%)")
    print(f"  Точек cam frame   : {len(points_cam):,}")
    if len(points_cam) > 0:
        print(f"  X (вправо) : [{points_cam[:,0].min():.2f}, {points_cam[:,0].max():.2f}] м")
        print(f"  Y (вниз)   : [{points_cam[:,1].min():.2f}, {points_cam[:,1].max():.2f}] м")
        print(f"  Z (вперёд) : [{points_cam[:,2].min():.2f}, {points_cam[:,2].max():.2f}] м")

    # ------------------------------------------------------------------
    # 4. cam frame → ego frame
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("Шаг 4/5: camera frame → ego-vehicle frame")
    print(sep)

    points_ego = projector.cam_to_ego(points_cam, cam2ego)
    print(f"  X (вперёд) : [{points_ego[:,0].min():.2f}, {points_ego[:,0].max():.2f}] м")
    print(f"  Y (влево)  : [{points_ego[:,1].min():.2f}, {points_ego[:,1].max():.2f}] м")
    print(f"  Z (вверх)  : [{points_ego[:,2].min():.2f}, {points_ego[:,2].max():.2f}] м")

    rf = RangeFilter(
        x_range=tuple(args.x_range),
        y_range=tuple(args.y_range),
        z_range=tuple(args.z_range),
    )

    # Показываем сколько точек срезается по каждой оси отдельно
    out_x = ((points_ego[:,0] < args.x_range[0]) | (points_ego[:,0] > args.x_range[1])).sum()
    out_y = ((points_ego[:,1] < args.y_range[0]) | (points_ego[:,1] > args.y_range[1])).sum()
    out_z = ((points_ego[:,2] < args.z_range[0]) | (points_ego[:,2] > args.z_range[1])).sum()
    print(f"  Вне x_range {args.x_range} : {out_x:,} точек")
    print(f"  Вне y_range {args.y_range} : {out_y:,} точек")
    print(f"  Вне z_range {args.z_range} : {out_z:,} точек")

    points_filtered = rf.filter(points_ego)
    removed = len(points_ego) - len(points_filtered)
    print(f"  После range-фильтра: {len(points_filtered):,} точек  (удалено {removed:,})")

    # ------------------------------------------------------------------
    # 5. Сохранение
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("Шаг 5/5: Сохранение")
    print(sep)

    save_bin(points_filtered, output_dir / f"{stem}_pseudo_lidar.bin")
    save_visualization(
        image=image,
        depth=depth_map,
        points_cam=points_cam,
        points_ego=points_filtered,
        intrinsic=intrinsic,
        output_dir=output_dir,
        stem=stem,
    )

    print(f"\n{sep}")
    print("Готово!")
    print(f"  Псевдо-LiDAR точек : {len(points_filtered):,}")
    print(f"  Формат .bin        : float32 XYZI (intensity=0)")
    print(sep)


if __name__ == "__main__":
    main()
