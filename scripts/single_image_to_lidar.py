"""
single_image_to_lidar.py

Пошаговый перевод ОДНОЙ картинки nuScenes в псевдо-лидарное облако точек.

Шаги:
    1. Загрузить одно изображение (CAM_FRONT) + intrinsics + extrinsics из nuScenes
    2. Предсказать метрическую карту глубины (Depth Anything V2 Metric)
    3. Back-projection: depth map → 3D точки в системе камеры
    4. Перевести точки в систему ego-vehicle через cam2ego
    5. Применить range-фильтр
    6. Сохранить облако точек в .bin (XYZI float32)
    7. Сохранить визуализацию: оригинал | карта глубины | точки на фото | BEV

Запуск:
    python scripts/single_image_to_lidar.py \
        --dataroot /home/max/Phd/Phase1/Sparse4D/data \
        --version v1.0-mini \
        --camera CAM_FRONT \
        --sample_idx 0 \
        --output_dir ./output
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth2lidar.back_projection import BackProjector
from depth2lidar.nuscenes_loader import NuScenesLoader
from depth2lidar.depth_estimator import DepthEstimator
from depth2lidar.pointcloud_filter import RangeFilter
from depth2lidar.visualize import colorize_depth, project_points_on_image, points_to_bev_image


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Одна картинка nuScenes → псевдо-лидар")
    p.add_argument("--dataroot", default="/home/max/Phd/Phase1/Sparse4D/data",
                   help="Путь к датасету nuScenes (папка с samples/, maps/, ...)")
    p.add_argument("--version", default="v1.0-mini",
                   help="Версия датасета: v1.0-mini | v1.0-trainval")
    p.add_argument("--camera", default="CAM_FRONT",
                   choices=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"],
                   help="Камера для обработки")
    p.add_argument("--sample_idx", type=int, default=0,
                   help="Индекс sample в датасете (0-based)")
    p.add_argument("--output_dir", default="./output",
                   help="Папка для сохранения результатов")
    p.add_argument("--min_depth", type=float, default=0.5,
                   help="Минимальная глубина в метрах")
    p.add_argument("--max_depth", type=float, default=60.0,
                   help="Максимальная глубина в метрах")
    p.add_argument("--x_range", type=float, nargs=2, default=[-51.2, 51.2],
                   metavar=("X_MIN", "X_MAX"), help="BEV диапазон по X (вперёд)")
    p.add_argument("--y_range", type=float, nargs=2, default=[-51.2, 51.2],
                   metavar=("Y_MIN", "Y_MAX"), help="BEV диапазон по Y (влево)")
    p.add_argument("--z_range", type=float, nargs=2, default=[-5.0, 3.0],
                   metavar=("Z_MIN", "Z_MAX"), help="Диапазон высоты (убирает землю/небо)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_bin(points: np.ndarray, path: Path):
    """Сохраняет облако точек в формате float32 XYZI (intensity=0)."""
    intensity = np.zeros((len(points), 1), dtype=np.float32)
    xyzi = np.concatenate([points.astype(np.float32), intensity], axis=1)  # (N, 4)
    xyzi.tofile(str(path))
    size_kb = xyzi.nbytes / 1024
    print(f"    → сохранено: {path}  ({len(points):,} точек, {size_kb:.1f} KB)")


def save_visualization(
    image: np.ndarray,
    depth: np.ndarray,
    points_cam: np.ndarray,
    points_ego: np.ndarray,
    intrinsic: np.ndarray,
    output_dir: Path,
    prefix: str,
):
    """
    Сохраняет 4-панельную визуализацию:
        [оригинал] [карта глубины] [точки на фото] [BEV вид сверху]
    """
    try:
        import cv2
    except ImportError:
        print("    opencv не установлен — визуализация пропущена")
        return

    # Панель 1: оригинальное изображение
    panel_orig = image.copy()

    # Панель 2: colorize depth
    panel_depth = colorize_depth(depth, min_depth=0.5, max_depth=60.0, cmap="plasma")

    # Панель 3: точки спроецированы обратно на изображение
    # Берём каждый 10-й point чтобы не перегружать картинку
    panel_proj = project_points_on_image(
        image=image,
        points_cam=points_cam[::10],
        intrinsic=intrinsic,
        point_size=1,
        cmap="plasma",
        max_depth=60.0,
    )

    # Панель 4: BEV (вид сверху, ego frame)
    bev_gray = points_to_bev_image(
        points=points_ego,
        x_range=(-51.2, 51.2),
        y_range=(-51.2, 51.2),
        resolution=0.1,
    )
    # BEV в RGB, масштабируем до высоты изображения
    bev_rgb = np.stack([bev_gray] * 3, axis=-1)
    h_img = panel_orig.shape[0]
    bev_h, bev_w = bev_rgb.shape[:2]
    bev_scale = h_img / bev_h
    bev_rgb = cv2.resize(bev_rgb, (int(bev_w * bev_scale), h_img))

    # Выравниваем высоту всех панелей
    target_h = panel_orig.shape[0]
    target_w = panel_orig.shape[1]

    def resize_to(img, h, w):
        return cv2.resize(img, (w, h))

    panels = [
        resize_to(panel_orig,  target_h, target_w),
        resize_to(panel_depth, target_h, target_w),
        resize_to(panel_proj,  target_h, target_w),
        resize_to(bev_rgb,     target_h, target_w),
    ]

    # Подписи панелей
    labels = ["Original", "Depth map", "Points on image", "BEV (top-down)"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (8, 28), font, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

    # Собираем в одну строку
    grid = np.concatenate(panels, axis=1)

    vis_path = output_dir / f"{prefix}_visualization.jpg"
    cv2.imwrite(str(vis_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"    → визуализация: {vis_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    separator = "=" * 60

    # ------------------------------------------------------------------
    # Шаг 1: Загрузка данных nuScenes
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("Шаг 1/6: Загрузка данных nuScenes")
    print(separator)

    loader = NuScenesLoader(
        dataroot=args.dataroot,
        version=args.version,
        cameras=[args.camera],  # загружаем только нужную камеру
    )

    tokens = loader.get_sample_tokens()
    if args.sample_idx >= len(tokens):
        raise IndexError(
            f"sample_idx={args.sample_idx} выходит за пределы ({len(tokens)} samples)"
        )

    sample_token = tokens[args.sample_idx]
    print(f"  Sample token : {sample_token}")
    print(f"  Камера       : {args.camera}")
    print(f"  Sample idx   : {args.sample_idx} / {len(tokens) - 1}")

    cam_data_dict = loader.load_sample(sample_token)
    cam = cam_data_dict[args.camera]

    print(f"  Изображение  : {cam.image.shape}  dtype={cam.image.dtype}")
    print(f"  Intrinsic    :\n{cam.intrinsic}")
    print(f"  cam2ego      :\n{cam.cam2ego}")

    # ------------------------------------------------------------------
    # Шаг 2: Предсказание карты глубины
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("Шаг 2/6: Предсказание метрической карты глубины")
    print(separator)

    estimator = DepthEstimator(max_depth=args.max_depth)
    depth_map = estimator.predict(cam.image)

    print(f"  Размер depth map : {depth_map.shape}")
    print(f"  Диапазон глубины : [{depth_map.min():.2f}, {depth_map.max():.2f}] м")
    print(f"  Средняя глубина  : {depth_map.mean():.2f} м")

    # Сохраняем depth map как PNG (16-bit, масштаб x256 для визуального контроля)
    depth_png_path = output_dir / f"{sample_token[:8]}_{args.camera}_depth.png"
    depth_u16 = (depth_map * 256).clip(0, 65535).astype(np.uint16)
    Image.fromarray(depth_u16).save(str(depth_png_path))
    print(f"  → depth PNG сохранён: {depth_png_path}")

    # ------------------------------------------------------------------
    # Шаг 3: Back-projection → точки в системе камеры
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("Шаг 3/6: Back-projection: depth map → 3D точки (camera frame)")
    print(separator)

    projector = BackProjector(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    points_cam, valid_mask = projector.depth_to_points_cam(
        depth_map=depth_map,
        intrinsic=cam.intrinsic,
    )

    total_pixels = depth_map.size
    valid_pixels = valid_mask.sum()
    print(f"  Пикселей всего    : {total_pixels:,}")
    print(f"  Валидных пикселей : {valid_pixels:,}  ({100*valid_pixels/total_pixels:.1f}%)")
    print(f"  Точек camera frame: {len(points_cam):,}")
    print(f"  X range: [{points_cam[:,0].min():.2f}, {points_cam[:,0].max():.2f}] м")
    print(f"  Y range: [{points_cam[:,1].min():.2f}, {points_cam[:,1].max():.2f}] м")
    print(f"  Z range: [{points_cam[:,2].min():.2f}, {points_cam[:,2].max():.2f}] м")

    # ------------------------------------------------------------------
    # Шаг 4: camera frame → ego-vehicle frame
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("Шаг 4/6: Перевод camera frame → ego-vehicle frame")
    print(separator)

    points_ego = projector.cam_to_ego(
        points_cam=points_cam,
        cam2ego=cam.cam2ego,
    )

    print(f"  Точек ego frame: {len(points_ego):,}")
    print(f"  X range: [{points_ego[:,0].min():.2f}, {points_ego[:,0].max():.2f}] м  (вперёд)")
    print(f"  Y range: [{points_ego[:,1].min():.2f}, {points_ego[:,1].max():.2f}] м  (влево)")
    print(f"  Z range: [{points_ego[:,2].min():.2f}, {points_ego[:,2].max():.2f}] м  (вверх)")

    # ------------------------------------------------------------------
    # Шаг 5: Range-фильтр
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("Шаг 5/6: Фильтрация по диапазону (BEV range + высота)")
    print(separator)

    range_filter = RangeFilter(
        x_range=tuple(args.x_range),
        y_range=tuple(args.y_range),
        z_range=tuple(args.z_range),
    )
    points_ego_filtered = range_filter.filter(points_ego)

    removed = len(points_ego) - len(points_ego_filtered)
    print(f"  До фильтра  : {len(points_ego):,} точек")
    print(f"  После фильтра: {len(points_ego_filtered):,} точек")
    print(f"  Удалено     : {removed:,}  ({100*removed/max(len(points_ego),1):.1f}%)")
    print(f"  X range    : {args.x_range}")
    print(f"  Y range    : {args.y_range}")
    print(f"  Z range    : {args.z_range}")

    # ------------------------------------------------------------------
    # Шаг 6: Сохранение .bin + визуализация
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("Шаг 6/6: Сохранение результатов")
    print(separator)

    prefix = f"{sample_token[:8]}_{args.camera}"

    # .bin файл
    bin_path = output_dir / f"{prefix}_pseudo_lidar.bin"
    save_bin(points_ego_filtered, bin_path)

    # Визуализация
    save_visualization(
        image=cam.image,
        depth=depth_map,
        points_cam=points_cam,
        points_ego=points_ego_filtered,
        intrinsic=cam.intrinsic,
        output_dir=output_dir,
        prefix=prefix,
    )

    # Итог
    print(f"\n{separator}")
    print("Готово!")
    print(separator)
    print(f"  Исходное изображение : {cam.image.shape[1]}x{cam.image.shape[0]} px")
    print(f"  Карта глубины        : {depth_map.shape[1]}x{depth_map.shape[0]} px")
    print(f"  Псевдо-LiDAR точек   : {len(points_ego_filtered):,}")
    print(f"  Формат .bin          : float32 XYZI  (intensity=0)")
    print(f"  Совместим с          : CenterPoint, BEVFusion, PointPillars")
    print()


if __name__ == "__main__":
    main()
