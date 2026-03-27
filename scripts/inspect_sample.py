"""
inspect_sample.py

Определяет рекомендуемые параметры depth2lidar БЕЗ реального LiDAR.
Использует только то, что доступно в инференсе:
    1. Калибровка камеры из метаданных nuScenes (intrinsics + extrinsics)
       — это не сенсорные данные, а статические параметры платформы,
         доступны всегда, даже без LiDAR.
    2. Предсказание depth-модели на реальной картинке.
    3. Back-projection предсказанного depth → анализ распределения точек.

Запуск:
    python scripts/inspect_sample.py \
        --dataroot /home/max/Phd/Phase1/Sparse4D/data \
        --version v1.0-mini \
        --camera CAM_FRONT \
        --sample_idx 0 \
        --save_hist ./output/inspect.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nuscenes.nuscenes import NuScenes
from depth2lidar.back_projection import BackProjector
from depth2lidar.depth_estimator import DepthEstimator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Инспектор параметров depth2lidar (без LiDAR)"
    )
    p.add_argument("--dataroot", default="/home/max/Phd/Phase1/Sparse4D/data")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--camera", default="CAM_FRONT",
                   choices=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"])
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--save_hist", default=None,
                   help="Путь для сохранения PNG с графиками (опционально)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def percentile_table(values: np.ndarray, name: str, unit: str = "м"):
    pcts = [1, 5, 25, 50, 75, 95, 99]
    vals = np.percentile(values, pcts)
    print(f"\n  Перцентили {name}:")
    print(f"  {'%':>5}  {'значение':>10}")
    print(f"  {'-'*20}")
    for pct, v in zip(pcts, vals):
        print(f"  {pct:>5}%  {v:>9.2f} {unit}")


def section(title: str):
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)


def recommend(label: str, value, comment: str = ""):
    c = f"  # {comment}" if comment else ""
    print(f"    --{label:<12} {str(value):<22}{c}")


# ---------------------------------------------------------------------------
# Геометрические расчёты из калибровки (без LiDAR)
# ---------------------------------------------------------------------------

def fov_from_intrinsic(K: np.ndarray, img_w: int, img_h: int):
    """Углы обзора по горизонтали и вертикали."""
    hfov = 2 * np.degrees(np.arctan(img_w / (2 * K[0, 0])))
    vfov = 2 * np.degrees(np.arctan(img_h / (2 * K[1, 1])))
    return hfov, vfov


def visible_xy_at_depth(K: np.ndarray, img_w: int, img_h: int, depth: float):
    """
    Максимальная ширина и высота видимой зоны в метрах на заданной глубине.
    Показывает, какой диапазон X/Y покрывает камера при данном max_depth.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_left  = -(cx / fx) * depth
    x_right = ((img_w - cx) / fx) * depth
    y_top   = -(cy / fy) * depth
    y_bot   = ((img_h - cy) / fy) * depth
    return x_left, x_right, y_top, y_bot


def ground_plane_z(cam_translation: list, cam_rotation_q: list) -> float:
    """
    Оценивает Z земли в ego frame по позиции и ориентации камеры.
    Камера смотрит немного вниз → луч вниз пересекает Z=0 (уровень земли).
    Возвращает Z ego-frame для точки на земле прямо под камерой.
    ego-origin нуScenes — это центр задней оси автомобиля на уровне земли,
    поэтому Z земли ≈ 0, а Z камеры = cam_translation[2].
    """
    # В nuScenes ego origin уже на уровне земли (Z=0 = земля).
    # Камера находится на высоте cam_translation[2] над землёй.
    return float(cam_translation[2])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Загрузка метаданных nuScenes (только калибровка — без LiDAR)
    # ------------------------------------------------------------------
    section("Загрузка калибровки камеры из nuScenes")

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    sample_rec  = nusc.sample[args.sample_idx]
    sample_token = sample_rec["token"]

    cam_token = sample_rec["data"][args.camera]
    cam_sd    = nusc.get("sample_data", cam_token)
    cs        = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])

    K        = np.array(cs["camera_intrinsic"], dtype=np.float64)
    cam_t    = cs["translation"]   # [x, y, z] позиция камеры в ego frame
    cam_q    = cs["rotation"]      # [w, x, y, z] ориентация камеры
    cam2ego  = BackProjector.build_transform(np.array(cam_q), np.array(cam_t))

    img_h, img_w = cam_sd["height"], cam_sd["width"]
    hfov, vfov   = fov_from_intrinsic(K, img_w, img_h)
    cam_height   = cam_t[2]        # высота камеры над ego-origin (= над землёй)

    print(f"\n  Камера        : {args.camera}")
    print(f"  Разрешение    : {img_w} x {img_h} px")
    print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print(f"  Горизонтальный FoV : {hfov:.1f}°")
    print(f"  Вертикальный FoV   : {vfov:.1f}°")
    print(f"  Высота камеры над землёй : {cam_height:.3f} м  (из extrinsics)")
    print(f"  Позиция в ego frame: X={cam_t[0]:.3f} Y={cam_t[1]:.3f} Z={cam_t[2]:.3f} м")

    # Что видит камера геометрически при разных глубинах
    print(f"\n  Охват камеры в плоскости XY (camera frame) при разных глубинах:")
    print(f"  {'Глубина':>10}  {'X влево':>10}  {'X вправо':>10}  {'Y вверх':>10}  {'Y вниз':>10}")
    print(f"  {'-'*55}")
    for d in [10, 20, 30, 50, 80]:
        xl, xr, yt, yb = visible_xy_at_depth(K, img_w, img_h, d)
        print(f"  {d:>8} м  {xl:>10.1f}  {xr:>10.1f}  {yt:>10.1f}  {yb:>10.1f}")

    # ------------------------------------------------------------------
    # Загрузка изображения + предсказание глубины
    # ------------------------------------------------------------------
    section("Предсказание глубины (Depth Anything V2 Metric)")

    img_path = Path(args.dataroot) / cam_sd["filename"]
    image    = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

    estimator  = DepthEstimator(max_depth=200.0)   # намеренно широкий диапазон
    depth_pred = estimator.predict(image)           # (H, W) float32, метры

    valid_px = depth_pred > 0.05
    depth_valid = depth_pred[valid_px]

    print(f"\n  Статистика предсказанной глубины:")
    print(f"  Пикселей всего   : {depth_pred.size:,}")
    print(f"  Валидных пикселей: {valid_px.sum():,}  ({100*valid_px.mean():.1f}%)")
    print(f"  min = {depth_valid.min():.2f} м")
    print(f"  max = {depth_valid.max():.2f} м")
    print(f"  mean= {depth_valid.mean():.2f} м")
    percentile_table(depth_valid, "предсказанной глубины")

    # ------------------------------------------------------------------
    # Back-projection → анализ 3D точек в ego frame
    # ------------------------------------------------------------------
    section("Back-projection: анализ 3D точек в ego frame")

    proj = BackProjector(min_depth=0.1, max_depth=200.0)
    points_ego, _ = proj.depth_to_ego(depth_pred, K, cam2ego)

    n = len(points_ego)
    print(f"\n  Точек после back-projection: {n:,}")

    dist_bev = np.sqrt(points_ego[:, 0]**2 + points_ego[:, 1]**2)

    print(f"\n  Диапазоны в ego frame:")
    for axis, label in enumerate(["X (вперёд)", "Y (влево) ", "Z (вверх) "]):
        mn  = points_ego[:, axis].min()
        mx  = points_ego[:, axis].max()
        med = np.median(points_ego[:, axis])
        p5  = np.percentile(points_ego[:, axis], 5)
        p95 = np.percentile(points_ego[:, axis], 95)
        print(f"    {label}: min={mn:7.2f}  p5={p5:7.2f}  median={med:7.2f}"
              f"  p95={p95:7.2f}  max={mx:7.2f} м")

    percentile_table(dist_bev, "BEV дальности (sqrt(X²+Y²))")
    percentile_table(points_ego[:, 2], "Z (высота в ego frame)")

    # Распределение Z с ASCII-баром
    print(f"\n  Распределение точек по высоте Z (ego frame):")
    z_vals  = points_ego[:, 2]
    z_range = np.linspace(z_vals.min(), z_vals.max(), 12)
    for i in range(len(z_range) - 1):
        lo, hi  = z_range[i], z_range[i + 1]
        cnt     = ((z_vals >= lo) & (z_vals < hi)).sum()
        bar_len = int(40 * cnt / n)
        print(f"    [{lo:6.2f}, {hi:6.2f}) м : {cnt:7,} ({100*cnt/n:5.1f}%)  {'█'*bar_len}")

    # ------------------------------------------------------------------
    # Графики (опционально)
    # ------------------------------------------------------------------
    if args.save_hist:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(18, 9))
            fig.suptitle(
                f"Depth inspection — {args.camera}, sample_idx={args.sample_idx}\n"
                f"(только камера, без LiDAR)",
                fontsize=12,
            )

            # 1: RGB изображение
            axes[0, 0].imshow(image)
            axes[0, 0].set_title("RGB изображение")
            axes[0, 0].axis("off")

            # 2: Карта глубины
            dm = axes[0, 1].imshow(depth_pred, cmap="plasma",
                                    vmin=np.percentile(depth_valid, 1),
                                    vmax=np.percentile(depth_valid, 99))
            plt.colorbar(dm, ax=axes[0, 1], label="м")
            axes[0, 1].set_title("Предсказанная глубина")
            axes[0, 1].axis("off")

            # 3: Гистограмма глубин
            axes[0, 2].hist(depth_valid, bins=100, color="steelblue", edgecolor="none")
            for pct, color in [(95, "orange"), (99, "red")]:
                v = np.percentile(depth_valid, pct)
                axes[0, 2].axvline(v, color=color, linestyle="--", label=f"p{pct}={v:.1f}м")
            axes[0, 2].set_title("Гистограмма предсказанной глубины")
            axes[0, 2].set_xlabel("Глубина, м")
            axes[0, 2].set_ylabel("Пикселей")
            axes[0, 2].legend()

            # 4: Гистограмма BEV дальности
            axes[1, 0].hist(dist_bev, bins=100, color="seagreen", edgecolor="none")
            for pct, color in [(95, "orange"), (99, "red")]:
                v = np.percentile(dist_bev, pct)
                axes[1, 0].axvline(v, color=color, linestyle="--", label=f"p{pct}={v:.1f}м")
            axes[1, 0].set_title("BEV дальность (ego frame)")
            axes[1, 0].set_xlabel("sqrt(X²+Y²), м")
            axes[1, 0].set_ylabel("Точек")
            axes[1, 0].legend()

            # 5: Гистограмма Z
            axes[1, 1].hist(points_ego[:, 2], bins=100, color="salmon", edgecolor="none")
            axes[1, 1].axvline(0, color="black", linestyle="-", linewidth=1, label="земля (Z=0)")
            axes[1, 1].axvline(cam_height, color="blue", linestyle="--",
                               label=f"камера Z={cam_height:.2f}м")
            axes[1, 1].set_title("Высота Z (ego frame)")
            axes[1, 1].set_xlabel("Z, м")
            axes[1, 1].set_ylabel("Точек")
            axes[1, 1].legend()

            # 6: BEV scatter
            idx = np.random.choice(n, min(n, 30000), replace=False)
            sc = axes[1, 2].scatter(
                points_ego[idx, 0], points_ego[idx, 1],
                s=0.2, c=points_ego[idx, 2], cmap="RdYlGn",
                vmin=np.percentile(points_ego[:, 2], 5),
                vmax=np.percentile(points_ego[:, 2], 95),
            )
            plt.colorbar(sc, ax=axes[1, 2], label="Z, м")
            axes[1, 2].set_title("BEV вид (X=вперёд, Y=влево, цвет=Z)")
            axes[1, 2].set_xlabel("X, м")
            axes[1, 2].set_ylabel("Y, м")
            axes[1, 2].set_aspect("equal")
            axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()
            Path(args.save_hist).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.save_hist, dpi=120)
            print(f"\n  Графики сохранены: {args.save_hist}")
        except ImportError:
            print("\n  matplotlib не установлен — графики пропущены")

    # ------------------------------------------------------------------
    # Рекомендации параметров (выведены из данных камеры)
    # ------------------------------------------------------------------
    section("РЕКОМЕНДУЕМЫЕ ПАРАМЕТРЫ  (только из камеры)")

    # max_depth: p99 предсказанной глубины (99% пикселей попадут)
    max_depth_rec = float(np.ceil(np.percentile(depth_valid, 99) / 5) * 5)  # округление до 5м

    # min_depth: p1 предсказанной глубины (убирает артефакты вблизи)
    min_depth_rec = float(np.floor(np.percentile(depth_valid, 1) * 10) / 10)
    min_depth_rec = max(min_depth_rec, 0.5)

    # x_range / y_range: что камера видит в BEV на max_depth
    # берём p99 BEV дальности с небольшим запасом
    xy_rec = float(np.ceil(np.percentile(dist_bev, 99) / 5) * 5)

    # z_range:
    #   - нижняя граница: земля = -(cam_height) относительно камеры → в ego frame ≈ 0
    #     но из back-projection земля может уходить ниже из-за наклона камеры
    #     берём p1 Z с небольшим запасом
    z_min_rec = float(np.floor(np.percentile(points_ego[:, 2], 1) * 2) / 2)
    #   - верхняя граница: p99 Z (всё выше — скорее всего небо и шум)
    z_max_rec = float(np.ceil(np.percentile(points_ego[:, 2], 99) * 2) / 2)

    print()
    print(f"  Источники рекомендаций:")
    print(f"    max_depth   ← p99 предсказанной глубины на этой картинке")
    print(f"    min_depth   ← p1  предсказанной глубины (убирает артефакты вблизи)")
    print(f"    x/y_range   ← p99 BEV дальности после back-projection")
    print(f"    z_range     ← p1..p99 Z ego frame (убирает землю-артефакты и небо)")
    print(f"    высота камеры = {cam_height:.3f} м  (из extrinsics, не из LiDAR)")
    print()

    recommend("min_depth",  f"{min_depth_rec:.1f}")
    recommend("max_depth",  f"{max_depth_rec:.0f}")
    recommend("x_range",    f"-{xy_rec:.0f} {xy_rec:.0f}")
    recommend("y_range",    f"-{xy_rec:.0f} {xy_rec:.0f}")
    recommend("z_range",    f"{z_min_rec:.1f} {z_max_rec:.1f}")

    print(f"""
  Готовая команда:

    python scripts/single_image_to_lidar.py \\
        --dataroot {args.dataroot} \\
        --version {args.version} \\
        --camera {args.camera} \\
        --sample_idx {args.sample_idx} \\
        --min_depth {min_depth_rec:.1f} \\
        --max_depth {max_depth_rec:.0f} \\
        --x_range -{xy_rec:.0f} {xy_rec:.0f} \\
        --y_range -{xy_rec:.0f} {xy_rec:.0f} \\
        --z_range {z_min_rec:.1f} {z_max_rec:.1f} \\
        --output_dir ./output
    """)

    print("  Примечания:")
    print(f"  • Параметры получены ТОЛЬКО из камеры — без LiDAR")
    print(f"  • z_range=[{z_min_rec}, {z_max_rec}] — типично под nuScenes:")
    print(f"      нижний порог убирает артефакты земли под камерой")
    print(f"      верхний порог убирает небо и далёкие стены")
    print(f"  • Если depth-модель на вашей картинке занижает/завышает глубину,")
    print(f"    запустите повторно на нескольких sample_idx и усредните значения")
    print()


if __name__ == "__main__":
    main()
