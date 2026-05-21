"""
visualize_bev_v11.py

Сравнение BEV реального LiDAR и pseudo-LiDAR v11 для всех сцен nuScenes mini.

Ориентация BEV (стандарт autonomous driving):
    ВВЕРХ  = вперёд (X+)
    ВЛЕВО  = левый бок авто (Y+)
    ВПРАВО = правый бок авто (Y-)

Раскладка на картинке (3 ряда):
    Ряд 1: FL | F | FR       ← фронтальные камеры
    Ряд 2: BL | B | BR       ← задние камеры
    Ряд 3: Real LiDAR BEV | Pseudo-LiDAR v11 BEV

Запуск:
    python scripts/visualize_bev_v11.py \\
        --dataroot /home/max/Phd/Phase1/Sparse4D/data/nuscenes \\
        --pseudo_dir ./output/pseudo_lidar_nuscenes \\
        --output_dir ./output/bev_vis_v11 \\
        --n_samples 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BEV_RANGE  = 54.0   # ±м
Z_MIN      = -3.0
Z_MAX      =  5.0
RESOLUTION =  0.1   # м/пиксель → 1080×1080

CAM_ORDER = [
    ("CAM_FRONT_LEFT",  "Front-Left"),
    ("CAM_FRONT",       "Front"),
    ("CAM_FRONT_RIGHT", "Front-Right"),
    ("CAM_BACK_LEFT",   "Back-Left"),
    ("CAM_BACK",        "Back"),
    ("CAM_BACK_RIGHT",  "Back-Right"),
]


def load_bin(path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)


def make_bev_height(points: np.ndarray,
                    bev_range: float = BEV_RANGE,
                    res: float = RESOLUTION) -> np.ndarray:
    """
    Height-coloured BEV image (H, W) float32, NaN = empty cell.

    Ориентация пикселей:
        row 0        = X = +bev_range  (ВПЕРЁД)
        row H-1      = X = -bev_range  (НАЗАД)
        col 0        = Y = +bev_range  (ВЛЕВО)
        col W-1      = Y = -bev_range  (ВПРАВО)
    """
    size = int(2 * bev_range / res)
    mask = (
        (points[:, 0] >= -bev_range) & (points[:, 0] <=  bev_range) &
        (points[:, 1] >= -bev_range) & (points[:, 1] <=  bev_range) &
        (points[:, 2] >= Z_MIN)      & (points[:, 2] <=  Z_MAX)
    )
    pts = points[mask]

    # nuScenes LIDAR_TOP sensor frame: Y+=forward, X-=left, X+=right
    # row: forward (Y+) → top   ↔  row = (bev_range - Y) / res
    # col: left   (X-) → left   ↔  col = (X + bev_range) / res
    row = np.clip(((bev_range - pts[:, 1]) / res).astype(np.int32), 0, size - 1)
    col = np.clip(((pts[:, 0] + bev_range) / res).astype(np.int32), 0, size - 1)

    h_sum = np.zeros((size, size), dtype=np.float64)
    count = np.zeros((size, size), dtype=np.int32)
    np.add.at(h_sum, (row, col), pts[:, 2])
    np.add.at(count, (row, col), 1)

    img = np.where(count > 0, h_sum / np.maximum(count, 1), np.nan).astype(np.float32)
    return img


def draw_bev(ax, img: np.ndarray, title: str, n_pts: int,
             cmap: str = "plasma", bev_range: float = BEV_RANGE):
    """Рисует BEV на оси ax с правильными подписями и аннотациями."""
    nan_mask = np.isnan(img)
    norm = (img - Z_MIN) / (Z_MAX - Z_MIN)
    norm = np.clip(norm, 0, 1)
    rgb = matplotlib.colormaps[cmap](np.nan_to_num(norm, nan=0.0))[:, :, :3]
    rgb[nan_mask] = 0.0

    # imshow: extent = [left, right, bottom, top]
    # В нашей системе:
    #   col=0  → Y=+bev_range → ВЛЕВО  → в «математических» единицах = +bev_range по горизонтали
    #   col=W  → Y=-bev_range → ВПРАВО → -bev_range по горизонтали
    # Поэтому extent по x = [+bev_range, -bev_range] (ось X «перевёрнута»)
    #   row=0  → X=+bev_range → ВПЕРЁД → top
    #   row=H  → X=-bev_range → НАЗАД  → bottom
    # Поэтому extent по y = [-bev_range, +bev_range]
    # extent: x = X axis (left=-54, right=+54), y = Y axis (bottom=-54, top=+54)
    ax.imshow(rgb, origin="upper",
              extent=[-bev_range, bev_range, -bev_range, bev_range],
              aspect="equal")

    # Концентрические окружности
    for r in [10, 20, 30, 40, 50]:
        c = plt.Circle((0, 0), r, color="white", fill=False, lw=0.4, alpha=0.35)
        ax.add_patch(c)
        ax.text(0, r + 0.8, f"{r}m", color="white", fontsize=5,
                ha="center", va="bottom", alpha=0.55)

    # Перекрестие
    ax.axhline(0, color="white", lw=0.4, alpha=0.3)
    ax.axvline(0, color="white", lw=0.4, alpha=0.3)

    # Ego
    ax.plot(0, 0, "w+", markersize=9, markeredgewidth=1.5, zorder=5)

    # Аннотации ориентации
    fs = 8
    ax.text(0,  bev_range * 0.93, "▲ FRONT", color="white",
            fontsize=fs, ha="center", va="top",    fontweight="bold")
    ax.text(0, -bev_range * 0.93, "▼ BACK",  color="white",
            fontsize=fs, ha="center", va="bottom", fontweight="bold")
    ax.text( bev_range * 0.93, 0, "LEFT ►",  color="white",
            fontsize=fs, ha="right",  va="center", fontweight="bold")
    ax.text(-bev_range * 0.93, 0, "◄ RIGHT", color="white",
            fontsize=fs, ha="left",   va="center", fontweight="bold")

    ax.set_xlim(-bev_range,  bev_range)   # лево (X-) → право (X+)
    ax.set_ylim(-bev_range,  bev_range)
    ax.set_xlabel("← Left (X-)  ·  X (m)  ·  Right (X+) →", fontsize=7, color="white")
    ax.set_ylabel("↑ Forward (Y+)  ·  Y (m)  ·  Backward (Y-) ↓", fontsize=7, color="white")
    ax.set_title(f"{title}\n{n_pts:,} pts", fontsize=9, color="white")
    ax.tick_params(colors="white", labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor("white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot",   default="/home/max/Phd/Phase1/Sparse4D/data/nuscenes")
    ap.add_argument("--pseudo_dir", default="./output/pseudo_lidar_nuscenes_v11")
    ap.add_argument("--output_dir", default="./output/bev_vis_v11")
    ap.add_argument("--version",    default="v1.0-mini")
    ap.add_argument("--n_samples",  type=int, default=10)
    ap.add_argument("--resolution", type=float, default=RESOLUTION)
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    outdir     = Path(args.output_dir)
    pseudo_dir = Path(args.pseudo_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # По одному среднему sample из каждой сцены
    tokens = []
    for scene in nusc.scene:
        s = nusc.get("sample", scene["first_sample_token"])
        buf = []
        while True:
            buf.append(s["token"])
            if s["next"] == "":
                break
            s = nusc.get("sample", s["next"])
        tokens.append(buf[len(buf) // 2])
    tokens = tokens[:args.n_samples]

    print(f"Генерация {len(tokens)} BEV-визуализаций → {outdir}")

    for idx, token in enumerate(tokens):
        sample = nusc.get("sample", token)
        scene_name = nusc.get("scene", sample["scene_token"])["name"]

        # --- LiDAR файлы ---
        lidar_sd   = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        real_path  = Path(args.dataroot) / lidar_sd["filename"]
        pseudo_path = pseudo_dir / lidar_sd["filename"]
        real_pts   = load_bin(real_path)
        pseudo_pts = load_bin(pseudo_path)

        # --- BEV изображения ---
        real_bev   = make_bev_height(real_pts,   res=args.resolution)
        pseudo_bev = make_bev_height(pseudo_pts, res=args.resolution)

        # --- Камеры ---
        cam_imgs = {}
        for cam_key, _ in CAM_ORDER:
            cam_sd  = nusc.get("sample_data", sample["data"][cam_key])
            cam_path = Path(args.dataroot) / cam_sd["filename"]
            cam_imgs[cam_key] = np.array(Image.open(cam_path))

        # --- Раскладка: 3 ряда × 6 столбцов ---
        fig = plt.figure(figsize=(24, 14))
        fig.patch.set_facecolor("#111111")

        gs = gridspec.GridSpec(
            3, 6,
            figure=fig,
            hspace=0.12,
            wspace=0.04,
            height_ratios=[1, 1, 2.2],
        )

        # Ряды 0–1: камеры
        front_cams = CAM_ORDER[:3]   # FL, F, FR
        back_cams  = CAM_ORDER[3:]   # BL, B, BR

        for ci, (cam_key, cam_label) in enumerate(front_cams):
            ax = fig.add_subplot(gs[0, ci * 2: ci * 2 + 2])
            ax.imshow(cam_imgs[cam_key])
            ax.set_title(cam_label, fontsize=8, color="white", pad=2)
            ax.axis("off")

        for ci, (cam_key, cam_label) in enumerate(back_cams):
            ax = fig.add_subplot(gs[1, ci * 2: ci * 2 + 2])
            ax.imshow(cam_imgs[cam_key])
            ax.set_title(cam_label, fontsize=8, color="white", pad=2)
            ax.axis("off")

        # Ряд 2: BEV
        ax_real   = fig.add_subplot(gs[2, :3])
        ax_pseudo = fig.add_subplot(gs[2, 3:])
        ax_real.set_facecolor("black")
        ax_pseudo.set_facecolor("black")

        draw_bev(ax_real,   real_bev,
                 title="Real LiDAR  (Velodyne HDL-64E)",
                 n_pts=len(real_pts), cmap="viridis")
        draw_bev(ax_pseudo, pseudo_bev,
                 title="Pseudo-LiDAR v11  (DA3METRIC-LARGE)",
                 n_pts=len(pseudo_pts), cmap="plasma")

        # Colorbar
        sm_r = plt.cm.ScalarMappable(cmap="viridis",
                    norm=plt.Normalize(Z_MIN, Z_MAX))
        sm_p = plt.cm.ScalarMappable(cmap="plasma",
                    norm=plt.Normalize(Z_MIN, Z_MAX))
        for sm, ax in [(sm_r, ax_real), (sm_p, ax_pseudo)]:
            cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
            cb.set_label("Height (m)", color="white", fontsize=7)
            cb.ax.yaxis.set_tick_params(color="white", labelsize=6)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

        fig.suptitle(
            f"{scene_name}  |  token: {token[:16]}…",
            color="white", fontsize=12, y=1.002
        )

        out_path = outdir / f"bev_{idx:02d}_{scene_name}.png"
        plt.savefig(out_path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        print(f"  [{idx+1}/{len(tokens)}] {out_path.name}  "
              f"real={len(real_pts):,}  pseudo={len(pseudo_pts):,}")

    print(f"\nГотово. Картинки: {outdir}")


if __name__ == "__main__":
    main()
