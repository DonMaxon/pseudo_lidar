"""
compute_scale_stats.py

Считает scale и shift для выравнивания DA3METRIC по GT LiDAR
на nuScenes mini. Результаты — медиана и std по всем sample'ам,
отдельно для каждой камеры и глобально.

Запуск:
    python scripts/compute_scale_stats.py \
        --dataroot /home/max/Phd/Phase1/Sparse4D/data/nuscenes \
        --version v1.0-mini
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth2lidar.depth_estimator import DepthEstimator
from depth2lidar.nuscenes_loader import NuScenesLoader, NUSCENES_CAMERAS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", default="/home/max/Phd/Phase1/Sparse4D/data/nuscenes")
    p.add_argument("--version",  default="v1.0-mini")
    p.add_argument("--max_depth", type=float, default=60.0)
    p.add_argument("--min_valid", type=int,   default=100,
                   help="Минимум GT-пикселей для включения sample в статистику")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Ограничить число sample'ов (для быстрого теста)")
    p.add_argument("--device", default="auto",
                   help="Устройство для инференса: 'cuda', 'cpu', 'auto'")
    p.add_argument("--out", default="scale_stats.npz",
                   help="Куда сохранить результаты")
    return p.parse_args()


def main():
    args = parse_args()

    loader    = NuScenesLoader(dataroot=args.dataroot, version=args.version)
    estimator = DepthEstimator(max_depth=args.max_depth, device=args.device)

    # scale_i, shift_i по каждой камере
    per_cam_scales: dict[str, list[float]] = defaultdict(list)
    per_cam_shifts: dict[str, list[float]] = defaultdict(list)

    tokens = loader.get_sample_tokens()
    if args.max_samples:
        tokens = tokens[: args.max_samples]

    n = len(tokens)
    for i, token in enumerate(tokens):
        print(f"[{i+1:3d}/{n}] {token[:12]}...")
        cam_data_dict = loader.load_sample(token)

        for cam_name, cam in cam_data_dict.items():
            # 1. Предсказать глубину
            depth_pred = estimator.predict(cam.image, cam.intrinsic)

            # 2. GT sparse depth (проекция реального лидара)
            depth_sparse = loader.get_sparse_lidar_depth(token, cam_name)

            # Привести к одному размеру если нужно
            if depth_sparse.shape != depth_pred.shape:
                import cv2
                depth_sparse = cv2.resize(
                    depth_sparse, (depth_pred.shape[1], depth_pred.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # 3. Считаем scale и shift через lstsq
            valid = (depth_sparse > 0.1) & (depth_pred > 0.1)
            if valid.sum() < args.min_valid:
                print(f"    {cam_name}: пропуск (только {valid.sum()} GT-пикселей)")
                continue

            p = depth_pred[valid]
            g = depth_sparse[valid]
            A = np.stack([p, np.ones_like(p)], axis=1)
            result, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
            scale = float(np.clip(result[0], 0.1, 10.0))
            shift = float(result[1])

            per_cam_scales[cam_name].append(scale)
            per_cam_shifts[cam_name].append(shift)

            print(f"    {cam_name}: scale={scale:.4f}  shift={shift:.4f}  "
                  f"(GT-пикс: {valid.sum()})")

    # ── Статистика ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Камера':<20} {'N':>4}  {'scale median':>12}  {'scale std':>9}  "
          f"{'shift median':>12}  {'shift std':>9}")
    print("-" * 65)

    all_scales, all_shifts = [], []
    out_dict = {}

    for cam in NUSCENES_CAMERAS:
        sc = per_cam_scales.get(cam, [])
        sh = per_cam_shifts.get(cam, [])
        if not sc:
            print(f"{cam:<20} {'—':>4}")
            continue
        sc_arr = np.array(sc)
        sh_arr = np.array(sh)
        all_scales.extend(sc)
        all_shifts.extend(sh)
        out_dict[f"{cam}_scales"] = sc_arr
        out_dict[f"{cam}_shifts"] = sh_arr
        print(f"{cam:<20} {len(sc):>4}  "
              f"{np.median(sc_arr):>12.4f}  {sc_arr.std():>9.4f}  "
              f"{np.median(sh_arr):>12.4f}  {sh_arr.std():>9.4f}")

    print("=" * 65)
    all_sc = np.array(all_scales)
    all_sh = np.array(all_shifts)
    print(f"{'GLOBAL':<20} {len(all_sc):>4}  "
          f"{np.median(all_sc):>12.4f}  {all_sc.std():>9.4f}  "
          f"{np.median(all_sh):>12.4f}  {all_sh.std():>9.4f}")
    print("=" * 65)

    out_dict["global_scale_median"] = float(np.median(all_sc))
    out_dict["global_shift_median"] = float(np.median(all_sh))
    out_dict["global_scale_std"]    = float(all_sc.std())
    out_dict["global_shift_std"]    = float(all_sh.std())

    np.savez(args.out, **out_dict)
    print(f"\nСтатистика сохранена: {args.out}")
    print(f"\nДля применения на test:\n"
          f"  depth_aligned = {np.median(all_sc):.4f} * depth_pred + {np.median(all_sh):.4f}")


if __name__ == "__main__":
    main()
