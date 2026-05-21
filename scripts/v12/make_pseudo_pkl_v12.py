"""
make_pseudo_pkl_v12.py

Создаёт bevdetv5-nuscenes_infos_val_pseudo_v12.pkl для BEVDet.

Запуск:
    python scripts/v12/make_pseudo_pkl_v12.py \\
        --v1_pkl  output/pseudo_lidar_nuscenes/bevdetv5-nuscenes_infos_val_pseudo.pkl \\
        --out_dir output/pseudo_lidar_nuscenes_v12
"""

import argparse
import pickle
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1_pkl",  required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    v1_pkl  = Path(args.v1_pkl)
    out_dir = Path(args.out_dir)
    out_pkl = out_dir / "bevdetv5-nuscenes_infos_val_pseudo_v12.pkl"

    with open(v1_pkl, "rb") as f:
        data = pickle.load(f)

    for info in data["infos"]:
        info["lidar_path"] = info["lidar_path"].replace(
            "nuscenes_pseudo/", "nuscenes_pseudo_v12/"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(data, f)
    print(f"Записано: {out_pkl}")


if __name__ == "__main__":
    main()
