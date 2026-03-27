"""
make_pseudo_pkl_v2.py

Создаёт bevdetv5-nuscenes_infos_val_pseudo_v2.pkl для BEVDet,
заменяя пути к LiDAR на пути к v2 pseudo-LiDAR.

Использует существующий v1 pkl как источник аннотаций,
заменяет только lidar_path на nuscenes_pseudo_v2.

Запуск:
    python scripts/v2/make_pseudo_pkl_v2.py \\
        --v1_pkl  output/pseudo_lidar_nuscenes/bevdetv5-nuscenes_infos_val_pseudo.pkl \\
        --out_dir output/pseudo_lidar_nuscenes_v2
"""

import argparse
import pickle
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1_pkl",  required=True, help="Путь к существующему v1 pkl")
    p.add_argument("--out_dir", required=True, help="Директория вывода v2")
    args = p.parse_args()

    v1_pkl  = Path(args.v1_pkl)
    out_dir = Path(args.out_dir)
    out_pkl = out_dir / "bevdetv5-nuscenes_infos_val_pseudo_v2.pkl"

    print(f"Читаем {v1_pkl} ...")
    with open(v1_pkl, "rb") as f:
        data = pickle.load(f)

    infos = data["infos"]
    print(f"  {len(infos)} записей")

    for info in infos:
        # nuscenes_pseudo → nuscenes_pseudo_v2
        info["lidar_path"] = info["lidar_path"].replace(
            "nuscenes_pseudo/", "nuscenes_pseudo_v2/"
        )

    print(f"Записываем {out_pkl} ...")
    with open(out_pkl, "wb") as f:
        pickle.dump(data, f)
    print("Готово.")


if __name__ == "__main__":
    main()
