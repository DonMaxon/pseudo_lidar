"""
make_pseudo_pkl_v14.py

v14 = v12 без реальных LiDAR-свипов.

В v12 (и всех предыдущих версиях) PKL содержит sweeps, указывающие на
реальный LiDAR из nuScenes. Модель получает смешанный ввод:
  - keyframe: pseudo-LiDAR (наш)
  - sweeps[0..9]: реальный LiDAR (nuScenes)

v14 обнуляет sweeps → модель получает чистый pseudo-LiDAR:
  - keyframe: pseudo-LiDAR
  - sweeps[]: пусто → pad_empty_sweeps=True дублирует keyframe 10 раз

Цель: честная оценка pseudo-LiDAR без утечки реальных данных.

Запуск:
    python scripts/v14/make_pseudo_pkl_v14.py \
        --v12_pkl output/pseudo_lidar_nuscenes_v12/bevdetv5-nuscenes_infos_val_pseudo_v12.pkl \
        --out_dir output/pseudo_lidar_nuscenes_v14
"""

import argparse
import pickle
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v12_pkl", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_pkl = out_dir / "bevdetv5-nuscenes_infos_val_pseudo_v14.pkl"

    with open(args.v12_pkl, "rb") as f:
        data = pickle.load(f)

    total_sweeps_removed = 0
    for info in data["infos"]:
        info["lidar_path"] = info["lidar_path"].replace(
            "nuscenes_pseudo_v12/", "nuscenes_pseudo_v14/"
        )
        total_sweeps_removed += len(info.get("sweeps", []))
        info["sweeps"] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(data, f)

    print(f"Записано: {out_pkl}")
    print(f"Удалено sweep-записей: {total_sweeps_removed} "
          f"(в среднем {total_sweeps_removed / len(data['infos']):.1f} на семпл)")


if __name__ == "__main__":
    main()
