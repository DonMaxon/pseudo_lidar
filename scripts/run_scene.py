"""
run_scene.py

Process all samples of a nuScenes scene sequentially.
For each sample produces a .bin pseudo-LiDAR file.

Usage:
    python scripts/run_scene.py \
        --dataroot /home/max/Phd/Phase1/Sparse4D/data \
        --version v1.0-mini \
        --scene_name scene-0001 \
        --output_dir ./output/scene-0001 \
        --voxel_size 0.1
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth2lidar import BackProjector, NuScenesLoader, DepthEstimator, PointCloudFilter


def parse_args():
    parser = argparse.ArgumentParser(description="Process a full nuScenes scene")
    parser.add_argument("--dataroot", type=str,
                        default="/home/max/Phd/Phase1/Sparse4D/data")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--scene_name", type=str, default=None,
                        help="Scene name, e.g. 'scene-0001'. If None, processes all scenes.")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=60.0)
    parser.add_argument("--align_lidar", action="store_true")
    return parser.parse_args()


def get_scene_sample_tokens(nusc, scene_name: str | None) -> dict[str, list[str]]:
    """Return {scene_name: [sample_token, ...]} for the requested scene(s)."""
    result = {}
    for scene in nusc.scene:
        if scene_name is not None and scene["name"] != scene_name:
            continue

        tokens = []
        sample_token = scene["first_sample_token"]
        while sample_token:
            tokens.append(sample_token)
            sample = nusc.get("sample", sample_token)
            sample_token = sample["next"]
        result[scene["name"]] = tokens

    return result


def process_sample(
    token: str,
    loader: NuScenesLoader,
    estimator: DepthEstimator,
    back_proj: BackProjector,
    pcloud_filter: PointCloudFilter,
    voxel_size: float,
    align_lidar: bool,
    output_dir: Path,
):
    cam_data_dict = loader.load_sample(token)
    all_points_ego = []

    for cam_name, cam in cam_data_dict.items():
        depth = estimator.predict(cam.image)

        if align_lidar:
            depth_sparse = loader.get_sparse_lidar_depth(token, cam_name)
            depth = DepthEstimator.align_scale_shift(depth, depth_sparse)

        points_ego, _ = back_proj.depth_to_ego(
            depth_map=depth,
            intrinsic=cam.intrinsic,
            cam2ego=cam.cam2ego,
        )
        points_ego = pcloud_filter.range_filter.filter(points_ego)
        all_points_ego.append(points_ego)

    merged = np.concatenate(all_points_ego, axis=0)
    downsampled = PointCloudFilter.voxel_downsample(merged, voxel_size=voxel_size)

    intensity = np.zeros((len(downsampled), 1), dtype=np.float32)
    pseudo_lidar = np.concatenate([downsampled.astype(np.float32), intensity], axis=1)

    out_path = output_dir / f"{token}.bin"
    pseudo_lidar.tofile(str(out_path))
    return len(downsampled)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = NuScenesLoader(dataroot=args.dataroot, version=args.version)
    estimator = DepthEstimator(max_depth=args.max_depth)
    back_proj = BackProjector(min_depth=0.5, max_depth=args.max_depth)
    pcloud_filter = PointCloudFilter()

    scene_tokens = get_scene_sample_tokens(loader.nusc, args.scene_name)

    total_samples = sum(len(v) for v in scene_tokens.values())
    print(f"Scenes to process: {len(scene_tokens)}, total samples: {total_samples}")

    processed = 0
    for scene_name, tokens in scene_tokens.items():
        scene_out = output_dir / scene_name
        scene_out.mkdir(exist_ok=True)
        print(f"\n=== Scene: {scene_name} ({len(tokens)} samples) ===")

        for i, token in enumerate(tokens):
            n_pts = process_sample(
                token, loader, estimator, back_proj,
                pcloud_filter, args.voxel_size, args.align_lidar, scene_out,
            )
            processed += 1
            print(f"  [{processed}/{total_samples}] {token[:8]}... → {n_pts:,} pts → {scene_out/token}.bin")

    print(f"\nAll done. Processed {processed} samples.")


if __name__ == "__main__":
    main()
