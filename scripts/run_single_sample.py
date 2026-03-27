"""
run_single_sample.py

End-to-end pipeline for ONE nuScenes sample:
    1. Load 6-camera images + calibration
    2. Predict metric depth per camera (Depth Anything V2 Metric)
    3. (Optional) Align depth scale to sparse LiDAR GT
    4. Back-project each depth map → camera-frame → ego-frame point cloud
    5. Filter by range, optionally by object masks
    6. Merge all cameras into one pseudo-LiDAR cloud
    7. Voxel downsample
    8. Save as .bin (float32 XYZI, intensity=0) compatible with BEVFusion/CenterPoint
    9. Save visualization

Usage:
    python scripts/run_single_sample.py \
        --dataroot /home/max/Phd/Phase1/Sparse4D/data \
        --version v1.0-mini \
        --sample_idx 0 \
        --output_dir ./output \
        --align_lidar
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add parent dir to path so we can import depth2lidar package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth2lidar import BackProjector, NuScenesLoader, DepthEstimator, PointCloudFilter, LidarComparator
from depth2lidar.visualize import make_summary_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Depth2LiDAR single sample pipeline")
    parser.add_argument("--dataroot", type=str,
                        default="/home/max/Phd/Phase1/Sparse4D/data",
                        help="Path to nuScenes dataset root")
    parser.add_argument("--version", type=str, default="v1.0-mini",
                        help="nuScenes version string")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of the sample to process")
    parser.add_argument("--sample_token", type=str, default=None,
                        help="nuScenes sample token (overrides --sample_idx)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save results")
    parser.add_argument("--align_lidar", action="store_true",
                        help="Align predicted depth scale to sparse LiDAR GT")
    parser.add_argument("--voxel_size", type=float, default=0.1,
                        help="Voxel size for downsampling (meters)")
    parser.add_argument("--max_depth", type=float, default=60.0,
                        help="Maximum depth in meters")
    parser.add_argument("--no_depth_model", action="store_true",
                        help="Skip depth model, use LiDAR projection as depth (debug mode)")
    parser.add_argument("--compare_lidar", action="store_true",
                        help="Compare pseudo-LiDAR to GT nuScenes LiDAR: "
                             "Chamfer Distance, Precision/Recall/F-score, "
                             "depth metrics, BEV visualisation")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load nuScenes sample
    # ------------------------------------------------------------------
    loader = NuScenesLoader(
        dataroot=args.dataroot,
        version=args.version,
    )

    tokens = loader.get_sample_tokens()
    if args.sample_token:
        token = args.sample_token
    else:
        if args.sample_idx >= len(tokens):
            raise IndexError(f"sample_idx {args.sample_idx} out of range ({len(tokens)} samples)")
        token = tokens[args.sample_idx]

    print(f"\nProcessing sample token: {token}")
    cam_data_dict = loader.load_sample(token)

    # ------------------------------------------------------------------
    # 2. Depth estimator
    # ------------------------------------------------------------------
    if not args.no_depth_model:
        estimator = DepthEstimator(max_depth=args.max_depth)

    # ------------------------------------------------------------------
    # 3-6. Per-camera processing
    # ------------------------------------------------------------------
    projector = PointCloudFilter(
        x_range=(-51.2, 51.2),
        y_range=(-51.2, 51.2),
        z_range=(-5.0, 3.0),
    )
    back_proj = BackProjector(min_depth=0.5, max_depth=args.max_depth)

    all_points_ego: list[np.ndarray] = []
    images_vis = []
    depths_vis = []
    cam_names_vis = []
    pred_depth_per_cam: dict[str, np.ndarray] = {}  # collected for --compare_lidar

    for cam_name, cam in cam_data_dict.items():
        print(f"  [{cam_name}] predicting depth ...", end=" ")

        if args.no_depth_model:
            # Debug: use sparse LiDAR projection as "depth"
            depth = loader.get_sparse_lidar_depth(token, cam_name)
        else:
            depth = estimator.predict(cam.image)

        print(f"depth shape={depth.shape}, range=[{depth.min():.1f}, {depth.max():.1f}] m")

        # Optional: scale alignment with sparse LiDAR
        if args.align_lidar and not args.no_depth_model:
            depth_sparse = loader.get_sparse_lidar_depth(token, cam_name)
            depth = DepthEstimator.align_scale_shift(depth, depth_sparse)
            print(f"    after alignment: range=[{depth.min():.1f}, {depth.max():.1f}] m")

        # Back-project: depth → camera frame → ego frame
        points_ego, _ = back_proj.depth_to_ego(
            depth_map=depth,
            intrinsic=cam.intrinsic,
            cam2ego=cam.cam2ego,
        )

        # Range filter
        points_ego = projector.range_filter.filter(points_ego)
        print(f"    points after range filter: {len(points_ego):,}")

        all_points_ego.append(points_ego)
        images_vis.append(cam.image)
        depths_vis.append(depth)
        cam_names_vis.append(cam_name)
        if args.compare_lidar and not args.no_depth_model:
            pred_depth_per_cam[cam_name] = depth

    # ------------------------------------------------------------------
    # 7. Merge + voxel downsample
    # ------------------------------------------------------------------
    merged = np.concatenate(all_points_ego, axis=0)
    print(f"\nMerged point cloud: {len(merged):,} points from {len(all_points_ego)} cameras")

    downsampled = PointCloudFilter.voxel_downsample(merged, voxel_size=args.voxel_size)
    print(f"After voxel downsample (size={args.voxel_size}m): {len(downsampled):,} points")

    # ------------------------------------------------------------------
    # 8. Save pseudo-LiDAR as .bin  (XYZI float32, intensity=0)
    # ------------------------------------------------------------------
    intensity = np.zeros((len(downsampled), 1), dtype=np.float32)
    pseudo_lidar = np.concatenate([downsampled.astype(np.float32), intensity], axis=1)  # (N, 4)

    bin_path = output_dir / f"{token}_pseudo_lidar.bin"
    pseudo_lidar.tofile(str(bin_path))
    print(f"Saved pseudo-LiDAR: {bin_path}  ({pseudo_lidar.nbytes / 1024:.1f} KB)")

    # ------------------------------------------------------------------
    # 9. Optional: compare with GT LiDAR
    # ------------------------------------------------------------------
    if args.compare_lidar:
        print("\nComparing with GT nuScenes LiDAR ...")
        comparator = LidarComparator(
            x_range=(-51.2, 51.2),
            y_range=(-51.2, 51.2),
            z_range=(-5.0, 3.0),
            thresholds=(0.5, 1.0, 2.0),
        )
        cmp_results = comparator.compare(
            pred_points=downsampled,
            nusc=loader.nusc,
            sample_token=token,
            pred_depth_per_cam=pred_depth_per_cam if pred_depth_per_cam else None,
        )
        comparator.print_report(cmp_results)

        bev_path = output_dir / f"{token}_bev_comparison.png"
        comparator.save_bev_comparison(
            pred_points=downsampled,
            gt_points=cmp_results["gt_points"],
            output_path=bev_path,
        )

    # ------------------------------------------------------------------
    # 10. Visualization
    # ------------------------------------------------------------------
    try:
        import cv2
        summary = make_summary_figure(images_vis, depths_vis, cam_names_vis, downsampled)
        vis_path = output_dir / f"{token}_summary.jpg"
        cv2.imwrite(str(vis_path), cv2.cvtColor(summary, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization: {vis_path}")
    except ImportError:
        print("cv2 not found — skipping visualization")

    print("\nDone.")


if __name__ == "__main__":
    main()
