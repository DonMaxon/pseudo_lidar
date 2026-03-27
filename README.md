# depth2lidar

Converts nuScenes camera images to pseudo-LiDAR point clouds using metric depth estimation.

## Pipeline

```
RGB Image (6 cameras)
    │
    ▼
[Depth Anything V2 Metric]     ← depth_estimator.py
    │  (H, W) depth in meters
    ▼
[Scale/Shift Alignment]        ← optional, uses sparse LiDAR GT
    │
    ▼
[Back-Projection]              ← back_projection.py
    │  depth + intrinsics → camera-frame points
    │  + cam2ego transform  → ego-frame points
    ▼
[Range / Mask / Box Filter]    ← pointcloud_filter.py
    │
    ▼
[Voxel Downsample]
    │
    ▼
Pseudo-LiDAR .bin (XYZI float32)
    │
    ▼
BEVFusion / CenterPoint / BEVDet
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Single sample

```bash
python scripts/run_single_sample.py \
    --dataroot /data/nuscenes \
    --version v1.0-mini \
    --sample_idx 0 \
    --output_dir ./output \
    --align_lidar
```

### Full scene

```bash
python scripts/run_scene.py \
    --dataroot /data/nuscenes \
    --version v1.0-mini \
    --scene_name scene-0001 \
    --output_dir ./output/scene-0001
```

## Output format

`.bin` files contain `float32` arrays of shape `(N, 4)` — columns: `X, Y, Z, Intensity`.
Intensity is set to 0 (can be filled with image color or reflectance if needed).
This format is directly compatible with **CenterPoint**, **BEVFusion**, and **PointPillars**.

## Module overview

| File | Description |
|---|---|
| `depth2lidar/back_projection.py` | Core depth→3D unprojection + coordinate transforms |
| `depth2lidar/nuscenes_loader.py` | nuScenes data loading (images, intrinsics, extrinsics) |
| `depth2lidar/depth_estimator.py` | Depth Anything V2 Metric inference wrapper |
| `depth2lidar/pointcloud_filter.py` | Range, mask, and box-based point cloud filtering |
| `depth2lidar/visualize.py` | Depth colorization, BEV rendering, summary figures |
| `scripts/run_single_sample.py` | End-to-end pipeline for one sample |
| `scripts/run_scene.py` | Batch processing for a full scene |
| `configs/nuscenes.yaml` | Configuration file |

## Notes

- **Depth model**: Outdoor metric model is preferred for driving scenarios.
- **Scale alignment**: Use `--align_lidar` to correct metric drift using sparse LiDAR GT.
- **Filtering**: Default is range-only. For object-only clouds, use `MaskFilter` with
  a 2D segmentation model (SAM, YOLOv8-seg) before back-projection.
- **BEV range**: Default ±51.2 m matches standard nuScenes detection benchmarks.
