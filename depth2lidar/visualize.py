"""
visualize.py

Visualization utilities for depth maps and pseudo-LiDAR point clouds.

All functions return matplotlib figures or numpy image arrays —
no display is triggered automatically (suitable for headless servers).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Depth map visualization
# ---------------------------------------------------------------------------

def colorize_depth(
    depth: np.ndarray,
    min_depth: float = 0.5,
    max_depth: float = 60.0,
    cmap: str = "plasma",
) -> np.ndarray:
    """
    Convert a (H, W) float depth map to a (H, W, 3) uint8 color image.

    Args:
        depth:     (H, W) float32 depth in meters.
        min_depth: Depth values below this are clipped.
        max_depth: Depth values above this are clipped.
        cmap:      Matplotlib colormap name.

    Returns:
        color_img: (H, W, 3) uint8 RGB.
    """
    import matplotlib.cm as cm

    depth_clipped = np.clip(depth, min_depth, max_depth)
    depth_norm = (depth_clipped - min_depth) / (max_depth - min_depth + 1e-8)

    colormap = cm.get_cmap(cmap)
    color = (colormap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    return color


def overlay_depth_on_image(
    image: np.ndarray,
    depth: np.ndarray,
    alpha: float = 0.5,
    **colorize_kwargs,
) -> np.ndarray:
    """
    Blend a colorized depth map over an RGB image.

    Args:
        image:  (H, W, 3) uint8 RGB.
        depth:  (H, W) float32 depth.
        alpha:  Weight for depth overlay (0 = only image, 1 = only depth).

    Returns:
        blended: (H, W, 3) uint8.
    """
    depth_color = colorize_depth(depth, **colorize_kwargs)
    blended = (
        (1 - alpha) * image.astype(np.float32) +
        alpha * depth_color.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return blended


def project_points_on_image(
    image: np.ndarray,
    points_cam: np.ndarray,
    intrinsic: np.ndarray,
    point_size: int = 2,
    cmap: str = "plasma",
    max_depth: float = 60.0,
) -> np.ndarray:
    """
    Project 3D camera-frame points onto the image and draw colored dots.

    Args:
        image:      (H, W, 3) uint8 RGB.
        points_cam: (N, 3) points in camera frame.
        intrinsic:  (3, 3).
        point_size: Radius of drawn dots in pixels.
        cmap:       Colormap for depth coloring.
        max_depth:  Used for normalization.

    Returns:
        vis: (H, W, 3) uint8.
    """
    import matplotlib.cm as cm

    H, W = image.shape[:2]
    vis = image.copy()

    # Filter points in front of camera
    in_front = points_cam[:, 2] > 0.1
    pts = points_cam[in_front]
    if len(pts) == 0:
        return vis

    # Project
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u = (pts[:, 0] * fx / pts[:, 2] + cx).astype(np.int32)
    v = (pts[:, 1] * fy / pts[:, 2] + cy).astype(np.int32)
    d = pts[:, 2]

    in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, d = u[in_image], v[in_image], d[in_image]

    # Colorize by depth
    d_norm = np.clip(d / max_depth, 0, 1)
    colors = (cm.get_cmap(cmap)(d_norm)[:, :3] * 255).astype(np.uint8)

    r = point_size
    for i in range(len(u)):
        v0, v1 = max(0, v[i] - r), min(H, v[i] + r + 1)
        u0, u1 = max(0, u[i] - r), min(W, u[i] + r + 1)
        vis[v0:v1, u0:u1] = colors[i]

    return vis


# ---------------------------------------------------------------------------
# BEV (top-down) point cloud visualization
# ---------------------------------------------------------------------------

def points_to_bev_image(
    points: np.ndarray,
    x_range: Tuple[float, float] = (-51.2, 51.2),
    y_range: Tuple[float, float] = (-51.2, 51.2),
    resolution: float = 0.1,
    bg_color: int = 0,
    point_color: int = 255,
) -> np.ndarray:
    """
    Render an ego-frame point cloud as a grayscale BEV occupancy image.

    Стандартная ориентация (как в nuScenes viewer):
        вперёд (X+) = вверх изображения
        влево  (Y+) = влево изображения

    Args:
        points:      (N, 3) XYZ in ego frame (X=forward, Y=left, Z=up).
        x_range:     Forward/backward range [m].
        y_range:     Left/right range [m].
        resolution:  Meters per pixel.
        bg_color:    Background pixel value (0–255).
        point_color: Foreground pixel value (0–255).

    Returns:
        bev_img: (H, W) uint8 grayscale image.
    """
    W = int((y_range[1] - y_range[0]) / resolution)  # Y → ширина
    H = int((x_range[1] - x_range[0]) / resolution)  # X → высота
    bev = np.full((H, W), bg_color, dtype=np.uint8)

    # Y+ (влево) → столбец 0 (левый край);  Y- (вправо) → столбец W-1
    px = ((y_range[1] - points[:, 1]) / resolution).astype(np.int32)
    # X+ (вперёд) → строка 0 (верхний край);  X- (назад) → строка H-1
    py = ((x_range[1] - points[:, 0]) / resolution).astype(np.int32)

    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    bev[py[valid], px[valid]] = point_color
    return bev


# ---------------------------------------------------------------------------
# Multi-camera summary figure
# ---------------------------------------------------------------------------

def make_summary_figure(
    images: list[np.ndarray],
    depths: list[np.ndarray],
    cam_names: list[str],
    points_ego: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compose a summary grid:
        - Row 0: raw RGB images
        - Row 1: colorized depth maps
        - Row 2 (optional): BEV of merged pseudo-LiDAR

    Returns:
        grid: (H_total, W_total, 3) uint8 RGB.
    """
    import cv2

    rows_rgb = []
    rows_dep = []

    target_w = 640
    for img, dep, name in zip(images, depths, cam_names):
        h, w = img.shape[:2]
        scale = target_w / w
        nh = int(h * scale)

        rgb_r = cv2.resize(img, (target_w, nh))
        dep_color = colorize_depth(dep)
        dep_r = cv2.resize(dep_color, (target_w, nh))

        # Add label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(rgb_r, name, (5, 20), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        rows_rgb.append(rgb_r)
        rows_dep.append(dep_r)

    row0 = np.concatenate(rows_rgb, axis=1)
    row1 = np.concatenate(rows_dep, axis=1)
    grid = np.concatenate([row0, row1], axis=0)

    if points_ego is not None:
        bev_gray = points_to_bev_image(points_ego)
        bev_rgb = np.stack([bev_gray] * 3, axis=-1)
        bev_rgb = cv2.resize(bev_rgb, (grid.shape[1], bev_gray.shape[0]))
        grid = np.concatenate([grid, bev_rgb], axis=0)

    return grid
