"""
pointcloud_filter.py

Filters a dense pseudo-LiDAR point cloud to retain only points
that belong to objects of interest.

Two strategies are supported:
  A) Mask-based  — uses 2D segmentation masks (e.g. from SAM or YOLOv8-seg)
                   to select pixels before back-projection.
  B) Box-based   — uses 3D / BEV bounding boxes to filter already-projected points.
  C) Range-based — simple spatial crop (BEV range, height range).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# A) Mask-based filter  (applied on depth map BEFORE back-projection)
# ---------------------------------------------------------------------------

class MaskFilter:
    """
    Given a set of 2D binary masks (H, W) for detected objects,
    produces a combined pixel mask to select relevant depth values.

    Usage:
        mask_filter = MaskFilter(target_classes=[0, 1, 2])  # COCO person, bicycle, car
        pixel_mask  = mask_filter.combine(seg_masks, class_ids)
        depth_filtered = depth_map * pixel_mask
    """

    def __init__(self, target_classes: Optional[List[int]] = None, dilate_px: int = 0):
        """
        Args:
            target_classes: COCO class IDs to keep. None = keep all masks.
            dilate_px:      Dilate each mask by this many pixels (fills depth holes).
        """
        self.target_classes = set(target_classes) if target_classes is not None else None
        self.dilate_px = dilate_px

    def combine(
        self,
        masks: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Merge individual object masks into one binary pixel mask.

        Args:
            masks:     (K, H, W) bool array — one mask per detection.
            class_ids: (K,) int array — class ID for each mask. Required if
                       target_classes was set.

        Returns:
            combined: (H, W) bool array — True where depth should be kept.
        """
        if masks.ndim == 2:
            masks = masks[np.newaxis]  # treat single mask as K=1

        K, H, W = masks.shape
        combined = np.zeros((H, W), dtype=bool)

        for i in range(K):
            if self.target_classes is not None and class_ids is not None:
                if int(class_ids[i]) not in self.target_classes:
                    continue

            m = masks[i].astype(bool)

            if self.dilate_px > 0:
                m = self._dilate(m, self.dilate_px)

            combined |= m

        return combined

    def apply_to_depth(
        self,
        depth_map: np.ndarray,
        masks: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Zero out depth pixels that don't belong to any target object.

        Returns:
            depth_filtered: (H, W) float32 with background zeroed.
        """
        pixel_mask = self.combine(masks, class_ids)
        return depth_map * pixel_mask.astype(np.float32)

    @staticmethod
    def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
        """Simple square dilation without scipy dependency."""
        from numpy.lib.stride_tricks import sliding_window_view
        pad = radius
        padded = np.pad(mask, pad, mode="constant", constant_values=False)
        windows = sliding_window_view(padded, (2 * pad + 1, 2 * pad + 1))
        return windows.any(axis=(-2, -1))


# ---------------------------------------------------------------------------
# B) Box-based filter  (applied on 3D points AFTER back-projection)
# ---------------------------------------------------------------------------

class BoxFilter:
    """
    Filters a point cloud by 3D axis-aligned bounding boxes (AABB).

    Each box is defined as (x_min, y_min, z_min, x_max, y_max, z_max)
    in the same coordinate frame as the points (ego or global).
    """

    def filter(
        self,
        points: np.ndarray,
        boxes: np.ndarray,
    ) -> np.ndarray:
        """
        Keep only points that fall inside at least one bounding box.

        Args:
            points: (N, 3) point cloud.
            boxes:  (M, 6) array of AABB boxes [xmin, ymin, zmin, xmax, ymax, zmax].

        Returns:
            filtered: (K, 3) point cloud with K <= N.
        """
        if len(boxes) == 0:
            return points

        keep = np.zeros(len(points), dtype=bool)
        for box in boxes:
            xmin, ymin, zmin, xmax, ymax, zmax = box
            inside = (
                (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
                (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
                (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
            )
            keep |= inside

        return points[keep]


# ---------------------------------------------------------------------------
# C) Range-based filter  (spatial crop, no external labels needed)
# ---------------------------------------------------------------------------

class RangeFilter:
    """
    Retains points within a BEV range rectangle and height interval.
    Useful as a post-processing step after back-projection to remove
    ground plane, sky artefacts, and out-of-range points.

    Default values match typical BEVFusion / CenterPoint detection ranges.
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = (-51.2, 51.2),
        y_range: Tuple[float, float] = (-51.2, 51.2),
        z_range: Tuple[float, float] = (-2.0, 1.5),
        ego_radius: float = 2.5,
    ):
        """
        Args:
            x_range:    (min, max) in meters along ego-X (forward).
            y_range:    (min, max) in meters along ego-Y (left).
            z_range:    (min, max) in meters along ego-Z (up). Filters ground/sky.
            ego_radius: Points closer than this radius (XY plane) are removed
                        to eliminate back-projected car body / hood artefacts.
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.ego_radius = ego_radius

    def filter(self, points: np.ndarray) -> np.ndarray:
        """
        Args:
            points: (N, 3+) — first 3 columns are X, Y, Z in ego frame.

        Returns:
            filtered: (K, 3+)
        """
        mask = (
            (points[:, 0] >= self.x_range[0]) & (points[:, 0] <= self.x_range[1]) &
            (points[:, 1] >= self.y_range[0]) & (points[:, 1] <= self.y_range[1]) &
            (points[:, 2] >= self.z_range[0]) & (points[:, 2] <= self.z_range[1])
        )
        if self.ego_radius > 0.0:
            dist_xy = points[:, 0] ** 2 + points[:, 1] ** 2
            mask &= dist_xy > self.ego_radius ** 2
        return points[mask]


# ---------------------------------------------------------------------------
# D) Statistical Outlier Removal  (applied on 3D points AFTER back-projection)
# ---------------------------------------------------------------------------

class StatisticalOutlierFilter:
    """
    Removes points whose mean distance to their k nearest neighbours
    exceeds  global_mean + std_ratio * global_std.

    Eliminates isolated noise (sky artefacts, tree crowns, reflections)
    while keeping dense, coherent structures (walls, road, vehicles).

    Requires scipy (scipy.spatial.KDTree).
    """

    def __init__(self, nb_neighbors: int = 20, std_ratio: float = 2.0):
        """
        Args:
            nb_neighbors: Number of nearest neighbours to consider per point.
            std_ratio:    How many standard deviations above the mean is the
                          outlier threshold. Lower = more aggressive removal.
        """
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def filter(self, points: np.ndarray) -> np.ndarray:
        """
        Args:
            points: (N, 3+) point cloud in any coordinate frame.

        Returns:
            inliers: (K, 3+) with outliers removed.
        """
        if len(points) <= self.nb_neighbors + 1:
            return points

        from scipy.spatial import KDTree
        tree = KDTree(points[:, :3])
        # query k+1 because the point itself is always the closest (dist=0)
        dists, _ = tree.query(points[:, :3], k=self.nb_neighbors + 1)
        mean_dists = dists[:, 1:].mean(axis=1)

        threshold = mean_dists.mean() + self.std_ratio * mean_dists.std()
        return points[mean_dists <= threshold]


# ---------------------------------------------------------------------------
# Composite filter
# ---------------------------------------------------------------------------

class PointCloudFilter:
    """
    Combines RangeFilter and (optionally) BoxFilter.
    MaskFilter should be applied earlier, directly on the depth map.
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = (-51.2, 51.2),
        y_range: Tuple[float, float] = (-51.2, 51.2),
        z_range: Tuple[float, float] = (-2.0, 1.5),
        ego_radius: float = 2.5,
    ):
        self.range_filter = RangeFilter(x_range, y_range, z_range, ego_radius)

    def apply(
        self,
        points: np.ndarray,
        boxes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply range filter, then optionally box filter.

        Args:
            points: (N, 3) ego-frame point cloud.
            boxes:  (M, 6) optional AABB boxes for object-only filtering.

        Returns:
            filtered points (K, 3).
        """
        points = self.range_filter.filter(points)

        if boxes is not None and len(boxes) > 0:
            box_filter = BoxFilter()
            points = box_filter.filter(points, boxes)

        return points

    @staticmethod
    def voxel_downsample_mean(points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
        """
        Voxel grid downsampling using the **mean** position of all points in each voxel.

        Unlike voxel_downsample (which keeps the first/closest point), this method
        averages coordinates within each occupied voxel.  The result is geometrically
        smoother — systematic depth-estimation bias within a voxel is partially
        cancelled out — at the cost of points no longer lying on the original rays.

        Args:
            points:     (N, 3+) point cloud.
            voxel_size: Side length of each voxel in metres.

        Returns:
            downsampled: (K, 3+) one point per occupied voxel (mean coordinates).
        """
        if len(points) == 0:
            return np.empty((0, points.shape[1]), dtype=np.float32)

        coords = np.floor(points[:, :3] / voxel_size).astype(np.int32)

        # Sort by voxel key (same as voxel_downsample for consistency)
        order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        sorted_coords = coords[order]
        sorted_points = points[order].astype(np.float64)

        # Voxel boundaries
        is_new = np.concatenate(
            [[True], np.any(sorted_coords[1:] != sorted_coords[:-1], axis=1)]
        )
        split_idx = np.where(is_new)[0]

        # Sum then divide — O(N) via reduceat
        sums   = np.add.reduceat(sorted_points, split_idx, axis=0)
        counts = np.diff(np.concatenate([split_idx, [len(sorted_points)]]))
        means  = sums / counts[:, None]

        return means.astype(np.float32)

    @staticmethod
    def voxel_downsample(points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
        """
        Uniform voxel grid downsampling. Reduces dense pseudo-LiDAR to
        a manageable size for LiDAR backbone (PointPillars, VoxelNet).

        Args:
            points:     (N, 3+) point cloud.
            voxel_size: Side length of each voxel in meters.

        Returns:
            downsampled: (K, 3+) one point per occupied voxel (first point).
        """
        if len(points) == 0:
            return np.empty((0, points.shape[1]), dtype=np.float32)

        coords = np.floor(points[:, :3] / voxel_size).astype(np.int32)

        # Lexsort by (z, y, x): numpy sort in C, 10-50x faster than Python dict loop
        order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        sorted_coords = coords[order]

        # Keep first point in each unique voxel
        unique = np.concatenate(
            [[True], np.any(sorted_coords[1:] != sorted_coords[:-1], axis=1)]
        )
        return points[order[unique]].astype(np.float32)

    @staticmethod
    def random_subsample(points: np.ndarray, max_points: int) -> np.ndarray:
        """
        Randomly subsample a point cloud to at most max_points points.

        Useful as a hard cap before feeding into BEVFusion / PointPillars
        to guarantee a fixed upper bound on inference time.

        Args:
            points:     (N, 3+) point cloud.
            max_points: Maximum number of points to keep.

        Returns:
            subsampled: (min(N, max_points), 3+)
        """
        if len(points) <= max_points:
            return points
        idx = np.random.choice(len(points), size=max_points, replace=False)
        return points[idx]
