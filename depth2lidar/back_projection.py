"""
back_projection.py

Converts a depth map to a 3D point cloud (Pseudo-LiDAR) using camera intrinsics,
then optionally transforms into ego-vehicle or global coordinate frame
using camera extrinsics.

Pipeline per camera:
    depth_map (H, W)  +  intrinsics (3x3)  -->  points_cam (N, 3)
    points_cam        +  extrinsics (4x4)  -->  points_ego (N, 3)
    points_ego        +  ego_pose   (4x4)  -->  points_global (N, 3)
"""

import numpy as np


class BackProjector:
    """
    Back-projects a depth map into 3D space.

    Args:
        min_depth (float): Discard points closer than this (meters).
        max_depth (float): Discard points farther than this (meters).
        depth_scale (float): Multiplier applied to raw depth values
                             (use 1.0 if the model already outputs meters).
    """

    def __init__(
        self,
        min_depth: float = 0.5,
        max_depth: float = 60.0,
        depth_scale: float = 1.0,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def depth_to_points_cam(
        self,
        depth_map: np.ndarray,
        intrinsic: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Unproject a depth map into camera-frame 3D points.

        Args:
            depth_map: (H, W) float32 array of depth values in meters.
            intrinsic:  (3, 3) camera intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].

        Returns:
            points (N, 3): XYZ in camera frame.
            valid_mask (H, W) bool: pixels that produced valid points.
        """
        depth_map = depth_map.astype(np.float32) * self.depth_scale

        H, W = depth_map.shape
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        # Build pixel grid
        u_coords = np.arange(W, dtype=np.float32)  # (W,)
        v_coords = np.arange(H, dtype=np.float32)  # (H,)
        uu, vv = np.meshgrid(u_coords, v_coords)   # (H, W)

        valid = (depth_map >= self.min_depth) & (depth_map <= self.max_depth)

        d = depth_map[valid]
        u = uu[valid]
        v = vv[valid]

        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d

        points = np.stack([X, Y, Z], axis=-1)  # (N, 3)
        return points, valid

    def cam_to_ego(
        self,
        points_cam: np.ndarray,
        cam2ego: np.ndarray,
    ) -> np.ndarray:
        """
        Transform points from camera frame to ego-vehicle frame.

        Args:
            points_cam: (N, 3) points in camera coordinate system.
            cam2ego:    (4, 4) homogeneous transform matrix (camera → ego).

        Returns:
            points_ego: (N, 3) points in ego frame.
        """
        N = points_cam.shape[0]
        ones = np.ones((N, 1), dtype=points_cam.dtype)
        points_h = np.concatenate([points_cam, ones], axis=1)  # (N, 4)
        points_ego_h = (cam2ego @ points_h.T).T                # (N, 4)
        return points_ego_h[:, :3]

    def ego_to_global(
        self,
        points_ego: np.ndarray,
        ego2global: np.ndarray,
    ) -> np.ndarray:
        """
        Transform points from ego-vehicle frame to global (world) frame.

        Args:
            points_ego:  (N, 3) points in ego frame.
            ego2global:  (4, 4) homogeneous transform matrix (ego → global).

        Returns:
            points_global: (N, 3) points in global frame.
        """
        N = points_ego.shape[0]
        ones = np.ones((N, 1), dtype=points_ego.dtype)
        points_h = np.concatenate([points_ego, ones], axis=1)  # (N, 4)
        points_global_h = (ego2global @ points_h.T).T           # (N, 4)
        return points_global_h[:, :3]

    def depth_to_ego(
        self,
        depth_map: np.ndarray,
        intrinsic: np.ndarray,
        cam2ego: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convenience method: depth map → ego-frame points in one call.

        Returns:
            points_ego: (N, 3)
            valid_mask: (H, W) bool
        """
        points_cam, valid_mask = self.depth_to_points_cam(depth_map, intrinsic)
        points_ego = self.cam_to_ego(points_cam, cam2ego)
        return points_ego, valid_mask

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """
        Build a 4x4 homogeneous transform from a rotation matrix and translation.

        Args:
            rotation:    (3, 3) rotation matrix  OR  (4,) quaternion [w, x, y, z].
            translation: (3,) translation vector.

        Returns:
            T: (4, 4) homogeneous matrix.
        """
        T = np.eye(4, dtype=np.float64)
        if rotation.shape == (4,):
            rotation = BackProjector._quat_to_rot(rotation)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        return T

    @staticmethod
    def _quat_to_rot(q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
        return R
