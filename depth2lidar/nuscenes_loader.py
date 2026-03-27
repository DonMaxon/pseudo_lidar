"""
nuscenes_loader.py

Wraps the official nuscenes-devkit to expose per-sample data needed
for the depth2lidar pipeline:
    - RGB images for each camera
    - Camera intrinsics  (3x3)
    - cam2ego transform  (4x4)
    - ego2global transform (4x4)
    - (optional) sparse LiDAR depth projected onto each camera for evaluation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# nuscenes-devkit imports (pip install nuscenes-devkit)
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

from .back_projection import BackProjector

# All 6 cameras available in nuScenes
NUSCENES_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class CameraData:
    """Container for one camera's data at a single timestamp."""

    def __init__(
        self,
        name: str,
        image: np.ndarray,
        intrinsic: np.ndarray,
        cam2ego: np.ndarray,
        ego2global: np.ndarray,
        token: str,
    ):
        self.name = name
        self.image = image          # (H, W, 3) uint8
        self.intrinsic = intrinsic  # (3, 3)
        self.cam2ego = cam2ego      # (4, 4)
        self.ego2global = ego2global  # (4, 4)
        self.token = token


class NuScenesLoader:
    """
    Loads nuScenes samples and returns per-camera calibration + images.

    Args:
        dataroot:  Path to nuScenes dataset root (contains 'samples/', 'maps/', etc.)
        version:   Dataset version string, e.g. 'v1.0-trainval' or 'v1.0-mini'.
        cameras:   List of camera channel names to load. Defaults to all 6.
    """

    def __init__(
        self,
        dataroot: str | Path,
        version: str = "v1.0-mini",
        cameras: Optional[List[str]] = None,
    ):
        self.dataroot = Path(dataroot)
        self.version = version
        self.cameras = cameras or NUSCENES_CAMERAS

        print(f"Loading nuScenes {version} from {self.dataroot} ...")
        self.nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=False)
        print(f"  Scenes: {len(self.nusc.scene)}, Samples: {len(self.nusc.sample)}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.nusc.sample)

    def get_sample_tokens(self) -> List[str]:
        return [s["token"] for s in self.nusc.sample]

    def load_sample(self, sample_token: str) -> Dict[str, CameraData]:
        """
        Load all cameras for a given sample token.

        Returns:
            Dict mapping camera_name → CameraData.
        """
        sample = self.nusc.get("sample", sample_token)
        result: Dict[str, CameraData] = {}

        for cam_name in self.cameras:
            if cam_name not in sample["data"]:
                continue

            cam_token = sample["data"][cam_name]
            cam_data = self.nusc.get("sample_data", cam_token)
            cs = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            ego_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])

            # --- Image ---
            img_path = self.dataroot / cam_data["filename"]
            image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

            # --- Intrinsics ---
            intrinsic = np.array(cs["camera_intrinsic"], dtype=np.float64)  # (3, 3)

            # --- cam2ego ---
            cam2ego = BackProjector.build_transform(
                rotation=np.array(cs["rotation"], dtype=np.float64),      # quaternion [w,x,y,z]
                translation=np.array(cs["translation"], dtype=np.float64),
            )

            # --- ego2global ---
            ego2global = BackProjector.build_transform(
                rotation=np.array(ego_pose["rotation"], dtype=np.float64),
                translation=np.array(ego_pose["translation"], dtype=np.float64),
            )

            result[cam_name] = CameraData(
                name=cam_name,
                image=image,
                intrinsic=intrinsic,
                cam2ego=cam2ego,
                ego2global=ego2global,
                token=cam_token,
            )

        return result

    def get_sparse_lidar_depth(
        self,
        sample_token: str,
        cam_name: str,
    ) -> np.ndarray:
        """
        Project real LiDAR points onto a camera image to obtain a sparse
        ground-truth depth map. Useful for evaluation / scale correction.

        Returns:
            depth_sparse (H, W) float32 — zero where no LiDAR point hit.
        """
        sample = self.nusc.get("sample", sample_token)

        # Load camera calibration
        cam_token = sample["data"][cam_name]
        cam_sd = self.nusc.get("sample_data", cam_token)
        cs_cam = self.nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        ego_pose_cam = self.nusc.get("ego_pose", cam_sd["ego_pose_token"])

        H, W = cam_sd["height"], cam_sd["width"]
        depth_sparse = np.zeros((H, W), dtype=np.float32)

        # Load LiDAR point cloud
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_sd = self.nusc.get("sample_data", lidar_token)
        lidar_path = self.dataroot / lidar_sd["filename"]
        pc = LidarPointCloud.from_file(str(lidar_path))

        cs_lidar = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        ego_pose_lidar = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        # Transform: lidar → ego (lidar time) → global → ego (cam time) → cam
        pc.rotate(BackProjector._quat_to_rot(np.array(cs_lidar["rotation"])))
        pc.translate(np.array(cs_lidar["translation"]))

        pc.rotate(BackProjector._quat_to_rot(np.array(ego_pose_lidar["rotation"])))
        pc.translate(np.array(ego_pose_lidar["translation"]))

        ego2global_cam = BackProjector.build_transform(
            np.array(ego_pose_cam["rotation"]),
            np.array(ego_pose_cam["translation"]),
        )
        global2ego_cam = np.linalg.inv(ego2global_cam)

        points_global = pc.points[:3, :].T  # (N, 3)
        N = points_global.shape[0]
        ones = np.ones((N, 1))
        points_ego_cam = (global2ego_cam @ np.hstack([points_global, ones]).T).T[:, :3]

        # ego → cam
        cam2ego_mat = BackProjector.build_transform(
            np.array(cs_cam["rotation"]), np.array(cs_cam["translation"])
        )
        ego2cam = np.linalg.inv(cam2ego_mat)
        points_cam = (ego2cam @ np.hstack([points_ego_cam, np.ones((N, 1))]).T).T[:, :3]

        # Keep points in front of camera
        in_front = points_cam[:, 2] > 0.1
        points_cam = points_cam[in_front]

        intrinsic = np.array(cs_cam["camera_intrinsic"], dtype=np.float64)
        pts_2d = view_points(points_cam.T, intrinsic, normalize=True)  # (3, N)

        u = pts_2d[0].astype(np.int32)
        v = pts_2d[1].astype(np.int32)
        d = points_cam[:, 2]

        in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, d = u[in_image], v[in_image], d[in_image]

        depth_sparse[v, u] = d
        return depth_sparse
