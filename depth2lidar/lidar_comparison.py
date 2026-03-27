"""
lidar_comparison.py

Optional module for comparing pseudo-LiDAR point clouds against
ground-truth nuScenes LiDAR point clouds.

Metrics
-------
1. Chamfer Distance (CD)
       Symmetric mean nearest-neighbour distance between the two clouds.
       CD = mean_NN(pred→gt) + mean_NN(gt→pred)

2. Precision / Recall / F-score  at configurable distance thresholds
       Precision@d = fraction of pred points that have a GT neighbour ≤ d m
       Recall@d    = fraction of GT points that have a pred neighbour ≤ d m
       F-score@d   = 2·P·R / (P+R)

3. Depth metrics (per camera, requires sparse GT depth from LiDAR projection)
       MAE, RMSE, AbsRel, SqRel, δ<1.25, δ<1.25², δ<1.25³

Visualisation
-------------
   save_bev_comparison() — BEV PNG with pseudo-LiDAR (blue) vs GT LiDAR (orange)

Dependencies
------------
   scipy  — for KDTree nearest-neighbour queries (pip install scipy)
   matplotlib — for BEV visualisation (already in requirements.txt)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .back_projection import BackProjector


class LidarComparator:
    """
    Compares pseudo-LiDAR point clouds to ground-truth nuScenes LiDAR.

    Args:
        x_range:    (min, max) metres along ego-X (forward). Used to crop GT cloud.
        y_range:    (min, max) metres along ego-Y (left).
        z_range:    (min, max) metres along ego-Z (up).
        thresholds: Distance thresholds (metres) for Precision / Recall / F-score.
    """

    DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.5, 1.0, 2.0)

    def __init__(
        self,
        x_range: Tuple[float, float] = (-51.2, 51.2),
        y_range: Tuple[float, float] = (-51.2, 51.2),
        z_range: Tuple[float, float] = (-5.0, 3.0),
        thresholds: Tuple[float, ...] = DEFAULT_THRESHOLDS,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.thresholds = thresholds

    # ------------------------------------------------------------------
    # GT loading
    # ------------------------------------------------------------------

    def load_gt_lidar_ego(self, nusc, sample_token: str) -> np.ndarray:
        """
        Load the real LiDAR point cloud from nuScenes in ego-vehicle frame.

        The transformation chain is:
            LiDAR sensor frame  →  ego frame  (at LiDAR capture timestamp)

        No temporal correction across cameras is applied because within a
        single nuScenes sample the time offset is negligible (<50 ms).

        Args:
            nusc:         NuScenes object (already initialised).
            sample_token: Token of the nuScenes sample.

        Returns:
            points_ego: (N, 3) float32 XYZ in ego frame.
        """
        from nuscenes.utils.data_classes import LidarPointCloud

        sample = nusc.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_sd = nusc.get("sample_data", lidar_token)
        lidar_path = Path(nusc.dataroot) / lidar_sd["filename"]

        pc = LidarPointCloud.from_file(str(lidar_path))  # (4, N): XYZI

        # sensor frame → ego frame
        cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        pc.rotate(BackProjector._quat_to_rot(np.array(cs["rotation"])))
        pc.translate(np.array(cs["translation"]))

        return pc.points[:3, :].T.astype(np.float32)  # (N, 3)

    def _range_filter(self, points: np.ndarray) -> np.ndarray:
        mask = (
            (points[:, 0] >= self.x_range[0]) & (points[:, 0] <= self.x_range[1]) &
            (points[:, 1] >= self.y_range[0]) & (points[:, 1] <= self.y_range[1]) &
            (points[:, 2] >= self.z_range[0]) & (points[:, 2] <= self.z_range[1])
        )
        return points[mask]

    # ------------------------------------------------------------------
    # Nearest-neighbour helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_kdtree(points: np.ndarray):
        try:
            from scipy.spatial import KDTree
            return KDTree(points)
        except ImportError as exc:
            raise ImportError(
                "scipy is required for LidarComparator. "
                "Install it with:  pip install scipy"
            ) from exc

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def chamfer_distance(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
    ) -> Dict[str, float]:
        """
        Symmetric Chamfer Distance between two point clouds.

        CD = mean_NN(pred→gt) + mean_NN(gt→pred)

        Args:
            pred: (N, 3) pseudo-LiDAR points in ego frame.
            gt:   (M, 3) ground-truth LiDAR points in ego frame.

        Returns:
            dict with keys:
                'cd'              — total Chamfer Distance
                'cd_pred_to_gt'   — mean NN distance from pred to GT
                'cd_gt_to_pred'   — mean NN distance from GT to pred
        """
        tree_gt = self._build_kdtree(gt)
        tree_pred = self._build_kdtree(pred)

        dists_p2g, _ = tree_gt.query(pred)
        dists_g2p, _ = tree_pred.query(gt)

        cd_p2g = float(np.mean(dists_p2g))
        cd_g2p = float(np.mean(dists_g2p))

        return {
            "cd": cd_p2g + cd_g2p,
            "cd_pred_to_gt": cd_p2g,
            "cd_gt_to_pred": cd_g2p,
        }

    def precision_recall_fscore(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
    ) -> Dict[str, Dict[float, float]]:
        """
        Precision, Recall, and F-score at each configured threshold.

        Precision@d = fraction of pred points with a GT neighbour within d m.
        Recall@d    = fraction of GT points with a pred neighbour within d m.
        F-score@d   = 2·P·R / (P+R)

        Returns:
            {
                'precision': {thr: value, ...},
                'recall':    {thr: value, ...},
                'fscore':    {thr: value, ...},
            }
        """
        tree_gt = self._build_kdtree(gt)
        tree_pred = self._build_kdtree(pred)

        dists_p2g, _ = tree_gt.query(pred)
        dists_g2p, _ = tree_pred.query(gt)

        precision: Dict[float, float] = {}
        recall: Dict[float, float] = {}
        fscore: Dict[float, float] = {}

        for thr in self.thresholds:
            p = float(np.mean(dists_p2g <= thr))
            r = float(np.mean(dists_g2p <= thr))
            f = 2.0 * p * r / (p + r) if (p + r) > 0.0 else 0.0
            precision[thr] = p
            recall[thr] = r
            fscore[thr] = f

        return {"precision": precision, "recall": recall, "fscore": fscore}

    @staticmethod
    def depth_metrics(
        pred_depth: np.ndarray,
        gt_sparse: np.ndarray,
    ) -> Dict[str, float]:
        """
        Standard monocular depth evaluation metrics computed at pixels
        where ground-truth sparse LiDAR depth is available (gt_sparse > 0).

        Args:
            pred_depth: (H, W) float32 — predicted depth in metres.
            gt_sparse:  (H, W) float32 — sparse GT depth from LiDAR projection,
                        0 where no LiDAR point hit the pixel.

        Returns:
            dict:
                mae        — Mean Absolute Error  [m]
                rmse       — Root Mean Square Error  [m]
                abs_rel    — Mean Absolute Relative Error
                sq_rel     — Mean Squared Relative Error
                d1         — % pixels with max(pred/gt, gt/pred) < 1.25
                d2         — % pixels with max(...) < 1.25²
                d3         — % pixels with max(...) < 1.25³
                n_valid    — number of valid GT pixels used
        """
        valid = gt_sparse > 0.0
        n_valid = int(valid.sum())
        if n_valid == 0:
            return {}

        p = pred_depth[valid].astype(np.float64)
        g = gt_sparse[valid].astype(np.float64)

        diff = np.abs(p - g)
        mae = float(np.mean(diff))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        abs_rel = float(np.mean(diff / g))
        sq_rel = float(np.mean((diff ** 2) / g))

        ratio = np.maximum(p / g, g / p)
        d1 = float(np.mean(ratio < 1.25))
        d2 = float(np.mean(ratio < 1.25 ** 2))
        d3 = float(np.mean(ratio < 1.25 ** 3))

        return {
            "mae": mae,
            "rmse": rmse,
            "abs_rel": abs_rel,
            "sq_rel": sq_rel,
            "d1": d1,
            "d2": d2,
            "d3": d3,
            "n_valid": n_valid,
        }

    # ------------------------------------------------------------------
    # Full comparison pipeline
    # ------------------------------------------------------------------

    def compare(
        self,
        pred_points: np.ndarray,
        nusc,
        sample_token: str,
        pred_depth_per_cam: Optional[Dict[str, np.ndarray]] = None,
        gt_sparse_per_cam: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict:
        """
        Full comparison pipeline: load GT LiDAR, filter, compute all metrics.

        Args:
            pred_points:        (N, 3) merged pseudo-LiDAR in ego frame.
            nusc:               NuScenes object.
            sample_token:       nuScenes sample token.
            pred_depth_per_cam: Optional {cam_name: depth_map (H,W)} for depth metrics.
            gt_sparse_per_cam:  Optional {cam_name: sparse_gt (H,W)} for depth metrics.
                                If None but pred_depth_per_cam is given, sparse GT is
                                loaded automatically from nuScenes LiDAR projection.

        Returns:
            dict with keys:
                n_pred, n_gt_full, n_gt_filtered,
                chamfer, prf, depth (optional), gt_points
        """
        gt_full = self.load_gt_lidar_ego(nusc, sample_token)
        gt_filtered = self._range_filter(gt_full)

        results: Dict = {
            "n_pred": len(pred_points),
            "n_gt_full": len(gt_full),
            "n_gt_filtered": len(gt_filtered),
            "gt_points": gt_filtered,
        }

        if len(pred_points) == 0 or len(gt_filtered) == 0:
            print("[LidarComparator] Warning: empty cloud(s), skipping metric computation.")
            return results

        results["chamfer"] = self.chamfer_distance(pred_points, gt_filtered)
        results["prf"] = self.precision_recall_fscore(pred_points, gt_filtered)

        # Per-camera depth metrics
        if pred_depth_per_cam:
            # Auto-load sparse GT from nuScenes if not provided
            if gt_sparse_per_cam is None:
                from .nuscenes_loader import NuScenesLoader
                gt_sparse_per_cam = {}
                for cam_name in pred_depth_per_cam:
                    try:
                        # Use the helper already on NuScenesLoader —
                        # we reach into nusc directly to avoid a second init.
                        gt_sparse_per_cam[cam_name] = _project_lidar_to_cam(
                            nusc, sample_token, cam_name
                        )
                    except Exception as exc:
                        print(f"[LidarComparator] Could not load sparse GT for {cam_name}: {exc}")

            depth_results: Dict[str, Dict] = {}
            for cam_name, pred_d in pred_depth_per_cam.items():
                gt_sp = (gt_sparse_per_cam or {}).get(cam_name)
                if gt_sp is not None:
                    depth_results[cam_name] = self.depth_metrics(pred_d, gt_sp)
            results["depth"] = depth_results

        return results

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    def print_report(self, results: Dict) -> None:
        """Pretty-print the comparison results to stdout."""
        sep = "=" * 62
        print(f"\n{sep}")
        print("  LiDAR Comparison Report")
        print(sep)
        print(f"  Pseudo-LiDAR points  : {results.get('n_pred', 0):>10,}")
        print(f"  GT LiDAR  (full)     : {results.get('n_gt_full', 0):>10,}")
        print(f"  GT LiDAR  (filtered) : {results.get('n_gt_filtered', 0):>10,}")

        cd = results.get("chamfer")
        if cd:
            print("\n  -- Chamfer Distance (lower is better) --")
            print(f"  CD total             : {cd['cd']:.4f} m")
            print(f"  CD pred → GT         : {cd['cd_pred_to_gt']:.4f} m")
            print(f"  CD GT   → pred       : {cd['cd_gt_to_pred']:.4f} m")

        prf = results.get("prf")
        if prf:
            print("\n  -- Precision / Recall / F-score --")
            print(f"  {'Threshold':>12}  {'Precision':>10}  {'Recall':>10}  {'F-score':>10}")
            for thr in self.thresholds:
                p = prf["precision"][thr]
                r = prf["recall"][thr]
                f = prf["fscore"][thr]
                print(f"  {thr:>10.1f} m  {p:>10.4f}  {r:>10.4f}  {f:>10.4f}")

        depth = results.get("depth")
        if depth:
            print("\n  -- Depth Metrics (pred depth vs sparse LiDAR GT) --")
            for cam_name, m in depth.items():
                if not m:
                    continue
                n = m.get("n_valid", 0)
                print(f"\n  [{cam_name}]  ({n:,} valid pixels)")
                print(f"    MAE={m['mae']:.3f} m   RMSE={m['rmse']:.3f} m   "
                      f"AbsRel={m['abs_rel']:.4f}   SqRel={m['sq_rel']:.4f}")
                print(f"    δ<1.25={m['d1']:.4f}   δ<1.25²={m['d2']:.4f}   δ<1.25³={m['d3']:.4f}")

        print(f"{sep}\n")

    # ------------------------------------------------------------------
    # BEV visualisation
    # ------------------------------------------------------------------

    def save_bev_comparison(
        self,
        pred_points: np.ndarray,
        gt_points: np.ndarray,
        output_path: str | Path,
        resolution: float = 0.1,
    ) -> None:
        """
        Save a BEV PNG overlaying pseudo-LiDAR (blue) and GT LiDAR (orange).

        Points drawn in order: GT first, then pseudo — so pseudo always visible
        on top where both occupy the same pixel.

        Args:
            pred_points: (N, 3) pseudo-LiDAR in ego frame.
            gt_points:   (M, 3) GT LiDAR in ego frame (already range-filtered).
            output_path: Destination .png path.
            resolution:  Metres per pixel (default 0.1 m → 1024×1024 for ±51.2 m).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        W = int((x_max - x_min) / resolution)
        H = int((y_max - y_min) / resolution)

        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        def _paint(pts: np.ndarray, color_rgb: Tuple[int, int, int]) -> None:
            # X → horizontal axis (forward), Y → vertical axis (left)
            px = ((pts[:, 0] - x_min) / resolution).astype(np.int32)
            py = ((pts[:, 1] - y_min) / resolution).astype(np.int32)
            valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            canvas[py[valid], px[valid]] = color_rgb

        _paint(gt_points, (220, 80, 20))     # orange-red  — GT LiDAR
        _paint(pred_points, (30, 120, 220))  # blue        — pseudo-LiDAR

        fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
        ax.imshow(canvas, origin="lower", interpolation="nearest")
        ax.set_title("BEV Comparison: Pseudo-LiDAR vs GT LiDAR", fontsize=14)
        ax.set_xlabel("Y axis — left/right  [pixels × {:.2f} m]".format(resolution))
        ax.set_ylabel("X axis — forward/back  [pixels × {:.2f} m]".format(resolution))
        ax.legend(
            handles=[
                mpatches.Patch(
                    color=(220 / 255, 80 / 255, 20 / 255),
                    label=f"GT LiDAR ({len(gt_points):,} pts)",
                ),
                mpatches.Patch(
                    color=(30 / 255, 120 / 255, 220 / 255),
                    label=f"Pseudo-LiDAR ({len(pred_points):,} pts)",
                ),
            ],
            loc="upper right",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(str(output_path), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved BEV comparison: {output_path}")


# ---------------------------------------------------------------------------
# Internal helper: project LiDAR onto a camera to get sparse GT depth
# (mirrors the logic in NuScenesLoader.get_sparse_lidar_depth)
# ---------------------------------------------------------------------------

def _project_lidar_to_cam(nusc, sample_token: str, cam_name: str) -> np.ndarray:
    """
    Project real LiDAR points onto a camera image plane.

    Returns:
        depth_sparse: (H, W) float32, zero where no LiDAR point hit.
    """
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import view_points

    sample = nusc.get("sample", sample_token)

    cam_token = sample["data"][cam_name]
    cam_sd = nusc.get("sample_data", cam_token)
    cs_cam = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    ego_pose_cam = nusc.get("ego_pose", cam_sd["ego_pose_token"])

    H, W = cam_sd["height"], cam_sd["width"]
    depth_sparse = np.zeros((H, W), dtype=np.float32)

    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sd = nusc.get("sample_data", lidar_token)
    lidar_path = Path(nusc.dataroot) / lidar_sd["filename"]
    pc = LidarPointCloud.from_file(str(lidar_path))

    cs_lidar = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    ego_pose_lidar = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

    # lidar sensor → ego (lidar time) → global → ego (cam time) → cam
    pc.rotate(BackProjector._quat_to_rot(np.array(cs_lidar["rotation"])))
    pc.translate(np.array(cs_lidar["translation"]))
    pc.rotate(BackProjector._quat_to_rot(np.array(ego_pose_lidar["rotation"])))
    pc.translate(np.array(ego_pose_lidar["translation"]))

    ego2global_cam = BackProjector.build_transform(
        np.array(ego_pose_cam["rotation"]),
        np.array(ego_pose_cam["translation"]),
    )
    global2ego_cam = np.linalg.inv(ego2global_cam)

    pts_global = pc.points[:3, :].T
    N = pts_global.shape[0]
    pts_ego_cam = (global2ego_cam @ np.hstack([pts_global, np.ones((N, 1))]).T).T[:, :3]

    cam2ego_mat = BackProjector.build_transform(
        np.array(cs_cam["rotation"]),
        np.array(cs_cam["translation"]),
    )
    ego2cam = np.linalg.inv(cam2ego_mat)
    pts_cam = (ego2cam @ np.hstack([pts_ego_cam, np.ones((N, 1))]).T).T[:, :3]

    in_front = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[in_front]

    intrinsic = np.array(cs_cam["camera_intrinsic"], dtype=np.float64)
    pts_2d = view_points(pts_cam.T, intrinsic, normalize=True)

    u = pts_2d[0].astype(np.int32)
    v = pts_2d[1].astype(np.int32)
    d = pts_cam[:, 2]

    in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    depth_sparse[v[in_image], u[in_image]] = d[in_image]

    return depth_sparse
