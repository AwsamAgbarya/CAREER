"""
Collects per-frame pose estimation results and writes a comprehensive
summary (CSV + human-readable text report) to disk.
"""

import os
import csv
import numpy as np
from dataclasses import dataclass
from typing import List
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict
import time

def wrap_deg(a):
    """Wrap degrees to (-180, 180]. Use this for theta_z error -- the old
    `abs(est - gt)` reports 359 deg for a 1 deg error near the wrap point and
    silently inflates your reported mean."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0

@dataclass
class FrameRecord:
    """One row of evaluation data per processed frame."""
    frame_idx: int

    # PnP diagnostics
    pnp_success: bool
    num_inliers: int
    reproj_error_left_px: float
    reproj_error_right_px: float

    # Rotation Z (object spin around vertical axis)
    theta_z_est_deg: float
    theta_z_gt_deg: float
    theta_z_err_deg: float  # |est - gt|, wrapped

    # Camera spherical coords (est)
    radius_est: float
    polar_est_deg: float
    azimuth_est_deg: float

    # Camera spherical coords (GT)
    radius_gt: float
    polar_gt_deg: float
    azimuth_gt_deg: float

    # Full rotation matrix angular error (geodesic distance)
    rotation_matrix_err_deg: float

    # Translation of the object in the LEFT camera frame (metres)
    t_est_x: float
    t_est_y: float
    t_est_z: float
    t_gt_x: float
    t_gt_y: float
    t_gt_z: float
    trans_err_x: float          # est - gt, signed, per axis
    trans_err_y: float
    trans_err_z: float
    trans_err_norm_m: float     # Euclidean translation error ||t_est - t_gt||


class PoseEvaluator:
    """
    Accumulates per-frame pose evaluation records and writes reports.

    Parameters
    ----------
    output_dir : str
        Directory where 'evaluation_results.csv' and
        'evaluation_summary.txt' will be written.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.records: List[FrameRecord] = []
        self.skipped_frames: List[int] = []

    def add_frame(
        self,
        frame_idx: int,
        pnp_result,
        view_model: dict,
        R_obj_est: np.ndarray,
        theta_z_est: float,
        r: float,
        theta: float,
        phi: float,
        t_cam_est: np.ndarray,
    ):
        """
        Record metrics for one successfully processed frame.

        Parameters
        ----------
        frame_idx   : integer frame index
        pnp_result  : PoseEstimationResult (must have .success == True)
        view_model  : dict loaded from the per-frame JSON params file
        R_obj_est   : (3,3) estimated object rotation matrix in world frame
        theta_z_est : estimated Z-rotation in degrees
        r, theta, phi : estimated camera spherical coordinates (world frame)
        t_cam_est   : (3,) or (3,1) estimated object translation in the
                      LEFT camera optical frame, metres (the PnP `t_cam_left`).
                      GT is derived from left_w2c_base under the convention
                      that the object's rest/rotation centre sits at the
                      world origin, so t_gt = left_w2c_base[:3, 3].
        """
        if not pnp_result.success:
            self.skipped_frames.append(frame_idx)
            return

        # --- Ground-truth values from the metadata JSON ---
        GT_deg = view_model["object_rotation_z_deg"]
        GT_R   = np.array(view_model["object_rotation_matrix"], dtype=np.float64)

        # GT camera spherical position in world (left camera)
        import torch
        left_w2c  = torch.tensor(view_model["left_w2c_base"])
        left_c2w  = torch.linalg.inv(left_w2c)
        cam_pos_gt = left_c2w.numpy()[:3, 3]
        r_gt     = float(np.linalg.norm(cam_pos_gt))
        theta_gt = float(np.degrees(np.arccos(np.clip(cam_pos_gt[2] / r_gt, -1, 1))))
        phi_gt   = float(np.degrees(np.arctan2(cam_pos_gt[1], cam_pos_gt[0])))

        # Full rotation matrix angular error (geodesic)
        R_err = R_obj_est @ GT_R.T
        trace_val = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        rot_err_deg = float(np.degrees(np.arccos(trace_val)))

        # --- Translation: object position in the LEFT camera frame ---
        # Object's rotation centre is fixed at the world origin, so its GT
        # position in the camera frame is just the world->camera translation.
        left_w2c_np = np.array(view_model["left_w2c_base"], dtype=np.float64)
        t_gt = left_w2c_np[:3, 3]

        t_est = np.asarray(t_cam_est, dtype=np.float64).reshape(3)
        t_diff = t_est - t_gt

        record = FrameRecord(
            frame_idx=frame_idx,
            pnp_success=True,
            num_inliers=pnp_result.num_inliers,
            reproj_error_left_px=float(pnp_result.reprojection_error_left),
            reproj_error_right_px=float(pnp_result.reprojection_error_right),
            theta_z_est_deg=float(theta_z_est),
            theta_z_gt_deg=float(GT_deg),
            theta_z_err_deg=float(abs(wrap_deg(theta_z_est - GT_deg))),
            radius_est=float(r),
            polar_est_deg=float(theta),
            azimuth_est_deg=float(phi),
            radius_gt=float(r_gt),
            polar_gt_deg=float(theta_gt),
            azimuth_gt_deg=float(phi_gt),
            rotation_matrix_err_deg=rot_err_deg,
            t_est_x=float(t_est[0]), t_est_y=float(t_est[1]), t_est_z=float(t_est[2]),
            t_gt_x=float(t_gt[0]),   t_gt_y=float(t_gt[1]),   t_gt_z=float(t_gt[2]),
            trans_err_x=float(t_diff[0]),
            trans_err_y=float(t_diff[1]),
            trans_err_z=float(t_diff[2]),
            trans_err_norm_m=float(np.linalg.norm(t_diff)),
        )
        self.records.append(record)

    def add_skipped_frame(self, frame_idx: int):
        """Mark a frame as skipped (no detection or PnP failed)."""
        self.skipped_frames.append(frame_idx)

    def write_report(self):
        """
        Write two output files:
            evaluation_results.csv   – one row per frame, all metrics
            evaluation_summary.txt   – human-readable aggregate statistics
        """
        csv_path = os.path.join(self.output_dir, "evaluation_results.csv")
        txt_path = os.path.join(self.output_dir, "evaluation_summary.txt")

        fieldnames = [
            "frame_idx", "pnp_success", "num_inliers",
            "reproj_err_left_px", "reproj_err_right_px",
            "theta_z_est_deg", "theta_z_gt_deg", "theta_z_err_deg",
            "radius_est", "polar_est_deg", "azimuth_est_deg",
            "radius_gt", "polar_gt_deg", "azimuth_gt_deg",
            "rotation_matrix_err_deg",
            "t_est_x", "t_est_y", "t_est_z",
            "t_gt_x", "t_gt_y", "t_gt_z",
            "trans_err_x", "trans_err_y", "trans_err_z",
            "trans_err_norm_m",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow({
                    "frame_idx":              r.frame_idx,
                    "pnp_success":            r.pnp_success,
                    "num_inliers":            r.num_inliers,
                    "reproj_err_left_px":     f"{r.reproj_error_left_px:.4f}",
                    "reproj_err_right_px":    f"{r.reproj_error_right_px:.4f}",
                    "theta_z_est_deg":        f"{r.theta_z_est_deg:.4f}",
                    "theta_z_gt_deg":         f"{r.theta_z_gt_deg:.4f}",
                    "theta_z_err_deg":        f"{r.theta_z_err_deg:.4f}",
                    "radius_est":             f"{r.radius_est:.4f}",
                    "polar_est_deg":          f"{r.polar_est_deg:.4f}",
                    "azimuth_est_deg":        f"{r.azimuth_est_deg:.4f}",
                    "radius_gt":              f"{r.radius_gt:.4f}",
                    "polar_gt_deg":           f"{r.polar_gt_deg:.4f}",
                    "azimuth_gt_deg":         f"{r.azimuth_gt_deg:.4f}",
                    "rotation_matrix_err_deg": f"{r.rotation_matrix_err_deg:.4f}",
                    "t_est_x":                f"{r.t_est_x:.5f}",
                    "t_est_y":                f"{r.t_est_y:.5f}",
                    "t_est_z":                f"{r.t_est_z:.5f}",
                    "t_gt_x":                 f"{r.t_gt_x:.5f}",
                    "t_gt_y":                 f"{r.t_gt_y:.5f}",
                    "t_gt_z":                 f"{r.t_gt_z:.5f}",
                    "trans_err_x":            f"{r.trans_err_x:.5f}",
                    "trans_err_y":            f"{r.trans_err_y:.5f}",
                    "trans_err_z":            f"{r.trans_err_z:.5f}",
                    "trans_err_norm_m":       f"{r.trans_err_norm_m:.5f}",
                })
        n_total    = len(self.records) + len(self.skipped_frames)
        n_success  = len(self.records)
        n_skipped  = len(self.skipped_frames)
        success_rt = 100.0 * n_success / n_total if n_total > 0 else 0.0

        if n_success > 0:
            z_errs      = [r.theta_z_err_deg          for r in self.records]
            rot_errs    = [r.rotation_matrix_err_deg  for r in self.records]
            reproj_l    = [r.reproj_error_left_px      for r in self.records]
            reproj_r    = [r.reproj_error_right_px     for r in self.records]
            inliers     = [r.num_inliers               for r in self.records]
            trans_errs  = [r.trans_err_norm_m          for r in self.records]
            trans_ex    = [r.trans_err_x               for r in self.records]
            trans_ey    = [r.trans_err_y               for r in self.records]
            trans_ez    = [r.trans_err_z               for r in self.records]

            def stats(vals):
                a = np.array(vals)
                return (float(np.mean(a)), float(np.median(a)),
                        float(np.std(a)),  float(np.min(a)), float(np.max(a)))

            mz, medzz, stdz, minz, maxz       = stats(z_errs)
            mr, medmr, stdr, minr, maxr       = stats(rot_errs)
            ml, medml, stdl, minl, maxl       = stats(reproj_l)
            mrr, medmrr, stdrr, minrr, maxrr  = stats(reproj_r)
            mt, medt, stdt, mint, maxt        = stats(trans_errs)
            metx, _, stdtx, _, _              = stats(trans_ex)
            mety, _, stdty, _, _              = stats(trans_ey)
            metz, _, stdtz, _, _              = stats(trans_ez)

            # Percentile breakdowns
            pct   = np.percentile(z_errs, [25, 50, 75, 90, 95])
            pct_t = np.percentile(trans_errs, [25, 50, 75, 90, 95])
        else:
            mz = medzz = stdz = minz = maxz = 0.0
            mr = medmr = stdr = minr = maxr = 0.0
            ml = medml = stdl = minl = maxl = 0.0
            mrr = medmrr = stdrr = minrr = maxrr = 0.0
            mt = medt = stdt = mint = maxt = 0.0
            metx = stdtx = mety = stdty = metz = stdtz = 0.0
            pct = [0] * 5
            pct_t = [0] * 5

        sep = "=" * 65
        lines = [
            sep,
            "  POSE ESTIMATION EVALUATION SUMMARY",
            sep,
            f"  Total frames attempted   : {n_total}",
            f"  Successful (PnP ok)      : {n_success}  ({success_rt:.1f}%)",
            f"  Skipped / failed         : {n_skipped}",
            sep,
            "  ROTATION Z ERROR  (|estimated - GT|, degrees)",
            f"    Mean   : {mz:.3f}°",
            f"    Median : {medzz:.3f}°",
            f"    Std    : {stdz:.3f}°",
            f"    Min    : {minz:.3f}°",
            f"    Max    : {maxz:.3f}°",
            f"    P25/P50/P75/P90/P95 : "
            f"{pct[0]:.2f}° / {pct[1]:.2f}° / {pct[2]:.2f}° / {pct[3]:.2f}° / {pct[4]:.2f}°",
            sep,
            "  FULL ROTATION MATRIX ERROR  (geodesic angle, degrees)",
            f"    Mean   : {mr:.3f}°",
            f"    Median : {medmr:.3f}°",
            f"    Std    : {stdr:.3f}°",
            f"    Min    : {minr:.3f}°",
            f"    Max    : {maxr:.3f}°",
            sep,
            "  TRANSLATION ERROR  (||t_est - t_gt||, metres, LEFT cam frame)",
            f"    Mean   : {mt:.4f} m",
            f"    Median : {medt:.4f} m",
            f"    Std    : {stdt:.4f} m",
            f"    Min    : {mint:.4f} m",
            f"    Max    : {maxt:.4f} m",
            f"    P25/P50/P75/P90/P95 : "
            f"{pct_t[0]:.4f} / {pct_t[1]:.4f} / {pct_t[2]:.4f} / {pct_t[3]:.4f} / {pct_t[4]:.4f} m",
            f"    Per-axis mean signed error (est - gt): "
            f"x={metx:+.4f} (std {stdtx:.4f})  "
            f"y={mety:+.4f} (std {stdty:.4f})  "
            f"z={metz:+.4f} (std {stdtz:.4f})  [m]",
            sep,
            "  REPROJECTION ERROR  (pixels)",
            f"    Left  – Mean: {ml:.2f}  Median: {medml:.2f}  Std: {stdl:.2f}  [{minl:.2f}, {maxl:.2f}]",
            f"    Right – Mean: {mrr:.2f}  Median: {medmrr:.2f}  Std: {stdrr:.2f}  [{minrr:.2f}, {maxrr:.2f}]",
            sep,
            "  INLIER COUNT",
            f"    Mean: {np.mean(inliers) if n_success else 0:.1f}   "
            f"Min: {min(inliers) if n_success else 0}   Max: {max(inliers) if n_success else 0}",
            sep,
        ]
        text = "\n".join(lines) + "\n"
        with open(txt_path, "w") as f:
            f.write(text)
        print(f"[Evaluation] CSV  → {csv_path}")
        print(f"[Evaluation] Text → {txt_path}")

class Mode(str, Enum):
    VERBOSE = "verbose"
    EVALUATION = "evaluation"
    PRODUCTION = "production"
class StageTimer:
    """No-op outside EVALUATION mode; accumulates per-stage wall clock in it."""

    def __init__(self, mode: Mode):
        self.mode = mode
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    @contextmanager
    def stage(self, name: str):
        if self.mode != Mode.EVALUATION:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] += time.perf_counter() - t0
            self.counts[name] += 1

    def report(self):
        if self.mode != Mode.EVALUATION or not self.totals:
            return
        print("\n=== Stage timing (mean per call) ===")
        order = sorted(self.totals, key=lambda k: -self.totals[k])
        grand = sum(self.totals.values())
        for name in order:
            n, total = self.counts[name], self.totals[name]
            print(f"  {name:<22s}: {total/n*1000:8.3f} ms/frame  "
                  f"(n={n:5d}, total {total:7.3f}s, {100*total/grand:5.1f}%)")
        print(f"  {'TOTAL (summed)':<22s}: "
              f"{grand/max(self.counts[order[0]],1)*1000:8.3f} ms/frame\n")