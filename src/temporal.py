"""
temporal.py
===========
SE(3) pose tracking: prediction, outlier gating, coasting and smoothing.
What it buys you, in the order you asked for it:

  * PREDICTION. A constant-velocity extrapolation of the previous accepted
    pose. Handed to the PnP solver it becomes a warm start; handed to the
    planar mirror decision it becomes the tie-breaker; handed to the ring
    matcher it becomes the continuity prior.
  * GATING. A frame whose pose differs from the prediction by more than the
    object can physically move in one frame time is rejected outright. This
    is what stops one bad detection from being logged as a 40 deg jump. The
    gate widens while coasting, so a genuinely fast move is not locked out
    forever -- it just has to survive a couple of consistent frames.
  * COASTING. Up to `max_coast` consecutive bad frames output the prediction
    instead of nothing, so the downstream consumer sees a continuous pose
    through a short dropout rather than a gap.
  * SMOOTHING. SLERP toward the measurement with a gain that adapts to the
    measurement's own quality, so noisy shallow-angle frames move the state
    less than clean head-on ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def log_so3(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, np.float64))
    return rvec.reshape(3)


def exp_so3(w: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(w, np.float64).reshape(3, 1))
    return R


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    d = np.asarray(Ra, np.float64).T @ np.asarray(Rb, np.float64)
    return float(np.degrees(np.arccos(np.clip((np.trace(d) - 1.0) / 2.0, -1.0, 1.0))))


def slerp_R(Ra: np.ndarray, Rb: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate on SO(3): Ra * exp(alpha * log(Ra^T Rb))."""
    return np.asarray(Ra, np.float64) @ exp_so3(alpha * log_so3(
        np.asarray(Ra, np.float64).T @ np.asarray(Rb, np.float64)))


@dataclass
class TrackerOutput:
    R: np.ndarray
    t: np.ndarray
    accepted: bool          # measurement passed the gate
    coasted: bool           # output is a prediction, not a measurement
    d_rot_deg: float
    d_trans: float
    reason: str = ""


class PoseTracker:
    """
    Parameters
    ----------
    max_rot_step_deg  per-frame rotation the object can plausibly make. Set
                      this from your actual slew rate; too tight and real
                      motion is rejected, too loose and the gate does nothing.
    max_trans_step    same for translation, in model units (metres).
    gate_growth       gate multiplier per consecutive rejected frame, so a
                      genuine fast move eventually gets through.
    max_coast         consecutive rejects tolerated before the track resets.
    alpha_rot/trans   base smoothing gain toward the measurement (1 = no
                      smoothing).
    warmup            frames accepted unconditionally after a reset.
    """

    def __init__(
        self,
        max_rot_step_deg: float = 12.0,
        max_trans_step: float = 0.04,
        gate_growth: float = 1.6,
        max_coast: int = 6,
        alpha_rot: float = 0.65,
        alpha_trans: float = 0.65,
        velocity_damping: float = 0.75,
        warmup: int = 2,
    ):
        self.max_rot_step_deg = float(max_rot_step_deg)
        self.max_trans_step = float(max_trans_step)
        self.gate_growth = float(gate_growth)
        self.max_coast = int(max_coast)
        self.alpha_rot = float(alpha_rot)
        self.alpha_trans = float(alpha_trans)
        self.velocity_damping = float(velocity_damping)
        self.warmup = int(warmup)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        self.R = None
        self.t = None
        self.w = np.zeros(3)          # rotation velocity, camera frame, per frame
        self.v = np.zeros(3)
        self.n_accepted = 0
        self.n_coast = 0

    @property
    def initialised(self) -> bool:
        return self.R is not None

    def predict(self):
        """(R, t) expected on the next frame, or None before initialisation."""
        if self.R is None:
            return None
        return exp_so3(self.w) @ self.R, self.t + self.v.reshape(3, 1)

    # ------------------------------------------------------------------ #
    def update(self, R_meas: np.ndarray, t_meas: np.ndarray,
               quality: float = 1.0) -> TrackerOutput:
        """
        quality in [0, 1]: 1 = clean measurement, 0 = barely usable. Drives
        the smoothing gain, so a shallow-angle frame with a big reprojection
        error nudges the state instead of yanking it.
        """
        R_meas = np.asarray(R_meas, np.float64)
        t_meas = np.asarray(t_meas, np.float64).reshape(3, 1)
        q = float(np.clip(quality, 0.0, 1.0))

        if self.R is None:
            self.R, self.t = R_meas.copy(), t_meas.copy()
            self.w[:] = 0.0
            self.v[:] = 0.0
            self.n_accepted = 1
            self.n_coast = 0
            return TrackerOutput(self.R.copy(), self.t.copy(), True, False, 0.0, 0.0, "init")

        R_pred, t_pred = self.predict()
        d_rot = geodesic_deg(R_pred, R_meas)
        d_tr = float(np.linalg.norm(t_meas - t_pred))

        grow = self.gate_growth ** self.n_coast
        rot_gate = self.max_rot_step_deg * grow
        tr_gate = self.max_trans_step * grow

        if self.n_accepted < self.warmup:
            ok, reason = True, "warmup"
        elif d_rot > rot_gate:
            ok, reason = False, f"rotation jump {d_rot:.1f}deg > {rot_gate:.1f}deg"
        elif d_tr > tr_gate:
            ok, reason = False, f"translation jump {d_tr:.3f} > {tr_gate:.3f}"
        else:
            ok, reason = True, ""

        if not ok:
            self.n_coast += 1
            if self.n_coast > self.max_coast:
                self.reset()
                return TrackerOutput(R_meas, t_meas, False, False, d_rot, d_tr,
                                     reason + " (track reset)")
            self.R, self.t = R_pred, t_pred
            self.w *= self.velocity_damping
            self.v *= self.velocity_damping
            return TrackerOutput(self.R.copy(), self.t.copy(), False, True,
                                 d_rot, d_tr, reason)

        a_r = self.alpha_rot * (0.35 + 0.65 * q)
        a_t = self.alpha_trans * (0.35 + 0.65 * q)
        R_new = slerp_R(R_pred, R_meas, a_r)
        t_new = t_pred + a_t * (t_meas - t_pred)

        self.w = (self.velocity_damping * self.w
                  + (1.0 - self.velocity_damping) * log_so3(R_new @ self.R.T))
        self.v = (self.velocity_damping * self.v
                  + (1.0 - self.velocity_damping) * (t_new - self.t).reshape(3))
        self.R, self.t = R_new, t_new
        self.n_accepted += 1
        self.n_coast = 0
        return TrackerOutput(self.R.copy(), self.t.copy(), True, False, d_rot, d_tr, "")

    def miss(self) -> TrackerOutput:
        """No usable measurement this frame (no detection, PnP failed)."""
        if self.R is None:
            return TrackerOutput(np.eye(3), np.zeros((3, 1)), False, False, 0.0, 0.0, "no track")
        self.n_coast += 1
        if self.n_coast > self.max_coast:
            self.reset()
            return TrackerOutput(np.eye(3), np.zeros((3, 1)), False, False, 0.0, 0.0,
                                 "track lost")
        R_pred, t_pred = self.predict()
        self.R, self.t = R_pred, t_pred
        self.w *= self.velocity_damping
        self.v *= self.velocity_damping
        return TrackerOutput(self.R.copy(), self.t.copy(), False, True, 0.0, 0.0, "coasting")


def quality_from_result(result, max_reproj_px: float = 4.0,
                        expected_points: int = 10) -> float:
    """Map a PoseEstimationResult onto [0, 1] for the tracker's gain."""
    if not getattr(result, "success", False):
        return 0.0
    err = max(result.reprojection_error_left, result.reprojection_error_right)
    q_err = float(np.clip(1.0 - err / max(max_reproj_px, 1e-6), 0.0, 1.0))
    q_inl = float(np.clip(result.num_inliers / max(expected_points, 1), 0.0, 1.0))
    q_amb = float(np.clip(getattr(result, "ambiguity_margin", 1.0) / 2.0, 0.0, 1.0))
    return float(q_err * (0.5 + 0.3 * q_inl + 0.2 * q_amb))