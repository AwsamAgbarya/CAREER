from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

_EPS = 1e-12
@dataclass
class PoseEstimationResult:
    success: bool
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    inliers: Optional[np.ndarray]
    reprojection_error_left: float
    reprojection_error_right: float
    num_inliers: int
    ambiguity_margin: float = 0.0     # runner-up score minus winner's
    view_cosine: float = 0.0          # >0 face turned toward the camera
    reason: str = ""


def view_cosine(R, t, n_obj) -> float:
    """
    Cosine of the angle between the outward face normal and the direction from
    the face to the camera centre.

        > 0  face turned toward the camera  (visible, keep)
        < 0  face turned away               (impossible, reject)
        ~ 0  edge-on, the test is degenerate -- reject the frame instead

    (R, t) is the object->camera pose, n_obj the outward face normal in OBJECT
    coordinates. The camera centre is the origin of the camera frame, so the
    face->camera direction is simply -t normalised.
    """
    R = np.asarray(R, np.float64).reshape(3, 3)
    t = np.asarray(t, np.float64).reshape(3)
    n_cam = R @ np.asarray(n_obj, np.float64).reshape(3)
    norm = float(np.linalg.norm(t))
    if norm < _EPS:
        return 0.0
    return float(n_cam @ (-t / norm))

def is_face_visible(R, t, n_obj, min_cos: float = 0.0) -> bool:
    """
    `min_cos` is a safety margin, expressed as cos(tilt). 0.0 accepts anything
    that is nominally front-facing. Note that a margin here is usually the
    wrong tool: at extreme tilt you want the FRAME rejected (layer 3), not the
    pose silently dropped, because at that point the pose is worthless whether
    or not it is front-facing.
    """
    if n_obj is None:
        return True
    return view_cosine(R, t, n_obj) > min_cos

def _failed(reason: str = "") -> PoseEstimationResult:
    return PoseEstimationResult(
        success=False, rotation_matrix=np.eye(3), translation_vector=np.zeros((3, 1)),
        rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)), inliers=None,
        reprojection_error_left=float("inf"), reprojection_error_right=float("inf"),
        num_inliers=0, reason=reason,
    )


class StereoPnPPoseEstimator:
    def __init__(
        self,
        camera_matrix_left: np.ndarray,
        camera_matrix_right: np.ndarray,
        dist_coeffs_left: Optional[np.ndarray] = None,
        dist_coeffs_right: Optional[np.ndarray] = None,
        R_stereo: Optional[np.ndarray] = None,
        t_stereo: Optional[np.ndarray] = None,
        reproj_threshold: float = 4.0,
        min_inliers: int = 5,
        planarity_tol: float = 1e-3,
        huber_scale: float = 3.0,
        face_normal_obj=None,
        min_view_cos: float = 0.0,
    ):
        self.K_left = np.asarray(camera_matrix_left, np.float64)
        self.K_right = np.asarray(camera_matrix_right, np.float64)
        self.dist_left = (np.zeros((5, 1)) if dist_coeffs_left is None
                          else np.asarray(dist_coeffs_left, np.float64))
        self.dist_right = (np.zeros((5, 1)) if dist_coeffs_right is None
                           else np.asarray(dist_coeffs_right, np.float64))
        self.R_stereo = np.eye(3) if R_stereo is None else np.asarray(R_stereo, np.float64)
        self.t_stereo = (np.zeros((3, 1)) if t_stereo is None
                         else np.asarray(t_stereo, np.float64).reshape(3, 1))
        self.reproj_threshold = float(reproj_threshold)
        self.min_inliers = int(min_inliers)
        self.planarity_tol = float(planarity_tol)
        self.huber_scale = float(huber_scale)
        self.face_normal_obj = (None if face_normal_obj is None
                                else np.asarray(face_normal_obj, np.float64).reshape(3))
        self.min_view_cos = float(min_view_cos)

    def _face_visible(self, rvec, tvec) -> bool:
        """Layer 2 of the visibility constraint (see face_orientation.py)."""
        if self.face_normal_obj is None:
            return True
        R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
        return is_face_visible(R, tvec, self.face_normal_obj, self.min_view_cos)

    def _view_cos(self, rvec, tvec) -> float:
        if self.face_normal_obj is None:
            return 0.0
        R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
        return view_cosine(R, tvec, self.face_normal_obj)

    # Projection helpers
    def _project(self, X, rvec, tvec, right: bool):
        if right:
            R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
            Rr = self.R_stereo @ R
            tr = self.R_stereo @ np.asarray(tvec, np.float64).reshape(3, 1) + self.t_stereo
            rv, _ = cv2.Rodrigues(Rr)
            p, _ = cv2.projectPoints(X, rv, tr, self.K_right, self.dist_right)
        else:
            p, _ = cv2.projectPoints(X, np.asarray(rvec, np.float64).reshape(3, 1),
                                     np.asarray(tvec, np.float64).reshape(3, 1),
                                     self.K_left, self.dist_left)
        return p.reshape(-1, 2)

    def _errors(self, X, uvL, uvR, rvec, tvec):
        eL = np.linalg.norm(self._project(X, rvec, tvec, False) - uvL, axis=1)
        eR = np.linalg.norm(self._project(X, rvec, tvec, True) - uvR, axis=1)
        return eL, eR

    def estimate_pose(
        self,
        object_points: np.ndarray,
        uv_left: np.ndarray,
        uv_right: np.ndarray,
        w_left: np.ndarray = None,
        w_right: np.ndarray = None,
        predicted_pose: tuple = None,
        use_right_view: bool = True,
    ) -> PoseEstimationResult:
        """
        object_points : (N, 3) model points in the object rest frame
        uv_left/right : (N, 2) pixel coordinates, SAME ordering
        w_left/right  : (N,)   optional positive weights (conf / sigma)
        predicted_pose: optional (R, t) from the temporal tracker, used both
                        as an extra initialisation and as the tie-breaker for
                        the planar mirror ambiguity.
        """
        X = np.ascontiguousarray(np.asarray(object_points, np.float64))
        uvL = np.ascontiguousarray(np.asarray(uv_left, np.float64))
        uvR = np.ascontiguousarray(np.asarray(uv_right, np.float64))
        N = len(X)
        if N < 4 or len(uvL) != N or len(uvR) != N:
            return _failed(f"need >=4 matched points, got {N}")

        wL = np.ones(N) if w_left is None else np.asarray(w_left, np.float64)
        wR = np.ones(N) if w_right is None else np.asarray(w_right, np.float64)
        wL = wL / max(wL.mean(), 1e-9)
        wR = wR / max(wR.mean(), 1e-9)
        if not use_right_view:
            wR = np.zeros_like(wR)

        hyps = self._initial_hypotheses(X, uvL, predicted_pose)
        if not hyps:
            return _failed("no PnP hypothesis could be computed")

        scored = []
        for rvec, tvec, tag in hyps:
            if tvec is None or not np.isfinite(tvec).all():
                continue
            eL, eR = self._errors(X, uvL, uvR, rvec, tvec)
            comb = eL + (eR if use_right_view else 0.0)
            thr = self.reproj_threshold * (2.0 if use_right_view else 1.0)
            inl = comb < max(thr, 1e-6)
            if inl.sum() < min(self.min_inliers, N):
                inl = comb <= np.partition(comb, min(self.min_inliers, N) - 1)[
                    min(self.min_inliers, N) - 1]
            if inl.sum() < 4:
                continue

            rv, tv = self._refine(X[inl], uvL[inl], uvR[inl], wL[inl], wR[inl], rvec, tvec)
            # Refinement is unconstrained, so re-test rather than assume the
            # hypothesis stayed on the front-facing side of the boundary.
            if not self._face_visible(rv, tv):
                continue
            eL, eR = self._errors(X, uvL, uvR, rv, tv)
            comb = eL + (eR if use_right_view else 0.0)
            inl = comb < max(thr, 1e-6)
            if inl.sum() < 4:
                continue

            rmsL = float(np.sqrt(np.average(eL[inl] ** 2, weights=np.maximum(wL[inl], 1e-6))))
            rmsR = float(np.sqrt(np.average(eR[inl] ** 2, weights=np.maximum(wR[inl], 1e-6)))) \
                if use_right_view else 0.0

            # The right view carries the mirror decision, so it gets full say.
            score = rmsL + rmsR
            score += 0.5 * (N - int(inl.sum()))
            score += self._temporal_cost(rv, tv, predicted_pose)
            scored.append((score, rv, tv, inl, rmsL, rmsR, tag))

        if not scored:
            # Distinguish the two failure causes: "everything was back-facing"
            # usually means the labelling was the mirrored one (fine, the
            # matcher will be told to try another) or that face_normal_obj is
            # INVERTED (not fine, and it would otherwise look like a quiet
            # drop in solve rate).
            return _failed("all hypotheses rejected"
                           + (" (all back-facing -- check face_normal_obj sign)"
                              if self.face_normal_obj is not None else ""))

        scored.sort(key=lambda s: s[0])
        score, rv, tv, inl, rmsL, rmsR, tag = scored[0]
        margin = float(scored[1][0] - score) if len(scored) > 1 else float("inf")

        if int(inl.sum()) < self.min_inliers:
            return _failed(f"only {int(inl.sum())} inliers (< {self.min_inliers})")

        R, _ = cv2.Rodrigues(rv)
        return PoseEstimationResult(
            success=True, rotation_matrix=R, translation_vector=tv.reshape(3, 1),
            rvec=rv.reshape(3, 1), tvec=tv.reshape(3, 1),
            inliers=np.where(inl)[0], reprojection_error_left=rmsL,
            reprojection_error_right=rmsR, num_inliers=int(inl.sum()),
            ambiguity_margin=margin, reason=tag,
        )

    def _initial_hypotheses(self, X, uvL, predicted_pose):
        """
        Collect every plausible starting pose. For a planar target IPPE
        returns the two mirror solutions; keeping BOTH and letting the right
        camera choose is the fix for the flipping.
        """
        hyps = []
        plane = self._plane_frame(X)

        if plane is not None:
            A, c, Xl = plane
            try:
                n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    np.ascontiguousarray(Xl), np.ascontiguousarray(uvL),
                    self.K_left, self.dist_left, flags=cv2.SOLVEPNP_IPPE)
                for i in range(int(n_sol)):
                    Rl, _ = cv2.Rodrigues(rvecs[i])
                    R = Rl @ A.T
                    t = tvecs[i].reshape(3, 1) - R @ c.reshape(3, 1)
                    rv, _ = cv2.Rodrigues(R)
                    hyps.append((rv, t, f"ippe{i}"))
            except cv2.error:
                pass

        for flag, name in ((cv2.SOLVEPNP_SQPNP, "sqpnp"), (cv2.SOLVEPNP_EPNP, "epnp")):
            try:
                ok, rv, tv = cv2.solvePnP(X, uvL, self.K_left, self.dist_left, flags=flag)
                if ok:
                    hyps.append((rv, tv, name))
            except cv2.error:
                pass

        if predicted_pose is not None:
            Rp, tp = predicted_pose
            try:
                ok, rv, tv = cv2.solvePnP(
                    X, uvL, self.K_left, self.dist_left,
                    cv2.Rodrigues(np.asarray(Rp, np.float64))[0],
                    np.asarray(tp, np.float64).reshape(3, 1),
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    hyps.append((rv, tv, "warmstart"))
            except cv2.error:
                pass
            hyps.append((cv2.Rodrigues(np.asarray(Rp, np.float64))[0],
                         np.asarray(tp, np.float64).reshape(3, 1), "prediction"))

        # Keep only hypotheses that put the object in front of the camera
        # AND turn the interface face toward it.
        return [(rv, tv, tag) for rv, tv, tag in hyps
                if tv is not None and np.isfinite(np.asarray(tv)).all()
                and float(np.asarray(tv).reshape(3)[2]) > 0
                and self._face_visible(rv, tv)]

    def _plane_frame(self, X):
        """Return (A, centroid, X_local) with X_local[:, 2] ~ 0, or None."""
        c = X.mean(axis=0)
        try:
            _, S, Vt = np.linalg.svd(X - c, full_matrices=True)
        except np.linalg.LinAlgError:
            return None
        if S[0] < 1e-12 or S[2] > self.planarity_tol * S[0]:
            return None
        A = Vt.T.copy()
        if np.linalg.det(A) < 0:
            A[:, 2] *= -1.0
        Xl = np.ascontiguousarray((X - c) @ A)
        Xl[:, 2] = 0.0
        return A, c, Xl

    def _refine(self, X, uvL, uvR, wL, wR, rvec, tvec):
        """Weighted Huber LM over BOTH views at once."""
        p0 = np.concatenate([np.asarray(rvec, np.float64).reshape(3),
                             np.asarray(tvec, np.float64).reshape(3)])

        def resid(p):
            rv, tv = p[:3], p[3:]
            rL = (self._project(X, rv, tv, False) - uvL) * wL[:, None]
            rR = (self._project(X, rv, tv, True) - uvR) * wR[:, None]
            return np.concatenate([rL.ravel(), rR.ravel()])

        try:
            sol = least_squares(resid, p0, method="trf", loss="huber",
                                f_scale=self.huber_scale, max_nfev=120)
            if sol.success or sol.status > 0:
                return sol.x[:3].reshape(3, 1), sol.x[3:].reshape(3, 1)
        except Exception:
            pass
        return np.asarray(rvec, np.float64).reshape(3, 1), np.asarray(tvec, np.float64).reshape(3, 1)

    @staticmethod
    def _temporal_cost(rvec, tvec, predicted_pose):
        if predicted_pose is None:
            return 0.0
        Rp, tp = predicted_pose
        R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
        dR = np.asarray(Rp, np.float64).T @ R
        ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
        dt = float(np.linalg.norm(np.asarray(tvec).reshape(3) - np.asarray(tp).reshape(3)))
        # Deliberately gentle: this nudges the mirror decision, it must never
        # be able to hold a genuinely moving object in place.
        return 0.04 * ang + 2.0 * dt

    def score_correspondence(self, uv_left, uv_right, object_points, predicted_pose=None) -> float:
        """
        Scorer for StereoRingMatcher's `pose_scorer` hook.

        Deliberately NOT `reprojection_error_left + right`: those are computed
        over INLIERS only, so a wrong labelling that manages to explain just
        five of the ten points scores better than the right one that explains
        all ten. The cost here is a truncated (M-estimator) mean over EVERY
        supplied point in both views, so failing to explain a point costs the
        full clamp instead of being quietly excluded.
        """
        X = np.ascontiguousarray(np.asarray(object_points, np.float64))
        uvL = np.ascontiguousarray(np.asarray(uv_left, np.float64))
        uvR = np.ascontiguousarray(np.asarray(uv_right, np.float64))
        r = self.estimate_pose(X, uvL, uvR, predicted_pose=predicted_pose)
        if not r.success:
            return float("inf")
        eL, eR = self._errors(X, uvL, uvR, r.rvec, r.tvec)
        clamp = 3.0 * self.reproj_threshold
        return float(np.mean(np.minimum(eL, clamp) + np.minimum(eR, clamp)))