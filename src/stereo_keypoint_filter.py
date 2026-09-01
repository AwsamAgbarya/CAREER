"""
Decides, jointly for the left and right views, which detected hole corresponds
to which 3D model point.

The matcher makes the decision jointly and geometrically:

  1. Fit the ellipse properly (direct conic + RANSAC, not the covariance --
     the covariance is biased for this point layout) and de-squash to get
     near-circular angles.
  2. Enumerate every cyclic-order-preserving assignment of detections to
     model slots, with explicit skips, via a dynamic program. Cyclic order
     is preserved exactly under any homography, so this constraint is exact
     no matter how oblique the view.
  3. Score each candidate with the SUM of the left and right angular
     residuals, so the two views cannot disagree -- there is one labelling,
     evaluated in both images.
  4. Re-score the survivors with the actual plane->image homography
     reprojection error in both views. This is the exact projective test and
     it is what breaks ties when the whitened-angle heuristic is ambiguous.
  5. Add a temporal continuity term, so a marginally-better-scoring flip has
     to genuinely beat the incumbent to win.
  6. Optionally swap in a channel's runner-up heat-map peak if that lowers
     the reprojection error -- when the network is unsure the right hole is
     often the second peak, and throwing the channel away loses information
     you already have.

The output is a SHARED model-index set with the matching 2D points from both
views, so "filter the same ones on the left and the right" is structural
rather than something enforced afterwards by an intersection of keep-masks.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TWO_PI = 2.0 * np.pi
def wrap_pi(x):
    """Wrap angle(s) (radians) into (-pi, pi]."""
    x = np.asarray(x, dtype=np.float64)
    return (x + np.pi) % TWO_PI - np.pi


# direct-conic ellipse fit (Fitzgibbon-style total least squares)

def _fit_conic(pts: np.ndarray) -> np.ndarray:
    """Fit a general conic a x^2+b xy+c y^2+d x+e y+f=0 to `pts` (K,2) via the
    right null vector of the design matrix (direct linear fit)."""
    x, y = pts[:, 0], pts[:, 1]
    D = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], axis=1)
    s = np.mean(np.sqrt(x * x + y * y)) + 1e-9
    D[:, 0] /= s * s; D[:, 1] /= s * s; D[:, 2] /= s * s
    D[:, 3] /= s;     D[:, 4] /= s
    _, _, Vt = np.linalg.svd(D, full_matrices=True)
    p = Vt[-1]
    return p / np.array([s * s, s * s, s * s, s, s, 1.0])


def _conic_to_center_shape(p: np.ndarray):
    """Convert general-conic coefficients to (centre, (eigvals, eigvecs)) of
    the centred quadratic form Q' with X^T Q' X = 1 on the ellipse. Returns
    (None, None) if the conic is not a real, non-degenerate ellipse."""
    a, b, c, d, e, f = p
    Q = np.array([[a, b / 2.0], [b / 2.0, c]], dtype=np.float64)
    lin = np.array([d, e], dtype=np.float64)
    try:
        ctr = np.linalg.solve(Q, -lin / 2.0)
    except np.linalg.LinAlgError:
        return None, None

    def _fp(coef, centre):
        aa, bb, cc, dd, ee, ff = coef
        return (aa * centre[0] ** 2 + bb * centre[0] * centre[1] + cc * centre[1] ** 2
                + dd * centre[0] + ee * centre[1] + ff)

    fp = _fp(p, ctr)
    if fp >= -1e-12:
        p = -p
        a, b, c, d, e, f = p
        Q = np.array([[a, b / 2.0], [b / 2.0, c]], dtype=np.float64)
        fp = _fp(p, ctr)
        if fp >= -1e-12:
            return None, None
    Qn = Q / (-fp)
    w, V = np.linalg.eigh(Qn)
    if np.any(w <= 1e-12):
        return None, None
    return ctr, (w, V)


def _algebraic_residual(pts: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Sampson-style (gradient-normalised algebraic) distance of `pts` to the
    conic `p`, cheap and good enough for RANSAC scoring."""
    x, y = pts[:, 0], pts[:, 1]
    a, b, c, d, e, f = p
    val = a * x * x + b * x * y + c * y * y + d * x + e * y + f
    grad = np.stack([2 * a * x + b * y + d, b * x + 2 * c * y + e], axis=1)
    gn = np.linalg.norm(grad, axis=1) + 1e-9
    return np.abs(val) / gn


def robust_whitening(pts: np.ndarray, center: np.ndarray, ransac_iters: int = 60,
                     thresh_px: float = None, min_inliers: int = 5,
                     rng: np.random.Generator = None):
    """
    Fit an ellipse to `pts` with a direct-conic model, robustified by RANSAC
    over minimal 5-point subsets, then refit on the consensus set.

    Returns
    -------
    c : (2,) fitted ellipse centre. `center` (e.g. the object's projected
        centroid) only seeds the RANSAC scale/threshold -- the fit is free
        to (and should) settle on the true ellipse centre.
    T : (2,2) symmetric PD whitening matrix. For a centred point X = p - c,
        ||T @ X|| == 1 on the fitted ellipse, so T maps the ellipse onto the
        unit circle. Its eigenvalues are the reciprocal semi-axis lengths.
    inlier_mask : (K,) bool, which input points fed the final fit.
    """
    pts = np.asarray(pts, np.float64)
    center = np.asarray(center, np.float64)
    K = len(pts)
    if K < 5:
        r = float(np.mean(np.linalg.norm(pts - center, axis=1))) if K else 1.0
        r = max(r, 1e-6)
        return center.copy(), np.eye(2) / r, np.ones(K, bool)

    if rng is None:
        rng = np.random.default_rng(0)
    if thresh_px is None:
        scale = float(np.mean(np.linalg.norm(pts - center, axis=1))) + 1e-6
        thresh_px = 0.05 * scale

    best_mask, best_count = None, -1
    idx_all = np.arange(K)
    for _ in range(ransac_iters):
        sample = idx_all if K == 5 else rng.choice(idx_all, size=5, replace=False)
        p = _fit_conic(pts[sample])
        ctr, shape = _conic_to_center_shape(p)
        if ctr is None:
            continue
        mask = _algebraic_residual(pts, p) < thresh_px
        cnt = int(mask.sum())
        if cnt > best_count:
            best_count, best_mask = cnt, mask

    if best_mask is None or best_count < min_inliers:
        best_mask = np.ones(K, bool)

    p = _fit_conic(pts[best_mask])
    ctr, shape = _conic_to_center_shape(p)
    if ctr is None:
        p = _fit_conic(pts)
        ctr, shape = _conic_to_center_shape(p)
    if ctr is None:
        r = max(float(np.mean(np.linalg.norm(pts - center, axis=1))), 1e-6)
        return center.copy(), np.eye(2) / r, best_mask

    w, V = shape
    T = (V * np.sqrt(w)[None, :]) @ V.T
    return ctr.astype(np.float64), T.astype(np.float64), best_mask


def whitened_angles(pts: np.ndarray, c: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Map `pts` through the ellipse->circle whitening transform `T` about
    centre `c` and return the resulting arctan2 angles (radians)."""
    pts = np.asarray(pts, np.float64)
    d = (pts - np.asarray(c, np.float64)) @ np.asarray(T, np.float64).T
    return np.arctan2(d[:, 1], d[:, 0])


# homography (DLT) fit / reprojection error

def _dlt_homography(model_pts: np.ndarray, img_pts: np.ndarray, weights: np.ndarray = None):
    """Weighted, normalised DLT homography fit, model_pts (N,2) -> img_pts
    (N,2), N >= 4. Returns the 3x3 H (H[2,2] == 1) or None."""
    model_pts = np.asarray(model_pts, np.float64)
    img_pts = np.asarray(img_pts, np.float64)
    N = len(model_pts)
    if N < 4:
        return None
    w = np.ones(N) if weights is None else np.sqrt(np.maximum(np.asarray(weights, np.float64), 1e-6))

    def _norm(pts):
        c = pts.mean(axis=0)
        d = pts - c
        s = np.sqrt(2.0) / (np.mean(np.linalg.norm(d, axis=1)) + 1e-9)
        Tn = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        ph = np.hstack([pts, np.ones((len(pts), 1))])
        return (Tn @ ph.T).T[:, :2], Tn

    mn, Tm = _norm(model_pts)
    ino, Ti = _norm(img_pts)

    A = np.zeros((2 * N, 9))
    for i in range(N):
        X, Y = mn[i]; u, v = ino[i]; wi = w[i]
        A[2 * i] = wi * np.array([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u])
        A[2 * i + 1] = wi * np.array([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Ti) @ Hn @ Tm
    return H / (H[2, 2] if abs(H[2, 2]) > 1e-12 else 1.0)


def _apply_H(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64)
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    proj = (H @ ph.T).T
    w = proj[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return proj[:, :2] / w


def homography_fit(model_pts: np.ndarray, img_pts: np.ndarray, labels: np.ndarray,
                   weights: np.ndarray = None):
    """
    Fit a plane->image homography from `model_pts[labels[i]]` -> `img_pts[i]`
    over the assigned rows (`labels[i] >= 0`), weighted by `weights`.

    Returns (rms_px, orientation): `orientation` is sign(det(H)), the
    chirality of the fitted map (a real, non-mirrored view of the face has a
    single, learnable sign; a mirrored/impossible labelling flips it). Both
    are (`inf`, 0.0) if the fit is impossible or degenerate.
    """
    labels = np.asarray(labels, np.int64)
    sel = labels >= 0
    n = int(sel.sum())
    if n < 4:
        return float("inf"), 0.0
    mp = np.asarray(model_pts, np.float64)[labels[sel]]
    ip = np.asarray(img_pts, np.float64)[sel]
    w = None if weights is None else np.asarray(weights, np.float64)[sel]
    H = _dlt_homography(mp, ip, w)
    if H is None or not np.all(np.isfinite(H)):
        return float("inf"), 0.0
    res = np.linalg.norm(_apply_H(H, mp) - ip, axis=1)
    if w is None:
        rms = float(np.sqrt(np.mean(res ** 2)))
    else:
        wn = w / (w.sum() + 1e-12)
        rms = float(np.sqrt(np.sum(wn * res ** 2)))
    det = np.linalg.det(H)
    orientation = float(np.sign(det)) if abs(det) > 1e-15 else 0.0
    return rms, orientation


def homography_rms(model_pts: np.ndarray, img_pts: np.ndarray, labels: np.ndarray,
                   weights: np.ndarray = None) -> float:
    """Same fit as `homography_fit` but returns only the rms reprojection
    error, for call sites (e.g. peak polishing) that don't need chirality."""
    rms, _ = homography_fit(model_pts, img_pts, labels, weights)
    return rms


# cyclic-order-preserving assignment DP #

def _cyclic_dp(cost_mat: np.ndarray, gate: float, skip_cost: float):
    """
    Order-preserving weighted alignment. `cost_mat` is (n, M): row i is a
    detection in fixed cyclic order, column p is a model ring slot in fixed
    cyclic order (both already rotated to a common, tried, reference by the
    caller). Finds the minimum-cost way to either skip each detection (cost
    `skip_cost`) or assign it to a column strictly greater than the column
    used by the previous assigned detection (any earlier/unused columns are
    simply skipped for free -- a missing hole costs nothing beyond the
    detections it would have anchored). This one-directional, no-reordering
    constraint is exactly cyclic order preservation once the seam has been
    fixed by rotation.

    Returns (total_cost, col_for_row) with col_for_row[i] in [-1, M), -1 for
    a skipped detection.
    """
    n, M = cost_mat.shape
    INF = float("inf")
    dp = np.full(M + 1, INF)
    dp[0] = 0.0
    back = [None] * n
    for i in range(n):
        dp_new = np.full(M + 1, INF)
        choice = [None] * (M + 1)
        for s in range(M + 1):
            if dp[s] + skip_cost < dp_new[s]:
                dp_new[s] = dp[s] + skip_cost
                choice[s] = (s, -1)
        for s in range(M + 1):
            if not np.isfinite(dp[s]):
                continue
            for p in range(s, M):
                c = cost_mat[i, p]
                if c > gate:
                    continue
                cand = dp[s] + c
                newstate = p + 1
                if cand < dp_new[newstate]:
                    dp_new[newstate] = cand
                    choice[newstate] = (s, p)
        dp = dp_new
        back[i] = choice

    j_best = int(np.argmin(dp))
    total = float(dp[j_best])
    col_for_row = np.full(n, -1, dtype=np.int64)
    j = j_best
    for i in range(n - 1, -1, -1):
        prev_j, p = back[i][j]
        if p >= 0:
            col_for_row[i] = p
        j = prev_j
    return total, col_for_row


def cyclic_match(cost_fn, model_ang: np.ndarray, aL: np.ndarray, gate: float,
                 skip_cost: float, topk: int, allow_mirror: bool = True):
    """
    Enumerate cyclic-order-preserving labellings of the `n` detections
    (already sorted into cyclic order in `aL`) against the `M` model ring
    slots described by `model_ang`, via the order-preserving DP in
    `_cyclic_dp`.

    The object's spin (which model slot the first detection actually
    corresponds to) and, if `allow_mirror`, the viewing handedness (does the
    detection sequence run the same rotational direction as the model, or
    the reverse) are both unknown a priori. Both are combinatorial, not
    continuous, unknowns, so every rotation of the model sequence and, if
    enabled, both traversal directions of the detection sequence are tried;
    for each such hypothesis `cost_fn(roll, m_rel)` supplies the (n, M)
    angular-residual matrix that `_cyclic_dp` then solves exactly.

    Also rotates which detection sits at the DP's start ("seam"), which is
    necessary because the DP itself is a linear (non-wraparound) alignment;
    trying every seam position guarantees the true cyclic-consistent
    alignment is found regardless of where it happens to fall.

    Returns up to `topk` hypothesis dicts (ascending cost), each with:
      labels      (n,) int64 model-slot index per detection in `aL`'s order,
                  or -1 for a skipped detection.
      cost        total DP cost (angular units).
      n_assigned  number of non-skipped detections.
      mirror      bool, whether this hypothesis used the reversed traversal.
    """
    n = len(aL)
    model_ang = np.asarray(model_ang, np.float64)
    M = len(model_ang)
    if n == 0 or M == 0:
        return []

    model_order = np.argsort(model_ang)
    mang_sorted = model_ang[model_order]

    seq_normal = np.arange(n)
    seq_mirror = np.arange(n)[::-1]
    mirror_options = [False, True] if allow_mirror else [False]

    hyps, seen = [], set()
    for mirror in mirror_options:
        base_seq = seq_mirror if mirror else seq_normal
        for start in range(n):
            roll = np.roll(base_seq, -start)
            for mstart in range(M):
                mroll_idx = np.roll(model_order, -mstart)
                mang_roll = np.roll(mang_sorted, -mstart)
                m_rel = (mang_roll - mang_roll[0]) % TWO_PI

                cost_mat = cost_fn(roll, m_rel)
                total, col_for_row = _cyclic_dp(cost_mat, gate, skip_cost)
                if not np.isfinite(total):
                    continue

                labels = np.full(n, -1, dtype=np.int64)
                for k in range(n):
                    p = col_for_row[k]
                    if p >= 0:
                        labels[roll[k]] = mroll_idx[p]
                n_assigned = int((labels >= 0).sum())
                if n_assigned == 0:
                    continue

                key = (tuple(labels.tolist()), mirror)
                if key in seen:
                    continue
                seen.add(key)
                hyps.append({"labels": labels, "cost": float(total),
                           "n_assigned": n_assigned, "mirror": bool(mirror)})

    hyps.sort(key=lambda h: (h["cost"], -h["n_assigned"]))
    return hyps[:topk]

# Result container
@dataclass
class MatchResult:
    ok: bool
    model_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    channel_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    uv_left: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float64))
    uv_right: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float64))
    w_left: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    w_right: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    rms_left: float = float("inf")
    rms_right: float = float("inf")
    score: float = float("inf")
    delta: float = 0.0
    mirror: bool = False
    orientation: float = 0.0
    axis_ratio: float = 1.0
    reason: str = ""

    @property
    def n(self) -> int:
        return len(self.model_idx)


class StereoRingMatcher:
    """
    Parameters
    ----------
    model_ring_xy    (M, 2) hole coordinates in the face plane, relative to
                     the face centre. Order = model/channel index order.
    gate_deg         max whitened-angle residual for a detection->slot pair.
                     Generous is fine: the DP's ordering constraint plus the
                     homography re-scoring do the real discrimination.
    skip_deg         penalty for leaving a detection unassigned. Too low and
                     the DP happily discards everything; too high and one
                     outlier drags a whole labelling off by a slot.
    temporal_weight  weight on continuity with the previous frame's spin.
    max_rms_px       reject the frame if the best labelling still cannot be
                     explained by a plane homography to this accuracy.
    min_points       minimum shared points required to emit a match.
    """

    def __init__(
        self,
        model_ring_xy: np.ndarray,
        gate_deg: float = 30.0,
        skip_deg: float = 42.0,
        topk: int = 6,
        temporal_weight: float = 0.6,
        angle_weight: float = 3.0,
        missing_penalty: float = 0.35,
        max_rms_px: float = 6.0,
        min_points: int = 5,
        min_axis_ratio: float = 0.10,
        arbitrate_topk: int = 3,
        tie_margin: float = 0.75,
        expected_orientation: float = None,
        polish_candidates: bool = True,
        max_delta_rate: float = np.deg2rad(25.0),
    ):
        self.model_xy = np.asarray(model_ring_xy, np.float64)
        # The face centre lies on the same plane, so it is a free extra constraint for the homography test
        self.model_xy_aug = np.vstack([self.model_xy, [[0.0, 0.0]]])
        self._center_slot = len(self.model_xy)
        self.model_ang = np.arctan2(self.model_xy[:, 1], self.model_xy[:, 0])
        self.gate = np.deg2rad(gate_deg)
        self.skip_cost = np.deg2rad(skip_deg)
        self.topk = int(topk)
        self.temporal_weight = float(temporal_weight)
        self.angle_weight = float(angle_weight)
        self.missing_penalty = float(missing_penalty)
        self.max_rms_px = float(max_rms_px)
        self.min_points = int(min_points)
        self.min_axis_ratio = float(min_axis_ratio)
        self.arbitrate_topk = int(arbitrate_topk)
        self.tie_margin = float(tie_margin)
        self.expected_orientation = (None if expected_orientation is None
                                     else float(np.sign(expected_orientation)))
        self._orientation_locked = self.expected_orientation is not None
        self.polish_candidates = bool(polish_candidates)
        self.max_delta_rate = float(max_delta_rate)

        self.prev_delta = None
        self.prev_rate = 0.0
        self.prev_mirror = None
        self._misses = 0

    def commit(self, res: MatchResult):
        """Call after the frame has SURVIVED PnP. Keeping the temporal state
        conditional on downstream success stops a bad frame from poisoning
        the prior for the frames that follow it."""
        if not res.ok:
            self._misses += 1
            if self._misses > 5:
                self.reset()
            return
        if not self._orientation_locked and res.orientation != 0.0:
            self.expected_orientation = float(res.orientation)
            self._orientation_locked = True
        if self.prev_delta is not None:
            rate = float(wrap_pi(res.delta - self.prev_delta))
            self.prev_rate = np.clip(0.6 * self.prev_rate + 0.4 * rate,
                                     -self.max_delta_rate, self.max_delta_rate)
        self.prev_delta = float(res.delta)
        self.prev_mirror = bool(res.mirror)
        self._misses = 0

    def __call__(self, det, pose_scorer=None) -> MatchResult:
        """
        det : anything exposing .coords (2,K,2), .conf (2,K), .sigma (2,K),
              .valid (2,K), .center (2,2) and optionally .cand_coords /
              .cand_conf. See heatmap_processing.RingDetections.
        pose_scorer : optional callable(uv_left, uv_right, model_idx) -> float
              (lower better, e.g. stereo PnP reprojection error). Consulted
              only when the top two candidates are within `tie` of each other.
        """
        coords = np.asarray(det.coords, np.float64)
        conf = np.asarray(det.conf, np.float64)
        sigma = np.asarray(det.sigma, np.float64)
        valid = np.asarray(det.valid, bool)
        center = np.asarray(det.center, np.float64)

        shared = valid[0] & valid[1]
        ch = np.where(shared)[0]
        if len(ch) < self.min_points:
            return MatchResult(ok=False, reason=f"only {len(ch)} channels in both views")

        ptsL, ptsR = coords[0][ch], coords[1][ch]
        wL = conf[0][ch] / np.maximum(sigma[0][ch], 0.25)
        wR = conf[1][ch] / np.maximum(sigma[1][ch], 0.25)

        cL, TL, _ = robust_whitening(ptsL, center[0])
        cR, TR, _ = robust_whitening(ptsR, center[1])
        angL = whitened_angles(ptsL, cL, TL)
        angR = whitened_angles(ptsR, cR, TR)

        ratio = min(_axis_ratio(TL), _axis_ratio(TR))
        if self.min_axis_ratio > 0 and ratio < self.min_axis_ratio:
            return MatchResult(ok=False, axis_ratio=ratio,
                               reason=f"view too close to edge-on "
                                      f"(ellipse axis ratio {ratio:.3f} < "
                                      f"{self.min_axis_ratio:.3f})")

        order = np.argsort(angL)                       # cyclic order, left view
        aL, aR = angL[order], angR[order]

        def cost_fn(roll, m_rel):
            dL = (aL[roll] - aL[roll[0]]) % TWO_PI
            dR = (aR[roll] - aR[roll[0]]) % TWO_PI
            cl = np.abs(wrap_pi(dL[:, None] - m_rel[None, :]))
            cr = np.abs(wrap_pi(dR[:, None] - m_rel[None, :]))
            return 0.5 * (cl + cr)

        # Once the orientation is known, only one handedness is physically possible, so half the search space disappears.
        hyps = cyclic_match(cost_fn, self.model_ang, aL, self.gate, self.skip_cost, self.topk, allow_mirror=not self._orientation_locked)
        if not hyps:
            return MatchResult(ok=False, reason="no cyclic-order-consistent labelling")

        scored = []
        for h in hyps:
            lab_sorted = h["labels"]
            rmsL, orL = self._hfit(ptsL[order], lab_sorted, wL[order], center[0])
            rmsR, orR = self._hfit(ptsR[order], lab_sorted, wR[order], center[1])
            if not np.isfinite(rmsL) or not np.isfinite(rmsR):
                continue
            orient = orL if orL == orR else 0.0
            # A labelling with the wrong handedness reprojects perfectly and is still geometrically impossible.
            # Reject it outright rather than hoping the reprojection error notices.
            if self.expected_orientation is not None and orient != self.expected_orientation:
                continue
            n_ass = h["n_assigned"]
            delta = self._delta_of(aL, lab_sorted)

            score = 0.5 * (rmsL + rmsR)
            score += self.angle_weight * np.degrees(h["cost"] / max(n_ass, 1)) * 0.1
            score += self.missing_penalty * (len(ch) - n_ass)
            score += self._temporal_penalty(delta, h["mirror"])
            scored.append(dict(h=h, rmsL=rmsL, rmsR=rmsR, delta=delta,
                               score=score, orient=orient, axis_ratio=ratio))

        if not scored:
            return MatchResult(ok=False, reason="no labelling explainable by a homography")

        scored.sort(key=lambda d: d["score"])
        best = scored[0]

        # Real geometric arbitration. Always consult the pose scorer while
        # the orientation is still unknown, because the mirror pair is
        # indistinguishable by reprojection error and only a PnP solve --
        # which cannot produce a proper rotation from a reflected
        # correspondence -- can separate them.
        need_orient = not self._orientation_locked and len({s["orient"] for s in scored}) > 1
        # A PnP scorer is a stronger test than the homography (6 DoF and a
        # real camera model vs 8 DoF), but running it on every frame turned
        # out to HURT on continuous sequences: it overrules the temporal
        # continuity term on frames where the two labellings are genuinely
        # near-degenerate, and near-degenerate is exactly when continuity is
        # the more reliable evidence. So it is consulted only when the
        # geometry is actually undecided.
        close = len(scored) > 1 and (scored[1]["score"] - scored[0]["score"]) < self.tie_margin
        if pose_scorer is not None and (need_orient or close):
            if need_orient:
                tied, seen, picked = [], set(), set()
                for k, s in enumerate(scored):         # best of each handedness
                    if s["orient"] not in seen:
                        seen.add(s["orient"]); tied.append(s); picked.add(k)
                for k in range(min(2, len(scored))):
                    if k not in picked:
                        tied.append(scored[k])
            else:
                tied = [x for x in scored
                        if x["score"] - scored[0]["score"] < self.tie_margin][:self.arbitrate_topk]

            def _pnp_cost(s):
                r = self._assemble(s, ch, order, ptsL, ptsR, wL, wR)
                if not r.ok:
                    return float("inf")
                try:
                    return float(pose_scorer(r.uv_left, r.uv_right, r.model_idx))
                except Exception:
                    return float("inf")

            costs = [_pnp_cost(s) for s in tied]
            if np.isfinite(min(costs)):
                best = tied[int(np.argmin(costs))]

        res = self._assemble(best, ch, order, ptsL, ptsR, wL, wR)
        if not res.ok:
            return res

        if self.polish_candidates:
            res = self._polish(res, det)

        if max(res.rms_left, res.rms_right) > self.max_rms_px:
            return MatchResult(ok=False, delta=res.delta,
                               reason=f"homography rms {max(res.rms_left, res.rms_right):.2f}px "
                                      f"exceeds {self.max_rms_px:.2f}px")
        return res

    def _hfit(self, uv, labels, w, center_uv):
        lab = np.r_[labels, self._center_slot]
        uva = np.vstack([uv, np.asarray(center_uv, np.float64).reshape(1, 2)])
        wa = np.r_[w, np.median(w[labels >= 0]) if (labels >= 0).any() else 1.0]
        return homography_fit(self.model_xy_aug, uva, lab, wa)

    def _delta_of(self, ang_sorted, labels_sorted) -> float:
        """Circular mean of (whitened detection angle - model angle) over the
        assigned points. A continuous, viewpoint-stable proxy for the object's
        spin, used only as a temporal prior -- the real pose comes from PnP."""
        sel = labels_sorted >= 0
        if not sel.any():
            return 0.0
        d = wrap_pi(ang_sorted[sel] - self.model_ang[labels_sorted[sel]])
        return float(np.arctan2(np.sin(d).mean(), np.cos(d).mean()))

    def _temporal_penalty(self, delta: float, mirror: bool) -> float:
        pen = 0.0
        if self.prev_mirror is not None and bool(mirror) != self.prev_mirror:
            pen += 1.5 * self.temporal_weight
        if self.prev_delta is None:
            return pen
        pred = self.prev_delta + self.prev_rate
        pen += self.temporal_weight * np.degrees(abs(wrap_pi(delta - pred))) * 0.05
        return float(pen)

    def _assemble(self, s, ch, order, ptsL, ptsR, wL, wR) -> MatchResult:
        lab_sorted = s["h"]["labels"]
        sel = lab_sorted >= 0
        if int(sel.sum()) < self.min_points:
            return MatchResult(ok=False, reason="too few assigned after labelling")

        model_idx = lab_sorted[sel]
        pos = order[sel]                                # index into ch/ptsL/ptsR
        srt = np.argsort(model_idx)                     # deterministic ordering
        return MatchResult(
            ok=True,
            model_idx=model_idx[srt].astype(np.int64),
            channel_idx=ch[pos][srt].astype(np.int64),
            uv_left=ptsL[pos][srt],
            uv_right=ptsR[pos][srt],
            w_left=wL[pos][srt],
            w_right=wR[pos][srt],
            rms_left=float(s["rmsL"]),
            rms_right=float(s["rmsR"]),
            score=float(s["score"]),
            delta=float(s["delta"]),
            mirror=bool(s["h"]["mirror"]),
            orientation=float(s.get("orient", 0.0)),
            axis_ratio=float(s.get("axis_ratio", 1.0)),
        )

    def _polish(self, res: MatchResult, det) -> MatchResult:
        """
        Try each channel's alternative heat-map peaks and keep whichever best
        fits the plane homography. Cheap, and it recovers exactly the case you
        described: the network is unsure, so the correct hole ends up as the
        runner-up peak instead of the winner.
        """
        cand = getattr(det, "cand_coords", None)
        if cand is None:
            return res
        cand = np.asarray(cand, np.float64)
        if cand.ndim != 4 or cand.shape[2] < 2:
            return res

        labels = np.arange(res.n)
        for view, uv in ((0, res.uv_left), (1, res.uv_right)):
            base = homography_rms(self.model_xy[res.model_idx], uv, labels)
            if not np.isfinite(base):
                continue
            for i, k in enumerate(res.channel_idx):
                best_uv, best_err = uv[i].copy(), base
                for c in range(1, cand.shape[2]):
                    trial = uv.copy()
                    trial[i] = cand[view, k, c]
                    if np.allclose(trial[i], uv[i]):
                        continue
                    err = homography_rms(self.model_xy[res.model_idx], trial, labels)
                    if err < best_err - 0.15:
                        best_uv, best_err = trial[i].copy(), err
                if best_err < base:
                    uv[i] = best_uv
                    base = best_err
            if view == 0:
                res.rms_left = float(base)
            else:
                res.rms_right = float(base)
        return res

    # ------------------------------------------------------------------ #
    def plot(self, match: MatchResult, output_path: str = None, figsize=(11, 5.2)):
        """
        Side-by-side left/right debug view of a ring-matching result:
        matched detections (blue, labelled by model index and joined in
        TRUE cyclic order -- i.e. `argsort(model_ang)`, not detection order,
        so a genuine mislabel shows up as a crossed/knotted polygon rather
        than a merely-rotated one) versus the model ring reprojected through
        the per-view homography fitted to the winning labelling (orange
        dashed, with any unmatched model slots marked). A good match hugs
        the dashed ring even when the ellipse is heavily foreshortened; a
        bad one shows a blue ring that visibly crosses itself or drifts off
        the orange one.

        On failure (`match.ok is False`) it renders the failure reason
        instead, since there's nothing else to show -- there is no shared
        labelling to plot.
        """
        fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)

        if not match.ok or match.n == 0:
            for ax, name in ((axL, "left"), (axR, "right")):
                ax.set_title(f"{name}: NO MATCH", fontsize=10)
                ax.text(0.5, 0.5, f"reason:\n{match.reason or '(unknown)'}",
                        ha="center", va="center", wrap=True, fontsize=10,
                        transform=ax.transAxes, color="crimson")
                ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle("Ring match FAILED", color="crimson")
            fig.tight_layout()
            if output_path:
                fig.savefig(output_path, dpi=140)
            plt.close(fig)
            return fig

        HL = _dlt_homography(self.model_xy[match.model_idx], match.uv_left, match.w_left)
        HR = _dlt_homography(self.model_xy[match.model_idx], match.uv_right, match.w_right)

        _plot_ring_view(axL, match, self.model_xy, self.model_ang, match.uv_left,
                        f"LEFT  rms={match.rms_left:.2f}px  n={match.n}", HL)
        _plot_ring_view(axR, match, self.model_xy, self.model_ang, match.uv_right,
                        f"RIGHT  rms={match.rms_right:.2f}px  n={match.n}", HR)

        fig.suptitle(
            f"delta={np.degrees(match.delta):.1f} deg   mirror={match.mirror}   "
            f"orientation={match.orientation:+.0f}   axis_ratio={match.axis_ratio:.2f}   "
            f"score={match.score:.2f}",
            fontsize=10,
        )
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=140)
        plt.close(fig)
        return fig


def _plot_ring_view(ax, match: MatchResult, model_xy, model_ang, uv, title, H):
    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    if uv is not None and len(uv):
        ax.scatter(uv[:, 0], uv[:, 1], s=45, c="tab:blue", zorder=3, label="matched detections")
        for (x, y), mi in zip(uv, match.model_idx):
            ax.annotate(str(int(mi)), (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color="tab:blue")

        # Ring polygon in the TRUE geometric cyclic order (not detection
        # order), skipping any model slot this frame didn't assign.
        cyc = np.argsort(model_ang)
        pos_of_model = {int(m): i for i, m in enumerate(match.model_idx)}
        pts_in_cyc = [uv[pos_of_model[int(slot)]] for slot in cyc if int(slot) in pos_of_model]
        pts_in_cyc = np.array(pts_in_cyc)
        if len(pts_in_cyc) >= 2:
            closed = np.vstack([pts_in_cyc, pts_in_cyc[:1]])
            ax.plot(closed[:, 0], closed[:, 1], "-", color="tab:blue", lw=1.0,
                   alpha=0.6, zorder=2)

    if H is not None:
        proj_all = _apply_H(H, model_xy)
        cyc = np.argsort(model_ang)
        ring = np.vstack([proj_all[cyc], proj_all[cyc[:1]]])
        ax.plot(ring[:, 0], ring[:, 1], "--", color="tab:orange", lw=1.2,
               alpha=0.9, zorder=2, label="model reprojected via fitted H")
        ax.scatter(proj_all[:, 0], proj_all[:, 1], s=15, c="tab:orange",
                  marker="x", zorder=3)
        present = set(match.model_idx.tolist())
        for i in range(len(model_xy)):
            if i not in present:
                ax.annotate(f"{i}(missing)", proj_all[i], textcoords="offset points",
                           xytext=(5, -8), fontsize=7, color="tab:orange")

    ax.legend(fontsize=7, loc="best")


# --------------------------------------------------------------------------- #
def gt_correspondence(gt_uv_left, gt_uv_right, model_idx):
    """
    The ground-truth 2D points are produced by projecting the model in model
    index order, so their correspondence is the identity. The old code ran the
    full largest-gap labeller on GT and then re-sorted it against the
    detection labels, which could only ever introduce error into the metric.
    """
    idx = np.asarray(model_idx, np.int64)
    return np.asarray(gt_uv_left, np.float64)[idx], np.asarray(gt_uv_right, np.float64)[idx]


def _axis_ratio(T) -> float:
    """
    Minor/major axis ratio of the fitted ellipse. T maps the ellipse onto the
    unit circle, so its eigenvalues are the reciprocal semi-axes and the ratio
    falls straight out. ~1.0 head on, ~0.0 edge on.
    """
    ev = np.abs(np.linalg.eigvalsh(np.asarray(T, np.float64)))
    hi = float(ev.max())
    return float(ev.min() / hi) if hi > 1e-12 else 0.0