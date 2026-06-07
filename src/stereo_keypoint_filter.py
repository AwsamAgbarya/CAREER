import numpy as np
import torch
import matplotlib.pyplot as plt

class StereoKeypointPipeline:
    """
    End-to-end stereo keypoint labeling and filtering pipeline.

    Input
    -----
    keypoints : torch.Tensor  shape (2, N, 2)  – pixel coords for L and R camera
    centers   : torch.Tensor  shape (2, 2)      – center coord for L and R camera

    The class processes both cameras through the full pipeline:
      1. Compute unit vectors from center to each keypoint
      2. Label keypoints CW from the largest angular gap
      3. Filter deterministically across both images (shared keep-mask when N > 6)
      4. Optionally save a 4-panel diagnostic plot

    Parameters
    ----------
    amp          : std amplification for outlier detection
    min_ang_deg  : minimum angular separation between kept vectors (degrees)
    max_kp       : only apply filters when N > this value (default 6)
    """

    def __init__(self, amp = 4.0, min_ang_deg = 15.0, max_kp = 6):
        self.amp = amp
        self.min_ang_deg = min_ang_deg
        self.max_kp      = max_kp

    def __call__(self, keypoints, centers, gt_2d, gt_c, amps, save_path = None):
        res_L = self.label(keypoints[0], centers[0], amps[0])
        res_R = self.label(keypoints[1], centers[1], amps[1])
        res_gt = self.label(gt_2d, gt_c, torch.ones((gt_2d.shape[0], 1)))
        gt_slot_to_idx = np.argsort(res_gt['labels'])
        gt_2d = gt_2d[gt_slot_to_idx[res_L['labels'] - 1]] 
        res_gt = self.label(gt_2d, gt_c, torch.ones((gt_2d.shape[0], 1)))

        assert len(res_L['valid']) == len(res_R['valid'])
        N = len(res_L['valid'])
        if N > self.max_kp:
            res_L, res_R = self.joint_filter(res_L, res_R)
            res_gt['keep'] = res_L['keep']

        if save_path is not None:
            vectors = np.stack([res_L['unit_vecs'], res_R['unit_vecs'], res_gt['unit_vecs']])
            valids = np.stack([res_L['keep'], res_R['keep'], res_gt['keep']])
            labels = np.stack([res_L['labels'], res_R['labels'], res_gt['labels']])
            plot_unit_vectors(vectors, valids, labels, output_path=save_path)

        return res_L, res_R, res_gt
    
    def label(self, keypoints, center, amps):
        """
        Run the full labeling + filtering pipeline for one camera image.
        """
        unit_vecs, angles, valid, norms = compute_unit_vectors(keypoints, center)

        ang_v  = angles[valid]
        labels_v = label_by_largest_gap(ang_v)
        labels_full = np.zeros(len(keypoints), dtype=int)
        labels_full[valid] = labels_v
        
        
        result = torch.empty_like(amps)
        result[labels_full-1] = amps

        return dict(
            keypoints  = keypoints,
            center     = center,
            unit_vecs  = unit_vecs,
            angles     = angles,
            valid      = valid,
            norms      = norms,
            labels     = labels_full,
            amps       = result,
        )


    def joint_filter(self, res_L, res_R):
        """
        Compute a *shared* keep-mask so that both images drop the same indices.
        """
        valid_L = res_L['valid']
        valid_R = res_R['valid']
        ang_L   = res_L['angles'][valid_L]
        ang_R   = res_R['angles'][valid_R]
        lbl_L   = res_L['labels'][valid_L]
        lbl_R   = res_R['labels'][valid_R]
        nrm_L   = res_L['norms'][valid_L]
        nrm_R   = res_R['norms'][valid_R]
        amps_L  = res_L['amps'][valid_L]
        amps_R  = res_R['amps'][valid_R]
        mean_amps = (amps_R + amps_L) / 2

        # Per-camera filter masks (valid-subset length)
        f1_L = filter_far_away(nrm_L, self.amp)
        f2_L = filter_min_angular_sep(ang_L, lbl_L, mean_amps, self.min_ang_deg)
        f1_R = filter_far_away(nrm_R, self.amp)
        f2_R = filter_min_angular_sep(ang_R, lbl_R, mean_amps, self.min_ang_deg)
        keep_l = (f1_L & f2_L)
        keep_r = (f1_R & f2_R)

        keep_by_label_L = {int(lbl): bool(kept) for lbl, kept in zip(lbl_L, keep_l)}
        keep_by_label_R = {int(lbl): bool(kept) for lbl, kept in zip(lbl_R, keep_r)}
        all_labels = set(keep_by_label_L) | set(keep_by_label_R)
        joint_by_label = {
            lbl: keep_by_label_L.get(lbl, False) and keep_by_label_R.get(lbl, False)
            for lbl in all_labels
        }
        keep_full_L = np.zeros(len(res_L['keypoints']), dtype=bool)
        keep_full_R = np.zeros(len(res_R['keypoints']), dtype=bool)

        valid_idx_L = np.where(valid_L)[0]
        valid_idx_R = np.where(valid_R)[0]

        for arr_idx, lbl in zip(valid_idx_L, lbl_L):
            keep_full_L[arr_idx] = joint_by_label.get(lbl, False)

        for arr_idx, lbl in zip(valid_idx_R, lbl_R):
            keep_full_R[arr_idx] = joint_by_label.get(lbl, False)
        
        res_L = {**res_L, 'keep': keep_full_L}
        res_R = {**res_R, 'keep': keep_full_R}
        return res_L, res_R


def compute_unit_vectors(keypoints, center):
    vecs  = keypoints.to(torch.float32) - center
    norms = np.linalg.norm(vecs, axis=1)
    valid = norms > 1e-6

    unit_vecs = np.zeros_like(vecs, dtype=float)
    unit_vecs[valid] = vecs[valid] / norms[valid, None]

    angles = np.arctan2(-unit_vecs[:, 1], unit_vecs[:, 0])
    return unit_vecs, angles, valid, norms

def label_by_largest_gap(angles):
    N = len(angles)
    if N == 0:
        return np.array([], dtype=int), 0.0, 0.0

    order_ccw   = np.argsort(angles)
    sorted_ang  = angles[order_ccw]

    # Gaps between consecutive CCW angles; last gap wraps around
    gaps = np.empty(N)
    for i in range(N - 1):
        gaps[i] = sorted_ang[i + 1] - sorted_ang[i]
    gaps[N - 1] = (sorted_ang[0] + 2 * np.pi) - sorted_ang[N - 1]

    big_i       = int(np.argmax(gaps))
    start_pos   = big_i % N
    labels      = np.zeros(N, dtype=int)

    for rank in range(1, N + 1):
        ccw_pos = (start_pos - (rank - 1)) % N
        labels[order_ccw[ccw_pos]] = rank

    return labels


def filter_min_angular_sep(angles, labels, scores, min_deg = 10.0):
    """
    Filter 2 Minimum angular separation on the unit circle.
    """
    min_rad = np.deg2rad(min_deg)
    N       = len(angles)
    keep    = np.ones(N, dtype=bool)
    order   = np.argsort(labels)

    for i in range(N):
        if not keep[order[i]]:
            continue
        for j in range(i + 1, N):
            if not keep[order[j]]:
                continue
            diff = abs(angles[order[i]] - angles[order[j]])
            diff = min(diff, 2 * np.pi - diff)
            if diff < min_rad:
                if scores[i] >= scores[j]:
                    keep[order[j]] = False
                else:
                    keep[order[i]] = False
                    break

    return keep

def filter_far_away(norms, amp = 4.0):
    means = norms.mean()
    stds   = norms.std()
    min_norm = means - stds * amp
    max_norm = means + stds * amp
    return np.logical_and(norms >= min_norm , norms <= max_norm)


def plot_unit_vectors(vectors, valid, indices, output_path, figsize = (6, 6), valid_color = "limegreen", invalid_color = "crimson", alpha = 0.85, arrow_scale = 0.35, titles=("Left", "Right", "GT")):
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    theta = np.linspace(0, 2 * np.pi, 300)

    for ax, vecs, valids, idxs, title in zip(axes, vectors, valid, indices, titles):
        # Unit circle for reference
        ax.plot(np.cos(theta), np.sin(theta), color="gray", lw=0.8, ls="--", zorder=0)

        for vec, is_valid, label in zip(vecs, valids, idxs):
            color = valid_color if is_valid else invalid_color
            dx, dy = vec * arrow_scale
            ax.annotate(
                "",
                xy=(dx, dy),
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.5,
                    mutation_scale=12,
                ),
                alpha=alpha,
                zorder=2,
            )
            ax.text(
                dx * 1.25, dy * 1.25,
                str(label),
                color=color,
                fontsize=8,
                ha="center",
                va="center",
            )

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect("equal")
        ax.axhline(0, color="lightgray", lw=0.5, zorder=0)
        ax.axvline(0, color="lightgray", lw=0.5, zorder=0)
        ax.scatter([0], [0], s=20, color="white", edgecolors="gray", zorder=3)
        ax.set_title(f"{title}")
        ax.invert_yaxis()
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)