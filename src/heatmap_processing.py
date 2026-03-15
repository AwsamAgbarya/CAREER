import torch
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
class HeatmapProcessor:
    def __init__(self):
        pass

    def __call__(self, prediction, imgs=None, save_path=None, n_peaks=10, peaks_per_c=2, min_local_dist=3, threshold_rel=0.10, sigma_suppress=15.0):
        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(-1)
        B, C, H, W = prediction.shape
        heatmaps = prediction.cpu().detach().to(torch.float32)
        center_coords = torch.zeros((B,2))
        keypoint_coords = torch.zeros((B,n_peaks,2))
        amps = torch.zeros((B,n_peaks))

        for i,heatmap in enumerate(heatmaps):
            # Remove background
            flat   = heatmap.reshape(C, -1)
            bg_est = torch.quantile(flat, 0.98, dim=1).reshape(C, 1, 1)  # 30th percentile
            heatmap = torch.clamp(heatmap - bg_est, min=0.0)
            mx     = heatmap.amax(dim=(1,2), keepdim=True)
            heatmap_normalized = heatmap / (mx + 1e-6)

            # Separate keypoints and center
            keypoints = heatmap_normalized[:-1]
            center = heatmap_normalized[-1]
            keypoints = torch.clamp(keypoints - center,min=0)

            # Extract 10 highest keypoint peaks
            keypoint_coords[i], amps[i] = soft_nms(
                keypoints,
                n_peaks           = n_peaks,
                peaks_per_channel = peaks_per_c,  # max local maxima extracted per channel
                min_local_dist    = min_local_dist, # neighbourhood radius for local-max search
                threshold_rel     = threshold_rel,  # per-channel floor: fraction of ch max
                sigma_suppress    = sigma_suppress,  # Gaussian σ (px) for score suppression
            )

            flat_idx = torch.argmax(center)
            y = flat_idx // center.shape[1]
            x = flat_idx %  center.shape[1]
            center_coords[i] = torch.stack([x, y])
        
        if not save_path is None:
            all_kpts = torch.concat([keypoint_coords, center_coords[:,None,:]], dim=1)
            all_amps = torch.concat([amps, torch.tensor([1,1]).reshape(2,1)], dim=1)
            plot_sidebyside_keypoints(imgs, all_kpts, save_path, labels=all_amps)

        return keypoint_coords, center_coords, amps

def find_local_maxima_2d(score_map, min_distance = 6, threshold_rel = 0.05):
    """
    Return (ys, xs, values) of all local maxima in *score_map*, sorted
    by descending value.

    Parameters
    ----------
    score_map      : 2-D array (H, W)
    min_distance   : half-size of the maximum-filter neighbourhood (pixels)
    threshold_rel  : fraction of the global max used as a floor

    Returns
    -------
    ys, xs, values : 1-D arrays of equal length, sorted descending by value
    """
    score_np = score_map.detach().cpu().numpy()
    neighbourhood = maximum_filter(score_np, size=2 * min_distance + 1)

    is_local_max = (
        (score_map == torch.as_tensor(neighbourhood, device=score_map.device)) &
        (score_map >= threshold_rel * score_map.max())
    )

    ys, xs = torch.where(is_local_max)
    vals   = score_map[ys, xs]

    order  = torch.argsort(vals, descending=True)
    return ys[order], xs[order], vals[order]

def soft_nms(heatmap, n_peaks = 10, peaks_per_channel = 3, min_local_dist = 5, threshold_rel = 0.10, sigma_suppress = 20.0):
    """
    Two-stage extraction designed for ambiguous multi-peak heatmaps.
    1. Find candidate pool across channels
    2. Apply greedy max with a gaussian penalty per choice

    Parameters
    ----------
    heatmap        : tensor of shape (C, H, W)
    sigma_suppress : set to roughly half the expected inter-keypoint
                     distance in pixels. Smaller → more peaks survive
                     near each other; larger → sparser output.

    Returns
    -------
    peaks : tensor of shape (N, 2) with columns [x, y], length ≤ n_peaks
    """
    device = heatmap.device
    candidates_yx  = []   # list of [y, x] int tensors
    candidates_score = [] # list of scalar tensors (raw score)

    for c in range(heatmap.shape[0]):
        ch     = heatmap[c]
        ch_max = ch.max()
        if ch_max < 1e-9:
            continue

        ch_norm = ch / ch_max
        ys, xs, _ = find_local_maxima_2d(
            ch_norm,
            min_distance=min_local_dist,
            threshold_rel=threshold_rel,
        )
        for y, x in zip(ys[:peaks_per_channel], xs[:peaks_per_channel]):
            candidates_yx.append(torch.stack([y, x]))
            candidates_score.append(heatmap[c, y, x])

    if not candidates_yx:
        return torch.zeros((0, 2), dtype=torch.long, device=device)

    # arr : (N, 2) — [y, x] per candidate
    arr    = torch.stack(candidates_yx).float()
    scores = torch.stack(candidates_score).float()
    scores = scores.clone()                                      # avoid in-place modification of heatmap values

    selected = []
    amps = []

    for idx in range(n_peaks):
        if scores.max() < 1e-9:
            break

        best = torch.argmax(scores)
        by   = arr[best, 0]
        bx   = arr[best, 1]
        selected.append(torch.stack([bx, by]))
        amps.append(scores[best].item())

        # Gaussian soft suppression
        d_sq    = (arr[:, 0] - by) ** 2 + (arr[:, 1] - bx) ** 2
        penalty = torch.exp(-d_sq / (2.0 * sigma_suppress ** 2))
        scores *= (1.0 - penalty)

    if not selected:
        return torch.zeros((0, 2), dtype=torch.long, device=device), torch.zeros((0, 2), dtype=torch.long, device=device)
    
    return torch.stack(selected).to(dtype=torch.long), torch.tensor(amps)


def plot_sidebyside_keypoints(images, coords, output_path, radius = 3, color = "lime", linewidth = 1.5, figsize = (16, 8), labels = None, titles = ("Left", "Right")):
    assert images.shape[0] == 2 and coords.shape[0] == 2, \
        "First dimension of images and coords must be 2."

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for idx, ax in enumerate(axes):
        img = images[idx]
        img_np = img.permute(1, 2, 0).cpu().numpy()

        if img_np.dtype != np.uint8:
            lo, hi = img_np.min(), img_np.max()
            if hi > lo:
                img_np = (img_np - lo) / (hi - lo)
            img_np = img_np.clip(0.0, 1.0)

        ax.imshow(img_np)
        ax.set_title(titles[idx])
        ax.axis("off")

        kps = coords[idx].cpu().numpy()
        for i, (x, y) in enumerate(kps):
            circle = patches.Circle(
                (x, y),
                radius=radius,
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(circle)

            ax.plot(x, y, "+", color=color, markersize=5, markeredgewidth=0.8)

            if labels is not None:
                ax.text(
                    x + radius + 2, y,
                    f'{labels[idx, i].item():.3f}',
                    color=color,
                    fontsize=8,
                    va="center",
                )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)