import torch
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
class HeatmapProcessor:
    def __init__(self):
        pass

    def __call__(self, prediction, imgs=None, save_path=None, n_peaks=10, min_local_dist=3, threshold_rel=0.10):
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
            keypoint_coords[i], amps[i] = extract_peaks_per_channel(
                keypoints,
                min_local_dist=min_local_dist,
                threshold_rel=threshold_rel,
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

def extract_peaks_per_channel(heatmap, min_local_dist=3, threshold_rel=0.10):
    """
    One peak per channel — no cross-channel suppression.
    Returns coords (C, 2) as [x, y] and amps (C,).
    """
    C, H, W = heatmap.shape
    coords, amps = [], []

    for c in range(C):
        ch = heatmap[c]
        ch_max = ch.max()

        if ch_max < 1e-9:                           # dead channel → image centre
            coords.append(torch.tensor([W // 2, H // 2], dtype=torch.long))
            amps.append(torch.tensor(0.0))
            continue

        ys, xs, vals = find_local_maxima_2d(
            ch / ch_max,
            min_distance=min_local_dist,
            threshold_rel=threshold_rel,
        )

        if len(ys) == 0:                            # no local max → global argmax
            flat_idx = torch.argmax(ch)
            y, x = flat_idx // W, flat_idx % W
            val  = ch[y, x]
        else:
            y, x, val = ys[0], xs[0], vals[0]      # sorted descending already

        coords.append(torch.stack([x, y]))
        amps.append(ch[y, x])                       # raw (not normalized) amplitude

    return torch.stack(coords).to(dtype=torch.long), torch.stack(amps)

def cross_channel_nms(coords, amps, min_dist: float = 8.0):
    """
    Suppress peaks from *different* channels that are too close together.
    Greedily keeps the highest-amplitude peak; any lower-amplitude peak from
    a different channel within `min_dist` pixels is suppressed (amplitude 
    zeroed, coord replaced with image-centre sentinel).

    Parameters
    ----------
    coords   : (C, 2) long tensor  [x, y]
    amps     : (C,)   float tensor
    min_dist : minimum allowed Euclidean distance between cross-channel peaks

    Returns
    -------
    coords, amps  — same shape, suppressed entries zeroed out
    suppressed    — (C,) bool mask so callers can distinguish dead vs suppressed
    """
    C = coords.shape[0]
    suppressed = torch.zeros(C, dtype=torch.bool)

    # Process in descending amplitude order so the strongest peak always wins
    order = torch.argsort(amps, descending=True)

    for rank_i in range(C):
        ci = order[rank_i]
        if suppressed[ci]:
            continue
        for rank_j in range(rank_i + 1, C):
            cj = order[rank_j]
            if suppressed[cj]:
                continue
            dist = torch.norm(coords[ci].float() - coords[cj].float())
            if dist < min_dist:
                suppressed[cj] = True

    coords = coords.clone()
    amps   = amps.clone()
    amps[suppressed] = 0.0
    # Sentinel so downstream code can tell these apart from true dead channels
    coords[suppressed] = -1
    return coords, amps, suppressed

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