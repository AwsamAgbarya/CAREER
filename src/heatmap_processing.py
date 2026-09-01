from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class RingDetections:
    """
    Per-view ring keypoint detections
    B is normally 2 (left, right) and the channel index k is the network's
    identity claim for hole k - treated as a prior.

    coords       (B, K, 2) float32  sub-pixel [x, y] in CROP pixels
    conf         (B, K)    float32  in [0, 1]
    sigma        (B, K)    float32  localisation std in px (>= 0.25)
    valid        (B, K)    bool
    reasons      (B, K)    list[list[str]]  human-readable reason each
                            channel was marked invalid ("" when valid)
    center       (B, 2)    float32  sub-pixel centre keypoint (channel K)
    center_conf  (B,)      float32
    center_sigma (B,)      float32
    cand_coords  (B, K, C, 2)  alternative peaks per channel, best first
    cand_conf    (B, K, C)
    """
    coords: torch.Tensor
    conf: torch.Tensor
    sigma: torch.Tensor
    valid: torch.Tensor
    reasons: list
    center: torch.Tensor
    center_conf: torch.Tensor
    center_sigma: torch.Tensor
    cand_coords: Optional[torch.Tensor] = field(default=None)
    cand_conf: Optional[torch.Tensor] = field(default=None)

    @property
    def num_views(self) -> int:
        return self.coords.shape[0]

    @property
    def num_channels(self) -> int:
        return self.coords.shape[1]

    def weights(self) -> torch.Tensor:
        """(B, K) PnP weights: confident and tightly localised -> heavy."""
        return self.conf / torch.clamp(self.sigma, min=0.25)

    def center_weights(self) -> torch.Tensor:
        return self.center_conf / torch.clamp(self.center_sigma, min=0.25)

    def to(self, device) -> "RingDetections":
        for f_ in self.__dataclass_fields__:
            v = getattr(self, f_)
            if isinstance(v, torch.Tensor):
                setattr(self, f_, v.to(device))
        return self

    def reason_for(self, b: int, k: int) -> str:
        """Convenience accessor: reasoning string for view b, channel k."""
        return self.reasons[b][k]

    # Visualization
    def plot(self, images: torch.Tensor, output_path: str, view: Optional[int] = None,
              radius: float = 3, color_valid: str = "lime", color_invalid: str = "red",
              center_color: str = "cyan", linewidth: float = 1.5, figsize=(16, 8),
              show_conf: bool = True, show_reason: bool = True, reason_fontsize: float = 6.5,
              reason_wrap: int = 26, titles=("Left", "Right")):
        """
        Plot detected keypoints on top of the (B, 3, H, W) crop images that were
        fed to the network. If ``view`` is None and B == 2, a side-by-side plot
        of both views is produced (mirrors the old plot_sidebyside_keypoints
        helper); otherwise a single view is plotted. Invalid channels are drawn
        faded in ``color_invalid`` so they stay visible for debugging.

        For invalid channels, when ``show_reason`` is True the stored
        ``reasons`` string is used as the point's text label instead of the
        confidence value, so it's immediately visible *why* a channel was
        rejected (dead peak, low confidence, centre leakage, NMS duplicate).
        """
        import textwrap

        single = view is not None or self.num_views == 1
        idxs = [view if view is not None else 0] if single else list(range(self.num_views))

        fig, axes = plt.subplots(1, len(idxs), figsize=figsize)
        if len(idxs) == 1:
            axes = [axes]

        for ax, b in zip(axes, idxs):
            img = images[b].detach()
            img_np = img.permute(1, 2, 0).cpu().numpy()
            if img_np.dtype != "uint8":
                lo, hi = img_np.min(), img_np.max()
                if hi > lo:
                    img_np = (img_np - lo) / (hi - lo)
                img_np = img_np.clip(0.0, 1.0)

            ax.imshow(img_np)
            ax.set_title(titles[b] if b < len(titles) else f"View {b}")
            ax.axis("off")

            coords_b = self.coords[b].detach().cpu()
            conf_b = self.conf[b].detach().cpu()
            valid_b = self.valid[b].detach().cpu()
            reasons_b = self.reasons[b] if self.reasons is not None else None

            for k in range(coords_b.shape[0]):
                x, y = coords_b[k].tolist()
                is_valid = bool(valid_b[k])
                color = color_valid if is_valid else color_invalid
                circle = patches.Circle((x, y), radius=radius, linewidth=linewidth,
                                          edgecolor=color, facecolor="none",
                                          alpha=1.0 if is_valid else 0.4)
                ax.add_patch(circle)
                ax.plot(x, y, "+", color=color, markersize=5, markeredgewidth=0.8,
                         alpha=1.0 if is_valid else 0.4)

                if not is_valid and show_reason and reasons_b is not None:
                    reason_txt = reasons_b[k] or "invalid"
                    wrapped = "\n".join(textwrap.wrap(f"k{k}: {reason_txt}", width=reason_wrap))
                    ax.text(x + radius + 2, y, wrapped,
                             color=color, fontsize=reason_fontsize, va="center",
                             alpha=0.85,
                             bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.35))
                elif show_conf:
                    ax.text(x + radius + 2, y, f"{conf_b[k].item():.3f}",
                             color=color, fontsize=8, va="center",
                             alpha=1.0 if is_valid else 0.4)

            cx, cy = self.center[b].detach().cpu().tolist()
            ax.plot(cx, cy, "x", color=center_color, markersize=8, markeredgewidth=2.0)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


class HeatmapProcessor:
    """
    Parameters
    ----------
    bg_quantile        per-channel quantile used as the background floor.
    peak_radius        half-width of the local-maximum neighbourhood (px).
    centroid_radius    half-width of the sub-pixel centroid window (px).
    n_candidates       alternative peaks kept per channel.
    min_conf           channels whose best peak is weaker than this are
                       marked invalid rather than hallucinated.
    dup_dist           two channels peaking within this many px are treated
                       as the same physical hole; the weaker one is dropped.
    center_exclusion   a ring peak this close to the centre channel's peak is
                       centre leakage, not a hole. Set to 0 to disable.
    subtract_center    subtract the centre channel from the ring channels
                       (the old behaviour) before peak-picking.
    """

    def __init__(self, bg_quantile: float = 0.50, peak_radius: int = 3,
                 centroid_radius: int = 2, n_candidates: int = 3,
                 min_conf: float = 0.05, dup_dist: float = 2.0,
                 center_exclusion: float = 4.0, subtract_center: bool = True):
        self.bg_quantile = float(bg_quantile)
        self.peak_radius = int(peak_radius)
        self.centroid_radius = int(centroid_radius)
        self.n_candidates = int(n_candidates)
        self.min_conf = float(min_conf)
        self.dup_dist = float(dup_dist)
        self.center_exclusion = float(center_exclusion)
        self.subtract_center = bool(subtract_center)

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor) -> RingDetections:
        """
        prediction : (B, C, H, W) or (C, H, W) raw network output. Channel
                     C-1 is the face-centre channel; 0..C-2 are the holes.
        Everything stays on prediction.device / dtype float32 the whole way.
        """
        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(0)
        if prediction.ndim != 4:
            raise ValueError(f"expected (B,C,H,W), got {tuple(prediction.shape)}")

        device = prediction.device
        hm = prediction.detach().to(torch.float32)
        B, C, H, W = hm.shape
        K = C - 1
        r_c = self.centroid_radius
        n_cand = min(self.n_candidates, H * W)

        raw_peak_all = hm.reshape(B, C, -1).amax(dim=2)              # (B, C)

        flat = hm.reshape(B, C, -1)
        bg = torch.quantile(flat, self.bg_quantile, dim=2).reshape(B, C, 1, 1)
        proc = torch.clamp(hm - bg, min=0.0)
        mx = proc.amax(dim=(2, 3), keepdim=True)
        proc = proc / (mx + 1e-6)

        center_map = proc[:, -1]                                     # (B, H, W)
        ring = proc[:, :K]                                           # (B, K, H, W)
        if self.subtract_center:
            ring = torch.clamp(ring - center_map.unsqueeze(1), min=0.0)

        rp = self.peak_radius
        pooled = F.max_pool2d(ring.reshape(B * K, 1, H, W), kernel_size=2 * rp + 1,
                                stride=1, padding=rp).reshape(B, K, H, W)
        is_max = (ring >= pooled - 1e-9) & (ring > 0)

        masked = torch.where(is_max, ring, torch.full_like(ring, -1.0))
        top_val, top_idx = masked.reshape(B, K, -1).topk(n_cand, dim=2)       # (B,K,n)
        top_y = torch.div(top_idx, W, rounding_mode="floor")
        top_x = top_idx % W

        # vectorized sub-pixel refinement helpers
        ring_flat = ring.reshape(B * K, H, W)
        ring_pad = F.pad(ring_flat.unsqueeze(1), (r_c, r_c, r_c, r_c), mode="replicate").squeeze(1)

        def extract_windows(padded, image_idx, xs, ys, radius):
            size = 2 * radius + 1
            dy = torch.arange(-radius, radius + 1, device=device)
            dx = torch.arange(-radius, radius + 1, device=device)
            yy, xx = torch.meshgrid(dy, dx, indexing="ij")
            ys_p = (ys + radius).view(-1, 1, 1) + yy.view(1, size, size)
            xs_p = (xs + radius).view(-1, 1, 1) + xx.view(1, size, size)
            img_idx = image_idx.view(-1, 1, 1).expand(-1, size, size)
            return padded[img_idx, ys_p, xs_p]

        def subpixel_batch(windows, xs, ys, radius):
            size = 2 * radius + 1
            win = windows.to(torch.float64)
            win = win - win.amin(dim=(1, 2), keepdim=True)
            s = win.sum(dim=(1, 2))
            offs = torch.arange(-radius, radius + 1, dtype=torch.float64, device=device)
            ys_grid = offs.view(1, size, 1) + ys.view(-1, 1, 1).to(torch.float64)
            xs_grid = offs.view(1, 1, size) + xs.view(-1, 1, 1).to(torch.float64)
            safe_s = torch.clamp(s, min=1e-12)
            cy = (win * ys_grid).sum(dim=(1, 2)) / safe_s
            cx = (win * xs_grid).sum(dim=(1, 2)) / safe_s
            vy = (win * (ys_grid - cy.view(-1, 1, 1)) ** 2).sum(dim=(1, 2)) / safe_s
            vx = (win * (xs_grid - cx.view(-1, 1, 1)) ** 2).sum(dim=(1, 2)) / safe_s
            sig = torch.sqrt(torch.clamp(0.5 * (vx + vy), min=1e-6))
            sig = torch.clamp(sig, min=0.25)
            degenerate = s <= 1e-12
            cx = torch.where(degenerate, xs.to(torch.float64), cx)
            cy = torch.where(degenerate, ys.to(torch.float64), cy)
            sig = torch.where(degenerate, torch.ones_like(sig), sig)
            return cx.to(torch.float32), cy.to(torch.float32), sig.to(torch.float32)

        # centre channel
        flat_c = center_map.reshape(B, -1)
        c_idx = flat_c.argmax(dim=1)
        cy0 = torch.div(c_idx, W, rounding_mode="floor")
        cx0 = c_idx % W
        center_pad = F.pad(center_map.unsqueeze(1), (r_c, r_c, r_c, r_c), mode="replicate").squeeze(1)
        c_img_idx = torch.arange(B, device=device)
        c_win = extract_windows(center_pad, c_img_idx, cx0, cy0, r_c)
        csx, csy, csig = subpixel_batch(c_win, cx0, cy0, r_c)
        center = torch.stack([csx, csy], dim=1)                       # (B, 2)
        center_conf = raw_peak_all[:, -1]
        center_sigma = csig

        # ring channels: best peak
        bk_img_idx = torch.arange(B * K, device=device)
        best_x = top_x[:, :, 0].reshape(-1)
        best_y = top_y[:, :, 0].reshape(-1)
        best_win = extract_windows(ring_pad, bk_img_idx, best_x, best_y, r_c)
        psx, psy, psig = subpixel_batch(best_win, best_x, best_y, r_c)
        px = psx.reshape(B, K)
        py = psy.reshape(B, K)
        sigma = psig.reshape(B, K)

        vals0 = top_val[:, :, 0]
        vals1 = top_val[:, :, 1] if n_cand > 1 else torch.zeros_like(vals0)
        dead = vals0 <= 0
        second = torch.where(vals1 > 0, vals1, torch.zeros_like(vals1))
        sep = 1.0 - second / torch.clamp(vals0, min=1e-6)

        py_idx = (torch.round(py).long() % H)
        px_idx = (torch.round(px).long() % W)
        b_idx = torch.arange(B, device=device).view(B, 1).expand(B, K)
        k_idx = torch.arange(K, device=device).view(1, K).expand(B, K)
        raw_at_peak = hm[b_idx, k_idx, py_idx, px_idx]

        conf = torch.clamp(raw_at_peak * (0.5 + 0.5 * sep), min=0.0, max=1.0)
        conf = torch.where(dead, torch.zeros_like(conf), conf)
        coords = torch.stack([px, py], dim=2)
        coords = torch.where(dead.unsqueeze(-1), torch.zeros_like(coords), coords)
        sigma = torch.where(dead, torch.ones_like(sigma), sigma)

        # reasoning bookkeeping
        low_conf_mask = (~dead) & (conf < self.min_conf)
        valid = (conf >= self.min_conf) & (~dead)

        # candidate peaks (sequential fill-forward over c, tiny loop)
        cand_coords = torch.zeros(B, K, n_cand, 2, device=device, dtype=torch.float32)
        cand_conf = torch.zeros(B, K, n_cand, device=device, dtype=torch.float32)
        for c in range(n_cand):
            xs_c = top_x[:, :, c].reshape(-1)
            ys_c = top_y[:, :, c].reshape(-1)
            vals_c = top_val[:, :, c]
            win_c = extract_windows(ring_pad, bk_img_idx, xs_c, ys_c, r_c)
            csx_c, csy_c, _ = subpixel_batch(win_c, xs_c, ys_c, r_c)
            cur_coords = torch.stack([csx_c, csy_c], dim=1).reshape(B, K, 2)
            cur_conf = raw_peak_all[:, :K] * vals_c
            has_peak = (vals_c > 0).unsqueeze(-1)
            prev_coords = cand_coords[:, :, max(c - 1, 0), :]
            cand_coords[:, :, c, :] = torch.where(has_peak, cur_coords, prev_coords)
            cand_conf[:, :, c] = torch.where(vals_c > 0, cur_conf, torch.zeros_like(cur_conf))

        # centre-leakage rejection
        d_center = torch.linalg.norm(coords - center.unsqueeze(1), dim=2)
        if self.center_exclusion > 0:
            center_leak_mask = valid & (d_center <= self.center_exclusion)
            valid = valid & (d_center > self.center_exclusion)
        else:
            center_leak_mask = torch.zeros_like(valid)

        # cross-channel NMS (vectorized over B, tiny loop over K)
        valid, suppressed_by, suppress_dist = self._cross_channel_nms(coords, conf, valid, self.dup_dist)

        reasons = self._build_reasons(
            dead=dead, low_conf_mask=low_conf_mask, center_leak_mask=center_leak_mask,
            suppressed_by=suppressed_by, suppress_dist=suppress_dist,
            valid_final=valid, conf=conf, d_center=d_center)

        return RingDetections(
            coords=coords, conf=conf, sigma=sigma, valid=valid, reasons=reasons,
            center=center, center_conf=center_conf, center_sigma=center_sigma,
            cand_coords=cand_coords, cand_conf=cand_conf,
        )

    def _build_reasons(self, dead, low_conf_mask, center_leak_mask,
                        suppressed_by, suppress_dist, valid_final, conf, d_center):
        """
        Build a (B, K) nested list of human-readable strings explaining why
        each channel was marked invalid. Priority mirrors the order the
        checks are actually applied in __call__: dead peak -> low confidence
        -> centre leakage -> cross-channel NMS duplicate. Valid channels get
        an empty string.
        """
        B, K = dead.shape
        dead_l = dead.detach().cpu().tolist()
        low_conf_l = low_conf_mask.detach().cpu().tolist()
        leak_l = center_leak_mask.detach().cpu().tolist()
        supp_by_l = suppressed_by.detach().cpu().tolist()
        supp_dist_l = suppress_dist.detach().cpu().tolist()
        valid_l = valid_final.detach().cpu().tolist()
        conf_l = conf.detach().cpu().tolist()
        d_center_l = d_center.detach().cpu().tolist()

        reasons = []
        for b in range(B):
            row = []
            for k in range(K):
                if valid_l[b][k]:
                    row.append("")
                elif dead_l[b][k]:
                    row.append("dead peak: no positive local maximum found in heatmap")
                elif low_conf_l[b][k]:
                    row.append(
                        f"low confidence: {conf_l[b][k]:.3f} < min_conf={self.min_conf:.3f}")
                elif leak_l[b][k]:
                    row.append(
                        f"center leakage: peak {d_center_l[b][k]:.2f}px from centre "
                        f"keypoint (<= center_exclusion={self.center_exclusion:.1f}px)")
                elif supp_by_l[b][k] >= 0:
                    row.append(
                        f"duplicate of channel {supp_by_l[b][k]}: "
                        f"{supp_dist_l[b][k]:.2f}px apart (< dup_dist={self.dup_dist:.1f}px), "
                        f"lower confidence suppressed")
                else:
                    row.append("invalid: unspecified reason")
            reasons.append(row)
        return reasons

    @staticmethod
    def _cross_channel_nms(coords: torch.Tensor, conf: torch.Tensor,
                             valid: torch.Tensor, min_dist: float):
        """
        Suppress peaks from DIFFERENT channels that landed on the same
        physical hole. Highest confidence wins. Fully torch, vectorized
        across the batch dim; only loops (K choose 2) times over the tiny
        channel count.

        Returns
        -------
        keep           (B, K) bool    surviving validity mask
        suppressed_by  (B, K) long    channel index that suppressed this one,
                                       -1 if not suppressed by NMS
        suppress_dist  (B, K) float   distance to the suppressing channel,
                                       0.0 if not suppressed
        """
        B, K, _ = coords.shape
        keep = valid.clone()
        suppressed_by = torch.full((B, K), -1, dtype=torch.long, device=coords.device)
        suppress_dist = torch.zeros((B, K), dtype=torch.float32, device=coords.device)
        conf_masked = torch.where(valid, conf, torch.full_like(conf, float("-inf")))
        order = torch.argsort(conf_masked, dim=1, descending=True)            # (B, K)
        dist = torch.cdist(coords, coords)                                    # (B, K, K)

        for i in range(K):
            a = order[:, i]
            a_keep = keep.gather(1, a.unsqueeze(1)).squeeze(1)
            d_a = dist.gather(1, a.view(B, 1, 1).expand(B, 1, K)).squeeze(1)   # (B, K)
            for j in range(i + 1, K):
                b = order[:, j]
                b_keep = keep.gather(1, b.unsqueeze(1)).squeeze(1)
                d_ab = d_a.gather(1, b.unsqueeze(1)).squeeze(1)
                suppress = a_keep & b_keep & (d_ab < min_dist)
                new_b_keep = b_keep & (~suppress)
                keep.scatter_(1, b.unsqueeze(1), new_b_keep.unsqueeze(1))
                if suppress.any():
                    suppress_col = suppress.unsqueeze(1)
                    new_supp_by = torch.where(
                        suppress_col, a.unsqueeze(1), suppressed_by.gather(1, b.unsqueeze(1)))
                    suppressed_by.scatter_(1, b.unsqueeze(1), new_supp_by)
                    new_supp_dist = torch.where(
                        suppress_col, d_ab.unsqueeze(1), suppress_dist.gather(1, b.unsqueeze(1)))
                    suppress_dist.scatter_(1, b.unsqueeze(1), new_supp_dist)
        return keep, suppressed_by, suppress_dist