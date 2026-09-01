"""
pose_visualizer.py
──────────────────
Draw estimated (and optionally GT) object coordinate axes on top of BOTH
the left and right camera images side by side, similar to the notebook's
cv2.projectPoints visualization.

Key fixes vs. the previous version:
  1. Axes are now anchored at the actual 3D "Center" keypoint location
     (in the object's rest frame) instead of the coordinate-frame origin
     (0,0,0), which almost never coincides with the object's visual
     center once the rest-frame keypoints are off-origin.
  2. Axis length auto-scales with the object's distance from the camera
     (axis_len_ratio * ||t_cam||) so the arrows stay visible regardless
     of depth/units, instead of a fixed world-unit constant that can end
     up sub-pixel after projection. An explicit axis_len still overrides
     this if you want a fixed value.
  3. Renders left AND right camera views side-by-side in one saved PNG.

The axes are projected via OpenCV's projectPoints and rendered with
matplotlib as coloured arrows overlaid on each image.

Usage
-----
    from pose_visualizer import save_axes_visualization

    save_axes_visualization(
        image_tensor_left  = left_img,   # (3, H, W) float [0,1] torch tensor
        image_tensor_right = right_img,  # (3, H, W) float [0,1] torch tensor
        R_cam_left   = R_object_wrt_leftCam,   # (3,3) np, object->left cam
        t_cam_left   = t_object_wrt_leftCam,   # (3,1) np
        K_left       = KL_crop,                # (3,3) np left intrinsics
        R_cam_right  = R_object_wrt_rightCam,  # (3,3) np, object->right cam
        t_cam_right  = t_object_wrt_rightCam,  # (3,1) np
        K_right      = KR_crop,                # (3,3) np right intrinsics
        save_path    = "frame_000_axes.png",
        center_3d    = center_3d_rest,         # (3,) 3D coord of Center kp
        R_gt_cam_left  = R_gt_in_leftCam,       # optional GT rotation
        t_gt_cam_left  = t_gt_in_leftCam,       # optional GT translation
        R_gt_cam_right = R_gt_in_rightCam,
        t_gt_cam_right = t_gt_in_rightCam,
        frame_idx    = 0,
    )
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend – safe for pipeline scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _project_axis_endpoints(R, t, K, dist, axis_len, center_3d=None):
    """
    Project the three axis endpoints (X, Y, Z) and the object's Center
    keypoint into the image plane.

    Parameters
    ----------
    center_3d : (3,) or None
        3D coordinate of the Center keypoint in the object's rest frame.
        If None, defaults to (0, 0, 0) (legacy behaviour).

    Returns
    -------
    origin2d  : (2,) pixel coordinate of the Center keypoint
    x2d, y2d, z2d : (2,) pixel coordinates of the axis tips
    """
    if center_3d is None:
        center_3d = np.zeros(3, dtype=np.float64)
    else:
        center_3d = np.asarray(center_3d, dtype=np.float64).reshape(3)

    origin_3d = center_3d.reshape(1, 3)
    axes_3d = center_3d.reshape(1, 3) + np.array(
        [[axis_len, 0, 0],
         [0, axis_len, 0],
         [0, 0, axis_len]], dtype=np.float64)

    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec    = t.reshape(3, 1).astype(np.float64)

    origin_px, _ = cv2.projectPoints(origin_3d, rvec, tvec, K, dist)
    axes_px,  _  = cv2.projectPoints(axes_3d,   rvec, tvec, K, dist)

    origin2d = origin_px.squeeze()
    x2d      = axes_px[0].squeeze()
    y2d      = axes_px[1].squeeze()
    z2d      = axes_px[2].squeeze()

    return origin2d, x2d, y2d, z2d


def _draw_axes_on_ax(ax, origin, x2d, y2d, z2d, linestyle="-", alpha=1.0,
                     label_suffix=""):
    """Draw X/Y/Z arrows on a matplotlib Axes object."""
    arrowprops = dict(arrowstyle="-|>", lw=3.0,
                      mutation_scale=22, alpha=alpha)

    def arrow(src, dst, color):
        ax.annotate(
            "", xy=dst, xytext=src,
            arrowprops={**arrowprops, "color": color,
                        "linestyle": linestyle},
        )

    arrow(origin, x2d, "red")
    arrow(origin, y2d, "green")
    arrow(origin, z2d, "blue")

    # Small text label near the tips
    offset = 8
    ax.text(x2d[0] + offset, x2d[1], f"X{label_suffix}", color="red",
            fontsize=10, fontweight="bold", alpha=alpha)
    ax.text(y2d[0] + offset, y2d[1], f"Y{label_suffix}", color="green",
            fontsize=10, fontweight="bold", alpha=alpha)
    ax.text(z2d[0] + offset, z2d[1], f"Z{label_suffix}", color="blue",
            fontsize=10, fontweight="bold", alpha=alpha)

    # Mark origin (Center keypoint)
    ax.plot(*origin, "o", color="yellow", markersize=7, alpha=alpha,
            markeredgecolor="black", markeredgewidth=0.8)


def _render_one_view(ax, image_tensor, R_cam, t_cam, K, dist_coeffs,
                     axis_len, center_3d, R_gt_cam=None, t_gt_cam=None,
                     view_label=""):
    """Render a single camera view (image + estimated/GT axes) on `ax`."""
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    ax.imshow(img_np)
    ax.axis("off")

    try:
        o, x, y, z = _project_axis_endpoints(R_cam, t_cam, K, dist_coeffs,
                                             axis_len, center_3d=center_3d)
        _draw_axes_on_ax(ax, o, x, y, z, linestyle="-", alpha=1.0, label_suffix="")
    except cv2.error as e:
        print(f"[pose_visualizer] Warning: could not project estimated axes ({view_label}): {e}")

    has_gt = (R_gt_cam is not None) and (t_gt_cam is not None)
    if has_gt:
        try:
            og, xg, yg, zg = _project_axis_endpoints(
                R_gt_cam, t_gt_cam, K, dist_coeffs, axis_len, center_3d=center_3d)
            _draw_axes_on_ax(ax, og, xg, yg, zg, linestyle="--",
                             alpha=0.55, label_suffix="_gt")
        except cv2.error as e:
            print(f"[pose_visualizer] Warning: could not project GT axes ({view_label}): {e}")
            has_gt = False

    ax.set_title(view_label, fontsize=10, pad=4)
    return has_gt


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def save_axes_visualization(
    image_tensor_left:  "torch.Tensor",   # (3, H, W) float [0,1]
    image_tensor_right: "torch.Tensor",   # (3, H, W) float [0,1]
    R_cam_left:   np.ndarray,       # (3,3) estimated rotation, object->left cam
    t_cam_left:   np.ndarray,       # (3,1) estimated translation, left cam
    K_left:       np.ndarray,       # (3,3) left camera intrinsics
    R_cam_right:  np.ndarray,       # (3,3) estimated rotation, object->right cam
    t_cam_right:  np.ndarray,       # (3,1) estimated translation, right cam
    K_right:      np.ndarray,       # (3,3) right camera intrinsics
    save_path:    str,
    center_3d:    np.ndarray = None,  # (3,) 3D coord of the Center keypoint
    R_gt_cam_left:   np.ndarray = None,
    t_gt_cam_left:   np.ndarray = None,
    R_gt_cam_right:  np.ndarray = None,
    t_gt_cam_right:  np.ndarray = None,
    axis_len:     float = None,       # explicit override; if None, auto-scaled
    axis_len_ratio: float = 0.4,      # fraction of camera distance used when auto-scaling
    frame_idx:    int   = -1,
    dist_coeffs_left:  np.ndarray = None,
    dist_coeffs_right: np.ndarray = None,
    reproj_err_left:  float = None,
    reproj_err_right: float = None,
    theta_z_est:  float = None,
    theta_z_gt:   float = None,
    num_inliers:  int   = None,
):
    """
    Render the estimated (and optionally GT) coordinate axes, anchored at
    the object's Center keypoint, on top of BOTH the left and right camera
    images side by side, and save as a single PNG file.

    The axis length auto-scales with the object's distance from the
    camera (axis_len_ratio * ||t_cam_left||) unless `axis_len` is given
    explicitly.
    """
    if dist_coeffs_left is None:
        dist_coeffs_left = np.zeros((4, 1), dtype=np.float64)
    if dist_coeffs_right is None:
        dist_coeffs_right = np.zeros((4, 1), dtype=np.float64)

    if axis_len is None:
        cam_distance = float(np.linalg.norm(np.asarray(t_cam_left).flatten()))
        axis_len = max(axis_len_ratio * cam_distance, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    has_gt_l = _render_one_view(
        axes[0], image_tensor_left, R_cam_left, t_cam_left, K_left,
        dist_coeffs_left, axis_len, center_3d,
        R_gt_cam=R_gt_cam_left, t_gt_cam=t_gt_cam_left,
        view_label="Left camera",
    )
    has_gt_r = _render_one_view(
        axes[1], image_tensor_right, R_cam_right, t_cam_right, K_right,
        dist_coeffs_right, axis_len, center_3d,
        R_gt_cam=R_gt_cam_right, t_gt_cam=t_gt_cam_right,
        view_label="Right camera",
    )
    has_gt = has_gt_l or has_gt_r

    # ── Legend (shared) ────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color="red",   label="X"),
        mpatches.Patch(color="green", label="Y"),
        mpatches.Patch(color="blue",  label="Z"),
    ]
    if has_gt:
        legend_items += [
            mpatches.Patch(facecolor="white", edgecolor="grey",
                           label="solid=est  dashed=GT", linestyle="--"),
        ]
    axes[1].legend(handles=legend_items, loc="upper right", fontsize=9,
                   framealpha=0.7)

    # ── Suptitle / annotation ───────────────────────────────────────────────
    title_parts = [f"Frame {frame_idx}" if frame_idx >= 0 else "Axes Visualization"]
    if theta_z_est is not None:
        est_str = f"θZ est: {theta_z_est:+.2f}°"
        gt_str  = f"  GT: {theta_z_gt:+.2f}°" if theta_z_gt is not None else ""
        err_str = (f"  Δ: {abs(theta_z_est - theta_z_gt):.2f}°"
                   if theta_z_gt is not None else "")
        title_parts.append(est_str + gt_str + err_str)
    if reproj_err_left is not None:
        title_parts.append(
            f"Reproj L:{reproj_err_left:.2f}px  R:{reproj_err_right:.2f}px"
            if reproj_err_right is not None else
            f"Reproj L:{reproj_err_left:.2f}px"
        )
    if num_inliers is not None:
        title_parts.append(f"Inliers: {num_inliers}")

    fig.suptitle("  |  ".join(title_parts), fontsize=11, y=0.99)

    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)