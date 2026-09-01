"""
realtime_visualizer.py
───────────────────────
Change #2: draw the estimated (and optional GT) pose axes directly on the
FULL, uncropped camera frame (not the cropped detector patch), using cv2
only (no matplotlib) so it's cheap enough for a realtime loop. Also
provides a small VideoRecorder to accumulate frames into an .mp4 so you
can watch the realtime run afterwards.

Because of the crop-invariance property (see pose_fullframe.py docstring),
the pose (R_cam_left, t_cam_left) coming out of the estimator is ALREADY
valid in the true camera frame. So to draw on the full image you simply
project with the ORIGINAL K_left / K_right (not KL_crop / KR_crop) — no
pixel-offset bookkeeping required for the axes themselves. We optionally
also draw the crop bounding box, for which the offset IS needed (purely
for a debug overlay, not for the pose math).
"""

import cv2
import numpy as np
import torch


def _to_bgr_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    """(3,H,W) float [0,1] torch tensor -> (H,W,3) uint8 BGR numpy array."""
    img = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _project_axes(R, t, K, dist, axis_len, center_3d):
    center_3d = np.asarray(center_3d, dtype=np.float64).reshape(1, 3)
    axes_3d = center_3d + np.array(
        [[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]], dtype=np.float64
    )
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec = t.reshape(3, 1).astype(np.float64)
    origin_px, _ = cv2.projectPoints(center_3d, rvec, tvec, K, dist)
    axes_px, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist)
    return origin_px.reshape(2), axes_px.reshape(3, 2)


def _draw_axes(img, origin, axes_2d, alpha=1.0, dashed=False, thickness=2):
    """Draw X/Y/Z arrows (red/green/blue) with cv2.arrowedLine; supports a
    faded 'alpha' look (via cv2.addWeighted on a scratch layer) for GT."""
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: X red, Y green, Z blue
    labels = ["X", "Y", "Z"]
    overlay = img.copy()
    o = tuple(np.round(origin).astype(int))
    for pt, color, label in zip(axes_2d, colors, labels):
        p = tuple(np.round(pt).astype(int))
        cv2.arrowedLine(overlay, o, p, color, thickness, tipLength=0.15,
                         line_type=cv2.LINE_AA)
        cv2.putText(overlay, label, (p[0] + 6, p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    cv2.circle(overlay, o, 5, (0, 255, 255), -1, cv2.LINE_AA)
    if alpha >= 1.0:
        img[:] = overlay
    else:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def render_full_frame_pose(
    image_tensor_full: torch.Tensor,      # (3, H, W) float [0,1] — FULL uncropped frame
    R_cam: np.ndarray,                     # (3,3) object->camera rotation (already full-frame valid)
    t_cam: np.ndarray,                     # (3,1) object->camera translation
    K_full: np.ndarray,                    # (3,3) ORIGINAL (uncropped) camera intrinsics
    center_3d: np.ndarray,                 # (3,) rest-frame coord of the anchor/"Center" keypoint
    axis_len: float = None,
    axis_len_ratio: float = 0.4,
    dist_coeffs: np.ndarray = None,
    R_gt_cam: np.ndarray = None,
    t_gt_cam: np.ndarray = None,
    crop_bbox_xyxy: tuple = None,          # (x0, y0, x1, y1) in full-image pixels, debug only
    frame_idx: int = -1,
    fps_est: float = None,
    reproj_err_left: float = None,
    num_inliers: int = None,
    rel_motion: dict = None,               # output of relative_motion_to_rest(), optional HUD
) -> np.ndarray:
    """
    Returns a (H, W, 3) uint8 BGR numpy image with axes (+ optional GT
    axes, crop box, and text HUD) drawn on the FULL camera frame — ready
    to hand to VideoRecorder.write() or cv2.imshow().
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    if axis_len is None:
        axis_len = max(axis_len_ratio * float(np.linalg.norm(t_cam.flatten())), 1e-6)

    img = _to_bgr_uint8(image_tensor_full)

    if crop_bbox_xyxy is not None:
        x0, y0, x1, y1 = map(int, crop_bbox_xyxy)
        cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1, cv2.LINE_AA)

    origin, axes2d = _project_axes(R_cam, t_cam, K_full, dist_coeffs, axis_len, center_3d)
    _draw_axes(img, origin, axes2d, alpha=1.0, thickness=2)

    if R_gt_cam is not None and t_gt_cam is not None:
        origin_gt, axes2d_gt = _project_axes(R_gt_cam, t_gt_cam, K_full, dist_coeffs, axis_len, center_3d)
        _draw_axes(img, origin_gt, axes2d_gt, alpha=0.5, thickness=1)

    hud = []
    if frame_idx >= 0:
        hud.append(f"frame {frame_idx}")
    if fps_est is not None:
        hud.append(f"{fps_est:.1f} FPS")
    if reproj_err_left is not None:
        hud.append(f"reproj {reproj_err_left:.2f}px")
    if num_inliers is not None:
        hud.append(f"inliers {num_inliers}")
    if rel_motion is not None:
        hud.append(f"d_rot {rel_motion['rotation_angle_deg']:.1f} deg")
        hud.append(f"d_t {rel_motion['translation_magnitude']:.3f} m")

    y = 20
    for line in hud:
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 20

    return img


class VideoRecorder:
    """Thin wrapper around cv2.VideoWriter to accumulate rendered frames
    into a single mp4 for reviewing realtime performance after the run."""

    def __init__(self, path: str, fps: float = 30.0, frame_size: tuple = None,
                 fourcc: str = "mp4v"):
        self.path = path
        self.fps = fps
        self.frame_size = frame_size  # (W, H); inferred from first frame if None
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._writer = None

    def write(self, frame_bgr: np.ndarray):
        if self._writer is None:
            h, w = frame_bgr.shape[:2]
            self.frame_size = (w, h)
            self._writer = cv2.VideoWriter(self.path, self.fourcc, self.fps, self.frame_size)
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter for {self.path}")
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.frame_size:
            frame_bgr = cv2.resize(frame_bgr, self.frame_size)
        self._writer.write(frame_bgr)

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()