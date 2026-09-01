"""
Stereo 6-DoF pose estimation for the docking-cube interface.

Per synchronised stereo pair the pipeline detects and crops the interface,
regresses 11 keypoint heatmaps (10 coplanar face holes + face centre), labels
them by the asymmetric largest-angular-gap pattern, filters outliers jointly
across both views, solves PnP against the rest-frame 3D model, resolves the
planar two-fold (mirror) ambiguity against the right camera, and reports the
object pose in the LEFT camera optical frame.

═══════════════════════════════════════════════════════════════════════════
INPUTS  — data and priors that must exist before a run 
═══════════════════════════════════════════════════════════════════════════
1. STEREO IMAGES          <data-dir>/left/*_rgb.jpeg
                          <data-dir>/right/*_rgb.jpeg
   Sorted lexicographically and paired by position: both folders must hold
   the same frames in the same order. Native 1280x720 RGB.

2. 3D OBJECT MODEL        <meta-dir>/object_keypoints.json
   {"1".."11": {"x":…, "y":…, "z":…}} in the object REST frame, metres.
     ids 1-10 : face holes. Forced coplanar at their mean z, so the face
                plane is the model's z = const plane and +Z is the face
                normal. The mirror resolver assumes this.
     id 11    : face centre — PnP anchor, axis origin, heatmap channel 11.
   Keypoint id k must correspond to heatmap channel k; the pipeline pairs
   2D detections to 3D model points by index.

3. CAMERA INTRINSICS      K_left, K_right — full-frame 3x3 pinhole, zero
   (prior)                distortion. Currently hardcoded fx=fy=800,
                          cx=640, cy=360. Swap in your calibration: crop
                          intrinsics, PnP and all rendering derive from
                          these. The crop is a pure principal-point shift
                          (fixed-size square crop, no resize) so fx/fy are
                          never rescaled.

4. STEREO EXTRINSICS      R_stereo, t_stereo — rigid LEFT→RIGHT transform.
   (prior)                Currently identity / [-1, 0, 0] m (rectified,
                          1 m baseline). Must be metric and in the same
                          units as the 3D model: the mirror disambiguation
                          scores hypotheses in the right view, so a wrong
                          baseline silently degrades it.

5. NETWORK WEIGHTS        --yolo-weights    YOLOv8-seg interface detector
                          --vmamba-weights  11-channel heatmap regressor,
                                            trained at --crop-size crops
                                            (the two must match).

6. OPTIONAL
   <meta-dir>/*_params.json   per-frame ground truth: object_rotation_matrix,
       object_rotation_z_deg, left_w2c_base, right_w2c_base. Read ONLY in
       verbose runs or with --evaluate. Not required in realtime.
   --rest-pose-json  {"R": 3x3, "t": [x,y,z]} — docked pose of the object in
       the LEFT-camera frame. A one-time station calibration constant, not
       re-estimated per frame. Enables the relative-motion-to-rest output.
   --background-dir + --process-depths — simulation-only background
       compositing; needs *_depth.png beside each image.

═══════════════════════════════════════════════════════════════════════════
OUTPUTS
═══════════════════════════════════════════════════════════════════════════
production + verbose
  <data-dir>/output/pose_log.json   one record per solved frame:
      frame, R_cam_left (3x3), t_cam_left (3,) metres, quat_xyzw and — when
      a rest pose was supplied — relative_to_rest {rotation_axis,
      rotation_angle_deg, translation_vector, translation_magnitude}.
      R/t are the object pose in the TRUE uncropped left-camera optical
      frame (crop-invariant; see pose_fullframe.py).
verbose only
  --video-out .mp4                  full-frame axis overlay + HUD
  <output>/view_XXXX_fullframe.png, view_XXXX_keypoints.png,
  <output>/view_XXXX_vecs.png       per-frame diagnostics
  <output>/evaluation_results.csv, evaluation_summary.txt
timing only
  per-stage latency table on stdout (mean / median / p95 ms, CUDA-synced)

Frames with no detection, a failed PnP, too few surviving keypoints, or a
reprojection error above --max-reproj-px emit no record and reset the
temporal trackers (PnP warm start, gap hysteresis, flip continuity).

═══════════════════════════════════════════════════════════════════════════
RUN MODES  (--mode, see MODE_PRESETS)
═══════════════════════════════════════════════════════════════════════════
  production : pose only. No GT read, no plots, no video, no prints.
  verbose    : everything — plots, video, evaluation report, progress bar.
  timing     : production compute path + per-stage timing. No side outputs.

============================================
Stereo 6-DoF pose estimation for the docking-cube interface.
"""

import os
import cv2
import glob
import argparse
from time import time as now
from pathlib import Path

import json
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torch.nn.functional import pad

from interface_detector import InterfaceDetector
from Keypoint_detector import KeypointPredictor

from heatmap_processing import HeatmapProcessor
from stereo_keypoint_filter import StereoRingMatcher, gt_correspondence
from pose_predictor import StereoPnPPoseEstimator
from temporal import PoseTracker, quality_from_result
from evaluation import PoseEvaluator, wrap_deg, StageTimer, Mode
from frame_builder import to_se3, build_pose_record
from realtime_visualizer import render_full_frame_pose, VideoRecorder

def parse_args():
    p = argparse.ArgumentParser(description='Pose estimation pipeline')
    # Non defaulted
    p.add_argument('--data-dir', help='dir containing "left" and "right"')
    p.add_argument('--face-normal-z', type=float,
                   help='+1 or -1: direction of the OUTWARD face normal along the '
                        'object Z axis. 0 = infer from the first GT frame '
                        '(verbose/evaluation only) and print it. This is what '
                        'resolves the 180 deg mirror ambiguity of the hole '
                        'pattern -- get it wrong and every pose flips.')
    # Defaulted
    p.add_argument('--background-dir', default=None)
    p.add_argument('--mode', choices=[m.value for m in Mode], default=Mode.PRODUCTION.value)
    p.add_argument('--meta-dir', default="./renders/meta_data")
    p.add_argument('--yolo-weights', default="./checkpoints/finetuned/yolov8s-seg-finetuned/weights/best.pt")
    p.add_argument('--vmamba-weights', default="./checkpoints/finetuned/vmamba_heat_compound2/best.pt")
    p.add_argument('--crop-size', default=256, type=int)
    p.add_argument('--video-out', default='./renders/test/output/realtime_track.mp4')
    p.add_argument('--video-fps', default=24.0, type=float)
    p.add_argument('--rest-pose-json', default=None)

    # ---- robustness knobs ------------------------------------------------
    p.add_argument('--max-reproj-px', type=float, default=8.0, help='reject the frame above this stereo reprojection RMS')
    p.add_argument('--max-rot-step-deg', type=float, default=20.0, help='largest per-frame rotation the object can make; the temporal gate rejects anything bigger')
    p.add_argument('--max-trans-step', type=float, default=0.1, help='same for translation, in model units (m)')
    p.add_argument('--no-smoothing', action='store_true', help='log the raw PnP pose instead of the tracker-smoothed one')
    p.add_argument('--no-temporal', action='store_true', help='disable prediction/gating/coasting entirely')
    return p.parse_args()


def main():
    args = parse_args()
    mode = Mode(args.mode)

    verbose = (mode == Mode.VERBOSE)
    evaluate = (mode in (Mode.VERBOSE, Mode.EVALUATION))
    write_video = (mode == Mode.VERBOSE)
    use_gt = (mode in (Mode.VERBOSE, Mode.EVALUATION))

    timer = StageTimer(mode)
    left_folder, right_folder = 'left', 'right'

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError("Data directory does not exist")
    for f in (left_folder, right_folder):
        if not os.path.exists(os.path.join(args.data_dir, f)):
            raise FileNotFoundError("Data directory does not contain left / right")

    output_dir = os.path.join(args.data_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 3D rest-frame model 
    obj_model = json.load(open(args.meta_dir + '/object_keypoints.json'))
    ids = [str(k) for k in range(1, 12)]
    kp_rest = torch.tensor([[obj_model[i]['x'], obj_model[i]['y'], obj_model[i]['z']]
                            for i in ids], dtype=torch.float64)
    # Force ALL eleven points coplanar, centre included.
    kp_rest[:, 2] = kp_rest[:-1, 2].mean()

    model_np = kp_rest.numpy().astype(np.float64)          # (11, 3)
    center_3d_rest_np = model_np[10]
    ring_xy = model_np[:10, :2] - center_3d_rest_np[:2]    # (10, 2) face-plane coords

    # Assumed intrinsics + Baseline
    K_left_np = np.array([[800, 0.0, 640.0], [0.0, 800, 360.0], [0.0, 0.0, 1.0]])
    K_right_np = K_left_np.copy()
    R_stereo = np.eye(3)
    t_stereo = np.array([[-1.0], [0.0], [0.0]])

    # Image loading
    left_images = sorted(glob.glob(f'{args.data_dir}/{left_folder}/*_rgb.jpeg'))
    right_images = sorted(glob.glob(f'{args.data_dir}/{right_folder}/*_rgb.jpeg'))
    left_depths = sorted(glob.glob(f'{args.data_dir}/{left_folder}/*_depth.png'))
    right_depths = sorted(glob.glob(f'{args.data_dir}/{right_folder}/*_depth.png'))
    json_files = sorted(glob.glob(f'{args.meta_dir}/*_params.json')) if use_gt else []

    # Background injection
    backgrounds = []
    if use_gt and args.background_dir:
        bg_files = [str(f) for f in Path(args.background_dir).rglob('*')
                    if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        print(f"Pre-loading {len(bg_files)} backgrounds...")
        for bg_file in bg_files:
            bg = Image.open(bg_file).convert("RGB").resize((640, 360), Image.Resampling.BILINEAR)
            backgrounds.append(pil_to_tensor(bg).float() / 255.0)

    # Rest pose loading
    T_cam_rest = None
    if args.rest_pose_json:
        rest = json.load(open(args.rest_pose_json))
        T_cam_rest = to_se3(np.array(rest['R'], np.float64),  np.array(rest['t'], np.float64).reshape(3, 1))

    # Face-normal convention (resolves the 180 deg mirror ambiguity)
    face_normal_obj = None
    if abs(args.face_normal_z) > 0.5:
        face_normal_obj = np.array([0.0, 0.0, float(np.sign(args.face_normal_z))])
    else:
        print("[pipeline] WARNING: no face-normal convention available. The 180 deg mirror ambiguity of the hole pattern will NOT be resolved")

    recorder = None
    if write_video:
        os.makedirs(os.path.dirname(args.video_out) or '.', exist_ok=True)
        recorder = VideoRecorder(args.video_out, fps=args.video_fps)

    pose_log = []
    n_backfacing = 0          # frames lost to the visibility constraint
    N = min(len(left_images), len(right_images))
    half_hw = args.crop_size // 2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Models
    detector = InterfaceDetector(weights_path=args.yolo_weights, device=device, mask_mode=False, crop_size=args.crop_size)
    predictor = KeypointPredictor(weights=args.vmamba_weights, num_keypoints=11, device=device, input_size=args.crop_size)
    # Post-processing
    processor = HeatmapProcessor()
    matcher = StereoRingMatcher(ring_xy)
    # PnP
    estimator = StereoPnPPoseEstimator(
        camera_matrix_left=K_left_np.copy(), camera_matrix_right=K_right_np.copy(),
        R_stereo=R_stereo, t_stereo=t_stereo,
        reproj_threshold=args.max_reproj_px, min_inliers=5,
        face_normal_obj=face_normal_obj,
    )
    tracker = PoseTracker(max_rot_step_deg=args.max_rot_step_deg,max_trans_step=args.max_trans_step)
    # Evaluation
    evaluator = PoseEvaluator(output_dir) if evaluate else None

    if mode != Mode.PRODUCTION:
        detector(torch.empty([3, 1280, 1024]))

    frame_iter = range(N)
    for i in frame_iter:
        loop_t0 = now()
        # Loading images and GT
        with timer.stage("load_frame"):
            left_img = pil_to_tensor(Image.open(left_images[i]).convert('RGB')).float() / 255.0
            right_img = pil_to_tensor(Image.open(right_images[i]).convert('RGB')).float() / 255.0
            view_model = json.load(open(json_files[i])) if use_gt else None
        
        # Preprocess the tensors
        with timer.stage("preprocess"):
            inp = torch.stack([left_img, right_img], dim=0).to(device)
            s = inp.shape
            if (s[2] / 32) % 1 or (s[3] / 32) % 1:
                inp = pad(inp, (0, int(((s[3] // 32) + 1 - (s[3] / 32)) * 32) % 32,
                                0, int(((s[2] // 32) + 1 - (s[2] / 32)) * 32) % 32), 'circular')
        # Detect the interface via Yolo
        with timer.stage("detection"):
            cropped_img, centers = detector(inp)
        if centers is None or centers[0] is None or centers[1] is None:
            if mode != Mode.PRODUCTION:
                print(f"No detection in frame {i}")
            if evaluate:
                evaluator.add_skipped_frame(i)
            tracker.miss()
            continue

        # Pure principal-point shift:
        cx_full, cy_full = 640.0, 360.0
        cx_off_L = float(centers[0][1]) - half_hw
        cy_off_L = float(centers[0][0]) - half_hw
        cx_off_R = float(centers[1][1]) - half_hw
        cy_off_R = float(centers[1][0]) - half_hw
        estimator.K_left = np.array([[800, 0, cx_full - cx_off_L],
                                     [0, 800, cy_full - cy_off_L],
                                     [0, 0, 1]], np.float64)
        estimator.K_right = np.array([[800, 0, cx_full - cx_off_R],
                                      [0, 800, cy_full - cy_off_R],
                                      [0, 0, 1]], np.float64)
        # Heatmap inference via VMamba
        with timer.stage("heatmap_inference"):
            heatmaps = predictor(cropped_img)

        # Heatmap processing
        with timer.stage("heatmap_processing"):
            det = processor(heatmaps)
            if verbose:
                det.plot(cropped_img, os.path.join(output_dir, f'view_{i:04d}_heatmaps.png'))

        # Estimated temporal movement
        predicted = None if args.no_temporal else tracker.predict()

        # Ring matching
        with timer.stage("ring_matching"):
            # PnP based Scorer to evaluate degenrate cases for better matching
            def _scorer(uL, uR, mi):
                return estimator.score_correspondence(uL, uR, model_np[mi], predicted_pose=predicted)
            # Match the stereo hole indices
            match = matcher(det, pose_scorer=_scorer)
            fig = matcher.plot(match, output_path=os.path.join(output_dir, f'view_{i:04d}_ring.png'))
            if not match.ok:
                if mode != Mode.PRODUCTION:
                    print(f"Frame {i}: no consistent labelling ({match.reason})")
                if evaluate:
                    evaluator.add_skipped_frame(i)
                matcher.commit(match)
                tracker.miss()
                continue
            # Labelled points
            object_pts = np.vstack([model_np[match.model_idx], center_3d_rest_np])
            uv_left = np.vstack([match.uv_left, det.center[0]])
            uv_right = np.vstack([match.uv_right, det.center[1]])
            # Confidence of labels
            w_left = np.concatenate([match.w_left, [det.center_weights()[0]]])
            w_right = np.concatenate([match.w_right, [det.center_weights()[1]]])

        # PnP Estimation
        with timer.stage("pnp_estimation"):
            result = estimator.estimate_pose(object_pts, uv_left, uv_right, w_left, w_right, predicted_pose=predicted)
            bad = (not result.success or max(result.reprojection_error_left, result.reprojection_error_right) > args.max_reproj_px)
            if bad:
                if "back-facing" in (result.reason or ""):
                    n_backfacing += 1
                if mode != Mode.PRODUCTION:
                    why = result.reason or (f"reproj {max(result.reprojection_error_left, result.reprojection_error_right):.2f}px")
                    print(f"Frame {i}: PnP rejected ({why})")
                if evaluate:
                    evaluator.add_skipped_frame(i)
                matcher.commit(type(match)(ok=False))
                tracker.miss()
                continue
            # Project onto the left camera space
            R_cam_left, t_cam_left = result.rotation_matrix.astype(np.float64), result.translation_vector.reshape(3, 1).astype(np.float64)
            
        # Temporal smoothing
        if args.no_temporal:
            accepted, coasted = True, False
        else:
            out = tracker.update(R_cam_left, t_cam_left, quality_from_result(result))
            accepted, coasted = out.accepted, out.coasted
            if not accepted and mode != Mode.PRODUCTION:
                print(f"Frame {i}: temporal gate rejected ({out.reason})")
            if not args.no_smoothing:
                R_cam_left, t_cam_left = out.R, out.t

        # A gated-out frame is a detection failure, not a pose. Don't let it poison the matcher's continuity prior.
        matcher.commit(match if accepted else type(match)(ok=False))
        if not accepted:
            if evaluate:
                evaluator.add_skipped_frame(i)
            continue

        # Pose logging
        with timer.stage("pose_logging"):
            record, rel_motion = build_pose_record(i, R_cam_left, t_cam_left, T_cam_rest)
            pose_log.append(record)
        
        # Evaluation of the performance into a file
        if evaluate:
            with timer.stage("evaluation_metrics"):
                left_w2c = np.array(view_model['left_w2c_base'], np.float64)
                R_c2w_left = np.linalg.inv(left_w2c)[:3, :3]
                R_obj_est = R_c2w_left @ R_cam_left
                theta_z_est = np.degrees(np.arctan2(R_obj_est[1, 0], R_obj_est[0, 0]))
                cam_pos_est = -(R_cam_left.T @ t_cam_left).flatten()
                cam_pos_world_est = R_obj_est @ cam_pos_est
                r = np.linalg.norm(cam_pos_world_est)
                theta = np.degrees(np.arccos(np.clip(cam_pos_world_est[2] / r, -1, 1)))
                phi = np.degrees(np.arctan2(cam_pos_world_est[1], cam_pos_world_est[0]))
                evaluator.add_frame(frame_idx=i, pnp_result=result, view_model=view_model,
                    R_obj_est=R_obj_est, theta_z_est=theta_z_est,
                    r=r, theta=theta, phi=phi,
                    t_cam_est=t_cam_left)
            if verbose:
                gt_err = _gt_keypoint_error(view_model, model_np, K_left_np, K_right_np, match, cx_off_L, cy_off_L, cx_off_R, cy_off_R)
                if gt_err is not None:
                    print(f"  frame {i}: n={match.n} matched, "
                          f"kp err L/R = {gt_err[0]:.2f}/{gt_err[1]:.2f} px, "
                          f"hom rms L/R = {match.rms_left:.2f}/{match.rms_right:.2f} px, "
                          f"theta_z err = {abs(wrap_deg(theta_z_est - view_model['object_rotation_z_deg'])):.2f} deg")

        # Visuaization error creation
        if write_video:
            with timer.stage("render_overlay"):
                frame_bgr = render_full_frame_pose(
                    image_tensor_full=inp[0].cpu(), R_cam=R_cam_left, t_cam=t_cam_left,
                    K_full=K_left_np, center_3d=center_3d_rest_np,
                    crop_bbox_xyxy=(cx_off_L, cy_off_L,
                                    cx_off_L + args.crop_size, cy_off_L + args.crop_size),
                    frame_idx=i, fps_est=1.0 / max(now() - loop_t0, 1e-6),
                    reproj_err_left=result.reprojection_error_left,
                    num_inliers=result.num_inliers, rel_motion=rel_motion)
                recorder.write(frame_bgr)
                if verbose:
                    cv2.imwrite(os.path.join(output_dir, f'view_{i:04d}_fullframe.png'), frame_bgr)

    if recorder is not None:
        recorder.close()
        print(f"Wrote realtime tracking video to {args.video_out}")

    with open(os.path.join(output_dir, 'pose_log.json'), 'w') as f:
        json.dump(pose_log, f, indent=2)

    if n_backfacing:
        rate = n_backfacing / max(N, 1)
        print(f"[pipeline] {n_backfacing}/{N} frames ({rate:.1%}) rejected because "
              f"every pose hypothesis was back-facing.")
        if rate > 0.3:
            print("[pipeline] That rate is high enough to suspect face_normal_obj "
                  "is INVERTED. An inverted sign keeps the flipped pose and "
                  "rejects the correct one, while reporting excellent "
                  "reprojection errors throughout.")

    timer.report()
    if evaluate:
        evaluator.write_report()

def _gt_keypoint_error(view_model, model_np, K_left, K_right, match,
                       cx_off_L, cy_off_L, cx_off_R, cy_off_R):
    """
    Verbose diagnostic: mean pixel error between the matched detections and
    the GT projections of the SAME model indices. This tells you directly
    whether a bad frame was a detection problem or a labelling problem --
    something the old vector plot could not distinguish.
    """
    try:
        R_obj = np.array(view_model['object_rotation_matrix'], np.float64)
        wl = np.array(view_model['left_w2c_base'], np.float64)
        wr = np.array(view_model['right_w2c_base'], np.float64)
    except (KeyError, TypeError):
        return None

    X = np.c_[model_np @ R_obj.T, np.ones(len(model_np))]

    def proj(w2c, K, ox, oy):
        c = X @ w2c.T
        c = c[:, :3] / c[:, 3:]
        c = c @ K.T
        uv = c[:, :2] / c[:, 2:]
        return uv - np.array([ox, oy])            # full-image -> crop pixels

    gt_L = proj(wl, K_left, cx_off_L, cy_off_L)
    gt_R = proj(wr, K_right, cx_off_R, cy_off_R)
    sel_L, sel_R = gt_correspondence(gt_L, gt_R, match.model_idx)
    return (float(np.linalg.norm(match.uv_left - sel_L, axis=1).mean()),
            float(np.linalg.norm(match.uv_right - sel_R, axis=1).mean()))


if __name__ == '__main__':
    main()