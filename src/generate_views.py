import os
import numpy as np
import torch
from gsplat.rendering import rasterization
from plyfile import PlyData
from viser import transforms as tf
from PIL import Image
import torch.nn.functional as F
import json
import argparse
import glob
from tqdm import tqdm 
from scipy.interpolate import CubicSpline
import cv2
from pathlib import Path

'''
Example usage:
Conda activate gsplat
python ./src/generate_views.py --stereo_test
Expects an object interface.ply and a keypoint directory such that each keypoint is saved in one separate file, named as an integer according to its class (+1 for technicality, i.e 1.ply keypoint file = class 2)

For more customization pease look at parse_args() function

'''


def parse_args():
    parser = argparse.ArgumentParser(description="Render random views from a Gaussian Splatting scene.")
    parser.add_argument("--base_ply", type=str, default="./splats/", help="Path to base PLY.")
    parser.add_argument("--keypoint_dir", type=str, default="./splats/keypoints", help="Path to keypoints.")
    parser.add_argument("--out_dir", type=str, default="./renders", help="Output directory.")
    parser.add_argument("--height", type=int, default=720, help="Image height.")
    parser.add_argument("--width", type=int, default=1280, help="Image width.")
    parser.add_argument("--focal", type=float, default=800.0, help="Focal length.")
    parser.add_argument("--near", type=float, default=0.01, help="Near plane.")
    parser.add_argument("--far", type=float, default=100.0, help="Far plane.")
    parser.add_argument("--num_views_train", type=int, default=2000, help="Number of views Train.")
    parser.add_argument("--num_views_test", type=int, default=500, help="Number of views Test.")
    parser.add_argument("--radius_range", type=tuple, default=(5.0,12.0), help="Radius range.")
    parser.add_argument("--theta_range", type=tuple, default=(125,180), help="Theta range.")
    parser.add_argument("--baseline", type=float, default=1.0, help="Stereo baseline (meters).")
    parser.add_argument("--object_rot_range", type=float, default=359.0, help="Max rotation (degrees) of the object around Z-axis for TEST data (e.g. 360 for full spin).")

    parser.add_argument("--stereo_test", action='store_true', help="Use stereo pairs for test set.")
    parser.add_argument("--num_keypoints", type=int, default=8, help="Number of keypoints to interpolate for test video trajectory")
    parser.add_argument("--min_segment_len", type=int, default=20)
    parser.add_argument("--max_segment_len", type=int, default=70)
    parser.add_argument("--speed_range_deg", type=tuple, default=(2.0, 4.0))
    parser.add_argument("--smooth_transition_frames", type=int, default=5)

    args = parser.parse_args()
    return args


def frames_to_video(frame_dir, output_name = "output.mp4", fps = 24,pattern = "*_rgb.jpeg"):
    """
    Compile rendered frames from a directory into an MP4 video.
    """
    left_dir = os.path.join(frame_dir, "left")
    right_dir = os.path.join(frame_dir, "right")

    left_paths  = sorted(Path(left_dir).glob(pattern))
    right_paths = sorted(Path(right_dir).glob(pattern))

    assert len(left_paths) > 0,  f"No frames found in {left_dir}"
    assert len(right_paths) > 0, f"No frames found in {right_dir}"
    assert len(left_paths) == len(right_paths), \
        f"Frame count mismatch: {len(left_paths)} left vs {len(right_paths)} right"

    output_path = os.path.join(frame_dir, output_name)

    first = cv2.imread(str(left_paths[0]))
    H, W = first.shape[:2]

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W * 2, H)
    )

    for left_p, right_p in zip(left_paths, right_paths):
        left_frame  = cv2.imread(str(left_p))
        right_frame = cv2.imread(str(right_p))
        composite   = np.concatenate([left_frame, right_frame], axis=1)  # hstack
        writer.write(composite)

    writer.release()
    print(f"Saved stereo video to {output_path} ({len(left_paths)} frames @ {fps} fps)")

def load_keypoint_3d_centers(json_path: str) -> np.ndarray:
    """
    Load keypoint 3D centers from JSON and return as ordered (N, 3) numpy array.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    sorted_ids = sorted(data.keys(), key=lambda k: int(k))
    centers = np.array([[data[k]["x"], data[k]["y"], data[k]["z"]] for k in sorted_ids], dtype=np.float32)
    return centers

def load_ply(path, keypoint_id, device="cuda"):
    """
    Load a PLY file and extract Gaussian point cloud data.
    """
    plydata = PlyData.read(path)
    v = plydata["vertex"]

    # Center of gaussians
    centers = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    # RGB colors
    SH_C0 = 0.28209479177387814
    rgbs = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    rgbs = np.clip(rgbs, 0.0, 1.0).astype(np.float32)

    # Opacities
    opacities = (1.0 / (1.0 + np.exp(-v["opacity"]))[:, None]).astype(np.float32)

    # Scales (originally saved in logscale by supersplat)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)).astype(np.float32)

    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)

    means = torch.from_numpy(centers).to(device).float()
    rgbs = torch.from_numpy(rgbs).to(device).float()
    opacities = torch.from_numpy(opacities).to(device).float()
    scales = torch.from_numpy(scales).to(device).float()
    quats = torch.from_numpy(quats).to(device).float()

    # Store keypoint ID for each Gaussian
    N = means.shape[0]
    kp_ids = torch.full((N, 1), keypoint_id, device=device, dtype=torch.float32)
    
    return means, scales, quats, rgbs, opacities, kp_ids


def render_image(
    ply_object_path: str,
    keypoint_dir: str,
    viewmat: torch.Tensor,
    focal: float,
    width: int,
    height: int,
    out_img_path: str,
    out_mask_path: str,
    out_depth_path: str,
    near: float = 0.01,
    far: float = 100.0,
    device = "cuda"
):
    """
    Render an image and semantic mask from object and multiple keypoint PLY files.
    """
    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_depth_path), exist_ok=True)
    viewmat_torch = viewmat.to(device).float()

    # Load object (keypoint_id=0)
    means_list, scales_list, quats_list, rgbs_list, ops_list, kp_ids_list = [], [], [], [], [], []
    
    means_obj, scales_obj, quats_obj, rgbs_obj, ops_obj, kp_ids_obj = load_ply(os.path.join(ply_object_path, "interface.ply"), 1, device)
    means_list.append(means_obj)
    scales_list.append(scales_obj)
    quats_list.append(quats_obj)
    rgbs_list.append(rgbs_obj)
    ops_list.append(ops_obj)
    kp_ids_list.append(kp_ids_obj)
    means_obj, scales_obj, quats_obj, rgbs_obj, ops_obj, kp_ids_obj = load_ply(os.path.join(ply_object_path, "front.ply"), 2, device)
    means_list.append(means_obj)
    scales_list.append(scales_obj)
    quats_list.append(quats_obj)
    rgbs_list.append(rgbs_obj)
    ops_list.append(ops_obj)
    kp_ids_list.append(kp_ids_obj)

    # Load all numbered keypoint files
    keypoint_files = sorted(glob.glob(os.path.join(keypoint_dir, "*.ply")))
    num_keypoints = len(keypoint_files)
    num_classes = num_keypoints + 3
    
    for kp_file in keypoint_files:
        # Extract keypoint number from filename (e.g., "1.ply" -> 1)
        filename = os.path.basename(kp_file)
        kp_num = int(os.path.splitext(filename)[0])+2
        
        means_kp, scales_kp, quats_kp, rgbs_kp, ops_kp, kp_ids_kp = load_ply(kp_file, kp_num, device)
        means_list.append(means_kp)
        scales_list.append(scales_kp)
        quats_list.append(quats_kp)
        rgbs_list.append(rgbs_kp)
        ops_list.append(ops_kp)
        kp_ids_list.append(kp_ids_kp)

    # Combine everything
    means = torch.cat(means_list, dim=0)
    scales = torch.cat(scales_list, dim=0)
    quats = torch.cat(quats_list, dim=0)
    all_rgbs = torch.cat(rgbs_list, dim=0)
    all_opacities = torch.cat(ops_list, dim=0)
    all_kp_ids = torch.cat(kp_ids_list, dim=0)

    # use homogeneous coords to project to image space
    means_hom = torch.cat([means, torch.ones(means.shape[0], 1, device=device, dtype=means.dtype)], dim=1)
    means_cam = (viewmat_torch @ means_hom.T).T  # [N, 4]
    depths = means_cam[:, 2:3]
    
    # Create one-hot encoded features for each Gaussian [N x num_classes]
    one_hot_features = torch.nn.functional.one_hot(
        all_kp_ids.long().squeeze(-1), 
        num_classes=num_classes
    ).float()

    # concat features: RGB (3) + keypoint_id (1) + depth (1) = 5 channels
    features = torch.cat([all_rgbs, one_hot_features, depths], dim=-1)
    features_premult = features * all_opacities

    # Add batch dims for rasterizer
    means_batch, scales_batch, quats_batch = (means.unsqueeze(0), scales.unsqueeze(0), quats.unsqueeze(0))
    alphas_batch = all_opacities.squeeze(-1).unsqueeze(0)
    features_batch = features_premult.unsqueeze(0)
    viewmats_batch = viewmat_torch.unsqueeze(0).unsqueeze(0)

    # Intrinsics
    Ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device, dtype=torch.float32,
    ).view(1, 1, 3, 3)

    render_output, render_alphas, _ = rasterization(
        means_batch, quats_batch, scales_batch, alphas_batch,
        features_batch, viewmats_batch, Ks, width, height, near, far,
        render_mode="RGB"
    )
    out_tensor = render_output[0, 0].detach().cpu().numpy()  # [H, W, 5]
    alpha_map = render_alphas[0, 0].detach().cpu().numpy()
    
    # Save data
    rgb_img = (out_tensor[..., :3].clip(0, 1) * 255).astype(np.uint8)

    class_contributions = out_tensor[..., 3:3+num_classes]  # [H, W, num_classes]
    raw_depth = out_tensor[..., -1]
    acc_alpha = alpha_map[..., 0]
    depth_map = raw_depth / (acc_alpha + 1e-6)
    depth_map[acc_alpha < 0.5] = 0 
    depth_vis = (depth_map / far * 255).astype(np.uint8) 

    label_mask = np.argmax(class_contributions, axis=-1).astype(np.uint8)   
    label_mask[alpha_map[..., 0] < 0.1] = 0

    # Ensure every keypoint has at least 1 pixel
    unique_classes = np.unique(label_mask)
    for class_id in range(2, num_classes):
        if class_id not in unique_classes:
            mask_gaussians = (all_kp_ids.squeeze(-1) == class_id).cpu().numpy()
            if mask_gaussians.sum() == 0:
                continue

            kp_means = means[mask_gaussians]
            
            # Project to 2D
            kp_means_hom = torch.cat([kp_means, torch.ones(kp_means.shape[0], 1, device=device)], dim=1)
            kp_means_cam = (viewmat_torch @ kp_means_hom.T).T  # [M, 4]
            
            # Camera to image plane (using pinhole projection)
            z = kp_means_cam[:, 2]
            x_img = (focal * kp_means_cam[:, 0] / z) + (width / 2.0)
            y_img = (focal * kp_means_cam[:, 1] / z) + (height / 2.0)
            
            # Compute mean projected position (Center of Mass in 2D)
            cx = int(torch.mean(x_img).item())
            cy = int(torch.mean(y_img).item())
            
            # Clamp to image bounds
            cx = np.clip(cx, 0, width - 1)
            cy = np.clip(cy, 0, height - 1)
            
            # Force this pixel to be the missing keypoint class
            label_mask[cy, cx] = class_id

    Image.fromarray(rgb_img).save(out_img_path)
    Image.fromarray(label_mask).save(out_mask_path)
    Image.fromarray(depth_vis).save(out_depth_path)


def look_at(camera_pos, target, up_axis='y'):
    """
    Constructs a World-to-Camera View Matrix.
    """
    # Define directions of view
    forward = target - camera_pos
    forward = F.normalize(forward, dim=-1)
    if up_axis == 'z':
        up = torch.tensor([0.0, 0.0, 1.0], device=camera_pos.device).expand_as(camera_pos)
    else:
        up = torch.tensor([0.0, -1.0, 0.0], device=camera_pos.device).expand_as(camera_pos)
    right = torch.cross(forward, up, dim=-1)
    right = F.normalize(right, dim=-1)
    down = torch.cross(forward, right, dim=-1)
    down = F.normalize(down, dim=-1)

    # Rotation Matrix (X, Y, Z axes as rows)
    R = torch.stack([right, down, forward], dim=1)  # (N, 3, 3)
    t = -torch.bmm(R, camera_pos.unsqueeze(-1)).squeeze(-1) # (N, 3)

    # compose projection homogenous matrix
    batch_size = camera_pos.shape[0]
    view_mats = torch.eye(4, device=camera_pos.device).unsqueeze(0).repeat(batch_size, 1, 1)
    view_mats[:, :3, :3] = R
    view_mats[:, :3, 3] = t
    
    return view_mats

def random_trajectory_bottom_cone(num_views, num_keypoints = 8, radius_range = (3.0, 5.0), theta_range_deg = (125, 180)):
    """
    Generate smooth temporal camera trajectories within a spherical cone constraint using a cubic spline between keypoints.

    Args:
        num_views (int):       Number of output frames.
        num_keypoints (int):   Number of random anchors to spline through.
        radius_range (tuple):  (min_radius, max_radius) for camera distance.
        theta_range_deg (tuple): (min_theta, max_theta) polar angle in degrees.
    Returns:
        Tensor: (num_views, 4, 4) view matrices (world-to-camera).
    """
    azimuth_kp = np.random.uniform(0, 2 * np.pi, num_keypoints)

    theta_min_rad = np.deg2rad(theta_range_deg[0])
    theta_max_rad = np.deg2rad(theta_range_deg[1])
    cos_max = np.cos(theta_min_rad)
    cos_min = np.cos(theta_max_rad)
    cos_theta_kp = np.random.uniform(cos_min, cos_max, num_keypoints)
    sin_theta_kp = np.sqrt(np.clip(1.0 - cos_theta_kp ** 2, 0, None))

    r_min, r_max = radius_range
    radius_kp = np.random.uniform(r_min, r_max, num_keypoints)

    x_kp = radius_kp * sin_theta_kp * np.cos(azimuth_kp)
    y_kp = radius_kp * sin_theta_kp * np.sin(azimuth_kp)
    z_kp = radius_kp * cos_theta_kp
    positions_kp = np.stack([x_kp, y_kp, z_kp], axis=-1)

    # Parameterize keypoints by arc-length approximation (chord length)nThis gives more uniform speed along the path vs. uniform t spacing.
    deltas = np.diff(positions_kp, axis=0)
    chord_lengths = np.linalg.norm(deltas, axis=-1)
    t_kp = np.concatenate([[0.0], np.cumsum(chord_lengths)])
    t_kp /= t_kp[-1]  # normalize to [0, 1]

    cs_x = CubicSpline(t_kp, positions_kp[:, 0])
    cs_y = CubicSpline(t_kp, positions_kp[:, 1])
    cs_z = CubicSpline(t_kp, positions_kp[:, 2])

    t_frames = np.linspace(0.0, 1.0, num_views, endpoint=True)
    x_frames = cs_x(t_frames)
    y_frames = cs_y(t_frames)
    z_frames = cs_z(t_frames)

    # clamp positions back into cone bounds
    cam_positions = np.stack([x_frames, y_frames, z_frames], axis=-1)

    radii = np.linalg.norm(cam_positions, axis=-1, keepdims=True)
    directions = cam_positions / (radii + 1e-8)
    clamped_radii = np.clip(radii, r_min, r_max)
    cos_theta_frames = directions[:, 2:3]
    cos_theta_clamped = np.clip(cos_theta_frames, cos_min, cos_max)

    # Reconstruct directions with clamped theta (preserve azimuth, adjust z)
    xy_norm = np.linalg.norm(directions[:, :2], axis=-1, keepdims=True) + 1e-8
    sin_theta_clamped = np.sqrt(np.clip(1.0 - cos_theta_clamped ** 2, 0, None))
    xy_dir = directions[:, :2] / xy_norm
    clamped_dirs = np.concatenate([
        xy_dir * sin_theta_clamped,
        cos_theta_clamped
    ], axis=-1)

    cam_positions = clamped_dirs * clamped_radii

    # Build view matrices
    cam_pos_torch = torch.from_numpy(cam_positions).float()
    target = torch.zeros_like(cam_pos_torch)

    return look_at(cam_pos_torch, target, up_axis='y')

def random_view_bottom_cone(num_views, radius_range=(3.0, 5.0), theta_range_deg=(180, 180)):
    """
    Generates N random view matrices from a specific band of the sphere.
    """
    # azimuth [0, 2pi]
    azimuth = torch.rand(num_views) * 2 * np.pi
    
    # elevation [Min, Max] (Usually 90-180)
    theta_min_rad = np.deg2rad(theta_range_deg[0])
    theta_max_rad = np.deg2rad(theta_range_deg[1])
    cos_max = np.cos(theta_min_rad)
    cos_min = np.cos(theta_max_rad)
    cos_theta = torch.rand(num_views) * (cos_max - cos_min) + cos_min
    sin_theta = torch.sqrt(1 - cos_theta**2)
    
    # radius random sampling
    r_min, r_max = radius_range
    radius = torch.rand(num_views) * (r_max - r_min) + r_min
    
    # spherical to cartesian
    x = radius * sin_theta * torch.cos(azimuth)
    y = radius * sin_theta * torch.sin(azimuth)
    z = radius * cos_theta
    camera_pos = torch.stack([x, y, z], dim=-1)
    target = torch.zeros_like(camera_pos)
    
    return look_at(camera_pos, target, up_axis='y')

def random_smooth_rotation_angles(num_views, rot_range_deg = 360.0, min_segment_len = 10, max_segment_len = 40, speed_range_deg = (1.0, 5.0), smooth_transition_frames = 5):
    """
    Generate N smooth rotation angles simulating realistic object rotation in video.

    Args:
        num_views (int):               Total number of frames / angle samples.
        rot_range_deg (float):         Max absolute rotation in degrees.
        min_segment_len (int):         Minimum frames per directional segment.
        max_segment_len (int):         Maximum frames per directional segment.
        speed_range_deg (tuple):       (min, max) angular speed in degrees/frame.
        smooth_transition_frames (int): Number of frames used to blend speed at direction changes (cosine ease in/out).

    Returns:
        np.ndarray: (num_views,) array of rotation angles in degrees.
    """
    angular_velocity = np.zeros(num_views)
    frame = 0
    current_dir = np.random.choice([-1, 1])

    while frame < num_views:
        # Sample segment length and speed
        seg_len = np.random.randint(min_segment_len, max_segment_len + 1)
        seg_len = min(seg_len, num_views - frame)
        speed = np.random.uniform(*speed_range_deg)

        # Fill the segment with constant angular velocity
        angular_velocity[frame:frame + seg_len] = current_dir * speed

        # Smooth the transition zone around the boundary
        t = min(smooth_transition_frames, seg_len // 2)
        if t > 0:
            ramp_out = np.array([0.5 * (1 + np.cos(np.pi * k / t)) for k in range(t)])
            end_start = frame + seg_len - t
            angular_velocity[end_start:frame + seg_len] *= ramp_out[::-1]
            if frame > 0:
                ramp_in = np.array([0.5 * (1 - np.cos(np.pi * k / t)) for k in range(t)])
                angular_velocity[frame:frame + t] *= ramp_in

        frame += seg_len
        current_dir *= -1

    # Integrate velocity → cumulative angle
    angles = np.cumsum(angular_velocity)
    # Clip to [-rot_range_deg, +rot_range_deg]
    angles = np.clip(angles, -rot_range_deg, rot_range_deg)

    return angles

def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy().tolist()
    return tensor.tolist()

def get_z_rotation_matrix(angle_deg, device='cuda'):
    """Returns a 3x3 rotation matrix for rotation around Z-axis."""
    rad = np.deg2rad(angle_deg)
    c = np.cos(rad)
    s = np.sin(rad)
    # Rotating the OBJECT around Z.
    return torch.tensor([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], device=device, dtype=torch.float32)

def generate_stereo_cameras_rigid(num_pairs, baseline, radius_range, theta_range_deg, rot_angles=None, num_keypoints=8):
    """
    Generates RIGID stereo camera pairs (Parallel axes) with optional Z-axis orbit (object rotation simulation).
    Args:
        rot_angles: Optional list/array of rotation angles (degrees) for each pair. 
                    If provided, orbits the camera by -angle around Z.
    """
    # Generate base Center views
    center_views_w2c = random_trajectory_bottom_cone(num_pairs, num_keypoints, radius_range, theta_range_deg)
    
    left_views = []
    right_views = []
    
    for i in range(num_pairs):
        w2c_center = center_views_w2c[i]
        c2w_center = torch.linalg.inv(w2c_center)
        
        # Apply Orbit Rotation
        if rot_angles is not None:
            angle_deg = rot_angles[i]
            R_orbit = get_z_rotation_matrix(-angle_deg, device=c2w_center.device)
            c2w_center[:3, 3] = R_orbit @ c2w_center[:3, 3]
            c2w_center[:3, :3] = R_orbit @ c2w_center[:3, :3]

        # Compute Stereo Offsets 
        R = c2w_center[:3, :3]
        center_pos = c2w_center[:3, 3]
        right_vec_world = c2w_center[:3, 0] 
        
        # Calculate Rigid Offsets
        pos_left = center_pos - (right_vec_world * (baseline * 0.5))
        pos_right = center_pos + (right_vec_world * (baseline * 0.5))
        
        # Construct W2C Matrices for Left/Right
        R_w2c = R.T 
        
        # Left Camera
        w2c_left = torch.eye(4, device=center_pos.device)
        w2c_left[:3, :3] = R_w2c
        w2c_left[:3, 3] = -torch.matmul(R_w2c, pos_left)
        
        # Right Camera
        w2c_right = torch.eye(4, device=center_pos.device)
        w2c_right[:3, :3] = R_w2c
        w2c_right[:3, 3] = -torch.matmul(R_w2c, pos_right)
        
        left_views.append(w2c_left)
        right_views.append(w2c_right)
        
    return torch.stack(left_views), torch.stack(right_views)

def extract_keypoint_3d_centers(keypoint_dir, output_json_path, baseline, focal, H, W, weight_by_opacity = True):
    """
    Extract the 3D center of mass for each keypoint PLY file and save to JSON.
    """
    keypoint_files = sorted(glob.glob(os.path.join(keypoint_dir, "*.ply")))
    
    if len(keypoint_files) == 0:
        raise FileNotFoundError(f"No PLY files found in: {keypoint_dir}")

    keypoint_centers = {}

    for kp_file in keypoint_files:
        filename = os.path.basename(kp_file)
        kp_id = int(os.path.splitext(filename)[0])

        plydata = PlyData.read(kp_file)
        v = plydata["vertex"]

        # 3D Gaussian centers — shape (N, 3)
        centers = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)

        if weight_by_opacity:
            # Sigmoid to convert raw logit opacity to [0, 1]
            raw_opacity = np.array(v["opacity"], dtype=np.float64)
            weights = 1.0 / (1.0 + np.exp(-raw_opacity))
            weights = weights / weights.sum()
            center_3d = (centers * weights[:, None]).sum(axis=0)
        else:
            center_3d = centers.mean(axis=0)

        keypoint_centers[str(kp_id)] = {
            "x": float(center_3d[0]),
            "y": float(center_3d[1]),
            "z": float(center_3d[2]),
        }
    

    keypoint_centers["baseline_rotation"] = [[1,0,0],[0,1,0],[0,0,1]]
    keypoint_centers["l2r_baseline_translation"] = [-baseline,0,0]
    keypoint_centers["baseline"] = baseline
    keypoint_centers['focal'] = focal
    keypoint_centers['H'] = H
    keypoint_centers['W'] = W
    # Save to JSON
    with open(output_json_path, "w") as f:
        json.dump(keypoint_centers, f, indent=2)

    print(f"\nSaved {len(keypoint_centers)} keypoint centers to: {output_json_path}")
    return keypoint_centers


if __name__ == "__main__":
    args = parse_args()
    base_ply = args.base_ply
    keypoint_ply = args.keypoint_dir
    out_dir = args.out_dir
    
    train_dir = os.path.join(out_dir, "train")
    # For stereo test, we use specific structure
    test_dir_left = os.path.join(out_dir, "test", "left")
    test_dir_right = os.path.join(out_dir, "test", "right")
    test_dir_single = os.path.join(out_dir, "test", "single")

    os.makedirs(train_dir, exist_ok=True)
    if args.stereo_test:
        os.makedirs(test_dir_left, exist_ok=True)
        os.makedirs(test_dir_right, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "meta_data"), exist_ok=True)
    else:
        os.makedirs(test_dir_single, exist_ok=True)
    
    H, W = args.height, args.width
    focal, near, far = args.focal, args.near, args.far

    # Calculate Split
    num_train = args.num_views_train
    num_test = args.num_views_test
    
    print(f"Train: {num_train} | Test: {num_test}")
    # 3D coordinates
    os.makedirs(os.path.join(out_dir, "meta_data"), exist_ok=True)
    extract_keypoint_3d_centers(keypoint_ply, os.path.join(out_dir, "meta_data", "object_keypoints.json"), args.baseline, focal, H, W)

    # TRAIN GENERATION
    print("Generating Training Data...")
    train_views = random_view_bottom_cone(num_train, args.radius_range, args.theta_range)
    rot_angles = np.random.uniform(-args.object_rot_range, args.object_rot_range, num_train)
    for i, view_w2c in tqdm(enumerate(train_views), total=num_train, desc="Training"):
        rgb_path = os.path.join(train_dir, f"view_{i:05d}_rgb.jpeg")
        mask_path = os.path.join(train_dir, f"view_{i:05d}_mask.png")
        depth_path = os.path.join(train_dir, f"view_{i:05d}_depth.png")
        json_path = os.path.join(train_dir, f"view_{i:05d}_params.json")
        
        # Save JSON with Pose
        c2w = torch.linalg.inv(view_w2c)
        # Augment
        angle_deg = rot_angles[i]
        R_orbit = get_z_rotation_matrix(-angle_deg, device=c2w.device)
        c2w[:3, 3] = R_orbit @ c2w[:3, 3]
        c2w[:3, :3] = R_orbit @ c2w[:3, :3]
        view_w2c = torch.linalg.inv(c2w)

        params = {
            "file_path": f"view_{i:05d}_rgb.jpeg",
            "deg":angle_deg,
            "c2w": tensor_to_list(c2w),
            "fl_x": focal, "fl_y": focal, "cx": W/2, "cy": H/2, "w": W, "h": H
        }
        with open(json_path, 'w') as f:
            json.dump(params, f, indent=4)
        

        render_image(
            ply_object_path=base_ply, keypoint_dir=keypoint_ply,
            viewmat=view_w2c, focal=focal, width=W, height=H,
            out_img_path=rgb_path, out_mask_path=mask_path, out_depth_path=depth_path,
            near=near, far=far
        )
    print(f"Generating Test Data ({'Stereo' if args.stereo_test else 'Mono'})...")

    rot_angles = np.random.uniform(-args.object_rot_range, args.object_rot_range, num_test)
    rot_angles = random_smooth_rotation_angles(
        num_views=num_test, 
        rot_range_deg=args.object_rot_range, 
        min_segment_len=args.min_segment_len, 
        max_segment_len=args.max_segment_len, 
        speed_range_deg=args.speed_range_deg,
        smooth_transition_frames=args.smooth_transition_frames
        )
    if args.stereo_test:
        left_views, right_views = generate_stereo_cameras_rigid(num_test, args.baseline, args.radius_range, args.theta_range, rot_angles=rot_angles, num_keypoints=args.num_keypoints)
        
        for i in tqdm(range(num_test), desc="Testing"):
            rot_deg = rot_angles[i]

            # Render Left
            render_image(
                ply_object_path=base_ply, keypoint_dir=keypoint_ply,
                viewmat=left_views[i], focal=focal, width=W, height=H,
                out_img_path=os.path.join(test_dir_left, f"view_{i:04d}_rgb.jpeg"),
                out_mask_path=os.path.join(test_dir_left, f"view_{i:04d}_mask.png"),
                out_depth_path=os.path.join(test_dir_left, f"view_{i:04d}_depth.png"),
                near=near, far=far
            )
            # Render Right
            render_image(
                ply_object_path=base_ply, keypoint_dir=keypoint_ply,
                viewmat=right_views[i], focal=focal, width=W, height=H,
                out_img_path=os.path.join(test_dir_right, f"view_{i:04d}_rgb.jpeg"),
                out_mask_path=os.path.join(test_dir_right, f"view_{i:04d}_mask.png"),
                out_depth_path=os.path.join(test_dir_right, f"view_{i:04d}_depth.png"),
                near=near, far=far
            )
            
            # Save Combined JSON
            c2w_left = torch.linalg.inv(left_views[i])
            c2w_right = torch.linalg.inv(right_views[i])
            
            params = {
                "left_file": f"view_{i:04d}_rgb.jpeg",
                "right_file": f"view_{i:04d}_rgb.jpeg",
                "c2w_left": tensor_to_list(c2w_left),
                "c2w_right": tensor_to_list(c2w_right),
                "left_w2c": tensor_to_list(left_views[i]),
                "object_rotation_z_deg": float(rot_deg),
                "object_rotation_matrix": get_z_rotation_matrix(rot_deg).cpu().numpy().tolist()
            }
            with open(os.path.join(out_dir, "meta_data", f"pair_{i:04d}_params.json"), 'w') as f:
                json.dump(params, f, indent=4)
        frames_to_video(os.path.join(out_dir, "test"))
                
    else:
        test_views = random_view_bottom_cone(num_test, args.radius_range, args.theta_range)
        for i, view_w2c in tqdm(enumerate(test_views), total=num_test, desc="Testing"):
            rot_deg = rot_angles[i]
            R_obj = get_z_rotation_matrix(rot_deg)

            # Update paths to use the determined subdirectory
            rgb_path = os.path.join(test_dir_single, f"view_{i:05d}_rgb.jpeg")
            mask_path = os.path.join(test_dir_single, f"view_{i:05d}_mask.png")
            depth_path = os.path.join(test_dir_single, f"view_{i:05d}_depth.png")
            json_path = os.path.join(test_dir_single, f"view_{i:05d}_params.json")
            c2w = torch.linalg.inv(view_w2c)

            params = {
                "file_path": f"view_{i:05d}_rgb.jpeg",
                "c2w": tensor_to_list(c2w),
                "object_rotation_z_deg": float(rot_deg),
                "object_rotation_matrix": tensor_to_list(R_obj),
            }
            with open(os.path.join(test_dir_single, f"pair_{i:04d}_params.json"), 'w') as f:
                json.dump(params, f, indent=4)

            render_image(
                ply_object_path=base_ply,
                keypoint_dir=keypoint_ply,
                viewmat=view_w2c,
                focal=focal,
                width=W,
                height=H,
                out_img_path=rgb_path,
                out_mask_path=mask_path,
                out_depth_path=depth_path,
                near=near,
                far=far,
            )