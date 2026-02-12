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

'''
Example usage:
Conda activate gsplat
python ./src/generate_views.py --num_views 2000 --base_ply ./splats/interface.ply --keypoint_dir ./splats/keypoints/ --out_dir ./renders

Expects an object interface.ply and a keypoint directory such that each keypoint is saved in one separate file, named as an integer according to its class (+1 for technicality, i.e 1.ply keypoint file = class 2)

For more customization pease look at parse_args() function

'''


def parse_args():
    parser = argparse.ArgumentParser(description="Render random views from a Gaussian Splatting scene.")
    parser.add_argument("--base_ply", type=str, default="./splats/interface.ply", help="Path to base PLY.")
    parser.add_argument("--keypoint_dir", type=str, default="./splats/keypoints", help="Path to keypoints.")
    parser.add_argument("--out_dir", type=str, default="./renders", help="Output directory.")
    parser.add_argument("--height", type=int, default=720, help="Image height.")
    parser.add_argument("--width", type=int, default=1280, help="Image width.")
    parser.add_argument("--focal", type=float, default=800.0, help="Focal length.")
    parser.add_argument("--near", type=float, default=0.01, help="Near plane.")
    parser.add_argument("--far", type=float, default=100.0, help="Far plane.")
    parser.add_argument("--num_views", type=int, default=2000, help="TOTAL number of views (Train + Test).")
    parser.add_argument("--train_split", type=float, default=0.8, help="Ratio of training data.")
    parser.add_argument("--radius_range", type=tuple, default=(5.0,15.0), help="Radius range.")
    parser.add_argument("--theta_range", type=tuple, default=(125,180), help="Theta range.")
    parser.add_argument("--stereo_test", action='store_true', help="Use stereo pairs for test set.")
    parser.add_argument("--baseline", type=float, default=1.0, help="Stereo baseline (meters).")
    parser.add_argument("--object_rot_range", type=float, default=359.0, help="Max rotation (degrees) of the object around Z-axis for TEST data (e.g. 360 for full spin).")

    args = parser.parse_args()
    return args



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
    
    means_obj, scales_obj, quats_obj, rgbs_obj, ops_obj, kp_ids_obj = load_ply(ply_object_path, 1, device)
    means_list.append(means_obj)
    scales_list.append(scales_obj)
    quats_list.append(quats_obj)
    rgbs_list.append(rgbs_obj)
    ops_list.append(ops_obj)
    kp_ids_list.append(kp_ids_obj)

    # Load all numbered keypoint files
    keypoint_files = sorted(glob.glob(os.path.join(keypoint_dir, "*.ply")))
    num_keypoints = len(keypoint_files)
    num_classes = num_keypoints + 2
    
    for kp_file in keypoint_files:
        # Extract keypoint number from filename (e.g., "1.ply" -> 1)
        filename = os.path.basename(kp_file)
        kp_num = int(os.path.splitext(filename)[0])+1
        
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

def generate_stereo_cameras_rigid(num_pairs, baseline, radius_range, theta_range_deg, rot_angles=None):
    """
    Generates RIGID stereo camera pairs (Parallel axes) with optional Z-axis orbit (object rotation simulation).
    Args:
        rot_angles: Optional list/array of rotation angles (degrees) for each pair. 
                    If provided, orbits the camera by -angle around Z.
    """
    # Generate base Center views
    center_views_w2c = random_view_bottom_cone(num_pairs, radius_range, theta_range_deg)
    
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
        os.makedirs(os.path.join(out_dir, "left_right_poses"), exist_ok=True)
    else:
        os.makedirs(test_dir_single, exist_ok=True)
    
    H, W = args.height, args.width
    focal, near, far = args.focal, args.near, args.far

    # Calculate Split
    num_total = args.num_views
    num_train = int(num_total * args.train_split)
    num_test = num_total - num_train
    
    print(f"Total: {num_total} | Train: {num_train} | Test: {num_test}")

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
    if args.stereo_test:
        left_views, right_views = generate_stereo_cameras_rigid(num_test, args.baseline, args.radius_range, args.theta_range, rot_angles=rot_angles)
        
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
                "object_rotation_z_deg": float(rot_deg),
                "object_rotation_matrix": get_z_rotation_matrix(rot_deg).cpu().numpy().tolist(),
                "baseline": args.baseline,
                "fl_x": focal, "fl_y": focal, "cx": W/2, "cy": H/2, "w": W, "h": H
            }
            with open(os.path.join(out_dir, "left_right_poses", f"pair_{i:04d}_params.json"), 'w') as f:
                json.dump(params, f, indent=4)
                
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
                "fl_x": focal, "fl_y": focal, "cx": W/2, "cy": H/2, "w": W, "h": H
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