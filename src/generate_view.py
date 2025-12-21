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
from tqdm import tqdm 

def load_ply(path, is_keypoint, device="cuda"):
    """
    Load a PLY file and extract Gaussian point cloud data.

    Args:
        path (str): Path to the PLY file.
        is_keypoint (bool): Whether the data represents keypoints or objects.
        device (str): Device to load tensors on ('cuda' or 'cpu').

    Returns:
        tuple: Tensors for means, scales, quaternions, RGBs, opacities, and semantic IDs.
    """
    plydata = PlyData.read(path)
    v = plydata["vertex"]

    # Center of gaussians
    centers = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    # RGB colors given the spherical harmonic coefficients
    SH_C0 = 0.28209479177387814
    rgbs = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    rgbs = np.clip(rgbs, 0.0, 1.0).astype(np.float32)

    # Extract opacities and sigmoid it
    opacities = (1.0 / (1.0 + np.exp(-v["opacity"]))[:, None]).astype(np.float32)

    # Extract scales (which are usually stored in log scale)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)).astype(np.float32)

    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32))
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)
    

    means = torch.from_numpy(centers).to(device).float()
    rgbs = torch.from_numpy(rgbs).to(device).float()
    opacities = torch.from_numpy(opacities).to(device).float()
    scales = torch.from_numpy(scales).to(device).float()
    quats = torch.from_numpy(quats).to(device).float()

    # Semantic Vector
    N = means.shape[0]
    sem_ids = torch.zeros((N, 2), device=device, dtype=torch.float32)
    
    if is_keypoint:
        sem_ids[:, 1] = 1.0 # mark as keypoints
    else:
        sem_ids[:, 0] = 1.0 # mark as object
    
    return means, scales, quats, rgbs, opacities, sem_ids

def render_image(
    ply_object_path: str,
    ply_keypoints_path: str,
    viewmat: torch.Tensor,
    focal: float,
    width: int,
    height: int,
    out_img_path: str,
    out_mask_path: str,
    near: float = 0.01,
    far: float = 100.0,
    device = "cuda"
):
    """
    Render an image and semantic mask from object and keypoint PLY data.

    Args:
        ply_object_path (str): Path to the object PLY file.
        ply_keypoints_path (str): Path to the keypoints PLY file.
        viewmat (torch.Tensor): 4x4 camera view matrix.
        focal (float): Camera focal length.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        out_img_path (str): Output path for the RGB image.
        out_mask_path (str): Output path for the mask image.
        near (float): Near plane clipping distance.
        far (float): Far plane clipping distance.
        device (str, optional): Device for computation ('cuda' or 'cpu').

    Returns:
        None
    """
    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
    viewmat_torch = viewmat.to(device).float()

    means1, scales1, quats1, rgbs1, ops1, sem1 = load_ply(ply_object_path, False)
    means2, scales2, quats2, rgbs2, ops2, sem2 = load_ply(ply_keypoints_path, True)

    # Combine everything
    means = torch.cat([means1, means2], dim=0)
    scales = torch.cat([scales1, scales2], dim=0)
    quats = torch.cat([quats1, quats2], dim=0)
    all_rgbs = torch.cat([rgbs1, rgbs2], dim=0)
    all_opacities = torch.cat([ops1, ops2], dim=0)
    all_sems = torch.cat([sem1, sem2], dim=0)

    # use homogeneous coords to project to image space
    means_hom = torch.cat([means, torch.ones_like(means[:, :1])], dim=1) 
    means_cam = (viewmat_torch @ means_hom.T).T  # [N, 4]
    depths = means_cam[:, 2:3]

    # concat features
    features = torch.cat([all_rgbs, all_sems, depths], dim=-1)
    features_premult = features * all_opacities

    # Add batch dims for rasterizer
    means_batch = means.unsqueeze(0)
    scales_batch = scales.unsqueeze(0)
    quats_batch = quats.unsqueeze(0)
    alphas_batch = all_opacities.squeeze(-1).unsqueeze(0)
    features_batch = features_premult.unsqueeze(0)
    viewmats_batch = viewmat.to(device).float().view(1, 1, 4, 4)

    # Instrinsics
    Ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device, dtype=torch.float32,
    ).view(1, 1, 3, 3)

    # Rasterize using gsplat
    render_output, __, _ = rasterization(
        means_batch, quats_batch, scales_batch, alphas_batch,
        features_batch,
        viewmats_batch, Ks, width, height, near, far,
        render_mode="RGB", 
    )

    out_tensor = render_output[0, 0].detach().cpu().numpy() # [H, W, 6]
    
    # Save data
    rgb_img = (out_tensor[..., :3].clip(0, 1) * 255).astype(np.uint8)
    label_mask = np.zeros((height, width), dtype=np.uint8)
    label_mask[out_tensor[..., 3] > 0.5] = 1
    label_mask[out_tensor[..., 4] > 0.5] = 2
    # label_mask = label_mask*127

    Image.fromarray(rgb_img).save(out_img_path)
    Image.fromarray(label_mask).save(out_mask_path)


def look_at(camera_pos, target, up_axis='y'):
    """
    Constructs a World-to-Camera View Matrix.
    
    Args:
        camera_pos: (N, 3) tensor of camera eye locations.
        target: (N, 3) tensor of look-at points.
        up_axis: 'y' or 'z'. Defines the world up vector for orientation.
                 For bottom-hemisphere views, 'y' is often more stable at the pole.
    
    Returns:
        (N, 4, 4) FloatTensor view matrices.
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

def random_view_bottom_cone(num_views, radius_range=(3.0, 5.0), theta_range_deg=(120, 150)):
    """
    Generates N random view matrices from a specific band of the sphere.
    
    Args:
        num_views (int): Number of random views.
        radius_range (tuple): (min_radius, max_radius).
        theta_range_deg (tuple): (min_angle, max_angle) in degrees.
                                 0   = North Pole (+Z)
                                 90  = Equator (Z=0)
                                 180 = South Pole (-Z)
                                 Example: (120, 150) samples a band in the bottom hemisphere,
                                 avoiding the equator and the direct bottom view.
    
    Returns:
        views (Tensor): (N, 4, 4) view matrices.
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
    return tensor.cpu().numpy().tolist()

def parse_args():
    parser = argparse.ArgumentParser(description="Render random views from a Gaussian Splatting scene.")
    parser.add_argument("--base_ply", type=str, default="../data/Docking.ply",
                        help="Path to the base object PLY file.")
    parser.add_argument("--keypoint_ply", type=str, default="../data/keypoints.ply",
                        help="Path to the keypoints PLY file.")
    parser.add_argument("--out_dir", type=str, default="./renders",
                        help="Output directory for rendered images and masks.")
    parser.add_argument("--height", type=int, default=1080,
                        help="Output image height in pixels.")
    parser.add_argument("--width", type=int, default=1280,
                        help="Output image width in pixels.")
    parser.add_argument("--focal", type=float, default=800.0,
                        help="Camera focal length.")
    parser.add_argument("--near", type=float, default=0.01,
                        help="Near plane distance.")
    parser.add_argument("--far", type=float, default=100.0,
                        help="Far plane distance.")
    parser.add_argument("--num_views", type=int, default=1,
                        help="Number of random views to render.")
    parser.add_argument("--radius_min", type=float, default=7.0,
                        help="Minimum radius for random camera placement.")
    parser.add_argument("--radius_max", type=float, default=12.0,
                        help="Maximum radius for random camera placement.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    base_ply = args.base_ply
    keypoint_ply = args.keypoint_ply
    out_dir = args.out_dir

    H, W = args.height, args.width
    focal = args.focal
    near = args.near
    far = args.far

    random_views = random_view_bottom_cone(
        num_views=args.num_views,
        radius_range=(args.radius_min, args.radius_max),
    )

    for i, view_w2c in tqdm(enumerate(random_views)):
        rgb_path = os.path.join(out_dir, f"view_{i:05d}_rgb.png")
        mask_path = os.path.join(out_dir, f"view_{i:05d}_mask.png")
        json_path = os.path.join(out_dir, f"view_{i:05d}_params.json")

        c2w = torch.linalg.inv(view_w2c)

        view_dict = {
            "file_path": f"view_{i:03d}_rgb.png",
            "transform_matrix": tensor_to_list(c2w),
            "camera_angle_x": 2 * np.arctan(W / (2 * focal)),
            "fl_x": focal,
            "fl_y": focal,
            "cx": W / 2.0,
            "cy": H / 2.0,
            "w": W,
            "h": H
        }
        with open(json_path, 'w') as f:
            json.dump(view_dict, f, indent=4)

        render_image(
            ply_object_path=base_ply,
            ply_keypoints_path=keypoint_ply,
            viewmat=view_w2c,
            focal=focal,
            width=W,
            height=H,
            out_img_path=rgb_path,
            out_mask_path=mask_path,
            near=near,
            far=far,
        )