import numpy as np
import cv2
import torch
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class PoseEstimationResult:
    """Container for pose estimation output."""
    success: bool
    rotation_matrix: np.ndarray  # 3x3
    translation_vector: np.ndarray  # 3x1
    rvec: np.ndarray  # 3x1 Rodrigues vector
    tvec: np.ndarray  # 3x1
    inliers: Optional[np.ndarray]  # Nx1 indices
    reprojection_error_left: float
    reprojection_error_right: float
    num_inliers: int


class StereoPnPPoseEstimator:
    """
    Robust 6DoF pose estimation using stereo keypoint correspondences.
    """
    
    def __init__(
        self,
        camera_matrix_left: np.ndarray,
        camera_matrix_right: np.ndarray,
        dist_coeffs_left: Optional[np.ndarray] = None,
        dist_coeffs_right: Optional[np.ndarray] = None,
        R_stereo: Optional[np.ndarray] = None,
        t_stereo: Optional[np.ndarray] = None,
        ransac_reproj_threshold: float = 4.0,
        ransac_confidence: float = 0.99,
        use_temporal_tracking: bool = True,
        max_temporal_deviation: float = 0.5,
        min_inliers: int = 4
    ):
        """
        Initialize stereo PnP pose estimator.
        
        Args:
            camera_matrix_left: 3x3 intrinsic matrix for left camera
            camera_matrix_right: 3x3 intrinsic matrix for right camera
            dist_coeffs_left: Distortion coefficients for left camera (can be None/zeros)
            dist_coeffs_right: Distortion coefficients for right camera
            R_stereo: 3x3 rotation from left to right camera (identity if None)
            t_stereo: 3x1 translation from left to right camera (zeros if None)
            ransac_reproj_threshold: RANSAC inlier threshold in pixels
            ransac_confidence: RANSAC confidence level (0-1)
            use_temporal_tracking: Enable warm-start from previous frame
            max_temporal_deviation: Maximum allowed pose deviation from prediction (meters)
            min_inliers: Minimum inliers required for valid pose
        """
        self.K_left = camera_matrix_left.astype(np.float64)
        self.K_right = camera_matrix_right.astype(np.float64)
        
        self.dist_left  = dist_coeffs_left if dist_coeffs_left is not None else np.zeros((4, 1), dtype=np.float64) 
        self.dist_right = dist_coeffs_right if dist_coeffs_right is not None else np.zeros((4, 1), dtype=np.float64) 
        
        # Stereo extrinsics (left->right transform)
        self.R_stereo = R_stereo if R_stereo is not None else np.eye(3, dtype=np.float64)
        self.t_stereo = t_stereo if t_stereo is not None else np.zeros((3, 1), dtype=np.float64)
        
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.ransac_confidence = ransac_confidence
        self.use_temporal_tracking = use_temporal_tracking
        self.max_temporal_deviation = max_temporal_deviation
        self.min_inliers = min_inliers
        
        # Temporal tracking state
        self.prev_rvec = None
        self.prev_tvec = None
        self.prev_velocity_rvec = None
        self.prev_velocity_tvec = None

    def _failed_result(self, reason = "Unknown"):
        """Return a failed pose estimation result."""
        return PoseEstimationResult(
            success=False,
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros((3, 1)),
            rvec=np.zeros((3, 1)),
            tvec=np.zeros((3, 1)),
            inliers=None,
            reprojection_error_left=float('inf'),
            reprojection_error_right=float('inf'),
            num_inliers=0
        )

    def estimate_pose(self, keypoints_2d, keypoints_3d, use_refinement = True, return_diagnostics = False):
        """
        Estimate 6DoF pose from stereo keypoint correspondences.
        
        Args:
            keypoints_2d: torch.Tensor of shape (2, N, 2) containing [left, right] pixel coords
            keypoints_3d: np.ndarray of shape (N, 3) containing 3D model keypoints in object frame
            use_refinement: Whether to apply LM refinement after RANSAC
            return_diagnostics: Whether to compute detailed diagnostics
            
        Returns:
            PoseEstimationResult containing pose, inliers, and reprojection errors
        """
        # Convert inputs to numpy
        kpts_left, kpts_right = self._prepare_inputs(keypoints_2d, keypoints_3d)
        object_points = keypoints_3d.cpu().numpy().astype(np.float64)
        # object_points[:, 2] = 0.0

        N = len(object_points)
        if N < 4:
            return self._failed_result("Insufficient keypoints (need ≥4)")
        
        # Initial solution
        initial_result = self._solve_pnp_left_ransac(object_points, kpts_left)
        if not initial_result['success']:
            return self._failed_result("Left camera RANSAC failed")
        
        rvec = initial_result['rvec']
        tvec = initial_result['tvec']
        inliers = initial_result['inliers']

        if len(inliers) < self.min_inliers:
            return self._failed_result(f"Insufficient inliers ({len(inliers)} < {self.min_inliers})")
        
        if use_refinement:
            rvec, tvec = self._refine_stereo_pnp(object_points, kpts_left, kpts_right, rvec, tvec, inliers)

        R, _ = cv2.Rodrigues(rvec)
        
        reproj_error_left = self._compute_reprojection_error(object_points[inliers], kpts_left[inliers], rvec, tvec, self.K_left, self.dist_left)
        reproj_error_right = self._compute_reprojection_error(object_points[inliers], kpts_right[inliers], rvec, tvec, self.K_right, self.dist_right,R_cam=self.R_stereo, t_cam=self.t_stereo)
        self._update_temporal_state(rvec, tvec)
        
        return PoseEstimationResult(
            success=True,
            rotation_matrix=R,
            translation_vector=tvec,
            rvec=rvec,
            tvec=tvec,
            inliers=inliers,
            reprojection_error_left=reproj_error_left,
            reprojection_error_right=reproj_error_right,
            num_inliers=len(inliers)
        )

    def _prepare_inputs(self, keypoints_2d, keypoints_3d):
        """Convert torch tensors to numpy and validate shapes."""
        if isinstance(keypoints_2d, torch.Tensor):
            keypoints_2d = keypoints_2d.cpu().numpy()
        
        assert keypoints_2d.shape[0] == 2, "First dim must be 2 (left, right)"
        assert keypoints_2d.shape[2] == 2, "Last dim must be 2 (x, y)"
        assert keypoints_2d.shape[1] == keypoints_3d.shape[0], "Keypoint count mismatch"
        
        kpts_left = keypoints_2d[0].astype(np.float32)
        kpts_right = keypoints_2d[1].astype(np.float32)
        
        return kpts_left, kpts_right
    
    def _predict_pose_from_velocity(self):
        """Predict current pose using constant velocity motion model."""
        if self.prev_velocity_rvec is None:
            return self.prev_rvec, self.prev_tvec
        
        # Constant velocity prediction
        rvec_pred = self.prev_rvec + self.prev_velocity_rvec
        tvec_pred = self.prev_tvec + self.prev_velocity_tvec
        
        return rvec_pred, tvec_pred
    
    def _solve_pnp_left_ransac(self, object_points, image_points):
        """Solve PnP with RANSAC on left camera, optionally using temporal warm-start."""
        flags = cv2.SOLVEPNP_SQPNP
        use_extrinsic_guess = False
        rvec_init = None
        tvec_init = None
        
        # Temporal warm-start if available
        if self.use_temporal_tracking and self.prev_rvec is not None:
            rvec_init, tvec_init = self._predict_pose_from_velocity()
            
            # Try warm-started iterative solver first (faster convergence)
            success, rvec_ws, tvec_ws = cv2.solvePnP(
                object_points, image_points,
                self.K_left, self.dist_left,
                rvec_init, tvec_init,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Validate pose is physically plausible (not a wild jump)
                deviation = np.linalg.norm(tvec_ws - tvec_init)
                if deviation < self.max_temporal_deviation:
                    # Use warm-start as initial guess for RANSAC
                    rvec_init = rvec_ws
                    tvec_init = tvec_ws
                    use_extrinsic_guess = True
                    flags = cv2.SOLVEPNP_ITERATIVE
        
        # RANSAC with optional warm-start
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points,
            self.K_left, None,
            rvec=rvec_init if use_extrinsic_guess else None,
            tvec=tvec_init if use_extrinsic_guess else None,
            useExtrinsicGuess=use_extrinsic_guess,
            iterationsCount=10000,
            reprojectionError=self.ransac_reproj_threshold,
            confidence=self.ransac_confidence,
            flags=flags
        )
        
        if not success or inliers is None:
            return {'success': False}
        
        return {
            'success': True,
            'rvec': rvec,
            'tvec': tvec,
            'inliers': inliers.flatten()
        }
    
    def _refine_stereo_pnp(self, object_points, kpts_left, kpts_right, rvec, tvec, inliers):
        obj_inliers = object_points[inliers]
        kpts_l      = kpts_left[inliers]
        kpts_r      = kpts_right[inliers]

        rvec_ref, tvec_ref = cv2.solvePnPRefineLM(obj_inliers, kpts_l, self.K_left, self.dist_left, rvec, tvec)
        stereo_calibrated = not np.allclose(self.R_stereo, np.eye(3)) or not np.allclose(self.t_stereo, 0)
        if not stereo_calibrated:
            return rvec_ref, tvec_ref 

        R_left, _ = cv2.Rodrigues(rvec_ref)
        R_right_cam = self.R_stereo @ R_left
        t_right_cam = self.R_stereo @ tvec_ref + self.t_stereo.reshape(3, 1)  # enforce (3,1)
        rvec_right, _ = cv2.Rodrigues(R_right_cam)

        try:
            rvec_ref_r, tvec_ref_r = cv2.solvePnPRefineLM(
                obj_inliers, kpts_r, self.K_right, self.dist_right,
                rvec_right, t_right_cam
            )
        except cv2.error:
            return rvec_ref, tvec_ref 

        R_ref_r, _ = cv2.Rodrigues(rvec_ref_r)
        t_stereo_col = self.t_stereo.reshape(3, 1)   # guarantee (3,1)
        tvec_ref_r   = tvec_ref_r.reshape(3, 1)       # guarantee (3,1)

        R_back = self.R_stereo.T @ R_ref_r
        t_back = self.R_stereo.T @ (tvec_ref_r - t_stereo_col)
        rvec_back, _ = cv2.Rodrigues(R_back)
        err_left_only  = self._compute_reprojection_error(obj_inliers, kpts_l, rvec_ref,  tvec_ref,  self.K_left, self.dist_left)
        err_stereo     = self._compute_reprojection_error(obj_inliers, kpts_l, rvec_back, t_back,    self.K_left, self.dist_left)

        if err_left_only <= err_stereo:
            return rvec_ref, tvec_ref
        return rvec_back, t_back
    
    def _compute_reprojection_error(self, object_points, image_points, rvec, tvec, K, dist, R_cam = None, t_cam = None):
        """
        Compute mean reprojection error in pixels.
        
        Args:
            R_cam, t_cam: Optional transform from reference camera to target camera
        """
        # Transform pose to target camera if stereo transform provided
        if R_cam is not None and t_cam is not None:
            R, _ = cv2.Rodrigues(rvec)
            R_transformed = R_cam @ R
            t_transformed = R_cam @ tvec + t_cam
            rvec_cam, _ = cv2.Rodrigues(R_transformed)
            tvec_cam = t_transformed
        else:
            rvec_cam = rvec
            tvec_cam = tvec
        
        # Project 3D points to image
        projected, _ = cv2.projectPoints(object_points, rvec_cam, tvec_cam, K, dist)
        projected = projected.squeeze()
        
        # Compute Euclidean distance
        errors = np.linalg.norm(projected - image_points, axis=1)
        return float(np.mean(errors))
    
    def _update_temporal_state(self, rvec, tvec):
        """Update temporal tracking state with current pose."""
        if not self.use_temporal_tracking:
            return
        
        if self.prev_rvec is not None:
            # Compute velocity as difference
            self.prev_velocity_rvec = rvec - self.prev_rvec
            self.prev_velocity_tvec = tvec - self.prev_tvec
        
        self.prev_rvec = rvec.copy()
        self.prev_tvec = tvec.copy()