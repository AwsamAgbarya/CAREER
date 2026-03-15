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
        
        self.dist_left  = dist_coeffs_left
        self.dist_right = dist_coeffs_right
        
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

        N = len(object_points)
        if N < 4:
            return self._failed_result("Insufficient keypoints (need ≥4)")
        
        # Initial solution
        initial_result = self._solve_pnp_left_ransac(object_points, kpts_left)
        if not initial_result['success']:
            return self._failed_result("Left camera RANSAC failed")

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
        flags = cv2.SOLVEPNP_IPPE
        print(flags)
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