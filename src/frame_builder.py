import numpy as np
import cv2

def to_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pack (R, t) into a 4x4 homogeneous transform."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def from_se3(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a 4x4 homogeneous transform into (R (3,3), t (3,1))."""
    return T[:3, :3].copy(), T[:3, 3:4].copy()


def rotation_to_axis_angle_deg(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (unit axis (3,), angle in degrees) for a rotation matrix."""
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    angle_rad = float(np.linalg.norm(rvec))
    axis = (rvec.flatten() / angle_rad) if angle_rad > 1e-12 else np.array([0.0, 0.0, 1.0])
    return axis, np.degrees(angle_rad)


def quaternion_from_R(R: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion — handy if your orchestrator's API wants quaternions."""
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    angle = np.linalg.norm(rvec)
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rvec.flatten() / angle
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)])


def relative_motion_to_rest(
    R_cam_obj: np.ndarray,
    t_cam_obj: np.ndarray,
    R_cam_rest: np.ndarray,
    t_cam_rest: np.ndarray,
    frame: str = "camera",
) -> dict:
    """
    Compute the rigid-body motion the object must undergo to go from its
    CURRENT observed pose to a known REST/docked pose, both expressed in
    the left-camera frame.

    T_cam_obj  : current object pose in left-cam frame  (from `object_pose_in_left_cam`)
    T_cam_rest : rest/docked pose of the object in left-cam frame — this is
                 a one-time calibration constant for your station (e.g.
                 measured once when the two interfaces are correctly
                 mated), NOT something re-estimated every frame.

    frame : "camera" -> motion expressed in the (fixed) left-camera frame:
                T_rel = T_cam_rest @ inv(T_cam_obj)
                Applying T_rel to the object's current pose (pre-multiply,
                in camera coordinates) yields the rest pose. Use this if
                your orchestrator's motion command is specified in a frame
                that is rigidly fixed to the camera (typical after a
                hand-eye calibration between camera and end-effector).

            "object" -> motion expressed in the object's OWN current frame:
                T_rel = inv(T_cam_obj) @ T_cam_rest
                Use this if commands are issued as "move the gripper that
                is holding the object by this much, in the object's local
                axes".

    Returns a dict with the relative SE(3) transform plus decomposed
    axis-angle rotation (deg) and translation (same units as t_cam_obj,
    typically meters) — the two scalars/vectors you hand to the
    orchestrator ("rotate by this much about this axis, translate by
    this vector").
    """
    T_obj  = to_se3(R_cam_obj, t_cam_obj)
    T_rest = to_se3(R_cam_rest, t_cam_rest)

    if frame == "camera":
        T_rel = T_rest @ np.linalg.inv(T_obj)
    elif frame == "object":
        T_rel = np.linalg.inv(T_obj) @ T_rest
    else:
        raise ValueError("frame must be 'camera' or 'object'")

    R_rel, t_rel = from_se3(T_rel)
    axis, angle_deg = rotation_to_axis_angle_deg(R_rel)

    return {
        "T_rel": T_rel,
        "R_rel": R_rel,
        "t_rel": t_rel,
        "rotation_axis": axis,
        "rotation_angle_deg": angle_deg,
        "translation_vector": t_rel.flatten(),
        "translation_magnitude": float(np.linalg.norm(t_rel)),
        "frame": frame,
    }


def build_pose_record(frame_idx: int, R_cam_obj: np.ndarray, t_cam_obj: np.ndarray, T_cam_rest: np.ndarray | None = None) -> dict:
    """
    Convenience: assemble the full per-frame message you'd publish to the
    orchestrator (e.g. serialize with json.dumps or send over a socket /
    ROS topic / shared memory).
    """
    quat = quaternion_from_R(R_cam_obj)
    record = {
        "frame": int(frame_idx),
        "R_cam_left": R_cam_obj.tolist(),
        "t_cam_left": t_cam_obj.flatten().tolist(),
        "quat_xyzw": quat.tolist(),
    }
    rel = None
    if T_cam_rest is not None:
        R_rest, t_rest = from_se3(T_cam_rest)
        rel = relative_motion_to_rest(R_cam_obj, t_cam_obj, R_rest, t_rest, frame="camera")
        record["relative_to_rest"] = {
            "rotation_axis": rel["rotation_axis"].tolist(),
            "rotation_angle_deg": rel["rotation_angle_deg"],
            "translation_vector": rel["translation_vector"].tolist(),
            "translation_magnitude": rel["translation_magnitude"],
        }
    return record, rel