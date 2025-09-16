# common/orientation.py
# Robust torso orientation estimation from a single 3D pose + optional temporal smoothing.
# - NumPy-facing API for easy integration
# - Internally reuses Torch quaternion helpers centralized in quaternion.py
# - Minimal, durable surface area

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch

# Centralized quaternion helpers (Torch) — already added to quaternion.py
try:
    # when imported as common.orientation
    from .quaternion import quat_from_rotmat, quat_slerp
except Exception:
    # fallback if orientation.py is run directly
    from quaternion import quat_from_rotmat, quat_slerp

# ----------------------------- Joint Layouts -----------------------------

@dataclass(frozen=True)
class JointLayout:
    l_hip: int
    r_hip: int
    spine: Optional[int]     # thorax or spine mid (None if not present)
    l_shoulder: int
    r_shoulder: int

H36M_LAYOUT = JointLayout(
    # H36M 17J (adapt if your indices differ)
    # 0: Hip(root), 1: RHip, 2: RKnee, 3: RAnkle, 4: LHip, 5: LKnee, 6: LAnkle,
    # 7: Spine, 8: Thorax, 9: Neck/Nose, 10: Head, 11: LShoulder, 12: LElbow,
    # 13: LWrist, 14: RShoulder, 15: RElbow, 16: RWrist
    l_hip=4, r_hip=1, spine=8, l_shoulder=11, r_shoulder=14
)

COCO17_LAYOUT = JointLayout(
    # 0: Nose, 1: LEye, 2: REye, 3: LEar, 4: REar, 5: LShoulder, 6: RShoulder,
    # 7: LElbow, 8: RElbow, 9: LWrist, 10: RWrist, 11: LHip, 12: RHip,
    # 13: LKnee, 14: RKnee, 15: LAnkle, 16: RAnkle
    l_hip=11, r_hip=12, spine=None, l_shoulder=5, r_shoulder=6
)

_LAYOUTS: Dict[str, JointLayout] = {
    "h36m": H36M_LAYOUT,
    "coco17": COCO17_LAYOUT,
}

# ----------------------------- Utils -----------------------------

def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n

# ----------------------------- Core API -----------------------------

def compute_torso_frame(
    joints_3d: np.ndarray,
    joints_conf: Optional[np.ndarray] = None,
    layout: str = "h36m",
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Estimate torso orientation from a single 3D pose (N x 3), returning:
      - quat_wxyz: rotation (wxyz) whose columns correspond to [right, up, forward]
      - forward, right, up: unit vectors in WORLD coords
      - confidence: [0..1] reliability score

    Frame construction (right-handed):
      up      = normalize(spine_or_shoulder_mid - pelvis_mid)
      forward = normalize(cross(shoulder_axis, up))             # shoulder_axis = RShoulder - LShoulder
      right   = normalize(cross(up, forward))
      (Gram-Schmidt clean-up for orthogonality)
    """
    if layout not in _LAYOUTS:
        raise ValueError(f"Unknown layout '{layout}'. Known: {list(_LAYOUTS)}")
    L = _LAYOUTS[layout]

    if joints_3d.ndim != 2 or joints_3d.shape[1] != 3:
        raise ValueError("joints_3d must be (N x 3)")

    # Pelvis midpoint
    lhip = joints_3d[L.l_hip]
    rhip = joints_3d[L.r_hip]
    pelvis = 0.5 * (lhip + rhip)

    # Spine/Thorax or shoulder midpoint
    if L.spine is not None:
        spine = joints_3d[L.spine]
    else:
        lsh = joints_3d[L.l_shoulder]
        rsh = joints_3d[L.r_shoulder]
        spine = 0.5 * (lsh + rsh)

    # Unit up direction: pelvis -> spine
    up = _normalize(spine - pelvis, eps)

    # Shoulder axis = anatomical right (R - L)
    shoulder_axis = _normalize(joints_3d[L.r_shoulder] - joints_3d[L.l_shoulder], eps)

    # Pin 'right' to shoulder axis
    right = _normalize(shoulder_axis, eps)

    # Define forward as up × right (front of chest)
    forward = _normalize(np.cross(up, right), eps)

    # Re-orthogonalize up so it's perpendicular to right
    up = _normalize(np.cross(right, forward), eps) #(Gram-Schmidt)

    # Rotation matrix with columns [right, up, forward]
    R_np = np.stack([right, up, forward], axis=1).astype(np.float32)  # (3,3) #right = X, up = Y, forward = Z.

    # Confidence: combine joint confidences (if provided) and geometric stability
    conf_j = 1.0
    if joints_conf is not None:
        needed = [L.l_hip, L.r_hip, L.l_shoulder, L.r_shoulder]
        if L.spine is not None:
            needed.append(L.spine)
        conf_j = float(np.clip(np.mean(joints_conf[needed]), 0.0, 1.0)) #averaging the conf to (0,1)

    ortho = (np.dot(right, up) ** 2 + np.dot(up, forward) ** 2 + np.dot(forward, right) ** 2)
    lengths = float(np.clip((np.linalg.norm(forward) + np.linalg.norm(up) + np.linalg.norm(right)) / 3.0, 0.0, 1.0))
    conf_g = float(np.clip(1.0 - ortho, 0.0, 1.0)) * lengths
    confidence = float(np.clip(0.5 * conf_j + 0.5 * conf_g, 0.0, 1.0))

    # Use Torch helper for quat conversion; keep NumPy I/O
    R = torch.from_numpy(R_np)  # (3,3)
    q = quat_from_rotmat(R).detach().cpu().numpy().astype(np.float32)  # (4,) wxyz

    return q, forward.astype(np.float32), right.astype(np.float32), up.astype(np.float32), confidence

def smooth_quat(prev_quat_wxyz: Optional[np.ndarray], cur_quat_wxyz: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    Temporal smoothing via SLERP-EMA: result = slerp(prev, cur, alpha).
    If prev is None, returns cur normalized.
    """
    if prev_quat_wxyz is None:
        q = np.asarray(cur_quat_wxyz, dtype=np.float32)
        n = np.linalg.norm(q) + 1e-8
        return (q / n).astype(np.float32)

    q0 = torch.from_numpy(np.asarray(prev_quat_wxyz, dtype=np.float32))
    q1 = torch.from_numpy(np.asarray(cur_quat_wxyz, dtype=np.float32))
    q = quat_slerp(q0, q1, float(np.clip(alpha, 0.0, 1.0)))
    return q.detach().cpu().numpy().astype(np.float32)
