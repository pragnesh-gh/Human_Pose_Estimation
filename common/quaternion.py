# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from typing import Tuple

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape)-1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape)-1)
    return (v + 2 * (q[..., :1] * uv + uuv))
    
    
def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape)-1)


def _safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    return torch.linalg.norm(x, dim=dim, keepdim=keepdim).clamp_min(eps)

def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize quaternion(s) (wxyz). Shape: (..., 4)
    """
    return q / _safe_norm(q, dim=-1, keepdim=True, eps=eps)

def quat_dot(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """
    Dot product between quaternion(s) along last dim. Shapes broadcast; returns (...,).
    """
    return (q0 * q1).sum(dim=-1)

def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Shortest-arc SLERP between unit quaternions (wxyz). Shapes broadcast. Returns same shape as q0/q1.
    t in [0,1].
    """
    t = float(t)
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    d = quat_dot(q0, q1)
    # Ensure shortest path
    q1 = torch.where(d[..., None] < 0.0, -q1, q1)
    d = d.abs()

    # If nearly parallel, use LERP
    lerp_mask = d > 0.9995
    # LERP branch
    q_lerp = quat_normalize(q0 + (q1 - q0) * t)

    # SLERP branch
    theta_0 = torch.arccos(d.clamp(-1.0, 1.0))
    theta = theta_0 * t
    sin_theta_0 = torch.sin(theta_0).clamp_min(1e-8)
    s0 = torch.sin(theta_0 - theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0
    q_slerp = quat_normalize(q0 * s0[..., None] + q1 * s1[..., None])

    return torch.where(lerp_mask[..., None], q_lerp, q_slerp)

def quat_from_rotmat(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert rotation matrix/matrices to quaternion(s) in wxyz.
    R shape: (..., 3, 3) -> returns (..., 4)
    """
    assert R.shape[-2:] == (3, 3), "R must be (...,3,3)"
    device, dtype = R.device, R.dtype

    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    trace = m00 + m11 + m22

    # Case 1: trace > 0
    t_pos = trace > 0.0
    s_pos = torch.sqrt(trace[t_pos] + 1.0) * 2.0
    w_pos = 0.25 * s_pos
    x_pos = (m21[t_pos] - m12[t_pos]) / s_pos
    y_pos = (m02[t_pos] - m20[t_pos]) / s_pos
    z_pos = (m10[t_pos] - m01[t_pos]) / s_pos

    # Cases where diagonal element is the largest
    idx = torch.stack([m00, m11, m22], dim=-1).argmax(dim=-1)
    i0 = (~t_pos) & (idx == 0)
    i1 = (~t_pos) & (idx == 1)
    i2 = (~t_pos) & (idx == 2)

    # i0
    s0 = torch.sqrt((1.0 + m00[i0] - m11[i0] - m22[i0]).clamp_min(eps)) * 2.0
    x0 = 0.25 * s0
    y0 = (m01[i0] + m10[i0]) / s0
    z0 = (m02[i0] + m20[i0]) / s0
    w0 = (m21[i0] - m12[i0]) / s0

    # i1
    s1 = torch.sqrt((1.0 - m00[i1] + m11[i1] - m22[i1]).clamp_min(eps)) * 2.0
    x1 = (m01[i1] + m10[i1]) / s1
    y1 = 0.25 * s1
    z1 = (m12[i1] + m21[i1]) / s1
    w1 = (m02[i1] - m20[i1]) / s1

    # i2
    s2 = torch.sqrt((1.0 - m00[i2] - m11[i2] + m22[i2]).clamp_min(eps)) * 2.0
    x2 = (m02[i2] + m20[i2]) / s2
    y2 = (m12[i2] + m21[i2]) / s2
    z2 = 0.25 * s2
    w2 = (m10[i2] - m01[i2]) / s2

    # Stitch results
    w = torch.empty_like(trace, dtype=dtype, device=device)
    x = torch.empty_like(trace, dtype=dtype, device=device)
    y = torch.empty_like(trace, dtype=dtype, device=device)
    z = torch.empty_like(trace, dtype=dtype, device=device)

    w[t_pos], x[t_pos], y[t_pos], z[t_pos] = w_pos, x_pos, y_pos, z_pos
    w[i0],    x[i0],    y[i0],    z[i0]    = w0,    x0,    y0,    z0
    w[i1],    x[i1],    y[i1],    z[i1]    = w1,    x1,    y1,    z1
    w[i2],    x[i2],    y[i2],    z[i2]    = w2,    x2,    y2,    z2

    q = torch.stack([w, x, y, z], dim=-1)
    return quat_normalize(q)

def rotmat_from_quat(q: torch.Tensor) -> torch.Tensor:
    """
    Quaternion(s) (wxyz) -> rotation matrix/matrices.
    q shape: (..., 4) -> returns (..., 3, 3)
    """
    q = quat_normalize(q)
    w, x, y, z = q.unbind(dim=-1)

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)

    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)

    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz

    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)
    return R

def basis_from_quat(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (forward, right, up) basis vectors in WORLD coords for quaternion(s) (wxyz).
    Shapes:
      q: (..., 4)
      returns forward,right,up each (..., 3)
    Note: Columns of R are [right, up, forward].
    """
    R = rotmat_from_quat(q)            # (..., 3, 3)
    right   = R[..., :, 0]             # (..., 3)
    up      = R[..., :, 1]             # (..., 3)
    forward = R[..., :, 2]             # (..., 3)
    return forward, right, up