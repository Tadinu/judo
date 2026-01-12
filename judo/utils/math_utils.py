# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Optional, Tuple, Union
import numpy as np


# Ref: brax.math
def np_euler_to_quat(v: np.ndarray) -> np.ndarray:
    """Converts euler rotations in degrees to quaternion."""
    # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''
    assert v.shape[-1] == 3
    cos_v = np.cos(v * np.pi / 360)
    c1, c2, c3 = [cos_v[..., [i]] for i in range(3)]
    sin_v = np.sin(v * np.pi / 360)
    s1, s2, s3 = [sin_v[..., [i]] for i in range(3)]
    w = c1 * c2 * c3 - s1 * s2 * s3
    x = s1 * c2 * c3 + c1 * s2 * s3
    y = c1 * s2 * c3 - s1 * c2 * s3
    z = c1 * c2 * s3 + s1 * s2 * c3
    return np.concatenate([w, x, y, z], axis=-1)


def np_safe_normalize_axis(axis: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Safely normalizes a batch of 3D axis vectors, avoiding division by zero.

    If the norm of an axis is less than `eps`, it defaults to [1, 0, 0].

    Args:
        axis: The unnormalized axis vectors. Shape = (..., 3).
        eps: Small threshold to avoid division by zero.

    Returns:
        Normalized axis vectors. Shape = (..., 3).
    """
    norm = np.linalg.norm(axis, axis=-1)
    small_angle_mask = norm < eps
    safe_norm = np.where(small_angle_mask, 1.0, norm)
    normalized = axis / safe_norm[..., None]
    return np.where(small_angle_mask[..., None], np.array([1.0, 0.0, 0.0]), normalized)


# Ref: mjx._src.math
def np_norm(x: np.ndarray, axis: Optional[Union[Tuple[int, ...], int]] = None) -> np.ndarray:
    """Calculates a linalg.norm(x) that's safe for gradients at x=0.

    Args:
      x: A np.ndarray
      axis: The axis along which to compute the norm

    Returns:
      Norm of the array x.
    """

    is_zero = np.allclose(x, 0.0)
    # temporarily swap x with ones if is_zero, then swap back
    x = np.where(is_zero, np.ones_like(x), x)
    n = np.linalg.norm(x, axis=axis)
    n = np.where(is_zero, 0.0, n)
    return n


# Ref: mjx._src.math
def np_normalize_with_norm(x: np.ndarray, axis: Optional[Union[Tuple[int, ...], int]] = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Normalizes an array.

    Args:
      x: A np.ndarray
      axis: The axis along which to compute the norm

    Returns:
      A tuple of (normalized array x, the norm).
    """
    n = np_norm(x, axis=axis)
    x = x / (n + 1e-6 * (n == 0.0))
    return x, n


# Ref: mjx._src.math
def np_normalize(x: np.ndarray, axis: Optional[Union[Tuple[int, ...], int]] = None) -> np.ndarray:
    """Normalizes an array.

    Args:
      x: A np.ndarray
      axis: The axis along which to compute the norm

    Returns:
      normalized array x
    """
    return np_normalize_with_norm(x, axis=axis)[0]


def np_quat_inv(u: np.ndarray) -> np.ndarray:
    """Inverts a quaternion in a way that can broadcast cleanly.

    Args:
        u: quat in wxyz order. Shape=(*u_dims, 4).

    Returns:
        result: quat in wxyz order. Shape=(*u_dims, 4).
    """
    return u * np.array([1, -1, -1, -1])


def np_quat_mul(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Multiplies two quaternions in a way that can broadcast cleanly.

    The leading dimensions of u and v do not have to match - only the trailing dims.

    Args:
        u: quat in wxyz order. Shape=(*u_dims, 4).
        v: quat in wxyz order. Shape=(*v_dims, 4).

    Returns:
        result: quat in wxyz order. Shape=(*(u_dims or v_dims), 4). The longer leading dims of u or v are used.
    """
    w = u[..., 0] * v[..., 0] - u[..., 1] * v[..., 1] - u[..., 2] * v[..., 2] - u[..., 3] * v[..., 3]
    x = u[..., 0] * v[..., 1] + u[..., 1] * v[..., 0] + u[..., 2] * v[..., 3] - u[..., 3] * v[..., 2]
    y = u[..., 0] * v[..., 2] - u[..., 1] * v[..., 3] + u[..., 2] * v[..., 0] + u[..., 3] * v[..., 1]
    z = u[..., 0] * v[..., 3] + u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1] + u[..., 3] * v[..., 0]
    return np.stack([w, x, y, z], axis=-1)


def np_quat_diff(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""Computes the 'quaternion difference' between two quaternions: u^* \otimes v.

    Args:
        u: quat in wxyz order. Shape=(*u_dims, 4).
        v: quat in wxyz order. Shape=(*v_dims, 4).

    Returns:
        result: quat in wxyz order. Shape=(*(u_dims or v_dims), 4). The longer leading dims of u or v are used.
    """
    return np_quat_mul(np_quat_inv(u), v)


def np_axis_angle_diff(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Computes the 'axis-angle difference' between two quaternions: 2 * vec(u^* \otimes v).

    Args:
        u: quat in wxyz order. Shape=(*u_dims, 4).
        v: quat in wxyz order. Shape=(*v_dims, 4).

    Returns:
        angle: The angle of rotation in radians. Shape=(*(u_dims or v_dims),).
        axis: The axis of rotation. Shape=(*(u_dims or v_dims), 3).
    """
    diff = np_quat_diff(u, v)
    axis = diff[..., 1:]
    sin_a_2 = np.linalg.norm(axis, axis=-1)

    # handle division by zero
    axis = np_safe_normalize_axis(axis, eps=1e-6)
    angle = 2.0 * np.arctan2(sin_a_2, diff[..., 0])

    # use correct angle comparison logic
    mask = angle > np.pi
    angle = np.where(mask, 2 * np.pi - angle, angle)
    axis = np.where(mask[..., None], -axis, axis)
    return angle, axis


def np_quat_diff_so3(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Computes the 'quaternion difference' between two quaternions and then takes the Log map."""
    diff = np_quat_diff(u, v)
    axis = diff[..., 1:]
    sin_a_2 = np.linalg.norm(axis, axis=-1)
    axis = np_safe_normalize_axis(axis, eps=1e-6)
    speed = 2.0 * np.arctan2(sin_a_2, diff[..., 0])
    speed = np.where(speed > np.pi, speed - 2 * np.pi, speed)
    output = axis * speed[..., None]
    return output


def np_quat_vel(u: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    r"""Estimates the angular velocity between two quaternions.

    Uses the formula
        omega = (2.0 * u^{-1} * [(v - u) / dt])[1:],
    where [1:] denotes taking the vector part only.

    This is derived from the general differential formula
        omega = 2 * vec(u^* \otimes \dot{q}).

    Source: mariogc.com/post/angular-velocity-quaternions
    """
    return 2.0 * np_quat_mul(np_quat_inv(u), (v - u) / dt)[..., 1:]


# Ref: mjx._src.math
def np_rotate_vec_quat(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Rotates a vector vec by a unit quaternion quat.

    Args:
      vec: (...,3) a batched vector
      quat: (...,4) a batched quaternion

    Returns:
      ndarray(..., 3) containing a batch of vec rotated by quat.
    """
    s, u = quat[..., [0]], quat[..., 1:]
    u_dot_vec = np.sum(u * vec, axis=1, keepdims=True)  # (N, 1)
    u_dot_u = np.sum(u * u, axis=1, keepdims=True)  # (N, 1)

    r = (2 * u_dot_vec * u) + (s * s - u_dot_u) * vec
    r = r + 2 * s * np.cross(u, vec)
    return r


def np_mul_pose(pos1: np.ndarray, quat1: np.ndarray, pos2: np.ndarray, quat2: np.ndarray) -> tuple[
    np.ndarray, np.ndarray]:
    # quat_res = quat1*quat2
    quat_res = np_quat_mul(quat1, quat2)
    quat_res = np_normalize(quat_res)

    # pos_res = quat1*pos2 + pos1
    pos_res = np_rotate_vec_quat(pos2, quat1)
    pos_res += pos1
    return pos_res, quat_res
