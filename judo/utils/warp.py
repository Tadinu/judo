from typing import Any
import numpy as np
import warp as wp


def wp_pose_to_mj(wp_pose: np.ndarray) -> np.ndarray:
    # NOTE: warp quaternion quat is XYZW
    return np.concatenate([wp_pose[:3], np.array([wp_pose[6], wp_pose[3],
                                                  wp_pose[4], wp_pose[5]])])


def wp_create_kernel_tile_array(array_size: int):
    ARRAY_SIZE_C = wp.constant(array_size)

    @wp.kernel
    def wp_kernel_tile_array(a: wp.array(dtype=Any),
                             # output
                             b: wp.array(dtype=Any)):
        i = wp.tid()
        b_offset = i * array_size
        a_tiled = wp.tile_load(a, shape=(ARRAY_SIZE_C,), offset=(0,), storage="shared")
        wp.tile_store(b, a_tiled, offset=(b_offset,))

    return wp_kernel_tile_array


@wp.kernel
def wp_kernel_default_set_joint_targets(
        control: wp.array(dtype=wp.float32, ndim=2),
        joint_id: wp.array(dtype=wp.int32, ndim=2),
        joint_qd_start: wp.array(dtype=wp.int32),
        joint_limit_lower: wp.array(dtype=wp.float32),
        joint_limit_upper: wp.array(dtype=wp.float32),
        world_time: wp.array(dtype=wp.float32),
        sim_dt: float,
        # outputs
        joint_target_pos: wp.array(dtype=wp.float32),
        joint_parent_xform: wp.array(dtype=wp.transform),
):
    pass
