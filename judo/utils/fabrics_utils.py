import enum

import numpy as np
import torch
from enum import Enum

# warp
import warp as wp

# mujoco
import mujoco as mj

# judo
from judo.utils.math_utils import np_euler_to_quat, np_mul_pose

# mjmanip
from mjmanip import DEFAULT_SCENE_XML_PATH, MJMANIP_DEVICE
from mjmanip.robot.world_base import WorldBase
from mjmanip.robot.leap_mjx import LeapMjx
from mjmanip.robot.leap_fabrics import LeapWithFabrics, LeapWithFabricsEnv, HAND_XML_PATH
from mjmanip.utils import mj_get_joints_qids, mj_get_site_pose
from mjmanip.control.fabrics.fabrics.arm_hand_pose_fabric import ArmHandPoseFabricConfig
from mjmanip.control.fabrics.fabrics_controller import FabricsController
from mjmanip.robot.leap_fabrics import LEAP_FABRIC_PALM_CONTROL_FRAME_NAMES, \
    LEAP_FABRIC_FINGER_CONTROL_FRAME_NAMES


@wp.func
def wp_project_to_plane(p: wp.vec3, c: wp.vec3, n: wp.vec3) -> wp.vec3:
    """Project point p onto the plane with point c and normal n."""
    return p - wp.dot(p - c, n) * n


@wp.kernel
def wp_kernel_get_projection_onto_object_surface(points: wp.array(dtype=float, ndim=2),
                                                 obj_mesh: wp.Mesh,
                                                 obj_mesh_vert_normals: wp.array(dtype=float, ndim=1),
                                                 obj_q: wp.array(dtype=float, ndim=1),
                                                 out_projection_points: wp.array(dtype=float, ndim=2)):
    tid = wp.tid()
    # --------------------------------------------------------------------- #
    #  Mesh query                                                           #
    # --------------------------------------------------------------------- #
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    max_dist = 1e8

    obj_mesh_id = obj_mesh.id
    point = points[tid]
    point_vec = wp.vec3(point[0], point[1], point[2])
    wp.mesh_query_point(
        obj_mesh_id, point_vec, max_dist, sign, face_index, face_u, face_v
    )

    face_w = 1.0 - face_u - face_v

    i0 = wp.mesh_get_index(obj_mesh_id, face_index * 3 + 0)
    i1 = wp.mesh_get_index(obj_mesh_id, face_index * 3 + 1)
    i2 = wp.mesh_get_index(obj_mesh_id, face_index * 3 + 2)

    p0 = wp.mesh_get_point(obj_mesh_id, face_index * 3 + 0)
    p1 = wp.mesh_get_point(obj_mesh_id, face_index * 3 + 1)
    p2 = wp.mesh_get_point(obj_mesh_id, face_index * 3 + 2)

    n0 = obj_mesh_vert_normals[i0]
    n1 = obj_mesh_vert_normals[i1]
    n2 = obj_mesh_vert_normals[i2]

    # Barycentric interpolation
    p = face_u * p0 + face_v * p1 + face_w * p2

    c0 = wp_project_to_plane(p, p0, n0)
    c1 = wp_project_to_plane(p, p1, n1)
    c2 = wp_project_to_plane(p, p2, n2)
    q = face_u * c0 + face_v * c1 + face_w * c2

    alpha = 0.75
    r = (1.0 - alpha) * p + alpha * q
    out_projection_points[tid] = r + obj_q


class FabricsMPCType(Enum):
    PCA_HAND_GRASP = enum.auto()
    FINGER_EE_SINGLE_TASK_SPACE = enum.auto()
    FINGER_EE_MULTI_TASK_SPACES = enum.auto()
    FINGER_EE_MULTI_CIRCULAR_TASK_SPACES = enum.auto()


FABRICS_MPC_TYPE = None  # FabricsMPCType.FINGER_EE_MULTI_TASK_SPACES


class FabricsAgent:
    USE_PCA_HAND_GRASP: bool = FABRICS_MPC_TYPE is FabricsMPCType.PCA_HAND_GRASP
    USE_FINGER_EE_MULTI_TASK_SPACES: bool = FABRICS_MPC_TYPE is FabricsMPCType.FINGER_EE_MULTI_TASK_SPACES
    USE_FINGER_EE_SINGLE_TASK_SPACE: bool = FABRICS_MPC_TYPE is FabricsMPCType.FINGER_EE_SINGLE_TASK_SPACE
    USE_FINGER_EE_MULTI_CIRCULAR_TASK_SPACES: bool = FABRICS_MPC_TYPE is FabricsMPCType.FINGER_EE_MULTI_CIRCULAR_TASK_SPACES
    HAND_DOFS_NO: int = LeapWithFabrics.HAND_DOFS_NO
    FINGER_EES_DOFS_NO: int = 6 * (len(LeapWithFabrics.FINGER_TIPS_NAMES) if USE_FINGER_EE_MULTI_TASK_SPACES else 1)
    FINGER_EES_TARGET_SITE: str = "mug_handle_loop_center"

    PALM_FABRIC_CONTROL_FRAMES = LEAP_FABRIC_PALM_CONTROL_FRAME_NAMES
    FINGER_FABRIC_CONTROL_FRAMES = LEAP_FABRIC_FINGER_CONTROL_FRAME_NAMES

    def __init__(self, sim_model: mj.MjModel, sim_data: mj.MjData,
                 fabric_cfg: ArmHandPoseFabricConfig,
                 num_rollout_worlds: int = 1,
                 num_fabrics_steps: int = 1):
        self.mj_model = sim_model
        self.mj_data = sim_data
        self.num_rollout_worlds = num_rollout_worlds
        self.num_fabrics_steps = 1 if self.USE_PCA_HAND_GRASP else num_fabrics_steps
        self.fabric_cfg = fabric_cfg
        self.init_fabrics(fabric_cfg)
        self.sampled_target_traces: list[np.ndarray] = []
        self.is_for_rollout = (num_rollout_worlds > 1)

    def init_fabrics(self, fabric_cfg: ArmHandPoseFabricConfig) -> None:
        LeapWithFabricsEnv.FINGER_FABRIC_CONTROL_FRAMES = ["if_ds_fabric2", "mf_ds_fabric2",
                                                           "rf_ds_fabric2", "th_ds_fabric2"]
        self.fabrics_env = LeapWithFabricsEnv(arm_hand_class=LeapWithFabrics,
                                              world_scene_xml=DEFAULT_SCENE_XML_PATH,
                                              arm_xml=HAND_XML_PATH,
                                              hand_xml=None,
                                              fabric_cfg=fabric_cfg,
                                              use_finger_fabrics=self.USE_FINGER_EE_MULTI_TASK_SPACES or
                                                                 self.USE_FINGER_EE_SINGLE_TASK_SPACE,
                                              use_cuda_graph=True,
                                              num_fabrics_steps=self.num_fabrics_steps,
                                              batch_size=self.num_rollout_worlds)
        self.fabrics_env.init()
        self.fabrics_world = self.fabrics_env.world

        # NOTE: MjData is created here-in if needed in robot's configuration
        self.fabrics_controller = self.fabrics_env.fabrics_controller
        if self.USE_PCA_HAND_GRASP:
            self.hand_pca_values = torch.zeros_like(self.fabrics_controller.hand_pca_targets)
            self.HAND_PCA_DIM = self.hand_pca_values.shape[-1]
            self.HAND_PCA_MATRIX = self.fabrics_controller.pose_fabric.pca_matrix.cpu().numpy()
        elif self.USE_FINGER_EE_MULTI_TASK_SPACES:
            self.finger_target_poses = {finger_ee_name: torch.clone(finger_target) for finger_ee_name, finger_target in
                                        self.fabrics_controller.finger_targets.items()}
        elif self.USE_FINGER_EE_SINGLE_TASK_SPACE:
            self.common_finger_ee_target = mj_get_site_pose(self.mj_data, self.FINGER_EES_TARGET_SITE, True)
            self.finger_target_poses = {finger_ee_name: torch.zeros((self.num_rollout_worlds, 7),
                                                                    device=MJMANIP_DEVICE) for
                                        finger_ee_name in list(self.fabrics_controller.finger_targets.keys())}

    def hand_pca_to_q(self, rollout_controls: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        num_rollouts, num_steps = rollout_controls.shape[:2]
        out_rollout_controls = np.zeros((num_rollouts, num_steps, self.mj_model.nu))

        # Wrist ctrls
        wrist_ctrls = rollout_controls[..., :-self.HAND_PCA_DIM]
        wrist_dofs_no = wrist_ctrls.shape[-1]
        out_rollout_controls[..., :wrist_dofs_no] = wrist_ctrls

        # PCA->Q ctrls
        for step in range(num_steps):
            rl_ctrl = rollout_controls[..., step, :]
            pca = rl_ctrl[..., -self.HAND_PCA_DIM:]

            # Save result q back to [rollout_controls]
            out_rollout_controls[..., step, wrist_dofs_no:] = pca @ np.linalg.pinv(self.HAND_PCA_MATRIX.T)
        return out_rollout_controls

    def fabrics_plan(self, rollout_controls: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        # Prep
        self.sampled_target_traces.clear()
        if self.USE_FINGER_EE_MULTI_TASK_SPACES:
            for finger_ee_name, finger_target_pose in self.finger_target_poses.items():
                finger_target_pose.copy_(torch.as_tensor(
                    mj_get_site_pose(self.mj_data, f"{finger_ee_name[:2]}_tip", True), device=MJMANIP_DEVICE))
        elif self.USE_FINGER_EE_SINGLE_TASK_SPACE:
            self.common_finger_ee_target = mj_get_site_pose(self.mj_data, self.FINGER_EES_TARGET_SITE, True)

        # Update [fabrics_controller]'s q, qdd with [current_state]
        cur_robot_q = current_state[self.fabrics_world.robot_qpos_ids]
        cur_robot_qd = current_state[self.fabrics_world.robot_dof_ids]
        # NOTE: objs here must be free-joint objs to have their qpos as poses
        cur_obj_poses = {obj_name: current_state[self.fabrics_world.obj_qpos_ids[obj_name]]
                         for obj_name in self.fabrics_world.OBJECT_NAMES}

        num_rollouts, num_steps = rollout_controls.shape[:2]
        out_rollout_controls = np.zeros((num_rollouts, num_steps, self.mj_model.nu))
        if self.USE_PCA_HAND_GRASP:
            # Rollouts from [current_state]
            wrist_ctrls = rollout_controls[..., :-self.HAND_PCA_DIM]
            wrist_dofs_no = wrist_ctrls.shape[-1]
            out_rollout_controls[..., :wrist_dofs_no] = wrist_ctrls

            for step in range(num_steps):
                rl_ctrl = rollout_controls[..., step, :]
                self.hand_pca_values.copy_(torch.from_numpy(rl_ctrl[..., -self.HAND_PCA_DIM:]).float())

                # Fabrics rollout
                # TODO: VERIFY IF cur_q, cur_qd ARE NEEDED HERE
                # -> LIKELY YES TO MAKE FABRICS SOLVE FROM CURRENT Q/QD, BUT IT CAN ALSO SLOW IT DOWN
                self.fabrics_controller.step(new_hand_pca_targets=self.hand_pca_values,
                                             cur_robot_q=torch.as_tensor(cur_robot_q, device=MJMANIP_DEVICE),
                                             cur_robot_qd=torch.as_tensor(cur_robot_qd, device=MJMANIP_DEVICE),
                                             cur_obj_poses=cur_obj_poses)

                # Save result q back to [rollout_controls]
                out_rollout_controls[..., step, wrist_dofs_no:] = (
                    self.fabrics_controller.q.clone().detach().cpu().numpy())
            return out_rollout_controls
        elif self.USE_FINGER_EE_SINGLE_TASK_SPACE:
            # Rollouts from [current_state]
            wrist_ctrls = rollout_controls[..., :-self.FINGER_EES_DOFS_NO]
            wrist_dofs_no = wrist_ctrls.shape[-1]
            out_rollout_controls[..., :wrist_dofs_no] = wrist_ctrls

            new_ee_target_pose = np.zeros((num_rollouts, 7))
            common_finger_ee_target = np.tile(self.common_finger_ee_target, (num_rollouts, 1))
            for step in range(num_steps):
                rl_ctrl = rollout_controls[..., step, :]
                # EE pose delta (6DOF in 3D)
                ee_pose_ctrl = rl_ctrl[..., -6:]
                ee_pose_delta = np.concatenate([ee_pose_ctrl[..., :3], np_euler_to_quat(ee_pose_ctrl[..., 3:])],
                                               axis=-1)
                new_ee_target_pose[..., :3], new_ee_target_pose[..., 3:] = np_mul_pose(
                    common_finger_ee_target[..., :3], common_finger_ee_target[..., 3:],
                    ee_pose_delta[..., :3], ee_pose_delta[..., 3:])
                for finger_ee_name, finger_target_pose in self.finger_target_poses.items():
                    finger_target_pose.copy_(torch.from_numpy(new_ee_target_pose))

                # Traces
                if not self.is_for_rollout:
                    self.sampled_target_traces.extend(new_ee_target_pose[..., :3].tolist())

                # Fabrics rollout
                self.fabrics_controller.step(new_finger_targets=self.finger_target_poses,
                                             cur_robot_q=torch.as_tensor(cur_robot_q, device=MJMANIP_DEVICE),
                                             cur_robot_qd=torch.as_tensor(cur_robot_qd, device=MJMANIP_DEVICE),
                                             cur_obj_poses=cur_obj_poses)

                # Save result q back to [rollout_controls]
                out_rollout_controls[..., step, wrist_dofs_no:] = (
                    self.fabrics_controller.q.clone().detach().cpu().numpy())
            return out_rollout_controls
        elif self.USE_FINGER_EE_MULTI_TASK_SPACES:
            # Rollouts from [current_state]
            wrist_ctrls = rollout_controls[..., :-self.FINGER_EES_DOFS_NO]
            wrist_dofs_no = wrist_ctrls.shape[-1]
            out_rollout_controls[..., :wrist_dofs_no] = wrist_ctrls

            for step in range(num_steps):
                rl_ctrl = rollout_controls[..., step, :]
                finger_ee_name_keys = list(self.finger_target_poses.keys())
                for finger_ee_name, finger_target_pose in self.finger_target_poses.items():
                    i = finger_ee_name_keys.index(finger_ee_name)
                    finger_target_pos = finger_target_pose[..., :3].detach().cpu().numpy()
                    finger_target_quat = finger_target_pose[..., 3:].detach().cpu().numpy()

                    # EE pose delta (6DOF in 3D)
                    ee_pose_ctrl = rl_ctrl[..., -(i + 1) * 6:-i * 6 if i > 0 else None]
                    ee_pose_delta = np.concatenate([ee_pose_ctrl[..., :3], np_euler_to_quat(ee_pose_ctrl[..., 3:])],
                                                   axis=-1)

                    new_ee_target_pose = np.zeros_like(ee_pose_delta)
                    new_ee_target_pose[..., :3], new_ee_target_pose[..., 3:] = np_mul_pose(
                        finger_target_pos, finger_target_quat,
                        ee_pose_delta[..., :3], ee_pose_delta[..., 3:])
                    finger_target_pose.copy_(torch.from_numpy(new_ee_target_pose).float())

                    # Traces
                    if not self.is_for_rollout:
                        self.sampled_target_traces.extend(new_ee_target_pose[..., :3].tolist())

                # Fabrics rollout
                self.fabrics_controller.step(new_finger_targets=self.finger_target_poses,
                                             cur_robot_q=torch.as_tensor(cur_robot_q, device=MJMANIP_DEVICE),
                                             cur_robot_qd=torch.as_tensor(cur_robot_qd, device=MJMANIP_DEVICE),
                                             cur_obj_poses=cur_obj_poses)

                # Save result q back to [rollout_controls]
                out_rollout_controls[..., step, wrist_dofs_no:] = (
                    self.fabrics_controller.q.clone().detach().cpu().numpy())
            return out_rollout_controls

        elif self.USE_FINGER_EE_MULTI_CIRCULAR_TASK_SPACES:
            from mjmanip.utils import mj_get_geom_mesh_data
            # TODO: TO MOVE THIS TO MPCAPP THEN ONLY PASS OBJ MESH NORMALS TO FabricsAgent
            mug_geom_id = self.mj_model.geom("mug").id
            mug_vertices, mug_faces, mug_vertex_normals = mj_get_geom_mesh_data(self.mj_model, mug_geom_id)

            obj_mesh_verts = wp.array(
                data=mug_vertices,
                dtype=wp.vec3,
                device="cuda",
                requires_grad=True,
            )
            obj_mesh_inds = wp.array(
                data=mug_faces,
                dtype=int,
                device="cuda",
                requires_grad=True,
            )
            obj_mesh_vert_normals = wp.array(
                data=mug_vertex_normals,
                dtype=wp.vec3,
                device="cuda",
                requires_grad=True,
            )
            obj_mesh = wp.Mesh(points=obj_mesh_verts, indices=obj_mesh_inds)
            obj_mesh.refit()

            new_ee_projected_points = wp.zeros((num_rollouts, 3), dtype=float, device="cuda")
            wp.launch(wp_kernel_get_projection_onto_object_surface(wp.array(new_ee_target_pose[..., :3], dtype=float),
                                                                   obj_mesh, obj_mesh_vert_normals,
                                                                   wp.array(cur_obj_poses[0], dtype=float),
                                                                   new_ee_projected_points))
            return rollout_controls
        else:
            return rollout_controls
