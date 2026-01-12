import enum

import numpy as np
import torch
from enum import Enum

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


class FabricsMPCType(Enum):
    PCA_HAND_GRASP = enum.auto()
    FINGER_EE_SINGLE_TASK_SPACE = enum.auto()
    FINGER_EE_MULTI_TASK_SPACES = enum.auto()


FABRICS_MPC_TYPE = None  # FabricsMPCType.FINGER_EE_SINGLE_TASK_SPACE


class FabricsAgent:
    USE_PCA_HAND_GRASP: bool = FABRICS_MPC_TYPE is FabricsMPCType.PCA_HAND_GRASP
    USE_FINGER_EE_MULTI_TASK_SPACES: bool = FABRICS_MPC_TYPE is FabricsMPCType.FINGER_EE_MULTI_TASK_SPACES
    USE_FINGER_EE_SINGLE_TASK_SPACE: bool = FABRICS_MPC_TYPE is FabricsMPCType.FINGER_EE_SINGLE_TASK_SPACE
    HAND_DOFS_NO: int = LeapWithFabrics.HAND_DOFS_NO
    FINGER_EES_DOFS_NO: int = 6 * (len(LeapWithFabrics.FINGER_TIPS_NAMES) if USE_FINGER_EE_MULTI_TASK_SPACES else 1)
    FINGER_EES_TARGET_SITE: str = "mug_handle_loop_center"
    NUM_FABRICS_STEPS: int = 10

    PALM_FABRIC_CONTROL_FRAMES = LEAP_FABRIC_PALM_CONTROL_FRAME_NAMES
    FINGER_FABRIC_CONTROL_FRAMES = LEAP_FABRIC_FINGER_CONTROL_FRAME_NAMES

    def __init__(self, sim_model: mj.MjModel, sim_data: mj.MjData,
                 fabric_cfg: ArmHandPoseFabricConfig,
                 num_rollout_worlds: int = 1):
        self.mj_model = sim_model
        self.mj_data = sim_data
        self.num_rollout_worlds = num_rollout_worlds
        self.fabric_cfg = fabric_cfg
        self.init_fabrics(fabric_cfg)
        self.sampled_target_traces: list[np.ndarray] = []
        self.is_for_rollout = (num_rollout_worlds > 1)

    def init_fabrics(self, fabric_cfg: ArmHandPoseFabricConfig) -> None:
        LeapWithFabricsEnv.FINGER_FABRIC_CONTROL_FRAMES = ["if_ds_fabric1", "mf_ds_fabric1",
                                                           "rf_ds_fabric1", "th_ds_fabric1"]
        fabrics_env = LeapWithFabricsEnv(arm_hand_class=LeapWithFabrics,
                                         world_scene_xml=DEFAULT_SCENE_XML_PATH,
                                         arm_xml=HAND_XML_PATH,
                                         hand_xml=None,
                                         fabric_cfg=fabric_cfg,
                                         use_finger_fabrics=self.USE_FINGER_EE_MULTI_TASK_SPACES or
                                                            self.USE_FINGER_EE_SINGLE_TASK_SPACE,
                                         use_cuda_graph=True,
                                         num_fabrics_steps=self.NUM_FABRICS_STEPS,
                                         batch_size=self.num_rollout_worlds)
        fabrics_env.init()
        self.fabrics_robot = fabrics_env.robot

        # NOTE: MjData is created here-in if needed in robot's configuration
        self.fabrics_controller = fabrics_env.fabrics_controller
        if self.USE_PCA_HAND_GRASP:
            self.hand_pca_values = torch.zeros_like(self.fabrics_controller.hand_pca_targets)
            self.HAND_PCA_DIM = self.hand_pca_values.shape[-1]
        elif self.USE_FINGER_EE_MULTI_TASK_SPACES:
            self.finger_target_poses = {finger_ee_name: torch.clone(finger_target) for finger_ee_name, finger_target in
                                        self.fabrics_controller.finger_targets.items()}
        elif self.USE_FINGER_EE_SINGLE_TASK_SPACE:
            self.common_finger_ee_target = mj_get_site_pose(self.mj_data, self.FINGER_EES_TARGET_SITE, True)
            self.finger_target_poses = {finger_ee_name: torch.zeros((self.num_rollout_worlds, 7),
                                                                    device=MJMANIP_DEVICE) for
                                        finger_ee_name in list(self.fabrics_controller.finger_targets.keys())}

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
        cur_q = current_state[..., self.fabrics_robot.robot_qpos_ids]
        cur_qd = current_state[..., self.fabrics_robot.robot_dof_ids]

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
                # -> LIKELY YES TO MAKE FABRICS SOLVE FROM CURRENT Q/QD
                self.fabrics_controller.step(new_hand_pca_targets=self.hand_pca_values,
                                             cur_q=torch.as_tensor(cur_q, device=MJMANIP_DEVICE),
                                             cur_qd=torch.as_tensor(cur_qd, device=MJMANIP_DEVICE))

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
                for _ in range(self.NUM_FABRICS_STEPS):
                    self.fabrics_controller.step(new_finger_targets=self.finger_target_poses,
                                                 cur_q=torch.as_tensor(cur_q, device=MJMANIP_DEVICE),
                                                 cur_qd=torch.as_tensor(cur_qd, device=MJMANIP_DEVICE))

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
                for _ in range(self.NUM_FABRICS_STEPS):
                    self.fabrics_controller.step(new_finger_targets=self.finger_target_poses,
                                                 cur_q=torch.as_tensor(cur_q, device=MJMANIP_DEVICE),
                                                 cur_qd=torch.as_tensor(cur_qd, device=MJMANIP_DEVICE))

                # Save result q back to [rollout_controls]
                out_rollout_controls[..., step, wrist_dofs_no:] = (
                    self.fabrics_controller.q.clone().detach().cpu().numpy())
            return out_rollout_controls
        else:
            return rollout_controls
