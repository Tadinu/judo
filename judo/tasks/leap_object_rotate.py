from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
import torch

# mujoco
import mujoco as mj

# newton
import newton
import warp as wp

# judo
from judo import BackendType, MODEL_PATH
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig
from judo.utils.fabrics_utils import FabricsAgent
from judo.utils.math_utils import np_quat_diff_so3, torch_quat_diff_so3
from judo.utils.mujoco import mj_get_qpos_ids, mj_get_mocap_id
from judo.utils.warp import wp_pose_to_mj

if TYPE_CHECKING:
    from judo.simulation.base import Simulation

# mjmanip
from mjmanip import DEFAULT_SCENE_XML_PATH, MJMANIP_DEVICE
from mjmanip.robot.leap_mjx import LeapMjx
from mjmanip.robot.leap_fabrics import (LeapWithFabrics, LeapWithFabricsEnv, HAND_XML_PATH,
                                        LEAP_FABRIC_HAND_BASE_CONTROL_FRAME_NAMES, \
                                        LEAP_FABRIC_FINGER_CONTROL_FRAME_NAMES)
from mjmanip.mj_utils import mj_model_joints_qids
from mjmanip.control.fabrics.fabrics.arm_hand_pose_fabric import ArmHandPoseFabricConfig
from mjmanip.control.fabrics.fabrics_controller import FabricsController
from mjmanip.mj_utils import IDENTITY_WXYZ

OBJ_NAME = "cube"  # LeapWithFabrics.OBJECT_NAMES[0]


@dataclass
class LeapObjectRotateConfig(LeapCubeConfig):
    """Reward configuration LEAP object rotation task."""

    sim_backend: str = BackendType.MUJOCO.name if MJMANIP_DEVICE == "cpu" else BackendType.MUJOCO_WARP.name
    task_name: str = "leap_object_rotate"
    # xml_path: str = str(MODEL_PATH / f"xml/leap_{OBJ_NAME}.xml")
    # xml_path: str = str(MODEL_PATH / f"xml/leap_rh_fabric_{OBJ_NAME}_sim.xml")
    sim_xml_path: Optional[str] = str(MODEL_PATH / f"xml/leap_{OBJ_NAME}_sim.xml")
    qpos_home: np.ndarray = field(default_factory=lambda: np.array(
        [
            0.0, 0.03, 0.1, 1.0, 0.0, 0.0, 0.0,  # cube
            0.5, -0.75, 0.75, 0.25,  # index
            0.5, 0.0, 0.75, 0.25,  # middle
            0.5, 0.75, 0.75, 0.25,  # ring
            0.65, 0.9, 0.75, 0.6,  # thumb
        ]
    ))
    reset_command: np.ndarray = field(default_factory=lambda: np.array(
        [
            0.5, -0.75, 0.75, 0.25,  # index
            0.5, 0.0, 0.75, 0.25,  # middle
            0.5, 0.75, 0.75, 0.25,  # ring
            0.65, 0.9, 0.75, 0.6,  # thumb
        ]
    ))  # fmt: skip
    goal_pos: np.ndarray = field(default_factory=lambda: np.array([-0.04, -0.035, -0.065]))
    goal_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    w_pos: float = 0.1
    w_rot: float = 50


class LeapObjectRotate(LeapCube):
    """Defines the free-joint LEAP object picking task."""

    config_t: type[LeapObjectRotateConfig] = LeapObjectRotateConfig

    def __init__(self, sim: Optional[Simulation] = None, num_rollout_worlds: int = 1) -> None:
        super().__init__(sim, num_rollout_worlds=num_rollout_worlds)
        self.obj_id: int = -1
        self.obj_qpos_ids: list[int]
        self.target_mocap_id: int = -1
        self.goal_pos_torch = torch.as_tensor(self.config.goal_pos, device=MJMANIP_DEVICE)
        self.goal_quat_torch = torch.as_tensor(self.config.goal_quat, device=MJMANIP_DEVICE)

    def init_ids(self):
        super().init_ids()
        if self.mj_model:
            self.obj_id = self.mj_model.body(OBJ_NAME).id
            self.obj = self.mj_data.body(self.obj_id)
            self.obj_qpos_ids = mj_get_qpos_ids(self.mj_model, [f"{OBJ_NAME}_freejoint"])
            self.target_mocap_id = mj_get_mocap_id(self.mj_model, "target")

            # distance sensors
            # NOTE: For rollout result analysis, these are only valid IF MuJoCo-C Rollout backend supports mocap_pos/quat
            # for the `initial_state`
            self.obj_quat_distance_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_orientation_from_target")
            self.obj_pos_distance_to_grasp_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_distance_to_grasp")
            self.obj_contact_with_palm_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_contact_with_palm")
        else:
            self.obj_id = -1
            self.obj = None

    @LeapCube.nu.getter
    def nu(self) -> int:
        """Number of control inputs. The same as the mj.MjModel for this task."""
        if self.mj_model and self.fabrics_agent:
            if FabricsAgent.USE_PCA_HAND_GRASP:
                return self.mj_model.nu - self.fabrics_agent.HAND_DOFS_NO + self.fabrics_agent.HAND_PCA_DIM
            elif FabricsAgent.USE_FINGER_EE_MULTI_TASK_SPACES or FabricsAgent.USE_FINGER_EE_SINGLE_TASK_SPACE:
                return self.mj_model.nu - self.fabrics_agent.HAND_DOFS_NO + self.fabrics_agent.FINGER_EES_DOFS_NO
        return super().nu

    @LeapCube.actuator_ctrlrange.getter
    def actuator_ctrlrange(self) -> np.ndarray:
        """Mujoco actuator limits for this task."""
        if self.mj_model and self.fabrics_agent:
            limits = self.mj_model.actuator_ctrlrange
            wrist_limits = limits[:-self.fabrics_agent.HAND_DOFS_NO]
            if FabricsAgent.USE_PCA_HAND_GRASP:
                hand_pca_mins = self.fabrics_agent.fabrics_controller.hand_pca_mins.clone().cpu().numpy()
                hand_pca_maxs = self.fabrics_agent.fabrics_controller.hand_pca_maxs.clone().cpu().numpy()
                hand_pca_ranges = np.vstack([hand_pca_mins, hand_pca_maxs]).transpose()
                limits = np.vstack([wrist_limits, hand_pca_ranges])
            elif FabricsAgent.USE_FINGER_EE_MULTI_TASK_SPACES:
                limits = np.vstack(
                    [wrist_limits,
                     np.tile(np.concatenate([np.array([[-0.01, 0.01]] * 3, dtype=limits.dtype),  # Linear dofs
                                             np.array([[-1.57, 1.57]] * 3, dtype=limits.dtype)]),  # Angular dofs
                             (int(FabricsAgent.FINGER_EES_DOFS_NO / 6), 1))])
            elif FabricsAgent.USE_FINGER_EE_SINGLE_TASK_SPACE:
                limits = np.vstack(
                    [wrist_limits,
                     np.array([[-0.01, 0.01]] * 3, dtype=limits.dtype),  # Linear dofs
                     np.array([[-1.57, 1.57]] * 3, dtype=limits.dtype)])  # Angular dofs
            return limits
        else:
            return super().actuator_ctrlrange

    def reward(self,
               states: np.ndarray,
               sensors: np.ndarray,
               controls: torch.Tensor,
               system_metadata: Optional[dict[str, Any]] = None) -> np.ndarray:
        """Implements the free-joint LEAP object picking task reward.
        NOTE: The idea is to make the total cost monolithically decrease through stages!
        """
        # rewards = super().reward(states, sensors, controls, system_metadata)

        # Position costs
        reaching_err = sensors[..., self.obj_pos_distance_to_grasp_sensor_idx:
                                    self.obj_pos_distance_to_grasp_sensor_idx + 2]
        squared_distance = np.square(reaching_err).sum(-1).mean(-1)
        position_cost = 0.1 * squared_distance + 100 * np.maximum(squared_distance - 0.05 ** 2, 0.0)
        # vertical_position_cost = 0.1 * np.abs(sensors[..., self.obj_pos_distance_to_grasp_sensor_idx + 2]).mean(-1)

        # Obj Rotating cost
        if self.MJ_C_ROLLOUT_FULL_PHYSICS_STATE_ONLY:
            obj_orientation = states[..., self.obj_qpos_ids[3:7]]
        else:
            # NOTE: Cannot use mocap's related quat sensor here since MuJoCo-C Rollout does not yet support mocap_pos/quat
            # obj_orientation_err = sensors[..., self.obj_quat_distance_sensor_idx:self.obj_quat_distance_sensor_idx + 4]
            # goal_err_quat = np.array([1.0, 0.0, 0.0, 0.0])
            pass
        orientation_cost = 0.05 * np.square(np_quat_diff_so3(obj_orientation, self.goal_quat)).sum(-1).mean(-1)

        # Contact cost
        contact_err = sensors[..., self.obj_contact_with_palm_sensor_idx]
        contact_cost = 0.01 * contact_err.sum(-1).mean(-1)

        total_reward = -(position_cost + orientation_cost + contact_cost)
        # print(total_reward.mean())
        return total_reward

    def reward_torch(self,
                     states: torch.Tensor,
                     sensors: torch.Tensor,
                     controls: torch.Tensor,
                     system_metadata: Optional[dict[str, Any]] = None) -> torch.Tensor:
        """Implements the free-joint LEAP object picking task reward.
        NOTE: The idea is to make the total cost monolithically decrease through stages!
        """
        # rewards = super().reward(states, sensors, controls, system_metadata)

        # Position costs
        reaching_err = sensors[..., self.obj_pos_distance_to_grasp_sensor_idx:
                                    self.obj_pos_distance_to_grasp_sensor_idx + 2]
        squared_distance = torch.square(reaching_err).sum(-1).mean(-1)
        position_cost = 0.1 * squared_distance + 100 * torch.maximum(squared_distance - 0.05 ** 2,
                                                                     torch.zeros(1, device=MJMANIP_DEVICE))
        # vertical_position_cost = 0.1 * np.abs(sensors[..., self.obj_pos_distance_to_grasp_sensor_idx + 2]).mean(-1)

        # Obj Rotating cost
        if self.MJ_C_ROLLOUT_FULL_PHYSICS_STATE_ONLY:
            obj_orientation = states[..., self.obj_qpos_ids[3:]]
        else:
            # NOTE: Cannot use mocap's related quat sensor here since MuJoCo-C Rollout does not yet support mocap_pos/quat
            # obj_orientation_err = sensors[..., self.obj_quat_distance_sensor_idx:self.obj_quat_distance_sensor_idx + 4]
            # goal_err_quat = torch.tensor(IDENTITY_WXYZ, device=MJMANIP_DEVICE)
            pass
        orientation_cost = (0.05 * torch.square(torch_quat_diff_so3(obj_orientation, self.goal_quat_torch))
                            .sum(-1).mean(-1))

        # Contact cost
        contact_err = sensors[..., self.obj_contact_with_palm_sensor_idx]
        contact_cost = 0.01 * contact_err.sum(-1).mean(-1)

        total_reward = -(position_cost + orientation_cost + contact_cost)
        # print(total_reward.mean())
        return total_reward

    def nt_get_obj_pose(self, state: newton.State, in_sim: bool) -> np.ndarray:
        num_worlds = 1 if in_sim else self.nt_num_rollout_worlds
        obj_poses = np.zeros((num_worlds, 7))

        for i in range(num_worlds):
            world_idx_offset = i * self.nt_num_bodies_per_world
            cube_body_idx = world_idx_offset + self.nt_cube_body_idx_offset
            with wp.ScopedDevice(state.body_q.device):
                body_q = state.body_q.numpy()[cube_body_idx]
                # body_qd = state.body_qd.numpy()[obj_body_id]
                obj_poses[i] = wp_pose_to_mj(body_q)
        return obj_poses.squeeze()

    def nt_reward(self,
                  states: list[newton.State],
                  contacts: list[newton.Contacts],
                  controls: list[newton.Control],
                  system_metadata: Optional[dict[str, Any]] = None) -> np.ndarray:
        """Implements the ALLEGRO cube rotation tracking Newton task reward."""
        if system_metadata is None:
            system_metadata = {}
        goal_quat = system_metadata.get("goal_quat", IDENTITY_WXYZ)

        # weights
        w_pos = self.config.w_pos
        w_rot = self.config.w_rot

        T = len(states)  # num of time steps per rollout
        pos_costs = np.zeros((self.nt_num_rollout_worlds, T, 3))
        rot_costs = np.zeros((self.nt_num_rollout_worlds, T, 3))
        for i in range(len(states)):
            obj_poses = self.nt_get_obj_pose(states[i], in_sim=False)
            for w in range(self.nt_num_rollout_worlds):
                # "standard" tracking task
                qo_pose_traj = obj_poses[w]
                qo_pos_traj = qo_pose_traj[:3]
                qo_quat_traj = qo_pose_traj[3:]
                qo_pos_diff = qo_pos_traj - self.goal_pos
                qo_quat_diff = np_quat_diff_so3(qo_quat_traj, goal_quat)

                pos_costs[w][i] = w_pos * 0.5 * np.square(qo_pos_diff)
                rot_costs[w][i] = w_rot * 0.5 * np.square(qo_quat_diff)

        pos_costs = pos_costs.sum(-1).mean(-1)
        rot_costs = rot_costs.sum(-1).mean(-1)
        rewards = -(pos_costs + rot_costs)
        return rewards

    def post_sim_step(self) -> None:
        """Checks if the obj has dropped and resets if so."""
        # obj_pose = self.mj_data.qpos[self.obj_qpos_ids]
        # print(obj_pose[2])
        # Reset obj pose if dropped to floor
        if self.obj:
            has_dropped = self.obj.xpos[2] < -0.05
            if has_dropped:
                self.reset()

    def pre_sim_step(self) -> None:
        self._update_goal()

    def _update_goal(self) -> None:
        if self.mj_data:
            self.goal_pos = self.mj_data.mocap_pos[self.target_mocap_id]
            self.goal_quat = self.mj_data.mocap_quat[self.target_mocap_id]
            self.goal_pos_torch = torch.as_tensor(self.goal_pos, device=MJMANIP_DEVICE)
            self.goal_quat_torch = torch.as_tensor(self.goal_quat, device=MJMANIP_DEVICE)

    def reset(self) -> None:
        """Resets the model to a default state with random goal."""
        if self.mj_model:
            """Resets the model to a default state with random goal."""
            self.mj_data.qpos[:] = self.config.qpos_home
            self.mj_data.qvel[self.qdof_ids] = 0.0
            self.mj_data.ctrl[:] = self.config.reset_command
            self._update_goal()
            mj.mj_forward(self.mj_model, self.mj_data)
