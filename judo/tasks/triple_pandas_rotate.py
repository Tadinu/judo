from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
import torch
# import warp as wp

# mujoco
import mujoco as mj

# judo
from judo.utils.math_utils import np_quat_diff_so3, torch_quat_diff_so3
from judo.utils.mujoco import mj_get_qpos_ids, mj_get_dof_ids, mj_get_mocap_id

if TYPE_CHECKING:
    from judo.simulation.base import Simulation

# mjmanip
from mjmanip import MJMANIP_DEVICE, DEFAULT_SCENE_XML_PATH, MJMANIP_DEVICE
from mjmanip.robot.arm_hand import ArmHand
from mjmanip.robot.multi_panda_nohands import MultiPandaNoHandsEnv, MultiPandaNoHands, ARM_SCENE_XML_PATH, ARM_XML_PATH
from mjmanip.mj_utils import mj_model_joints_qids

# judo
from judo import BackendType
from judo.tasks.base import Task, TaskConfig

OBJ_NAME = MultiPandaNoHands.OBJECT_NAMES[0]


@dataclass
class TriplePandasRotateConfig(TaskConfig):
    """Reward configuration LEAP object rotation task."""

    task_name: str = "triple_pandas_rotate"
    sim_backend: str = BackendType.MUJOCO.name if MJMANIP_DEVICE == "cpu" else BackendType.MUJOCO_WARP.name
    robot_class: ArmHand = MultiPandaNoHands

    xml_path = None
    sim_xml_path = None
    qpos_home: Optional[np.ndarray] = field(default_factory=
                                            lambda: MultiPandaNoHands.ARM_HOME_QPOS * MultiPandaNoHands.NINSTANCES +
                                                    MultiPandaNoHands.OBJECT_INIT_POSES[OBJ_NAME].tolist())

    reset_command: Optional[np.ndarray] = field(default_factory=
                                                lambda: np.array(
                                                    MultiPandaNoHands.ARM_HOME_QPOS * MultiPandaNoHands.NINSTANCES))  # fmt: skip
    goal_pos: np.ndarray = field(default_factory=lambda: np.array([-0.04, -0.035, -0.065]))
    goal_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    w_pos: float = 0.1
    w_rot: float = 50


class TriplePandasRotate(Task[TriplePandasRotateConfig]):
    """Defines the free-joint LEAP object picking task."""

    config_t: type[TriplePandasRotateConfig] = TriplePandasRotateConfig

    def __init__(self, sim: Optional[Simulation] = None, num_rollout_worlds: int = 1) -> None:
        super().__init__(sim, num_rollout_worlds=num_rollout_worlds)
        self.obj_id: int = -1
        self.obj_qpos_ids: list[int]
        self.obj_dof_ids: list[int]
        self.target_mocap_id: int = -1
        # distance sensors
        self.obj_pos_distance_to_palms_sensor_idxes = [self.get_sensor_start_index(
            f"{OBJ_NAME}_distance_to_{MultiPandaNoHands.arm_item_full_name('palm', i)}")
            for i in range(MultiPandaNoHands.NINSTANCES)]
        self.obj_contact_with_palms_sensor_idxs = [self.get_sensor_start_index(
            f"{OBJ_NAME}_contact_with_{MultiPandaNoHands.arm_item_full_name('palm', i)}")
            for i in range(MultiPandaNoHands.NINSTANCES)]

    def init_ids(self):
        super().init_ids()
        self.obj_id = self.mj_model.body(OBJ_NAME).id
        self.obj_name = OBJ_NAME
        self.obj = self.mj_data.body(self.obj_id)
        self.obj_qpos_ids = mj_get_qpos_ids(self.mj_model, [f"{OBJ_NAME}_freejoint"])
        self.obj_dof_ids = mj_get_dof_ids(self.mj_model, [f"{OBJ_NAME}_freejoint"])
        self.target_mocap_id = mj_get_mocap_id(self.mj_model, f"{OBJ_NAME}_goal")

        # distance sensors
        # NOTE: For rollout result analysis, these are only valid IF MuJoCo-C Rollout backend supports mocap_pos/quat
        # for the `initial_state`
        self.obj_quat_distance_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_orientation_distance_to_goal")

    def mj_compose_spec(self) -> Optional[mj.MjSpec]:
        self.robot_env = MultiPandaNoHandsEnv(world_scene_xml=ARM_SCENE_XML_PATH,
                                              arm_xml=ARM_XML_PATH,
                                              kinematics_mode=False,
                                              diff_ik_enabled=False)
        spec = self.robot_env.construct_main_spec(self.robot_env.meshdir, self.robot_env.texturedir)
        spec.option.timestep = 0.005
        return spec

    def reward(self,
               states: np.ndarray,
               sensors: np.ndarray,
               controls: np.ndarray,
               system_metadata: Optional[dict[str, Any]] = None) -> np.ndarray:
        """Implements the free-joint LEAP object picking task reward.
        NOTE: The idea is to make the total cost monolithically decrease through stages!
        """
        # rewards = super().reward(states, sensors, controls, system_metadata)

        # Position costs
        squared_distance = np.zeros(states.shape[0])
        for palm_distance_sensor_idx in self.obj_pos_distance_to_palms_sensor_idxes:
            reaching_err = self.sensor_value(sensors, palm_distance_sensor_idx, 3)
            squared_distance += np.square(reaching_err).sum(-1).mean(-1)
        position_cost = 0.1 * squared_distance + 100 * np.maximum(squared_distance - 0.1, 0.0)
        # vertical_position_cost = 0.1 * np.abs(sensors[..., self.obj_pos_distance_to_palms_sensor_idxes + 2]).mean(-1)

        # Obj Rotating cost
        if self.MJ_C_ROLLOUT_FULL_PHYSICS_STATE_ONLY:
            obj_orientation = states[..., self.obj_qpos_ids[3:]]
        else:
            # NOTE: Cannot use mocap's related quat sensor here since MuJoCo-C Rollout does not yet support mocap_pos/quat
            # obj_orientation_err = sensors[..., self.obj_quat_distance_sensor_idx:self.obj_quat_distance_sensor_idx + 4]
            # goal_err_quat = np.array([1.0, 0.0, 0.0, 0.0])
            pass
        orientation_cost = 50 * np.square(np_quat_diff_so3(obj_orientation, self.goal_quat)).sum(-1).mean(-1)

        # Contact cost
        palms_contact_reward = 0.01 * self.sensors_contact_cost(sensors, self.obj_contact_with_palms_sensor_idxs)

        total_reward = -(position_cost + orientation_cost) + palms_contact_reward
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
        # rewards = super().reward_torch(states, sensors, controls, system_metadata)

        # Position costs
        squared_distance = torch.zeros(states.shape[0], device=MJMANIP_DEVICE)
        for palm_distance_sensor_idx in self.obj_pos_distance_to_palms_sensor_idxes:
            reaching_err = self.sensor_value(sensors, palm_distance_sensor_idx, 3)
            squared_distance += torch.square(reaching_err).sum(-1).mean(-1)
        position_cost = 0.1 * squared_distance + 100 * torch.max(squared_distance - 0.1,
                                                                 torch.zeros(1, device=MJMANIP_DEVICE))
        # vertical_position_cost = 0.1 * torch.abs(sensors[..., self.obj_pos_distance_to_palms_sensor_idxes + 2]).mean(-1)

        # Obj Rotating cost
        if self.MJ_C_ROLLOUT_FULL_PHYSICS_STATE_ONLY:
            obj_orientation = states[..., self.obj_qpos_ids[3:]]
        else:
            # NOTE: Cannot use mocap's related quat sensor here since MuJoCo-C Rollout does not yet support mocap_pos/quat
            # obj_orientation_err = sensors[..., self.obj_quat_distance_sensor_idx:self.obj_quat_distance_sensor_idx + 4]
            # goal_err_quat = torch.tensor((1.0, 0.0, 0.0, 0.0))
            pass
        orientation_cost = (50 * torch.square(torch_quat_diff_so3(obj_orientation, self.goal_quat_torch))
                            .sum(-1).mean(-1))

        # Object vel cost
        obj_vel = states[..., self.obj_dof_ids]
        velocity_cost = 0.001 * torch.square(obj_vel).sum(-1).mean(-1)

        # Contact cost
        palms_contact_reward = 0.01 * self.sensors_contact_cost(sensors, self.obj_contact_with_palms_sensor_idxs)

        total_reward = -(position_cost + orientation_cost + velocity_cost) + palms_contact_reward
        # print(total_reward.mean())
        return total_reward

    def post_sim_step(self) -> None:
        """Checks if the obj has dropped and resets if so."""
        # obj_pose = self.mj_data.qpos[self.obj_qpos_ids]
        # print(self.obj.xpos)
        # Reset obj pose if dropped to floor
        obj_pos = self.obj.xpos
        has_dropped = ((obj_pos[0] < 0. or obj_pos[0] > 1.5) or (obj_pos[1] < 0. or obj_pos[1] > 1.5)
                       and obj_pos[2] < 0.2)
        if has_dropped:
            self.reset()

    def pre_sim_step(self) -> None:
        self._update_goal()

    def _update_goal(self) -> None:
        self.goal_pos = self.mj_data.mocap_pos[self.target_mocap_id]
        self.goal_quat = self.mj_data.mocap_quat[self.target_mocap_id]
        self.goal_pos_torch = torch.tensor(self.goal_pos, device=MJMANIP_DEVICE)
        self.goal_quat_torch = torch.tensor(self.goal_quat, device=MJMANIP_DEVICE)

    def reset(self) -> None:
        """Resets the model to a default state with random goal."""
        if self.mj_model:
            """Resets the model to a default state with random goal."""
            self.mj_data.qpos[:] = self.config.qpos_home
            self.mj_data.qpos[self.obj_qpos_ids] = self.config.robot_class.OBJECT_INIT_POSES[self.obj_name]
            self.mj_data.qvel[:] = 0.0
            self.mj_data.ctrl[:] = self.config.reset_command
            self._update_goal()
            mj.mj_forward(self.mj_model, self.mj_data)
        else:
            pass
