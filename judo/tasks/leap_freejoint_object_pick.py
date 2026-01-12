from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

# mujoco
import mujoco as mj

# judo
from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig
from judo.utils.math_utils import quat_diff_so3
from judo.utils.mujoco import mj_get_qpos_ids, mj_get_mocap_id

if TYPE_CHECKING:
    from judo.simulation.base import Simulation

OBJ_NAME = "cube"


@slider("w_pos", 0.0, 200.0)
@slider("w_rot", 0.0, 1.0)
@dataclass
class LeapFreeJointObjectPickConfig(LeapCubeConfig):
    """Reward configuration LEAP cube rotation task."""

    task_name: str = "leap_freejoint_object_pick"
    xml_path: str = str(MODEL_PATH / f"xml/leap_rh_fabric_freejoint_{OBJ_NAME}_sim.xml")
    sim_xml_path: Optional[str] = None
    qpos_home: np.ndarray = field(default_factory=lambda: np.array(
        [
            0.0, 0.03, 0.1, 1.0, 0.0, 0.0, 0.0,  # object
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,  # leap base free joint (3 pos + 4 quat)
            1.0, 0.0, 0.8, 0.8,  # index
            1.0, 0.0, 0.8, 0.8,  # middle
            1.0, 0.0, 0.8, 0.8,  # ring
            1.0, 1.0, 0.4, 0.9,  # thumb
        ]
    ))
    reset_command: np.ndarray = field(default_factory=lambda: np.array(
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # leap base free joint - 6DOF
            1.0, 0.0, 0.8, 0.8,  # index
            1.0, 0.0, 0.8, 0.8,  # middle
            1.0, 0.0, 0.8, 0.8,  # ring
            1.0, 1.0, 0.4, 0.9,  # thumb
        ]
    ))  # fmt: skip
    goal_pos: np.ndarray = field(default_factory=lambda: np.array([-0.04, -0.035, -0.065]))
    goal_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    w_pos: float = 0.1
    w_rot: float = 50


class LeapFreeJointObjectPick(LeapCube):
    """Defines the free-joint LEAP object picking task."""

    config_t: type[LeapFreeJointObjectPickConfig] = LeapFreeJointObjectPickConfig

    def __init__(self, sim: Optional[Simulation] = None, num_rollout_worlds: int = 1) -> None:
        """Initializes the LEAP cube rotation task."""
        super().__init__(sim, num_rollout_worlds=num_rollout_worlds)

        self.obj_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_BODY, OBJ_NAME)
        self.obj_qpos_ids = mj_get_qpos_ids(self.mj_model, [f"{OBJ_NAME}_freejoint"])
        self.obj_pos_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_position")
        self.leap_base_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_BODY, "leap_mount")
        self.obj_pos_distance_to_grasp_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_distance_to_grasp")

        if self.MJ_C_ROLLOUT_FULL_PHYSICS_STATE_ONLY:
            self.target_mocap_id = mj_get_mocap_id(self.mj_model, "target")
        else:
            # NOTE: This is only useful once MuJoCo-C Rollout backend supports mocap_pos/quat also for the `initial_state`
            self.obj_quat_distance_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_orientation_from_target")
            # self.grasp_pos_distance_to_goal_sensor_idx = self.get_sensor_start_index("grasp_distance_from_target")
            self.obj_pos_distance_to_goal_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_distance_to_target")
        self.reach_threshold = 0.015

    def reward(self,
               states: np.ndarray,
               sensors: np.ndarray,
               controls: np.ndarray,
               system_metadata: Optional[dict[str, Any]] = None) -> np.ndarray:
        """Implements the free-joint LEAP object picking task reward.
        NOTE: The idea is to make the total cost monolithically decrease through stages!
        """
        # rewards = super().reward(states, sensors, controls, system_metadata)

        # obj_pos = sensors[..., self.obj_pos_sensor_idx:self.obj_pos_sensor_idx + 3]

        # Stage 1: Obj Reaching cost - Ignore Z
        reaching_err = sensors[..., self.obj_pos_distance_to_grasp_sensor_idx:
                                    self.obj_pos_distance_to_grasp_sensor_idx + 2]
        squared_distance = np.square(reaching_err).sum(-1).mean(-1)
        reaching_cost = 0.1 * squared_distance + 100 * np.maximum(squared_distance - self.reach_threshold ** 2, 0.0)

        # Stage 2: Obj Rotating cost
        if self.MJ_C_ROLLOUT_FULL_PHYSICS_STATE_ONLY:
            obj_orientation = states[..., self.obj_qpos_ids[3:7]]
        else:
            # NOTE: Cannot use mocap's related quat sensor here since MuJoCo-C Rollout does not yet support mocap_pos/quat
            # obj_orientation_err = sensors[..., self.obj_quat_distance_sensor_idx:self.obj_quat_distance_sensor_idx + 4]
            # goal_err_quat = np.array([1.0, 0.0, 0.0, 0.0])
            pass
        orientation_cost = 0.05 * np.square(quat_diff_so3(obj_orientation, self.goal_quat)).sum(-1).mean(-1)

        # Stage 3: Obj Grasping cost
        grasp_cost = 0.001 * np.sum(np.square(controls))

        # Stage 4: Obj Bringing-To-Goal cost
        if self.MJ_C_ROLLOUT_FULL_PHYSICS_STATE_ONLY:
            obj_to_goal_distance = np.square(states[..., self.obj_qpos_ids[:3]] - self.goal_pos)
        else:
            obj_to_goal_distance = sensors[..., self.obj_pos_distance_to_goal_sensor_idx:
                                                self.obj_pos_distance_to_goal_sensor_idx + 3]
        bring_cost = np.square(obj_to_goal_distance).sum(-1).mean(-1)

        total_reward = - (reaching_cost + orientation_cost + grasp_cost + bring_cost)
        # print(total_reward.mean())
        return total_reward

    def post_sim_step(self) -> None:
        pass

    def pre_sim_step(self) -> None:
        self._update_goal()

    def _update_goal(self) -> None:
        self.goal_pos = self.mj_data.mocap_pos[self.target_mocap_id]
        self.goal_quat = self.mj_data.mocap_quat[self.target_mocap_id]

    def reset(self) -> None:
        pass
