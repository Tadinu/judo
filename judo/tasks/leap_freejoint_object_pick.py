from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
from enum import Enum

import numpy as np

# mujoco
import mujoco as mj

# judo
from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig
from judo.utils.fabrics_utils import FabricsAgent
from judo.utils.math_utils import np_quat_diff_so3
from judo.utils.mujoco import mj_get_qpos_ids, mj_get_mocap_id

if TYPE_CHECKING:
    from judo.simulation.base import Simulation

USE_MJX = True
USE_FABRICS = False and not USE_MJX

# mjmanip
from mjmanip import DEFAULT_SCENE_XML_PATH
from mjmanip.mj_utils import IDENTITY_WXYZ

if USE_FABRICS:
    from mjmanip.robot.leap_fabrics import LeapWithFabrics, LeapWithFabricsEnv, HAND_XML_PATH

    hand_class = LeapWithFabrics
    hand_class_env = LeapWithFabricsEnv
elif USE_MJX:
    from mjmanip.robot.leap_mjx import LeapMjx, LeapMjxEnv, FreeLeapMjx, HAND_XML_PATH

    hand_class = FreeLeapMjx
    hand_class_env = LeapMjxEnv

else:
    from mjmanip.robot.leap import LeapEnv, FreeLeap, HAND_XML_PATH

    hand_class = FreeLeap
    hand_class_env = LeapEnv
from mjmanip.mj_utils import mj_model_joints_qids, mj_body_free_joint_name, mj_data_site_pose
from mjmanip.control.fabrics.fabrics.arm_hand_pose_fabric import ArmHandPoseFabricConfig
from mjmanip.control.fabrics.fabrics_controller import FabricsController
from mjmanip.robot.leap_fabrics import LEAP_FABRIC_PALM_CONTROL_FRAME_NAMES, \
    LEAP_FABRIC_FINGER_CONTROL_FRAME_NAMES

# judo
from judo.utils.math_utils import np_quat_diff_so3, np_euler_to_quat, np_mul_pose
from judo.utils.mujoco import mj_get_qpos_ids, mj_get_dof_ids, mj_get_mocap_id

USE_MUG = False


class ObjectRelocatingPhase(Enum):
    """Defines the phases of the Leap Free-joint object pick task."""

    REACHING_OBJ = enum.auto()
    GRASPING_OBJ = enum.auto()
    ORIENTATING_OBJ = enum.auto()
    BRINGING_OBJ_TO_GOAL = enum.auto()


@slider("w_pos", 0.0, 200.0)
@slider("w_rot", 0.0, 1.0)
@dataclass
class LeapFreeJointObjectPickConfig(LeapCubeConfig):
    """Reward configuration freejoint-LEAP object picking task."""

    task_name: str = "leap_freejoint_object_pick"
    xml_path: str = str(
        MODEL_PATH / f"xml/leap_rh_fabric_freejoint_{hand_class.OBJECT_NAMES[0]}_sim.xml") if USE_FABRICS else None
    sim_xml_path: Optional[str] = None
    qpos_home: np.ndarray = field(default_factory=lambda: np.array(
        [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,  # leap base free joint (3 pos + 4 quat)
            1.0, 0.0, 0.8, 0.8,  # index
            1.0, 0.0, 0.8, 0.8,  # middle
            1.0, 0.0, 0.8, 0.8,  # ring
            1.0, 1.0, 0.4, 0.9,  # thumb
            0.0, 0.03, 0.1, 1.0, 0.0, 0.0, 0.0,  # object
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
        """Initializes the LEAP obj bringing task."""
        if USE_MUG:
            hand_class.OBJECT_NAMES = ["mug"]
            hand_class.OBJECT_MODEL_PATHS: dict[str, str] = {
                hand_class.OBJECT_NAMES[0]: f"{MODEL_PATH}/xml/objects/mug/mug.xml"
            }
            hand_class.OBJECT_INIT_POSES: dict[str, np.ndarray] = {
                hand_class.OBJECT_NAMES[0]: np.hstack([np.array([0, 0.5, 0.05]), np.array([1, 0, 0, 0])])
            }
        super().__init__(sim, num_rollout_worlds=num_rollout_worlds)

    def init_ids(self):
        super().init_ids()
        self.OBJ_NAME = hand_class.OBJECT_NAMES[0]
        OBJ_NAME = self.OBJ_NAME
        self.obj_id = self.mj_model.body(OBJ_NAME).id
        self.obj_qpos_ids = mj_get_qpos_ids(self.mj_model, [mj_body_free_joint_name(OBJ_NAME)])
        self.obj_dof_ids = mj_get_dof_ids(self.mj_model, [mj_body_free_joint_name(OBJ_NAME)])
        self.hand_dof_ids = mj_get_dof_ids(self.mj_model, hand_class.HAND_JOINTS_NAMES)
        self.target_mocap_id = mj_get_mocap_id(self.mj_model, hand_class.goal_name(OBJ_NAME))
        self.grasp_site_name = hand_class.HAND_GRASP_SITE_NAME
        self.grasp_site_id = self.mj_model.site(self.grasp_site_name).id
        self.grasp_direction_site_name = f"direction_{hand_class.HAND_GRASP_SITE_NAME}"
        self.obj_pos_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_position")

        # distance sensors
        # NOTE: For rollout result analysis, these are only valid IF MuJoCo-C Rollout backend supports mocap_pos/quat
        # for the `initial_state`
        # distance sensors
        self.obj_pos_distance_to_grasp_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_distance_to_grasp")
        self.obj_pos_distance_to_goal_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_distance_to_goal")
        self.obj_quat_distance_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_orientation_distance_to_goal")

        # grasp site sensors
        self.grasp_site_pos_sensor_idx = self.get_sensor_start_index(f"{self.grasp_site_name}_position")
        self.grasp_direction_site_pos_sensor_idx = self.get_sensor_start_index(
            f"{self.grasp_direction_site_name}_position")

        # contact sensors
        self.obj_contact_with_finger_palm_sensors = {
            finger_palm_geom: self.get_sensor_start_index(f"{OBJ_NAME}_contact_with_{finger_palm_geom}")
            for finger_palm_geom in hand_class.FINGER_PALMS_GEOM_NAMES
        }

        # NOTE: For rollout result analysis, these are only valid IF MuJoCo-C Rollout backend supports mocap_pos/quat
        # for the `initial_state`
        self.reach_threshold = 0.015 if self.OBJ_NAME == "cube" else 0.15
        self.orientation_threshold = 0.01
        self.last_obj_distance_to_goal = 0.

    def mj_compose_spec(self) -> Optional[mj.MjSpec]:
        self.robot_env = hand_class_env(world_scene_xml=DEFAULT_SCENE_XML_PATH,
                                        leap_xml=HAND_XML_PATH,
                                        base_free=True)
        self.robot_env.hand_xml = HAND_XML_PATH
        spec: mj.MjSpec = self.robot_env.construct_main_spec(self.robot_env.meshdir, self.robot_env.texturedir)
        spec.option.timestep = 0.01
        return spec

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

    @property
    def cur_phase(self) -> ObjectRelocatingPhase:
        # Phase 1: Reaching obj
        if True:
            cur_sensor_data = self.mj_data.sensordata
            obj_reaching_err = cur_sensor_data[self.obj_pos_distance_to_grasp_sensor_idx:
                                               self.obj_pos_distance_to_grasp_sensor_idx + 3]
            if np.square(obj_reaching_err).sum() > self.reach_threshold ** 2:
                return ObjectRelocatingPhase.REACHING_OBJ
        else:
            cur_obj_position = self.mj_data.body(self.OBJ_NAME).xpos
            cur_grasp_site_position = self.mj_data.site(self.grasp_site_name).xpos
            cur_obj_grasp_site_distance_square = np.square(cur_obj_position - cur_grasp_site_position).sum()
            is_cur_obj_within_grasp = cur_obj_grasp_site_distance_square < self.reach_threshold ** 2
            if not is_cur_obj_within_grasp:
                return ObjectRelocatingPhase.REACHING_OBJ

        # Phase 2: Orientating Obj/Bringing Obj to Goal
        obj_orientation_err = cur_sensor_data[self.obj_quat_distance_sensor_idx:self.obj_quat_distance_sensor_idx + 4]
        goal_err_quat = IDENTITY_WXYZ
        goal_err = np.square(np_quat_diff_so3(obj_orientation_err, goal_err_quat)).sum()
        if goal_err > self.orientation_threshold ** 2:
            obj_distance_to_goal = np.square(cur_sensor_data[self.obj_pos_distance_to_goal_sensor_idx:
                                                             self.obj_pos_distance_to_goal_sensor_idx + 3]).sum()
            if obj_distance_to_goal < self.last_obj_distance_to_goal:
                self.last_obj_distance_to_goal = obj_distance_to_goal
            return ObjectRelocatingPhase.ORIENTATING_OBJ

        return ObjectRelocatingPhase.BRINGING_OBJ_TO_GOAL

    def sensors_contact_cost(self, sensors_data: np.ndarray, sensor_idxs: dict[str, int]) -> float:
        return np.sum(np.array([sensors_data[..., s] for _, s in sensor_idxs.items()]))

    def sensor_value(self, sensors_data: np.ndarray, sensor_idx: int, sensor_dim: int) -> np.ndarray:
        return sensors_data[..., sensor_idx:sensor_idx + sensor_dim]

    def reward(self,
               states: np.ndarray,
               sensors: np.ndarray,
               controls: np.ndarray,
               system_metadata: Optional[dict[str, Any]] = None) -> np.ndarray:
        """Implements the free-joint LEAP object picking task reward.
        NOTE: The idea is to make the total cost monolithically decrease through stages!
        Ref: https://vikashplus.github.io/Projects/RLwithDemo/dapg-supplementary-materials.pdf
        """
        # rewards = super().reward(states, sensors, controls, system_metadata)
        # obj_pos = sensors[..., self.obj_pos_sensor_idx:self.obj_pos_sensor_idx + 3]
        is_cur_obj_within_grasp = self.cur_phase != ObjectRelocatingPhase.REACHING_OBJ

        # Stage 1: Obj Reaching cost - Ignore Z
        reaching_err = self.sensor_value(sensors, self.obj_pos_distance_to_grasp_sensor_idx, 2)
        squared_distance = np.square(reaching_err).sum(-1).mean(-1)
        reaching_cost = 0.1 * squared_distance + 100 * np.maximum(squared_distance - self.reach_threshold ** 2, 0.0)

        # Stage 2: Obj Rotating cost
        obj_position = states[..., self.obj_qpos_ids[:3]]
        if self.MJ_C_ROLLOUT_FULL_PHYSICS_STATE_ONLY:
            obj_orientation = states[..., self.obj_qpos_ids[3:7]]
        else:
            # NOTE: Cannot use mocap's related quat sensor here since MuJoCo-C Rollout does not yet support mocap_pos/quat
            # obj_orientation_err = sensors[..., self.obj_quat_distance_sensor_idx:self.obj_quat_distance_sensor_idx + 4]
            # goal_err_quat = np.array([1.0, 0.0, 0.0, 0.0])
            pass
        orientation_cost = 0.5 * np.square(np_quat_diff_so3(obj_orientation, self.goal_quat)).sum(-1).mean(-1)

        # Stage 3: Obj Grasping cost
        grasp_site_pos = self.sensor_value(sensors, self.grasp_site_pos_sensor_idx, 3)
        grasp_direction_site_pos = self.sensor_value(sensors, self.grasp_direction_site_pos_sensor_idx, 3)
        grasp_direction = (grasp_direction_site_pos - grasp_site_pos) / np.linalg.norm(
            grasp_direction_site_pos - grasp_site_pos, axis=2)[..., np.newaxis]
        grasp_obj_direction = (obj_position - grasp_site_pos) / np.linalg.norm(obj_position - grasp_site_pos,
                                                                               axis=2)[..., np.newaxis]
        grasp_direction_cost = 0.01 * np.square(grasp_direction - grasp_obj_direction).sum(-1).mean(-1)
        grasp_cost = 0.001 * np.sum(np.square(controls)) + grasp_direction_cost
        if is_cur_obj_within_grasp:
            finger_contact_cost = self.sensors_contact_cost(sensors, self.obj_contact_with_finger_palm_sensors)
            grasp_cost -= finger_contact_cost

        # Final reward
        if is_cur_obj_within_grasp:
            # Stage 4: Obj Bringing-To-Goal cost
            if self.MJ_C_ROLLOUT_FULL_PHYSICS_STATE_ONLY:
                obj_to_goal_distance = np.square(states[..., self.obj_qpos_ids[:3]] - self.goal_pos)
            else:
                obj_to_goal_distance = self.sensor_value(sensors, self.obj_pos_distance_to_goal_sensor_idx, 3)
            bring_cost = 50 * np.square(obj_to_goal_distance).sum(-1).mean(-1)
            total_reward = - (reaching_cost + orientation_cost + grasp_cost + bring_cost)
            # print(total_reward.mean())
            return total_reward
        else:
            return -reaching_cost - grasp_cost - grasp_direction_cost

    def post_sim_step(self) -> None:
        pass

    def pre_sim_step(self) -> None:
        self._update_goal()

    def _update_goal(self) -> None:
        self.goal_pos = self.mj_data.mocap_pos[self.target_mocap_id]
        self.goal_quat = self.mj_data.mocap_quat[self.target_mocap_id]

    def reset(self) -> None:
        pass
