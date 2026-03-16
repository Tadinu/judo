from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
import numpy as np

# mujoco
import mujoco as mj

# mjmanip
from mjmanip import DEFAULT_SCENE_MJX_XML_PATH, DEFAULT_SCENE_XML_PATH
from mjmanip.robot.arm_hand import ArmHandDiffIK
# NOTE: LeapMjx hand is more robust than Leap, so use [panda_leap_mjx] for now!
from mjmanip.robot.panda_leap_mjx import PandaLeapMjxEnv, PandaLeapMjx, ARM_XML_PATH, HAND_XML_PATH

if PandaLeapMjx:
    PandaLeapMjx.NINSTANCES = 1
    PANDA_LEAP = PandaLeapMjx
    PANDA_LEAP_ENV = PandaLeapMjxEnv
else:
    from mjmanip.robot.panda_leap import PandaLeapEnv, PandaLeap, ARM_XML_PATH

    PANDA_LEAP = PandaLeap
    PANDA_LEAP_ENV = PandaLeapEnv

from mjmanip.utils import mj_body_free_joint_name, mj_get_site_pose

# judo
from judo import BackendType
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig
from judo.utils.math_utils import np_quat_diff_so3, np_euler_to_quat, np_mul_pose
from judo.utils.mujoco import mj_get_qpos_ids, mj_get_dof_ids, mj_get_mocap_id
from judo.tasks.leap_freejoint_object_pick import ObjectRelocatingPhase

if TYPE_CHECKING:
    from judo.simulation.base import Simulation

OBJ_NAME = PANDA_LEAP.OBJECT_NAMES[0]
USE_EE_MPC = False
EE_DOFS_NO = 6


@slider("w_pos", 0.0, 200.0)
@slider("w_rot", 0.0, 1.0)
@dataclass
class PandaLeapPickConfig(TaskConfig):
    """Reward configuration Panda-Leap obj picking task."""

    task_name: str = "panda_leap_pick"
    sim_backend: str = BackendType.MUJOCO.name

    xml_path = None
    sim_xml_path = None
    qpos_home: Optional[np.ndarray] = field(default_factory=
                                            lambda: PANDA_LEAP.ARM_HOME_QPOS +
                                                    PANDA_LEAP.HAND_HOME_QPOS +
                                                    PANDA_LEAP.OBJECT_INIT_POSES[OBJ_NAME].tolist())  # fmt: skip
    reset_command: Optional[np.ndarray] = field(default_factory=
                                                lambda: np.array(PANDA_LEAP.ARM_HOME_QPOS +
                                                                 PANDA_LEAP.HAND_HOME_QPOS))  # fmt: skip
    goal_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.01, 0.03]))
    goal_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    w_pos: float = 0.1
    w_rot: float = 50

    def __post_init__(self):
        self.joint_names = []


class PandaLeapPick(Task[PandaLeapPickConfig]):
    """Defines the Panda-Leap object picking task."""

    config_t: type[PandaLeapPickConfig] = PandaLeapPickConfig

    def __init__(self, sim: Optional[Simulation] = None, num_rollout_worlds: int = 1) -> None:
        """Initializes the Panda-Leap obj picking task."""
        super().__init__(sim, num_rollout_worlds=num_rollout_worlds)
        self.goal_pos = self.config.goal_pos
        self.goal_quat = self.config.goal_quat
        self.qpos_home = self.config.qpos_home
        self.robot_env = None
        self.robot = None
        self.map_controls = self.map_ee_to_arm_controls if USE_EE_MPC else None
        self.rollout_diff_iks = None
        self.diff_ik = ArmHandDiffIK(self.mj_model, self.mj_data, PANDA_LEAP, self.qpos_home)
        self.diff_ik.DT = self.mj_model.opt.timestep
        self.diff_ik.init()

    def init_ids(self):
        super().init_ids()
        self.obj_id = self.mj_model.body(OBJ_NAME).id
        self.obj_qpos_ids = mj_get_qpos_ids(self.mj_model, [mj_body_free_joint_name(OBJ_NAME)])
        self.obj_dof_ids = mj_get_dof_ids(self.mj_model, [mj_body_free_joint_name(OBJ_NAME)])
        self.hand_dof_ids = mj_get_dof_ids(self.mj_model,
                                           PANDA_LEAP.hand_items_full_names(PANDA_LEAP.HAND_JOINTS_NAMES))
        self.target_mocap_id = mj_get_mocap_id(self.mj_model, PANDA_LEAP.goal_name(OBJ_NAME))
        self.grasp_site_name = PANDA_LEAP.hand_item_full_name(PANDA_LEAP.GRASP_SITE_NAME)
        self.grasp_site_id = self.mj_model.site(self.grasp_site_name).id
        self.grasp_direction_site_name = PANDA_LEAP.hand_item_full_name(f"direction_{PANDA_LEAP.GRASP_SITE_NAME}")

        # distance sensors
        # NOTE: For rollout result analysis, these are only valid IF MuJoCo-C Rollout backend supports mocap_pos/quat
        # for the `initial_state`
        # distance sensors
        self.obj_pos_distance_to_grasp_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_distance_to_grasp")
        self.obj_pos_distance_to_goal_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_distance_to_goal")
        self.obj_quat_distance_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_orientation_distance_to_goal")

        # grasp site sensors
        self.grasp_site_pos_sensor_idx = self.get_sensor_start_index(f"{self.grasp_site_name}_position")
        self.grasp_direction_site_pose_sensor_idx = self.get_sensor_start_index(
            f"{self.grasp_direction_site_name}_position")

        # contact sensors
        self.obj_contact_with_finger_palm_sensors = [
            self.get_sensor_start_index(f"{OBJ_NAME}_contact_with_{finger_palm_geom}")
            for finger_palm_geom in PANDA_LEAP.hand_items_full_names(PANDA_LEAP.FINGER_PALMS_GEOM_NAMES, 0)
        ]

        self.obj_contact_with_arm_sensors = [
            self.get_sensor_start_index(f"{OBJ_NAME}_contact_with_{arm_geom}")
            for arm_geom in PANDA_LEAP.ARM_GEOMS_NAMES
        ]

        self.reach_threshold_squared = 0.01
        self.orientation_threshold = 0.0001
        self.last_obj_distance_to_goal = 0.

    def mj_compose_spec(self) -> Optional[mj.MjSpec]:
        self.robot_env = PANDA_LEAP_ENV(
            world_scene_xml=DEFAULT_SCENE_MJX_XML_PATH if PandaLeapMjxEnv else DEFAULT_SCENE_XML_PATH,
            arm_xml=ARM_XML_PATH,
            hand_xml=HAND_XML_PATH)
        spec = self.robot_env.construct_main_spec(self.robot_env.meshdir, self.robot_env.texturedir)
        spec.option.timestep = 0.005
        return spec

    @Task.nu.getter
    def nu(self) -> int:
        """Number of control inputs. The same as the mj.MjModel for this task."""
        if self.mj_model and USE_EE_MPC:
            return self.mj_model.nu - PANDA_LEAP.ARM_DOFS_NO + EE_DOFS_NO
        return super().nu

    @Task.actuator_ctrlrange.getter
    def actuator_ctrlrange(self) -> np.ndarray:
        """Mujoco actuator limits for this task."""
        if self.mj_model and USE_EE_MPC:
            limits = self.mj_model.actuator_ctrlrange
            hand_limits = limits[PANDA_LEAP.ARM_DOFS_NO:]
            limits = np.vstack(
                [np.array([[-0.01, 0.01]] * 3, dtype=limits.dtype),  # Linear dofs
                 np.array([[-1.57, 1.57]] * 3, dtype=limits.dtype),  # Angular dofs
                 hand_limits])
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
            # print("Cur grasp<->obj dist", np.square(obj_reaching_err).sum())
            if np.square(obj_reaching_err).sum() > self.reach_threshold_squared:
                return ObjectRelocatingPhase.REACHING_OBJ
        else:
            cur_obj_position = self.mj_data.body(self.OBJ_NAME).xpos
            cur_grasp_site_position = self.mj_data.site(self.grasp_site_name).xpos
            cur_obj_grasp_site_distance_square = np.square(cur_obj_position - cur_grasp_site_position).sum()
            is_cur_obj_within_grasp = cur_obj_grasp_site_distance_square < self.reach_threshold_squared
            if not is_cur_obj_within_grasp:
                return ObjectRelocatingPhase.REACHING_OBJ

        # Phase 2: Orientating Obj/Bringing Obj to Goal
        obj_orientation_err = cur_sensor_data[self.obj_quat_distance_sensor_idx:self.obj_quat_distance_sensor_idx + 4]
        goal_err_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if np.square(np_quat_diff_so3(obj_orientation_err, goal_err_quat)).sum() > self.orientation_threshold ** 2:
            obj_distance_to_goal = np.square(cur_sensor_data[self.obj_pos_distance_to_goal_sensor_idx:
                                                             self.obj_pos_distance_to_goal_sensor_idx + 3]).sum()
            if self.last_obj_distance_to_goal > obj_distance_to_goal:
                self.last_obj_distance_to_goal = obj_distance_to_goal
                return ObjectRelocatingPhase.ORIENTATING_OBJ

        return ObjectRelocatingPhase.BRINGING_OBJ_TO_GOAL

    def reward(self,
               states: np.ndarray,
               sensors: np.ndarray,
               controls: np.ndarray,
               system_metadata: Optional[dict[str, Any]] = None) -> np.ndarray:
        """Implements the ALLEGRO cube rotation tracking task reward."""
        is_cur_obj_within_grasp = self.cur_phase != ObjectRelocatingPhase.REACHING_OBJ

        # Stage 1: Obj Reaching cost - Ignore Z
        # Always avoid arm contact, to focus on hand-based grasping only
        reaching_err = self.sensor_value(sensors, self.obj_pos_distance_to_grasp_sensor_idx, 3)
        squared_distance = np.square(reaching_err).sum(-1).mean(-1)
        reaching_cost = squared_distance + 100 * np.maximum(squared_distance - self.reach_threshold_squared, 0.0)

        # Stage 2: Obj Rotating cost
        obj_position = states[..., self.obj_qpos_ids[:3]]
        if self.MJ_C_ROLLOUT_FULL_PHYSICS_STATE_ONLY:
            obj_orientation = states[..., self.obj_qpos_ids[3:7]]
        else:
            # NOTE: Cannot use mocap's related quat sensor here since MuJoCo-C Rollout does not yet support mocap_pos/quat
            # obj_orientation_err = sensors[..., self.obj_quat_distance_sensor_idx:self.obj_quat_distance_sensor_idx + 4]
            # goal_err_quat = np.array([1.0, 0.0, 0.0, 0.0])
            pass
        orientation_cost = 0.05 * np.square(np_quat_diff_so3(obj_orientation, self.goal_quat)).sum(-1).mean(-1)

        # Stage 3: Obj Grasping cost
        grasp_site_pos = self.sensor_value(sensors, self.grasp_site_pos_sensor_idx, 3)
        grasp_direction_site_pos = self.sensor_value(sensors, self.grasp_direction_site_pose_sensor_idx, 3)
        grasp_direction = (grasp_direction_site_pos - grasp_site_pos) / np.linalg.norm(
            grasp_direction_site_pos - grasp_site_pos, axis=2)[..., np.newaxis]
        grasp_obj_direction = (obj_position - grasp_site_pos) / np.linalg.norm(obj_position - grasp_site_pos,
                                                                               axis=2)[..., np.newaxis]
        grasp_direction_cost = 10 * np.square(grasp_direction - grasp_obj_direction).sum(-1).mean(-1)
        grasp_cost = 0.001 * np.sum(np.square(controls)) + grasp_direction_cost
        if True:
            arm_contact_cost = self.sensors_contact_cost(sensors, self.obj_contact_with_arm_sensors)
            finger_contact_cost = 10 * self.sensors_contact_cost(sensors, self.obj_contact_with_finger_palm_sensors)
            grasp_cost += arm_contact_cost - finger_contact_cost

        # Stage 4: Obj vel cost
        obj_velocity = states[..., self.mj_model.nq + self.obj_dof_ids]
        obj_vel_cost = 100 * np.sum(np.square(obj_velocity))
        hand_velocity = states[..., self.mj_model.nq + self.hand_dof_ids]
        hand_vel_cost = np.sum(np.square(hand_velocity))

        # Stage 5: Obj Bringing-To-Goal cost
        if self.MJ_C_ROLLOUT_FULL_PHYSICS_STATE_ONLY:
            obj_to_goal_distance = np.square(obj_position - self.goal_pos)
        else:
            obj_to_goal_distance = self.sensor_value(sensors, self.obj_pos_distance_to_goal_sensor_idx, 3)
        bring_cost = 50 * np.square(obj_to_goal_distance).sum(-1).mean(-1)

        # Final reward
        if is_cur_obj_within_grasp:
            # print(arm_contact_cost, finger_contact_cost)
            total_reward = - (reaching_cost + orientation_cost + grasp_cost + obj_vel_cost + hand_vel_cost + bring_cost)
            # print(total_reward.mean())
            return total_reward
        else:
            return -reaching_cost - grasp_cost - grasp_direction_cost

    def reset(self) -> None:
        """Resets the model to a default state with random goal."""
        if self.mj_model:
            """Resets the model to a default state with random goal."""
            self.mj_data.qpos[:] = self.qpos_home
            self.mj_data.qvel[:] = 0.0
            self.mj_data.ctrl[:] = self.config.reset_command
            self._update_goal()
            mj.mj_forward(self.mj_model, self.mj_data)
        else:
            pass

    def get_sim_metadata(self) -> dict[str, Any]:
        """Returns the simulation's goal quat."""
        return {"goal_quat": self.goal_quat}

    def post_sim_step(self) -> None:
        pass

    def pre_sim_step(self) -> None:
        self._update_goal()

    def _update_goal(self) -> None:
        self.goal_pos = self.mj_data.mocap_pos[self.target_mocap_id]
        self.goal_quat = self.mj_data.mocap_quat[self.target_mocap_id]

    def map_ee_to_arm_controls(self, model_data_pairs: list[tuple[mj.MjModel, mj.MjData]],
                               rollout_controls: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        assert self.config.sim_backend_type() == BackendType.MUJOCO
        assert EE_DOFS_NO == 6
        if not model_data_pairs:
            self.optimal_target_traces.clear()

        # DiffIK for each rollout
        if self.rollout_diff_iks is None:
            self.rollout_diff_iks = [ArmHandDiffIK(mj_model, mj_data, PandaLeap, self.qpos_home)
                                     for mj_model, mj_data in model_data_pairs]
            for diff_ik in self.rollout_diff_iks:
                diff_ik.DT = self.mj_model.opt.timestep
                diff_ik.init()

        # Rollouts from [current_state]
        num_rollouts, num_steps = rollout_controls.shape[:2]
        out_rollout_controls = np.zeros((num_rollouts, num_steps, self.mj_model.nu))
        out_rollout_controls[..., PANDA_LEAP.ARM_DOFS_NO:] = rollout_controls[..., EE_DOFS_NO:]

        for rollout_idx in range(num_rollouts):
            rl_pair = model_data_pairs[rollout_idx] if model_data_pairs else None
            mj_model = rl_pair[0] if rl_pair else self.mj_model
            mj_data = rl_pair[1] if rl_pair else self.mj_data
            diff_ik = self.rollout_diff_iks[rollout_idx] if model_data_pairs else self.diff_ik
            obj = mj_data.body(OBJ_NAME)
            for step_idx in range(num_steps):
                # Step [mj_data] kinematics only
                if model_data_pairs:
                    mj.mj_kinematics(mj_model, mj_data)

                # EE pose delta ctrl (6DOF in 3D)
                ee_pose_ctrl = rollout_controls[rollout_idx, step_idx, :EE_DOFS_NO]

                # EE -> Arm control
                if False:
                    q = ArmHandDiffIK.dls_ik(mj_model, mj_data, ee_pose_ctrl,
                                             self.grasp_site_name, dt=diff_ik.DT)
                else:
                    ee_quat_ctrl = np.zeros(4)
                    mj.mju_euler2Quat(ee_quat_ctrl, ee_pose_ctrl[3:], "XYZ")
                    target_pose = np.zeros(7)
                    grasp_site_pos, grasp_site_quat = mj_get_site_pose(mj_data, self.grasp_site_id)
                    mj.mju_mulPose(target_pose[:3], target_pose[3:],
                                   grasp_site_pos, grasp_site_quat,
                                   ee_pose_ctrl[:3], ee_quat_ctrl)
                    q = diff_ik.plan(target_ee_pose=target_pose, use_solver=True)
                    if q is None:
                        q = diff_ik.plan(target_ee_pose=target_pose, use_solver=False)
                    # Traces
                    if not model_data_pairs:
                        self.optimal_target_traces.append(target_pose[:3])

                # Save result q back to [rollout_controls]
                out_rollout_controls[rollout_idx, step_idx, :PANDA_LEAP.ARM_DOFS_NO] = q[:PANDA_LEAP.ARM_DOFS_NO]
        return out_rollout_controls
