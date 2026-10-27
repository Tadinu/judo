from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Union, TYPE_CHECKING
from pathlib import Path
import re
import numpy as np
import torch

# mujoco
import mujoco as mj

# newton
import newton
import warp as wp

# mjmanip
from mjmanip import MJMANIP_DEVICE
from mjmanip.robot.arm_hand import ArmHand
from mjmanip.robot.allegro import Allegro
from mjmanip.newton.newton_utils import nt_get_body_id

# judo
from judo import BackendType
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig
from judo.utils.math_utils import np_quat_diff, np_quat_diff_so3, torch_quat_diff_so3
from judo.utils.mujoco import mj_get_qpos_ids, mj_get_mocap_id
from judo.utils.warp import wp_pose_to_mj

if TYPE_CHECKING:
    from judo.simulation.base import Simulation

# mjmanip
from mjmanip import DEFAULT_SCENE_XML_PATH
from mjmanip.mj_utils import IDENTITY_WXYZ
from mjmanip.robot.allegro import (HAND_XML_PATH as ALLEGRO_HAND_XML_PATH,
                                   AllegroEnv, Allegro)

OBJ_NAME = Allegro.OBJECT_NAMES[0]
Allegro.BASE_POSES = [np.array([0.0, 0.0, 0.1, 0, 1, 0, 1])] * Allegro.NINSTANCES


@slider("w_pos", 0.0, 200.0)
@slider("w_rot", 0.0, 1.0)
@dataclass
class AllegroCubeRotateConfig(TaskConfig):
    """Reward configuration ALLEGRO cube rotation task."""

    sim_backend: str = BackendType.MUJOCO.name if MJMANIP_DEVICE == "cpu" else BackendType.MUJOCO_WARP.name
    task_name: str = "allegro_cube"
    robot_class: ArmHand = Allegro
    xml_path: Optional[Union[Path, str]] = str()
    sim_xml_path: Optional[Union[Path, str]] = str()
    qpos_home: Optional[np.ndarray] = field(default_factory=
                                            lambda: Allegro.HAND_HOME_QPOS * Allegro.NINSTANCES +
                                                    Allegro.OBJECT_INIT_POSES[OBJ_NAME].tolist())

    reset_command: Optional[np.ndarray] = field(default_factory=
                                                lambda: np.array(Allegro.HAND_HOME_QPOS * Allegro.NINSTANCES))
    usd_path: Optional[Union[Path, str]] = str()  # str(MODEL_PATH / "usd" / "allegro_left_hand_with_cube.usda")
    goal_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.03, 0.1]))
    goal_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    nt_total_joint_q_size: int = 24
    nt_total_joint_dq_size: int = 23
    nt_total_body_q_size: int = 23
    nt_total_body_qd_size: int = 23
    nt_total_body_f_size: int = 23
    w_pos: float = 100.0
    w_rot: float = 0.1

    def __post_init__(self):
        self.joint_names = Allegro.HAND_JOINTS_NAMES
        self.act_names = Allegro.HAND_ACTS_NAMES
        self.reset_command = Allegro.HAND_HOME_QPOS


class AllegroCubeRotate(Task[AllegroCubeRotateConfig]):
    """Defines the ALLEGRO cube rotation task."""

    config_t: type[AllegroCubeRotateConfig] = AllegroCubeRotateConfig

    def __init__(self, sim: Optional[Simulation] = None, num_rollout_worlds: int = 1) -> None:
        """Initializes the ALLEGRO cube rotation task."""
        super().__init__(sim, num_rollout_worlds=num_rollout_worlds)

        self.goal_pos_torch = torch.as_tensor(self.config.goal_pos, device=MJMANIP_DEVICE).float()
        self.goal_quat_torch = torch.as_tensor(self.config.goal_quat, device=MJMANIP_DEVICE).float()

    def mj_compose_spec(self) -> Optional[mj.MjSpec]:
        self.robot_env = AllegroEnv(world_scene_xml=DEFAULT_SCENE_XML_PATH,
                                    allegro_xml=ALLEGRO_HAND_XML_PATH,
                                    base_free=False)
        spec = self.robot_env.construct_main_spec(self.robot_env.meshdir, self.robot_env.texturedir)
        spec.option.timestep = 0.002
        return spec

    def init_ids(self):
        super().init_ids()
        self.obj_name = OBJ_NAME
        if self.mj_model:
            self.obj_id = self.mj_model.body(OBJ_NAME).id
            self.obj_data = self.mj_data.body(self.obj_id)
            self.obj_qpos_ids = mj_get_qpos_ids(self.mj_model, [f"{OBJ_NAME}_freejoint"])
            self.goal_mocap_id = self.mj_model.body_mocapid[self.mj_model.body(f"{OBJ_NAME}_goal").id]
        else:
            self.obj_id = -1
            self.obj_data = None

            if self.nt_sim_model_builder:
                self.nt_cube_body_idx_offset = nt_get_body_id(self.nt_sim_model_builder, Allegro.OBJECT_NAMES[0])

    def nt_configure_custom_model_builder(self, model_builder: newton.ModelBuilder):
        model_builder.default_shape_cfg.ke = 1.0e3
        model_builder.default_shape_cfg.kd = 1.0e2

        # hide collision shapes for the hand links
        for i, key in enumerate(model_builder.shape_label):
            if re.match(".*Robot/.*?/collision", key):
                model_builder.shape_flags[i] &= ~newton.ShapeFlags.VISIBLE

        # set joint targets and joint drive gains
        for i in range(model_builder.joint_dof_count):
            model_builder.joint_target_ke[i] = 150
            model_builder.joint_target_kd[i] = 5
            model_builder.joint_target_q[i] = 0.0

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

    def reward(self,
               states: np.ndarray,
               sensors: np.ndarray,
               controls: np.ndarray,
               system_metadata: Optional[dict[str, Any]] = None) -> np.ndarray:
        """Implements the ALLEGRO cube rotation tracking task reward."""
        if system_metadata is None:
            system_metadata = {}
        goal_quat = system_metadata.get("goal_quat", np.array([1.0, 0.0, 0.0, 0.0]))

        # weights
        w_pos = self.config.w_pos
        w_rot = self.config.w_rot

        # "standard" tracking task
        qo_pos_traj = states[..., :3]
        qo_quat_traj = states[..., 3:7]
        qo_pos_diff = qo_pos_traj - self.config.goal_pos
        qo_quat_diff = np_quat_diff_so3(qo_quat_traj, goal_quat)

        pos_cost = w_pos * 0.5 * np.square(qo_pos_diff).sum(-1).mean(-1)
        rot_cost = w_rot * 0.5 * np.square(qo_quat_diff).sum(-1).mean(-1)
        rewards = -(pos_cost + rot_cost)
        return rewards

    def reward_torch(self,
                     states: torch.Tensor,
                     sensors: torch.Tensor,
                     controls: torch.Tensor,
                     system_metadata: Optional[dict[str, Any]] = None) -> torch.Tensor:
        """Implements the ALLEGRO cube rotation tracking task reward."""
        if system_metadata is None:
            system_metadata = {}

        # weights
        w_pos = self.config.w_pos
        w_rot = self.config.w_rot

        # "standard" tracking task
        qo_pos_traj = states[..., :3]
        qo_quat_traj = states[..., 3:7]
        qo_pos_diff = qo_pos_traj - self.goal_pos_torch
        qo_quat_diff = torch_quat_diff_so3(qo_quat_traj, self.goal_quat_torch)

        pos_cost = w_pos * 0.5 * torch.square(qo_pos_diff).sum(-1).mean(-1)
        rot_cost = w_rot * 0.5 * torch.square(qo_quat_diff).sum(-1).mean(-1)
        rewards = -(pos_cost + rot_cost)
        return rewards

    def nt_reward(self,
                  states: list[newton.State],
                  contacts: list[newton.Contacts],
                  controls: list[newton.Control],
                  system_metadata: Optional[dict[str, Any]] = None) -> np.ndarray:
        """Implements the ALLEGRO cube rotation tracking Newton task reward."""
        if system_metadata is None:
            system_metadata = {}
        goal_quat = system_metadata.get("goal_quat", np.array([1.0, 0.0, 0.0, 0.0]))

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
        """Checks if the cube has dropped and resets if so."""
        nt_obj_pose = self.nt_get_obj_pose(self.nt_sim.sim_state, in_sim=True) if self.nt_sim else None
        has_dropped = self.obj_data.xpos[2] < 0.1 if self.mj_model else (nt_obj_pose[2] < 0.01)

        # Reset cube if dropped
        if has_dropped:
            self.reset()

        # Check whether goal quat needs to be updated
        goal_quat = self.goal_quat
        obj_quat = self.mj_data.qpos[3:7] if self.mj_data else nt_obj_pose[3:]
        q_diff = np_quat_diff(obj_quat, goal_quat)
        sin_a_2 = np.linalg.norm(q_diff[1:])
        angle = 2 * np.arctan2(sin_a_2, q_diff[0])
        if angle > np.pi:
            angle -= 2 * np.pi
        at_goal = np.abs(angle) < 0.4
        if at_goal:
            self._update_goal()

    def _update_goal(self) -> None:
        """Updates the goal quaternion."""
        # generate uniformly random quaternion
        # https://stackoverflow.com/a/44031492
        uvw = np.random.rand(3)
        goal_quat = np.array(
            [
                np.sqrt(1 - uvw[0]) * np.sin(2 * np.pi * uvw[1]),
                np.sqrt(1 - uvw[0]) * np.cos(2 * np.pi * uvw[1]),
                np.sqrt(uvw[0]) * np.sin(2 * np.pi * uvw[2]),
                np.sqrt(uvw[0]) * np.cos(2 * np.pi * uvw[2]),
            ]
        )
        if self.mj_data:
            self.mj_data.mocap_quat[self.goal_mocap_id] = goal_quat
        self.goal_quat = goal_quat

    def reset(self) -> None:
        """Resets the model to a default state with random goal."""
        if self.mj_model:
            """Resets the model to a default state with random goal."""
            self.mj_data.qpos[self.qpos_ids] = 0.0  # self.config.robot_class.HAND_HOME_QPOS
            self.mj_data.qpos[self.obj_qpos_ids] = self.config.robot_class.OBJECT_INIT_POSES[self.obj_name]
            self.mj_data.qvel[self.qdof_ids] = 0.0
            self.mj_data.ctrl[self.ctrl_ids] = self.config.reset_command
            self._update_goal()
            mj.mj_forward(self.mj_model, self.mj_data)
        else:
            pass

    def get_sim_metadata(self) -> dict[str, Any]:
        """Returns the simulation's goal quat."""
        return {"goal_quat": self.goal_quat}
