from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Union, TYPE_CHECKING
from pathlib import Path
import re
import numpy as np

# mujoco
import mujoco as mj

# newton
import newton
import warp as wp

# judo
from judo import MODEL_PATH, BackendType
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig
from judo.tasks.leap_cube import LeapCubeConfig
from judo.tasks.caltech_leap_cube import CaltechLeapCubeConfig
from judo.utils.math_utils import np_quat_diff, np_quat_diff_so3
from judo.utils.warp import wp_pose_to_mj

if TYPE_CHECKING:
    from judo.simulation.base import Simulation


@slider("w_pos", 0.0, 200.0)
@slider("w_rot", 0.0, 1.0)
@dataclass
class AllegroCubeRotateConfig(TaskConfig):
    """Reward configuration ALLEGRO cube rotation task."""

    sim_backend: str = BackendType.MUJOCO.name
    task_name: str = "allegro_cube"
    xml_path: Optional[Union[Path, str]] = LeapCubeConfig().xml_path  # CaltechLeapCubeConfig().xml_path
    sim_xml_path: Optional[Union[Path, str]] = LeapCubeConfig().sim_xml_path  # CaltechLeapCubeConfig().sim_xml_path
    usd_path: Optional[Union[Path, str]] = str(MODEL_PATH / "usd" / "allegro_left_hand_with_cube.usda")
    goal_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.03, 0.1]))
    goal_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    total_joint_q_size: int = 24
    total_joint_dq_size: int = 23
    total_body_q_size: int = 23
    total_body_qd_size: int = 23
    total_body_f_size: int = 23
    w_pos: float = 100.0
    w_rot: float = 0.1

    def __post_init__(self):
        # Finger joints only
        self.joint_names = [
            '/World/envs/env_0/Robot/palm_link/index_joint_0',
            '/World/envs/env_0/Robot/index_link_0/index_joint_1',
            '/World/envs/env_0/Robot/index_link_1/index_joint_2',
            '/World/envs/env_0/Robot/index_link_2/index_joint_3',
            '/World/envs/env_0/Robot/palm_link/middle_joint_0',
            '/World/envs/env_0/Robot/middle_link_0/middle_joint_1',
            '/World/envs/env_0/Robot/middle_link_1/middle_joint_2',
            '/World/envs/env_0/Robot/middle_link_2/middle_joint_3',
            '/World/envs/env_0/Robot/middle_link_3/middle_biotac_tip_joint',
            '/World/envs/env_0/Robot/palm_link/ring_joint_0',
            '/World/envs/env_0/Robot/ring_link_0/ring_joint_1',
            '/World/envs/env_0/Robot/ring_link_1/ring_joint_2',
            '/World/envs/env_0/Robot/ring_link_2/ring_joint_3',
            '/World/envs/env_0/Robot/ring_link_3/ring_biotac_tip_joint',
            '/World/envs/env_0/Robot/palm_link/thumb_joint_0',
            '/World/envs/env_0/Robot/thumb_link_0/thumb_joint_1',
            '/World/envs/env_0/Robot/thumb_link_1/thumb_joint_2',
            '/World/envs/env_0/Robot/thumb_link_2/thumb_joint_3',
            '/World/envs/env_0/Robot/thumb_link_3/thumb_biotac_tip_joint'
        ]

        self.qpos_home = CaltechLeapCubeConfig().qpos_home if 'caltech' in self.xml_path else np.array(
            [
                0.0, 0.03, 0.1, 1.0, 0.0, 0.0, 0.0,  # cube
                0.5, -0.75, 0.75, 0.25,  # index
                0.5, 0.0, 0.75, 0.25,  # middle
                0.5, 0.75, 0.75, 0.25,  # ring
                0.65, 0.9, 0.75, 0.6,  # thumb
            ])

        self.reset_command = self.qpos_home[7:]


class AllegroCubeRotate(Task[AllegroCubeRotateConfig]):
    """Defines the ALLEGRO cube rotation task."""

    config_t: type[AllegroCubeRotateConfig] = AllegroCubeRotateConfig

    def __init__(self, sim: Optional[Simulation] = None, num_rollout_worlds: int = 1) -> None:
        """Initializes the ALLEGRO cube rotation task."""
        super().__init__(sim, num_rollout_worlds=num_rollout_worlds)

        goal_mocap_body_id = mj.mj_name2id(self.mj_sim_model, mj.mjtObj.mjOBJ_BODY, "target")
        self.goal_mocap_id = self.mj_sim_model.body_mocapid[goal_mocap_body_id] if goal_mocap_body_id > -1 else -1
        if self.nt_sim_model_builder:
            cube_names = ['cube', '/World/envs/env_0/object/DexCube']
            self.nt_cube_body_idx_offset = [self.nt_sim_model_builder.body_key.index(cube_name)
                                            for cube_name in cube_names
                                            if cube_name in self.nt_sim_model_builder.body_key][0]

    def nt_configure_custom_model_builder(self, model_builder: newton.ModelBuilder):
        model_builder.default_shape_cfg.ke = 1.0e3
        model_builder.default_shape_cfg.kd = 1.0e2

        # hide collision shapes for the hand links
        for i, key in enumerate(model_builder.shape_key):
            if re.match(".*Robot/.*?/collision", key):
                model_builder.shape_flags[i] &= ~newton.ShapeFlags.VISIBLE

        # set joint targets and joint drive gains
        for i in range(model_builder.joint_dof_count):
            model_builder.joint_target_ke[i] = 150
            model_builder.joint_target_kd[i] = 5
            model_builder.joint_target_pos[i] = 0.0

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
        has_dropped = self.mj_data.qpos[2] < -0.3 if self.mj_data else (nt_obj_pose[2] < 0.01)

        # we reset here if the cube has dropped
        if has_dropped:
            self.reset()

        # check whether goal quat needs to be updated
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
            self.mj_data.qpos[:] = self.config.qpos_home
            self.mj_data.qvel[:] = 0.0
            self.mj_data.ctrl[:] = self.config.reset_command
            self._update_goal()
            mj.mj_forward(self.mj_model, self.mj_data)
        else:
            pass

    def get_sim_metadata(self) -> dict[str, Any]:
        """Returns the simulation's goal quat."""
        return {"goal_quat": self.goal_quat}
