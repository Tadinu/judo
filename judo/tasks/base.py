# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, Optional, TYPE_CHECKING, Union

# MuJoCo
import mujoco
import numpy as np
from mujoco import MjData, MjModel, MjSpec

# Newton
import warp as wp
import newton

# judo
from judo import BackendType
from judo.utils.warp import wp_create_kernel_tile_array

if TYPE_CHECKING:
    from judo.simulation.base import Simulation
    from judo.simulation import NTSimulation


@dataclass
class TaskConfig:
    """Base task configuration dataclass."""
    sim_backend: str = BackendType.MUJOCO.name
    task_name: str = ""
    joint_names: Optional[list[str]] = None
    total_joint_q_size: int = 0
    total_joint_dq_size: int = 0
    total_body_q_size: int = 0
    total_body_qd_size: int = 0
    total_body_f_size: int = 0

    def __post_init__(self):
        self.joint_names = []

    def sim_backend_type(self) -> BackendType:
        return BackendType[self.sim_backend]


ConfigT = TypeVar("ConfigT", bound=TaskConfig)


class Task(ABC, Generic[ConfigT]):
    """Task definition."""

    config_t: type[ConfigT]

    def __init__(self, sim: Optional[Simulation] = None,
                 num_rollout_worlds: int = 1,
                 xml_path: Optional[Union[Path, str]] = None,
                 usd_path: Optional[Union[Path, str]] = None,
                 sim_xml_path: Optional[Union[Path, str]] = None) -> None:
        """Initialize the task."""
        self.sim = sim
        self.config = self.config_t()

        # MuJoCo
        self.xml_path = xml_path
        self.mj_spec = MjSpec.from_file(str(xml_path)) if xml_path else None
        self.mj_model = self.mj_spec.compile() if xml_path else None
        self.mj_data = MjData(self.mj_model) if xml_path else None
        self.mj_sim_model = MjModel.from_xml_path(str(sim_xml_path)) if sim_xml_path else self.mj_model

        # Newton models (sim + rollout)
        self.usd_path = usd_path
        from judo.simulation import NTSimulation
        self.nt_sim: NTSimulation = sim if isinstance(sim, NTSimulation) else None
        self.nt_rollout_model: newton.Model = None
        self.nt_rollout_model_builder: newton.ModelBuilder = None
        self.nt_sim_model: newton.Model = None
        self.nt_sim_model_builder: newton.ModelBuilder = None

        # Rollout Model batch info
        self.nt_num_rollout_worlds: int = 0
        self.nt_num_bodies_per_world: int = 0
        self.nt_initial_world_positions = None

        # Init state sim->rollout copy funcs
        self.nt_init_sim_to_rollout_copy_functions()

        # Init sim & rollout models
        if self.nt_sim:
            self.nt_init_models(num_rollout_worlds)

    def nt_init_models(self, num_rollout_worlds: int = 1, init_pose: wp.transform = wp.transform_identity()):
        if not self.nt_sim_model_builder:
            self.nt_sim_model_builder = self.nt_create_sim_model_builder(init_pose)
            self.nt_sim_model_builder.add_ground_plane()
            self.nt_sim_model = self.nt_sim_model_builder.finalize(device='cuda')

        # Finalize models
        if num_rollout_worlds > 1:
            if self.nt_rollout_model_builder and self.nt_rollout_model_builder.num_worlds == num_rollout_worlds:
                return
            # Scene model: multi-replications of sim model (before replicating it to the rollout model)
            self.nt_rollout_model_builder = newton.ModelBuilder()
            model_builder = self.nt_create_sim_model_builder(init_pose)
            self.nt_rollout_model_builder.replicate(model_builder, num_rollout_worlds)
            self.nt_rollout_model_builder.default_shape_cfg.ke = 1.0e3
            self.nt_rollout_model_builder.default_shape_cfg.kd = 1.0e2
            # NOTE: Rollout model must be pure replications of sim model, -> NOT add ground here!
            # self.nt_rollout_model_builder.add_ground_plane()
            self.nt_rollout_model = self.nt_rollout_model_builder.finalize(device='cuda')
        else:
            self.nt_rollout_model_builder = self.nt_sim_model_builder
            self.nt_rollout_model = self.nt_sim_model

        self.nt_num_rollout_worlds = self.nt_rollout_model_builder.num_worlds
        self.nt_num_bodies_per_world = self.nt_rollout_model.body_count // self.nt_rollout_model_builder.num_worlds
        self.nt_initial_world_positions = self.nt_rollout_model.body_q.numpy()[
            :: self.nt_sim_model_builder.body_count, :3].copy() if self.nt_rollout_model else None

    def nt_create_sim_model_builder(self, init_pose: wp.transform = wp.transform_identity()):
        # Sim model
        sim_model_builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(sim_model_builder)
        if self.usd_path.endswith(".usd") or self.usd_path.endswith(".usda"):
            sim_model_builder.add_usd(self.usd_path, xform=init_pose, enable_self_collisions=True)
        elif self.usd_path.endswith(".xml"):
            sim_model_builder.add_mjcf(self.usd_path, xform=init_pose, verbose=True)
        # Configure model builders
        self.nt_configure_custom_model_builder(sim_model_builder)
        return sim_model_builder

    def nt_configure_custom_model_builder(self, model_builder: newton.ModelBuilder):
        pass

    def nt_init_sim_to_rollout_copy_functions(self):
        self.nt_kernel_tile_joint_q = wp_create_kernel_tile_array(self.config.total_joint_q_size)
        self.nt_kernel_tile_joint_qd = wp_create_kernel_tile_array(self.config.total_joint_dq_size)
        self.nt_kernel_tile_body_q = wp_create_kernel_tile_array(self.config.total_body_q_size)
        self.nt_kernel_tile_body_qd = wp_create_kernel_tile_array(self.config.total_body_qd_size)
        self.nt_kernel_tile_body_f = wp_create_kernel_tile_array(self.config.total_body_f_size)

    def nt_copy_sim_to_rollout_state(self, sim_state: newton.State, rollout_state: newton.State,
                                     num_rollout_worlds: int):
        wp.launch_tiled(self.nt_kernel_tile_joint_q, dim=(num_rollout_worlds,),
                        inputs=[sim_state.joint_q],
                        outputs=[rollout_state.joint_q],
                        block_dim=len(sim_state.joint_q))
        wp.launch_tiled(self.nt_kernel_tile_joint_qd, dim=(num_rollout_worlds,),
                        inputs=[sim_state.joint_qd],
                        outputs=[rollout_state.joint_qd],
                        block_dim=len(sim_state.joint_qd))
        wp.launch_tiled(self.nt_kernel_tile_body_q, dim=(num_rollout_worlds,),
                        inputs=[sim_state.body_q],
                        outputs=[rollout_state.body_q],
                        block_dim=len(sim_state.body_q))
        wp.launch_tiled(self.nt_kernel_tile_body_qd, dim=(num_rollout_worlds,),
                        inputs=[sim_state.body_qd],
                        outputs=[rollout_state.body_qd],
                        block_dim=len(sim_state.body_qd))
        wp.launch_tiled(self.nt_kernel_tile_body_f, dim=(num_rollout_worlds,),
                        inputs=[sim_state.body_f],
                        outputs=[rollout_state.body_f],
                        block_dim=len(sim_state.body_f))

    @property
    def time(self) -> float:
        """Returns the current simulation time."""
        return self.mj_data.time if self.mj_data else (self.nt_sim.sim_backend.sim_time if self.nt_sim else 0)

    @time.setter
    def time(self, value: float) -> None:
        """Sets the current simulation time."""
        if self.mj_data:
            self.mj_data.time = value
        else:
            self.nt_sim.sim_backend.sim_time = value

    @abstractmethod
    def reward(self,
               states: np.ndarray,
               sensors: np.ndarray,
               controls: np.ndarray,
               system_metadata: dict[str, Any] | None = None) -> np.ndarray:
        """Abstract reward function for task.

        Args:
            states: The rolled out states (after the initial condition). Shape=(num_rollouts, T, nq + nv).
            sensors: The rolled out sensors readings. Shape=(num_rollouts, T, total_num_sensor_dims).
            controls: The rolled out controls. Shape=(num_rollouts, T, nu).
            config: The current task config (passed in from the top-level controller).
            system_metadata: Any additional metadata from the system that is useful for computing the reward. For
                example, in the cube rotation task, the system could pass in new goal cube orientations to the
                controller here.

        Returns:
            rewards: The reward for each rollout. Shape=(num_rollouts,).
        """

    def nt_reward(self,
                  states: list[newton.State],
                  contacts: list[newton.Contacts],
                  controls: list[newton.Control],
                  system_metadata: dict[str, Any] | None = None) -> np.ndarray:
        """Abstract Newton reward function for task.

        Args:
            states: list[newton.State], The rolled out states (after the initial condition). Shape=(T, state(num_rollouts,)).
            contacts: list[newton.Contacts], The rolled out contact readings. Shape=(num_rollouts, T, total_num_sensor_dims).
            controls: list[newton.Control], The rolled out controls. Shape=(num_rollouts, T, nu).
            config: The current task config (passed in from the top-level controller).
            system_metadata: Any additional metadata from the system that is useful for computing the reward. For
                example, in the cube rotation task, the system could pass in new goal cube orientations to the
                controller here.

        Returns:
            rewards: The reward for each rollout. Shape=(num_rollouts,).
        """
        return None

    @property
    def nu(self) -> int:
        """Number of control inputs. The same as the MjModel for this task."""
        return self.mj_model.nu if self.mj_model else self.nt_sim_model.joint_dof_count

    @property
    def actuator_ctrlrange(self) -> np.ndarray:
        if self.mj_model:
            """Mujoco actuator limits for this task."""
            limits = self.mj_model.actuator_ctrlrange
            limited: np.ndarray = self.mj_model.actuator_ctrllimited.astype(bool)  # type: ignore
            limits[~limited] = np.array([-np.inf, np.inf], dtype=limits.dtype)  # if not limited, set to inf
        else:
            assert self.nt_sim_model
            """Newton actuator limits for this task."""
            limits = []
            for i in range(self.nt_sim_model_builder.joint_dof_count):
                limits.append([self.nt_sim_model_builder.joint_limit_lower[i],
                               self.nt_sim_model_builder.joint_limit_upper[i]])
            limits = np.array(limits)

        return limits  # type: ignore

    def reset(self) -> None:
        """Reset behavior for task. Sets config + velocities to zeros."""
        if self.mj_model:
            self.mj_data.qpos = np.zeros_like(self.mj_data.qpos)
            self.mj_data.qvel = np.zeros_like(self.mj_data.qvel)
            mujoco.mj_forward(self.mj_model, self.mj_data)
        else:
            self.nt_sim.reset()

    @property
    def dt(self) -> float:
        """Returns Mujoco physics timestep for default physics task."""
        return self.mj_model.opt.timestep if self.mj_model else (self.nt_sim.sim_backend.sim_dt if self.nt_sim else 0)

    def pre_rollout(self, curr_state: np.ndarray) -> None:
        """Pre-rollout behavior for task (does nothing by default).

        Args:
            curr_state: Current state of the task. Shape=(nq + nv,).
        """

    def post_rollout(
            self,
            states: np.ndarray,
            sensors: np.ndarray,
            controls: np.ndarray,
            system_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Post-rollout behavior for task (does nothing by default).

        Same inputs as in reward function.
        """

    def pre_sim_step(self) -> None:
        """Pre-simulation step behavior for task (does nothing by default)."""

    def post_sim_step(self) -> None:
        """Post-simulation step behavior for task (does nothing by default)."""

    def get_sim_metadata(self) -> dict[str, Any]:
        """Returns metadata from the simulation.

        We need this function because the simulation thread runs separately from the controller thread, but there are
        task objects in both. This function is used to pass information about the simulation's version of the task to
        the controller's version of the task.

        For example, the LeapCube task has a goal quaternion that is updated in the simulation thread based on whether
        the goal was reached (which the controller thread doesn't know about). When a new goal is set, it must be passed
        to the controller thread via this function.
        """
        return {}

    def optimizer_warm_start(self) -> np.ndarray:
        """Returns a warm start for the optimizer.

        This is used to provide an initial guess for the optimizer when optimizing the task before any iterations.
        """
        return np.zeros(self.nu)

    def get_sensor_start_index(self, sensor_name: str) -> int:
        """Returns the starting index of a sensor in the 'sensors' array given the sensor's name.

        Args:
            sensor_name: The name of the sensor to get the index of.
        """
        return self.mj_model.sensor(sensor_name).adr[0]

    def get_joint_position_start_index(self, joint_name: str) -> int:
        """Returns the starting index of a joint's position in the 'states' array given the joint's name.

        Args:
            joint_name: The name of the joint to get the starting index in the position of the state array.
        """
        return self.mj_model.jnt_qposadr[self.mj_model.joint(joint_name).id]

    def get_joint_velocity_start_index(self, joint_name: str) -> int:
        """Returns the starting index of a joint's velocity in the 'states' array given the joint's name.

        NOTE: This is the index of the joint's velocity in the state array, which is after the position indices!

        Args:
            joint_name: The name of the joint to get the starting index in the state array of.
        """
        return self.mj_model.nq + self.mj_model.jnt_dofadr[self.mj_model.joint(joint_name).id]
