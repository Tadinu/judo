# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Literal, Union, Optional
import enum

import newton
import mujoco as mj
import numpy as np
import warp as wp
from omegaconf import DictConfig
from scipy.interpolate import interp1d

# judo
from judo import BackendType
from judo.app.structs import MujocoState, SplineData
from judo.app.utils import register_optimizers_from_cfg, register_tasks_from_cfg
from judo.config import OverridableConfig
from judo.gui import slider
from judo.optimizers import Optimizer, OptimizerConfig, get_registered_optimizers
from judo.tasks import Task, TaskConfig, get_registered_tasks
from judo.utils.mujoco import MJRolloutBackend, mj_make_model_data_pairs
from judo.utils.mujoco_warp import MJWarpBackend
from judo.utils.normalization import (
    IdentityNormalizer,
    Normalizer,
    NormalizerType,
    make_normalizer,
    normalizer_registry,
)
from judo.visualizers.utils import get_mj_trace_sensors
from judo.simulation.mj_simulation import MJSimulation
from judo.simulation.nt_simulation import NTSimulation
from judo.utils.newton import NewtonBackend

# mjmanip
from mjmanip.robot.arm_hand import ArmHandDiffIK


class SplineType(enum.Enum):
    ZERO = enum.auto()
    LINEAR = enum.auto()
    QUADRATIC = enum.auto()
    CUBIC = enum.auto()


@slider("horizon", 0.1, 10.0, bounded=True)
@slider("control_freq", 0.25, 50.0)
@dataclass
class MPControllerConfig(OverridableConfig):
    """Base controller config."""

    horizon: float = 1.0
    spline_order: str = SplineType.LINEAR.name
    control_freq: float = 20.0
    max_opt_iters: int = 1
    max_num_traces: int = 5
    action_normalizer: str = NormalizerType.NONE.name


class MPController:
    """The controller object."""

    def __init__(
            self,
            controller_config: MPControllerConfig,
            task: Task,
            optimizer: Optimizer,
            rollout_backend: BackendType = BackendType.MUJOCO,
    ) -> None:
        """Initialize the controller.

        Args:
            controller_config: The controller configuration.
            task: The task to use.
            optimizer: The optimizer to use.
            rollout_backend: The backend to use for rollouts. Currently only BackendType.MUJOCO is supported.
        """
        self._controller_cfg = controller_config
        self.task = task
        self.optimizer = optimizer

        self.available_optimizers = get_registered_optimizers()
        self.available_tasks = get_registered_tasks()

        # Newton
        is_newton = rollout_backend == BackendType.NEWTON
        self.nt_rollout_backend = NewtonBackend(task.config.joint_names, num_substeps=20,
                                                for_rollout=True,
                                                rollout_timesteps=self.num_timesteps) if is_newton else None
        if self.nt_rollout_backend:
            assert task.nt_num_rollout_worlds == self.optimizer_cfg.num_rollouts
            self.nt_rollout_backend.set_model(self.task.nt_rollout_model, self.task.nt_rollout_model_builder)
        else:
            assert self.task.mj_model, f"MuJoCo model must be valid for the controller with rollout backend {rollout_backend}"

        # MuJoCo
        self.mj_model = self.task.mj_model if (rollout_backend == BackendType.MUJOCO or
                                               rollout_backend == BackendType.MUJOCO_WARP) else None
        self.mj_model_data_pairs = mj_make_model_data_pairs(self.mj_model,
                                                            self.optimizer_cfg.num_rollouts) \
            if rollout_backend == BackendType.MUJOCO else None
        self.mj_state_type = self.task.mj_state_type

        # - MJ-C backend
        self.mj_rollout_backend = MJRolloutBackend(num_threads=self.optimizer_cfg.num_rollouts, backend=rollout_backend) \
            if rollout_backend == BackendType.MUJOCO else None

        # - MJ-Warp backend
        self.mjw_rollout_backend = MJWarpBackend(rollout_timesteps=self.num_timesteps,
                                                 state_type=self.mj_state_type) \
            if rollout_backend == BackendType.MUJOCO_WARP else None
        if self.mjw_rollout_backend:
            self.mjw_rollout_backend.set_model(self.mj_model, self.task.mjw_model, self.task.mjw_data)

        # - MJ States
        # Shape:
        # + [MUJOCO]: (self.optimizer_cfg.num_rollouts, self.num_timesteps, self.mj_model.nq + self.mj_model.nv)
        # + [MUJOCO_WARP]: (self.optimizer_cfg.num_rollouts, self.num_timesteps, mj.mj_stateSize(self.mj_model, state_type))
        self.mj_states: np.ndarray = None
        self.mj_current_state: np.ndarray = None

        # - MJ Sensors
        # + [MUJOCO]: (self.optimizer_cfg.num_rollouts, self.num_timesteps, self.mj_model.nsensordata)
        # + [MUJOCO_WARP]: (self.optimizer_cfg.num_rollouts, self.num_timesteps, self.mjw_rollout_backend.mjw_model.nsensordata)
        self.mj_sensors: np.ndarray = None

        # Controls
        self.rollout_controls = np.zeros((self.optimizer_cfg.num_rollouts, self.num_timesteps, self.mj_model.nu)) \
            if self.mj_model else None

        # Action (must be after rollout backend init)
        self.action_normalizer = self._init_action_normalizer()

        # A container for any metadata from the system that we want to pass to the task
        self.system_metadata = {}

        # Rewards
        self.rewards = np.zeros((self.optimizer_cfg.num_rollouts,))
        self.reset()

        # Traces
        self.traces = None
        self.mj_trace_sensors = get_mj_trace_sensors(self.mj_model) if self.mj_model else None
        self.num_trace_elites = min(self.max_num_traces, len(self.rewards)) if self.mj_model else 0
        self.num_trace_sensors = len(self.mj_trace_sensors) if self.mj_model else 0
        self.sensor_rollout_size = self.num_timesteps - 1
        self.all_traces_rollout_size = self.sensor_rollout_size * self.num_trace_sensors

    @property
    def nt_rollout_model(self) -> newton.Model:
        return self.task.nt_rollout_model

    @property
    def nt_rollout_model_builder(self) -> newton.ModelBuilder:
        return self.task.nt_rollout_model_builder

    @property
    def horizon(self) -> float:
        """Helper function to recalculate the horizon for simulation."""
        return self.controller_cfg.horizon

    @property
    def nu(self) -> int:
        """Helper function to get the number of control inputs."""
        return self.task.nu

    @property
    def max_num_traces(self) -> int:
        """Helper function to recalculate the max number of traces for simulation."""
        return self.controller_cfg.max_num_traces

    @property
    def max_opt_iters(self) -> int:
        """Helper function to recalculate the max number of optimization iterations for simulation."""
        return self.controller_cfg.max_opt_iters

    @property
    def spline_order(self) -> SplineType:
        """Helper function to recalculate the spline order for simulation."""
        return SplineType[self.controller_cfg.spline_order]

    @property
    def nominal_spline_data(self) -> SplineData:
        """Helper function to get the spline data."""
        return SplineData(self.times, self.nominal_knots)

    @property
    def action_normalizer_type(self) -> NormalizerType:
        """Helper function to get the type of action normalizer."""
        return NormalizerType[self.controller_cfg.action_normalizer]

    @property
    def num_timesteps(self) -> int:
        """Helper function to recalculate the number of timesteps for simulation."""
        # return 50
        return np.ceil(self.horizon / self.task.dt).astype(int)

    @property
    def rollout_times(self) -> np.ndarray:
        """Helper function to calculate the rollout times based on the horizon length."""
        return self.task.dt * np.arange(self.num_timesteps)

    @property
    def spline_timesteps(self) -> np.ndarray:
        """Helper function to create new timesteps for spline queries."""
        return np.linspace(0, self.horizon, self.optimizer_cfg.num_nodes, endpoint=True)

    @property
    def optimizer_cfg(self) -> OptimizerConfig:
        """Helper function to get the optimizer config."""
        return self.optimizer.config

    @optimizer_cfg.setter
    def optimizer_cfg(self, optimizer_cfg: OptimizerConfig) -> None:
        """Helper function to set the optimizer config."""
        self.optimizer.config = optimizer_cfg

    @property
    def optimizer_cls(self) -> type:
        """Returns the optimizer class."""
        return self.optimizer.__class__

    @property
    def optimizer_config_cls(self) -> type:
        """Returns the optimizer config class."""
        return self.optimizer.config.__class__

    @property
    def task_config(self) -> TaskConfig:
        """Returns the task config, which is uniquely defined by the task."""
        return self.task.config

    @task_config.setter
    def task_config(self, task_cfg: TaskConfig) -> None:
        """Sets the task config."""
        self.task.config = task_cfg

    @property
    def time(self) -> float:
        """Returns the current simulation time."""
        return self.task.time

    @time.setter
    def time(self, value: float) -> None:
        """Sets the current simulation time."""
        self.task.time = value

    @property
    def controller_cfg(self) -> MPControllerConfig:
        """Returns the controller config."""
        return self._controller_cfg

    @controller_cfg.setter
    def controller_cfg(self, controller_cfg: MPControllerConfig) -> None:
        """Sets the controller config."""
        self._controller_cfg = controller_cfg
        self.action_normalizer = self._init_action_normalizer()

    def update_action(self) -> None:
        """Abstract method for updating controller actions from current state/time."""
        assert self.optimizer_cfg.num_rollouts > 0, "Need at least one rollout!"

        if self.optimizer_cfg.num_nodes < 4 and self.spline_order == SplineType.CUBIC:
            warnings.warn("Cubic splines require at least 4 nodes. Setting num_nodes=4.", stacklevel=2)
            self.optimizer_cfg.num_nodes = 4

        # Adjust time + move policy forward.
        new_times = self.time + self.spline_timesteps
        nominal_knots = self.nominal_spline(new_times)
        nominal_knots_normalized = self.action_normalizer.normalize(nominal_knots)

        # resizing any variables due to changes in the GUI
        if self.mjw_rollout_backend:
            if self.task.mjw_data.nworld != self.optimizer_cfg.num_rollouts:
                self.task.mjw_init_data(self.optimizer_cfg.num_rollouts)
                self.mjw_rollout_backend.set_model(self.mj_model, self.task.mjw_model, self.task.mjw_data)
        elif self.mj_model:
            state_size = mj.mj_stateSize(self.mj_model, self.mj_state_type)
            current_state_size = len(self.mj_current_state)
            assert current_state_size == state_size, \
                (f"[MuJoCo-C Rollout backend]: Current state's size {current_state_size} does not match"
                 f"size {state_size} of the configured state type {self.mj_state_type}.")
            if len(self.mj_model_data_pairs) != self.optimizer_cfg.num_rollouts:
                self.mj_model_data_pairs = mj_make_model_data_pairs(self.mj_model, self.optimizer_cfg.num_rollouts)
                self.mj_rollout_backend.update(self.optimizer_cfg.num_rollouts)
        else:
            assert self.nt_rollout_model
            if self.nt_rollout_model_builder.num_worlds != self.optimizer_cfg.num_rollouts:
                self.task.nt_init_models(self.optimizer_cfg.num_rollouts)
                if self.nt_rollout_backend.model != self.nt_rollout_model:
                    self.nt_rollout_backend.set_model(self.nt_rollout_model, self.nt_rollout_model_builder)

        normalizer_cls = normalizer_registry.get(self.action_normalizer_type)
        if normalizer_cls is None:
            warnings.warn(
                f"Invalid action normalizer type '{self.action_normalizer_type}'. "
                f"Available types: {list(normalizer_registry.keys())}. "
                "Falling back to 'none' normalizer.",
                stacklevel=2,
            )
            normalizer_cls = IdentityNormalizer

        # force the normalizer to be re-initialized when the type changes in GUI
        # TODO(yunhai): check for changes in the normalizer config and update when appropriate
        if not isinstance(self.action_normalizer, normalizer_cls):
            self.action_normalizer = self._init_action_normalizer()

        # call entrypoint prior to optimization
        self.optimizer.pre_optimization(self.times, new_times)

        # run optimization loop
        i = 0
        while i < self.max_opt_iters and not self.optimizer.stop_cond():
            # sample controls and clamp to action bounds
            candidate_knots_normalized = self.optimizer.sample_control_knots(nominal_knots_normalized)
            candidate_knots_normalized = np.clip(
                candidate_knots_normalized,
                self.action_normalizer.normalize(self.task.actuator_ctrlrange[:, 0]),
                self.action_normalizer.normalize(self.task.actuator_ctrlrange[:, 1]),
            )
            self.candidate_knots = self.action_normalizer.denormalize(candidate_knots_normalized)

            # Evaluate rollout controls at sim timesteps.
            candidate_spline = make_spline(new_times, self.candidate_knots, self.spline_order)
            self.rollout_controls = candidate_spline(self.time + self.rollout_times)

            # Map algo's candidate controls to final ones that match model's configuration space.
            # Eg: Jacobian map to transform EE vel to joint vels, PCA map to transform PCA grasp values to finger joints
            if self.task.map_controls:
                for model, data in self.mj_model_data_pairs:
                    mj.mj_setState(model, data, self.mj_current_state, self.mj_state_type)
                self.rollout_controls = self.task.map_controls(self.mj_model_data_pairs, self.rollout_controls,
                                                               self.mj_current_state)

            # Roll out dynamics with action sequences.
            if self.mj_model:
                self.task.pre_rollout(self.mj_current_state)
                if self.mjw_rollout_backend:
                    self.mj_states, self.mj_sensors = self.mjw_rollout_backend.rollout(self.mj_current_state,
                                                                                       self.mj_state_type,
                                                                                       self.rollout_controls)
                else:
                    self.mj_states, self.mj_sensors = self.mj_rollout_backend.rollout(self.mj_model_data_pairs,
                                                                                      self.mj_current_state,
                                                                                      self.rollout_controls)
                self.task.post_rollout(
                    self.mj_states,
                    self.mj_sensors,
                    self.rollout_controls,
                    self.system_metadata,
                )
                self.rewards = self.task.reward(
                    self.mj_states,
                    self.mj_sensors,
                    self.rollout_controls,
                    self.system_metadata,
                )
            else:
                # [self.rollout_controls] -> [self.nt_rollout_backend.rollout_model_ctrls' joint_target_pos]
                for i in range(len(self.nt_rollout_backend.rollout_model_ctrls)):
                    control = self.nt_rollout_backend.rollout_model_ctrls[i]
                    control.joint_target_pos = wp.array(np.hstack(self.rollout_controls[:, i, :].squeeze()),
                                                        dtype=control.joint_target_pos.dtype,
                                                        device=control.joint_target_pos.device)

                # Rollout backend with updated [joint_target_controls]
                self.nt_rollout_backend.step()

                # Get rollout rewards
                self.rewards = self.task.nt_reward(
                    self.nt_rollout_backend.rollout_states,
                    self.nt_rollout_backend.rollout_contacts,
                    self.nt_rollout_backend.rollout_model_ctrls,
                    self.system_metadata,
                )

            # Update nominal knots for next optimization iteration
            nominal_knots_normalized = self.optimizer.update_nominal_knots(candidate_knots_normalized, self.rewards)

            # Update action normalizer
            self.action_normalizer.update(self.candidate_knots)

            i += 1

        # Update nominal controls and spline.
        self.nominal_knots = self.action_normalizer.denormalize(nominal_knots_normalized)
        self.times = new_times
        self.update_optimal_spline(self.times, self.nominal_knots)
        self.update_traces()

    def action(self, time: float) -> np.ndarray:
        """Current best action of policy."""
        return self.nominal_spline(time)

    def update_optimal_spline(self, times: np.ndarray, nominal_controls: np.ndarray) -> None:
        """Update the spline with new timesteps / controls."""
        self.nominal_spline = make_spline(times, nominal_controls, self.spline_order)

    def reset(self) -> None:
        """Reset the controls, candidate controls and the spline to their default values."""
        self.task.reset()
        if self.optimizer_cfg.num_nodes < 4 and self.spline_order == SplineType.CUBIC:
            warnings.warn("Cubic splines require at least 4 nodes. Setting num_nodes=4.", stacklevel=2)
            self.optimizer_cfg.num_nodes = 4
        self.nominal_knots = np.tile(self.task.optimizer_warm_start(), (self.optimizer_cfg.num_nodes, 1))
        self.candidate_knots = np.tile(self.nominal_knots, (self.optimizer_cfg.num_rollouts, 1, 1))
        self.times = self.task.time + self.spline_timesteps
        self.update_optimal_spline(self.times, self.nominal_knots)

    def update_traces(self) -> None:
        """Update traces by extracting data from sensors readings.

        We need to have num_spline_points - 1 line segments. Sensors will initially be of shape
        (num_rollout x num_timesteps x nsensordata) and needs to end up being in shape
        (num_elite * num_trace_sensors * size of a single rollout x 2 (first and last point of spline) x 3 (3d pos))
        """
        # Resize traces if forced by config change.
        if self.mj_sensors is not None:
            self.sensor_rollout_size = self.num_timesteps - 1
            self.all_traces_rollout_size = self.sensor_rollout_size * self.num_trace_sensors
            new_num_rollouts = min(self.max_num_traces, self.optimizer_cfg.num_rollouts)
            if self.num_trace_elites != new_num_rollouts:
                self.num_trace_elites = new_num_rollouts
            sensors = np.repeat(self.mj_sensors, 2, axis=1)

            # Order the actions from best to worst so that the first `num_trace_sensors` x `num_nodes` traces
            # correspond to the best rollout and are using a special colors
            elite_actions = np.argsort(self.rewards)[-self.num_trace_elites:][::-1]

            total_traces_rollouts = int(self.num_trace_elites * self.num_trace_sensors * self.sensor_rollout_size)
            # Calculates list of the elite indicies
            trace_inds = [self.mj_model.sensor_adr[id] + pos for id in self.mj_trace_sensors for pos in range(3)]

            # Filter out the non-elite indices we don't care about
            sensors = sensors[elite_actions, :, :]
            # Remove everything but the trace sensors we care about, leaving htis column as size num_trace_sensors * 3
            sensors = sensors[:, :, trace_inds]
            # Remove the first and last part the trajectory to form line segments properly
            # Array will be doubled and look something like: [(0, 0), (1, 1), (4, 4)]
            # We want it to look like: [(0, 1), (1, 4)]
            sensors = sensors[:, 1:-1, :]

            # We doubled it so the number of entries is going to be the size of the rollout * 2
            separated_sensors_size = (self.num_trace_elites, self.sensor_rollout_size, 2, 3)

            # Each block of (i, self.sensor_rollout_size) needs to be interleaved together into a stack of
            # [block(i, ), block (i + 1, ), ..., block(i + n)]
            elites = np.zeros((self.num_trace_sensors * self.num_trace_elites, self.sensor_rollout_size, 2, 3))
            for sensor in range(self.num_trace_sensors):
                s1 = np.reshape(sensors[:, :, sensor * 3: (sensor + 1) * 3], separated_sensors_size)
                elites[sensor:: self.num_trace_sensors] = s1
            self.traces = np.reshape(elites, (total_traces_rollouts, 2, 3))

    def update_state(self, state: Union[MujocoState, np.ndarray, newton.State]) -> None:
        """Updates the states."""
        if self.mjw_rollout_backend:
            assert isinstance(state, np.ndarray)
            assert len(state) == mj.mj_stateSize(self.mj_model, self.mj_state_type)
            self.mj_current_state = state
        elif self.mj_model:
            assert isinstance(state, MujocoState)
            self.mj_current_state = state.data
            self.time = state.time
            self.system_metadata = state.sim_metadata
        else:
            # Write sim-backend's state -> rollout-backend's state
            assert isinstance(state, newton.State)
            self.task.nt_copy_sim_to_rollout_state(sim_state=state, rollout_state=self.nt_rollout_backend.state_0)

    def _init_action_normalizer(self) -> Normalizer:
        """Initialize the action normalizer."""
        action_normalizer_kwargs = {}
        match self.action_normalizer_type:
            case NormalizerType.MIN_MAX:
                action_normalizer_kwargs["min"] = self.task.actuator_ctrlrange[:, 0]
                action_normalizer_kwargs["max"] = self.task.actuator_ctrlrange[:, 1]
            case NormalizerType.RUNNING:
                action_normalizer_kwargs["init_std"] = 1.0  # TODO(yunhai): make this configurable
        return make_normalizer(self.action_normalizer_type, self.nu, **action_normalizer_kwargs)


def make_spline(times: np.ndarray, controls: np.ndarray, spline_order: SplineType) -> interp1d:
    """Helper function for creating spline objects.

    Args:
        times: array of times for knot points, shape (T,).
        controls: (possibly batched) array of controls to interpolate, shape (..., T, m).
        spline_order: Order to use for interpolation. Same as parameter for scipy.interpolate.interp1d.
        extrapolate: Whether to allow extrapolation queries. Default true (for re-initialization).
    """
    # fill values for "before" and "after" spline extrapolation.
    fill_value = (controls[..., 0, :], controls[..., -1, :])
    return interp1d(
        times,
        controls,
        kind=spline_order.name.lower(),
        axis=-2,
        copy=False,
        fill_value=fill_value,  # interp1d is incorrectly typed # type: ignore
        bounds_error=False,
    )


def make_controller(
        sim: Union[MJSimulation, NTSimulation],
        init_task: Union[Task, str],
        init_optimizer: str,
        task_registration_cfg: Optional[DictConfig] = None,
        optimizer_registration_cfg: Optional[DictConfig] = None,
        rollout_backend: BackendType = BackendType.MUJOCO,
) -> MPController:
    """Make a controller."""
    available_optimizers = get_registered_optimizers()
    available_tasks = get_registered_tasks()
    if task_registration_cfg is not None:
        register_tasks_from_cfg(task_registration_cfg)
    if optimizer_registration_cfg is not None:
        register_optimizers_from_cfg(optimizer_registration_cfg)

    task_entry = available_tasks.get(init_task)
    optimizer_entry = available_optimizers.get(init_optimizer)
    assert optimizer_entry is not None, f"Optimizer {init_optimizer} not found in optimizer registry."

    # instantiate the task/optimizer/controller
    if isinstance(init_task, Task):
        task = init_task
    else:
        assert task_entry is not None, f"Task {init_task} not found in task registry."
        task_cls, _ = task_entry
        task = task_cls(sim)
    print("Task:", task.config)
    task_name = task.config.task_name

    optimizer_cls, optimizer_config_cls = optimizer_entry
    # Refer to optimizers/overrides.py for task-specific configs
    optimizer = optimizer_cls(optimizer_config_cls(), task.nu, override_task_name=task_name)
    print("Optimizer:", optimizer.config)

    controller_cfg = MPControllerConfig()
    controller_cfg.set_override(task_name)
    print("Controller:", controller_cfg)

    return MPController(
        controller_config=controller_cfg,
        task=task,
        optimizer=optimizer,
        rollout_backend=rollout_backend,
    )
