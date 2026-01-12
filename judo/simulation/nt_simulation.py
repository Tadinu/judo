from typing import Callable, Optional
from omegaconf import DictConfig

import numpy as np

# judo
from judo.simulation.base import Simulation
from judo.utils.newton import NewtonBackend

# Newton
import warp as wp
import newton


class NTSimulation(Simulation):
    """Newton simulation object.

    This class contains the data required to run a Newton simulation. This includes configurations, a control spline,
    and task information.

    Middleware nodes should instantiate this class and implement methods to send, process, and receive data.
    """

    def __init__(
            self,
            init_task: str,
            is_monkey_sim: bool = False,
            num_rollout_worlds: int = 1,
            num_substeps: int = 8,
            kernel_set_joint_targets: Optional[Callable] = None,
            task_registration_cfg: Optional[DictConfig] = None
    ) -> None:
        """Initialize the simulation node."""
        super().__init__(init_task=init_task, num_rollout_worlds=num_rollout_worlds,
                         task_registration_cfg=task_registration_cfg)

        # Init newton backend
        self.sim_backend = NewtonBackend(self.task.config.joint_names,
                                         num_substeps=num_substeps,
                                         for_rollout=False,
                                         headless=False)
        if kernel_set_joint_targets:
            self.sim_backend.kernel_set_joint_targets = kernel_set_joint_targets

        # Set backend's model as either sim or rollout model
        if is_monkey_sim:
            self.sim_backend.set_model(self.task.nt_rollout_model, self.task.nt_rollout_model_builder)
        else:
            self.sim_backend.set_model(self.task.nt_sim_model, self.task.nt_sim_model_builder)
        self.model = self.sim_backend.model
        self.model_builder = self.sim_backend.model_builder
        self.nominal_action: np.ndarray = None

        # Warm up to get stabilized sim initial state
        self.step()

    def step(self):
        """Step the simulation"""
        if not self.paused:
            self.task.pre_sim_step()
            # Prepare [sim_backend]'s [joint_target_controls]
            self.setup_joint_targets()
            self.sim_backend.step()
            self.task.post_sim_step()

    def setup_joint_targets(self):
        if self.nominal_control_spline is not None or self.nominal_action is not None:
            # NOTE: These are the same
            # print(self.nominal_action)
            # print(self.nominal_control_spline(self.sim_backend.sim_time))
            with wp.ScopedDevice(self.sim_backend.wp_device):
                if self.nominal_action is not None:
                    self.sim_backend.joint_target_controls.assign(
                        wp.array(np.full((self.model_builder.num_worlds, self.model_builder.joint_dof_count),
                                         self.nominal_action),
                                 dtype=wp.float32))

                else:
                    # Update target controls with [control_spline(self.sim_time)]
                    self.sim_backend.joint_target_controls.assign(
                        wp.array(np.full((self.model_builder.num_worlds, self.model_builder.joint_dof_count),
                                         self.nominal_control_spline(self.sim_backend.sim_time)),
                                 dtype=wp.float32))

    @property
    def sim_state(self) -> newton.State:
        """Returns the current simulation state."""
        return self.sim_backend.state_0

    @property
    def timestep(self) -> float:
        """Returns the simulation timestep."""
        return self.task.mj_sim_model.opt.timestep if self.task.mj_sim_model else self.sim_backend.sim_dt

    def reset(self):
        self.control_spline = None
        self.sim_backend.reset()

    def spin(self):
        viewer = self.sim_backend.viewer
        while viewer.is_running():
            if not viewer.is_paused():
                with wp.ScopedTimer("step", active=False):
                    self.step()

            with wp.ScopedTimer("render", active=False):
                self.sim_backend.render()

        # Close viewer
        viewer.close()
