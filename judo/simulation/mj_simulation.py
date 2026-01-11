# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Optional, Union
from mujoco import mj_step
from omegaconf import DictConfig

import numpy as np
import mujoco as mj

from judo import BackendType
from judo.app.structs import MujocoState
from judo.simulation.base import Simulation


class MJSimulation(Simulation):
    """Mujoco simulation object.

    This class contains the data required to run a Mujoco simulation. This includes configurations, a control spline,
    and task information.

    Middleware nodes should instantiate this class and implement methods to send, process, and receive data.
    """

    def __init__(
            self,
            init_task: str = "cylinder_push",
            num_rollout_worlds: int = 1,
            task_registration_cfg: Optional[DictConfig] = None,
    ) -> None:
        """Initialize the simulation node."""
        super().__init__(init_task=init_task, num_rollout_worlds=num_rollout_worlds,
                         task_registration_cfg=task_registration_cfg)

    def step(self) -> None:
        """Step the simulation forward by one timestep."""
        if self.nominal_control_spline is not None and not self.paused:
            try:
                self.task.mj_data.ctrl[:] = self.nominal_control_spline(self.task.mj_data.time)[:self.task.nu]
                self.task.pre_sim_step()
                mj_step(self.task.mj_sim_model, self.task.mj_data)
                self.task.post_sim_step()
            except ValueError:
                # we're switching tasks and the new task has a different number of actuators
                pass

    @property
    def sim_state(self) -> Union[MujocoState, np.ndarray]:
        """Returns the current simulation state."""
        backend_type = self.task.config.sim_backend_type()
        if backend_type == BackendType.MUJOCO_WARP:
            state_type = mj.mjtState.mjSTATE_PHYSICS
            current_state = np.zeros(mj.mj_stateSize(self.task.mj_sim_model, state_type))
            mj.mj_getState(self.task.mj_sim_model, self.task.mj_data, current_state, state_type)
            return current_state
        else:
            assert backend_type == BackendType.MUJOCO
            data = self.task.mj_data
            return MujocoState(
                time=data.time,
                qpos=data.qpos,
                qvel=data.qvel,
                xpos=data.xpos,
                xquat=data.xquat,
                mocap_pos=data.mocap_pos,
                mocap_quat=data.mocap_quat,
                sim_metadata=self.task.get_sim_metadata(),
            )

    @property
    def timestep(self) -> float:
        """Returns the simulation timestep."""
        return self.task.mj_sim_model.opt.timestep
