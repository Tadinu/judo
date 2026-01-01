# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from abc import ABC, abstractmethod
from typing import Callable, Optional

from omegaconf import DictConfig

from judo.app.utils import register_tasks_from_cfg
from judo.tasks import get_registered_tasks
from judo.tasks.base import Task


class Simulation(ABC):
    """Base class for a simulation object.

    This class contains the data required to run a simulation. This includes configurations, a control spline, and task
    information. It can be inherited from to implement specific simulation backends.

    Middleware nodes should instantiate this class and implement methods to send, process, and receive data.
    """

    def __init__(
            self,
            init_task: str,
            num_rollout_worlds: int = 1,
            task_registration_cfg: Optional[DictConfig] = None,
    ) -> None:
        """Initialize the simulation node."""
        # handling custom task registration
        if task_registration_cfg is not None:
            register_tasks_from_cfg(task_registration_cfg)

        self.nominal_control_spline: Callable | None = None
        self.paused = False
        self.task: Task = self._create_task(init_task, num_rollout_worlds)

    def _create_task(self, task_name: str, num_rollout_worlds: int = 1) -> Task:
        """Helper to initialize task from task name."""
        task_entry = get_registered_tasks().get(task_name)
        if task_entry is None:
            raise ValueError(f"Init task {task_name} not found in task registry")

        task_cls, _ = task_entry
        task: Task = task_cls(self, num_rollout_worlds=num_rollout_worlds)
        task.reset()
        return task

    @abstractmethod
    def step(self) -> None:
        """Step the simulation forward by one timestep."""

    def pause(self) -> None:
        """Event handler for processing pause status updates."""
        self.paused = not self.paused

    def update_nominal_control_spline(self, control_spline: Callable) -> None:
        """Event handler for processing controls received from controller node."""
        self.nominal_control_spline = control_spline

    @property
    @abstractmethod
    def timestep(self) -> float:
        """Timestep the simulation expects to run at."""
