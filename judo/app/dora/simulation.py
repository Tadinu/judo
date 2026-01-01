# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Optional
import time
import warnings

from dora_utils.dataclasses import from_arrow, to_arrow
from dora_utils.node import DoraNode, on_event
from omegaconf import DictConfig

# Judo
from judo import BackendType
from judo.app.structs import SplineData
from judo.simulation import get_simulation_backend


class SimulationNode(DoraNode):
    """The simulation node."""

    def __init__(
            self,
            node_id: str = "simulation",
            init_task: str = "cylinder_push",
            max_workers: Optional[int] = None,
            task_registration_cfg: Optional[DictConfig] = None,
            simulation_backend: str = BackendType.NEWTON.name,
    ) -> None:
        """Initialize the simulation node."""
        super().__init__(node_id=node_id, max_workers=max_workers)
        sim_backend_cls = get_simulation_backend(BackendType[simulation_backend])
        self.sim = sim_backend_cls(init_task=init_task, task_registration_cfg=task_registration_cfg)
        self.write_states()

    @on_event("INPUT", "task")
    def update_task(self, event: dict) -> None:
        """Event handler for processing task updates."""
        new_task = event["value"].to_numpy(zero_copy_only=False)[0]
        self.sim.create_task(new_task)

    def spin(self) -> None:
        """Spin logic for the simulation node."""
        while True:
            start_time = time.time()
            self.parse_messages()
            self.sim.step()
            self.write_states()

            # Force simulation node to run at fixed rate specified by simulation timestep (specified in the model).
            dt_des = self.sim.timestep
            dt_elapsed = time.time() - start_time
            if dt_elapsed < dt_des:
                time.sleep(dt_des - dt_elapsed)
            else:
                warnings.warn(
                    f"Sim step {dt_elapsed:.3f} longer than desired step {dt_des:.3f}!",
                    stacklevel=2,
                )

    def write_states(self) -> None:
        """Reads data from simulation and writes to output topic."""
        arr, metadata = to_arrow(self.sim.sim_state)
        self.node.send_output("states", arr, metadata)

    @on_event("INPUT", "sim_pause")
    def set_paused_status(self, event: dict) -> None:
        """Event handler for processing pause status updates."""
        self.sim.pause()

    @on_event("INPUT", "task_reset")
    def reset_task(self, event: dict) -> None:
        """Resets the task."""
        self.sim.task.reset()

    @on_event("INPUT", "controls")
    def update_control(self, event: dict) -> None:
        """Event handler for processing controls received from controller node."""
        spline_data = from_arrow(event["value"], event["metadata"], SplineData)
        print("AAAAAAAA", spline_data)
        control = spline_data.nominal_spline()
        self.sim.update_nominal_control_spline(control)
