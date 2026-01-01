# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Callable

import numpy as np

from judo.simulation import NTSimulation
from judo.tasks.cartpole import Cartpole, CartpoleConfig
from judo.tasks.cylinder_push import CylinderPush, CylinderPushConfig


def test_simulation_data_basics() -> None:
    """Test the simulation data basics."""
    simulation_data = NTSimulation(init_task="cylinder_push")
    assert simulation_data.task is not None
    assert simulation_data.control_spline is None
    simulation_data.pause()
    assert simulation_data.paused


def test_simulation_data_update_task() -> None:
    """Test the simulation data update task."""
    simulation_data = NTSimulation(init_task="cylinder_push")
    assert isinstance(simulation_data.task.config, CylinderPushConfig)
    assert isinstance(simulation_data.task, CylinderPush)
    simulation_data.create_task("cartpole")
    assert simulation_data.task.nu == 1
    assert isinstance(simulation_data.task.config, CartpoleConfig)
    assert isinstance(simulation_data.task, Cartpole)


def test_simulation_data_step(temp_np_seed: Callable) -> None:
    """Test the simulation data step."""
    simulation_data = NTSimulation(init_task="cylinder_push")

    def mock_control(t: float) -> np.ndarray:
        return np.zeros(simulation_data.task.nu) + 1.234

    with temp_np_seed(42):
        original_ctrl = np.copy(simulation_data.task.mj_data.ctrl)
        simulation_data.update_nominal_control_spline(control_spline=mock_control)
        assert simulation_data.control_spline is not None
        simulation_data.step()
        assert not np.allclose(simulation_data.task.mj_data.ctrl, original_ctrl)
        assert simulation_data.sim_state is not None
