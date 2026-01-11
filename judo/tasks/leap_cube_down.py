# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig

if TYPE_CHECKING:
    from judo.simulation.base import Simulation


@slider("w_pos", 0.0, 200.0)
@slider("w_rot", 0.0, 1.0)
@dataclass
class LeapCubeDownConfig(LeapCubeConfig):
    """Reward configuration LEAP cube rotation task."""

    task_name: str = "leap_cube_down"
    xml_path = str(MODEL_PATH / "xml/leap_cube_palm_down.xml")
    sim_xml_path = str(MODEL_PATH / "xml/leap_cube_palm_down_sim.xml")
    qpos_home: np.ndarray = field(default_factory=lambda: np.array(
        [
            -0.04, -0.035, -0.065, 1.0, 0.0, 0.0, 0.0,  # cube
            1.0, 0.0, 0.8, 0.8,  # index
            1.0, 0.0, 0.8, 0.8,  # middle
            1.0, 0.0, 0.8, 0.8,  # ring
            1.0, 1.0, 0.4, 0.9,  # thumb
        ]
    ))  # fmt: skip
    reset_command: np.ndarray = field(default_factory=lambda: np.array(
        [
            1.0, 0.0, 0.8, 0.8,  # index
            1.0, 0.0, 0.8, 0.8,  # middle
            1.0, 0.0, 0.8, 0.8,  # ring
            1.0, 1.0, 0.4, 0.9,  # thumb
        ]
    ))  # fmt: skip
    goal_pos: np.ndarray = field(default_factory=lambda: np.array([-0.04, -0.035, -0.065]))
    goal_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    w_rot: float = 0.05


class LeapCubeDown(LeapCube):
    """Defines the LEAP cube with palm down rotation task."""

    config_t: type[LeapCubeDownConfig] = LeapCubeDownConfig

    def __init__(self, sim: Optional[Simulation] = None, num_rollout_worlds: int = 1) -> None:
        """Initializes the LEAP cube rotation task."""
        super().__init__(sim, num_rollout_worlds=num_rollout_worlds)
