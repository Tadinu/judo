# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from __future__ import annotations
from typing import Any, Optional, Union, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig

if TYPE_CHECKING:
    from judo.simulation.base import Simulation


@slider("w_pos", 0.0, 200.0)
@slider("w_rot", 0.0, 1.0)
@dataclass
class CaltechLeapCubeConfig(LeapCubeConfig):
    """Reward configuration LEAP cube rotation task."""
    task_name: str = "caltech_leap_cube"
    xml_path: Optional[Union[Path, str]] = str(MODEL_PATH / "xml/caltech_leap_cube.xml")
    sim_xml_path: Optional[Union[Path, str]] = str(MODEL_PATH / "xml/caltech_leap_cube_sim.xml")
    qpos_home: Optional[np.ndarray] = field(default_factory=lambda: np.array(
        [
            0.11, 0.005, 0.04, 1.0, 0.0, 0.0, 0.0,  # cube
            0.5, -0.75, 0.75, 0.25,  # index
            0.5, 0.0, 0.75, 0.25,  # middle
            0.5, 0.75, 0.75, 0.25,  # ring
            0.65, 0.9, 0.75, 0.6,  # thumb
        ]
    ))  # fmt: skip
    reset_command: Optional[np.ndarray] = field(default_factory=lambda: np.array(
        [
            0.5, -0.75, 0.75, 0.25,  # index
            0.5, 0.0, 0.75, 0.25,  # middle
            0.5, 0.75, 0.75, 0.25,  # ring
            0.65, 0.9, 0.75, 0.6,  # thumb
        ]
    ))  # fmt: skip
    goal_pos: np.ndarray = field(default_factory=lambda: np.array([0.11, 0.005, 0.03]))
    goal_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))


class CaltechLeapCube(LeapCube):
    """Defines the LEAP cube rotation task."""

    config_t: type[CaltechLeapCubeConfig] = CaltechLeapCubeConfig

    def __init__(self, sim: Optional[Simulation] = None, num_rollout_worlds: int = 1) -> None:
        """Initializes the LEAP cube rotation task."""
        super().__init__(sim, num_rollout_worlds=num_rollout_worlds)
