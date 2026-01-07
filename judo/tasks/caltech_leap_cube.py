# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig


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


class CaltechLeapCube(LeapCube):
    """Defines the LEAP cube rotation task."""

    config_t: type[CaltechLeapCubeConfig] = CaltechLeapCubeConfig

    def __init__(self) -> None:
        """Initializes the LEAP cube rotation task."""
        super().__init__()
        self.goal_pos = np.array([0.11, 0.005, 0.03])
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.qpos_home = self.config.qpos_home
        self.reset_command = np.array(
            [
                0.5, -0.75, 0.75, 0.25,  # index
                0.5, 0.0, 0.75, 0.25,  # middle
                0.5, 0.75, 0.75, 0.25,  # ring
                0.65, 0.9, 0.75, 0.6,  # thumb
            ]
        )  # fmt: skip
        self.reset()
