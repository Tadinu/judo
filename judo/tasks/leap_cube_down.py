# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path

import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig


@slider("w_pos", 0.0, 200.0)
@slider("w_rot", 0.0, 1.0)
@dataclass
class LeapCubeDownConfig(LeapCubeConfig):
    """Reward configuration LEAP cube rotation task."""

    task_name: str = "leap_cube_down"
    xml_path = str(MODEL_PATH / "xml/leap_cube_palm_down.xml")
    sim_xml_path = str(MODEL_PATH / "xml/leap_cube_palm_down_sim.xml")
    qpos_home = np.array(
        [
            -0.04, -0.035, -0.065, 1.0, 0.0, 0.0, 0.0,  # cube
            1.0, 0.0, 0.8, 0.8,  # index
            1.0, 0.0, 0.8, 0.8,  # middle
            1.0, 0.0, 0.8, 0.8,  # ring
            1.0, 1.0, 0.4, 0.9,  # thumb
        ]
    )  # fmt: skip

    w_rot: float = 0.05


class LeapCubeDown(LeapCube):
    """Defines the LEAP cube with palm down rotation task."""

    config_t: type[LeapCubeDownConfig] = LeapCubeDownConfig

    def __init__(self) -> None:
        """Initializes the LEAP cube rotation task."""
        super().__init__()
        self.goal_pos = np.array([-0.04, -0.035, -0.065])
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.qpos_home = self.config.qpos_home
        self.reset_command = np.array(
            [
                1.0, 0.0, 0.8, 0.8,  # index
                1.0, 0.0, 0.8, 0.8,  # middle
                1.0, 0.0, 0.8, 0.8,  # ring
                1.0, 1.0, 0.4, 0.9,  # thumb
            ]
        )  # fmt: skip
        self.reset()
