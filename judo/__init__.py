# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from pathlib import Path
import enum

PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PACKAGE_ROOT / "models"


class BackendType(enum.Enum):
    MUJOCO = enum.auto()
    MUJOCO_WARP = enum.auto()
    NEWTON = enum.auto()
    GENESIS = enum.auto()
