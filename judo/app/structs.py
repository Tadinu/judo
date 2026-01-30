# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal

import numpy as np
import torch
from scipy.interpolate import interp1d

# judo
from judo.utils.interp1d_torch import Interp1dTorch


class EventType(Enum):
    """Enum for event types."""

    START_SIMULATION = auto()
    PAUSE_SIMULATION = auto()
    START_CONTROLLER = auto()
    PAUSE_CONTROLLER = auto()
    CHANGE_TASK = auto()
    CHANGE_CONTROLLER = auto()


@dataclass
class JudoEvent:
    """Struct for judo events."""

    event: EventType
    value: str | None = None


@dataclass
class MujocoState:
    """Struct for writing simulation states between different threads."""

    time: float
    data: np.ndarray  # Result of mj.mj_getState()
    sim_metadata: dict[str, Any]


KindType = Literal[
    "linear",
    "nearest",
    "nearest-up",
    "zero",
    "linear",
    "quadratic",
    "cubic",
    "previous",
    "next",
]


@dataclass
class SplineData:
    """Struct for (possibly batched) spline data."""

    t: torch.Tensor
    """array of times for knot points, shape (T,)"""
    x: torch.Tensor
    """(possibly batched) array of values to interpolate, shape (..., T, m)."""
    kind: KindType = "zero"
    """Spline type to use for interpolation. Same as parameter for scipy.interpolate.interp1d."""
    extrapolate: bool = True
    """Flag for whether to allow extrapolation queries. Default true (for re-initialization)."""

    @property
    def spline(self) -> Interp1dTorch:
        """Helper function for creating spline objects."""
        # fill values for "before" and "after" spline extrapolation.
        fill_value = (self.x[..., 0, :], self.x[..., -1, :])

        return Interp1dTorch(
            x=self.t[..., None, None].squeeze(),
            y=self.x[..., None, None].squeeze()
        )
