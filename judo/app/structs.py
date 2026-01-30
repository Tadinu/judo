# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal

import numpy as np
import torch

# judo
from judo.utils.spline import SplineType, TorchSpline


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
    def spline(self) -> TorchSpline:
        """Helper function for creating spline objects."""
        # fill values for "before" and "after" spline extrapolation.
        # fill_value = (self.x[..., 0, :], self.x[..., -1, :])
        return TorchSpline(
            self.t,
            self.x,
            SplineType.LINEAR if self.kind == "linear" else SplineType.CUBIC if self.kind == "cubic" else SplineType.ZERO,
            self.extrapolate
        )
