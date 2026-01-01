# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from judo import BackendType
from judo.simulation.base import Simulation
from judo.simulation.mj_simulation import MJSimulation
from judo.simulation.nt_simulation import NTSimulation

simulation_registry = {
    BackendType.MUJOCO: MJSimulation,
    BackendType.NEWTON: NTSimulation,
}


def get_simulation_backend(simulation_backend: BackendType) -> type:
    """Get the simulation class for a given backend."""
    return simulation_registry[simulation_backend]


__all__ = [
    "get_simulation_backend",
    "Simulation",
    "MJSimulation",
    "NTSimulation",
]
