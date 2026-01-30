# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.interpolate import interp1d

from judo.optimizers.base import Optimizer, OptimizerConfig
from judo.utils.interp1d_torch import Interp1dTorch


@dataclass
class CrossEntropyMethodConfig(OptimizerConfig):
    """Configuration for cross-entropy method."""

    sigma_min: float = 0.1
    sigma_max: float = 1.0
    num_elites: int = 2


class CrossEntropyMethod(Optimizer[CrossEntropyMethodConfig]):
    """The cross-entropy method."""

    def __init__(self, config: CrossEntropyMethodConfig, nu: int, override_task_name: Optional[str] = None) -> None:
        """Initialize cross-entropy method optimizer."""
        super().__init__(config, nu, override_task_name)
        num_nodes = config.num_elites
        self.sigma = ((self.sigma_min + self.sigma_max) / 2) * torch.ones((num_nodes, nu))

    @property
    def sigma_min(self) -> float:
        """Get the minimum sigma value."""
        return self.config.sigma_min

    @property
    def sigma_max(self) -> float:
        """Get the maximum sigma value."""
        return self.config.sigma_max

    @property
    def num_elites(self) -> int:
        """Get the number of elites."""
        return self.config.num_elites

    def pre_optimization(self, old_times: torch.Tensor, new_times: torch.Tensor) -> None:
        """Update sigma if the number of nodes has changed."""
        if len(self.sigma) != self.num_nodes:
            self.sigma = Interp1dTorch(
                x=old_times,
                y=self.sigma[:self.num_nodes, ...].view(-1, self.num_nodes),
            ).apply(new_times).view(self.num_nodes, -1)

    def sample_control_knots(self, nominal_knots: torch.Tensor) -> torch.Tensor:
        """Samples control knots given a nominal control input.

        CEM adds fitted Gaussian noise to the nominal control input.

        Args:
            nominal_knots: The nominal control input to sample from. Shape=(num_nodes, nu).

        Returns:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
        """
        num_nodes = self.num_nodes
        num_rollouts = self.num_rollouts
        noise_ramp = self.noise_ramp

        if self.use_noise_ramp:
            ramp = torch.linspace(noise_ramp / num_nodes, noise_ramp, num_nodes)[:, None]
            self.sigma = torch.clip(self.sigma * ramp, self.sigma_min, self.sigma_max)
        noised_knots = nominal_knots + self.sigma[None] * torch.randn(num_rollouts - 1, num_nodes, self.nu)
        return torch.concatenate([nominal_knots[None], noised_knots])

    def update_nominal_knots(self, sampled_knots: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Update the nominal control knots based on the sampled controls and rewards.

        CEM takes the top k sampled control inputs and fits a Gaussian to them.

        Args:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
            rewards: The rewards for each sampled control input. Shape=(num_rollouts,).

        Returns:
            nominal_knots: The updated nominal control input. Shape=(num_nodes, nu).
        """
        elite_inds = torch.flip(torch.argsort(rewards), dims=[0])[: self.num_elites]
        elite_knots = sampled_knots[elite_inds]
        nominal_knots = elite_knots.mean(0)
        self.sigma = torch.clip(torch.sqrt(elite_knots.var(0)), self.sigma_min, self.sigma_max)
        return nominal_knots
