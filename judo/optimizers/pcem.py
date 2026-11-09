# Ref: https://github.com/hzyu17/PCEM-PO/blob/master/scripts/pcem.py
#
# Proximal cross-entropy method (PCEM) in judo's Optimizer API, mirroring judo's cem.py:
# judo owns the rollouts, this class only implements the two hooks
#   sample_control_knots(nominal)  -> (num_rollouts, num_nodes, nu)
#   update_nominal_knots(samples, rewards) -> (num_nodes, nu)
# One sample/update pair is one PCEM iteration (one call to judo's optimizer step).

import math
from dataclasses import dataclass
from typing import Optional

import torch

from judo.optimizers.base import Optimizer, OptimizerConfig
from judo.utils.spline import TorchSpline, SplineType


@dataclass
class PCEMConfig(OptimizerConfig):
    """Configuration for the proximal cross-entropy method."""

    # Number of optimizer calls over which std / eta are annealed; they hold their final value afterwards.
    horizon: int = 50
    num_rollouts: int = 128
    num_iterations: float = 100
    elite_ratio: float = 0.05
    # Proposal std (scalar multiplier of the smoothness-correlated noise), cosine-annealed.
    std_initial: float = 10.0
    std_final: float = 0.1
    # Adaptive eta scheduling
    # Proximal step size eta, gamma = eta / (eta + 1); schedule: "cosine" | "linear" | "exponential".
    eta_initial: float = 2.0
    eta_final: float = 10.0
    eta_schedule: str = "cosine"
    temperature: float = 0.1
    ema_alpha: float = 0.5
    smoothness_weight: float = 0.1
    # Mean update: RAdam (use_adam=True) or plain momentum. lr == ema_alpha in the original.
    use_adam: bool = True
    lr: float = 0.5
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-4
    # Keep the nominal trajectory as sample 0 (judo/CEM convention). The original PCEM has no such sample.
    include_nominal: bool = True
    # Knot bounds used for sampling and for the mean update (None = unbounded).
    action_min: Optional[float] = -1.0
    action_max: Optional[float] = 1.0


class PCEM(Optimizer[PCEMConfig]):
    """The proximal cross-entropy method."""

    def __init__(self, config: PCEMConfig, nu: int, override_task_name: Optional[str] = None) -> None:
        """Initialize PCEM optimizer."""
        super().__init__(config, nu, override_task_name)
        self._R: Optional[torch.Tensor] = None
        self._L: Optional[torch.Tensor] = None
        self._K = max(1, self.num_elites)
        self.reset_state()

    # ------------------------------------------------------------ properties
    @property
    def num_elites(self) -> int:
        """Get the number of elites."""
        return int(self.config.num_rollouts * self.config.elite_ratio)

    def reset_state(self) -> None:
        """Restart the annealing schedules and the optimizer moments."""
        self._iter = 0
        self._t = 0
        self._m: Optional[torch.Tensor] = None
        self._v: Optional[torch.Tensor] = None
        self._nominal: Optional[torch.Tensor] = None
        self._eps_noise: Optional[torch.Tensor] = None

    # ------------------------------------------------------------- schedules
    @staticmethod
    def schedule_values(progress: float, cfg: PCEMConfig) -> tuple[float, float, float]:
        """(std, eta, gamma) at progress in [0, 1]."""
        std = cfg.std_final + 0.5 * (cfg.std_initial - cfg.std_final) * (1 + math.cos(math.pi * progress))
        if cfg.eta_schedule == "linear":
            eta = cfg.eta_initial + (cfg.eta_final - cfg.eta_initial) * progress
        elif cfg.eta_schedule == "exponential":
            ratio = max(cfg.eta_final / max(cfg.eta_initial, 1e-8), 1e-8)
            eta = cfg.eta_initial * ratio ** progress
        elif cfg.eta_schedule == "cosine":
            eta = cfg.eta_final - 0.5 * (cfg.eta_final - cfg.eta_initial) * (1 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown schedule: {cfg.eta_schedule}")
        return std, eta, eta / (eta + 1.0)

    def _schedule(self) -> tuple[float, float]:
        N = self.config.num_iterations
        progress = 1.0 if N <= 1 else min(self._iter / (N - 1), 1.0)
        std, _, gamma = self.schedule_values(progress, self.config)
        return std, gamma

    # ------------------------------------------------------------ smoothness
    def _ensure_smoothness(self, n: int, like: torch.Tensor) -> None:
        """(Re)build R and L = chol(R^-1) for n nodes, in float64, on like's device/dtype."""
        if self._R is not None and self._R.shape[0] == n \
                and self._R.device == like.device and self._R.dtype == like.dtype:
            return
        f64 = torch.float64
        if n < 3:
            R = torch.eye(n, dtype=f64)
            L = torch.eye(n, dtype=f64)
        else:
            ds = 1.0 / (n - 1)
            scale = 1.0 / (ds ** 3)
            reg = 1e-3 * scale
            diag = torch.ones(n, dtype=f64) * 6.0
            diag[0] = 1.0
            diag[n - 1] = 1.0
            diag[1] = 5.0
            diag[n - 2] = 5.0
            R = torch.diag(diag * scale + reg)
            off1 = torch.ones(n - 1, dtype=f64) * (-4.0)
            off1[0] = -2.0
            off1[n - 2] = -2.0
            R = R + torch.diag(off1 * scale, 1) + torch.diag(off1 * scale, -1)
            R = R + torch.diag(torch.ones(n - 2, dtype=f64) * scale, 2) \
                + torch.diag(torch.ones(n - 2, dtype=f64) * scale, -2)

            try:
                L = torch.linalg.cholesky(R)
            except torch.linalg.LinAlgError:
                R = R + torch.eye(n, dtype=f64) * 1e-4 * scale
                L = torch.linalg.cholesky(R)
        self._R = R.to(device=like.device, dtype=like.dtype)
        self._L = L.to(device=like.device, dtype=like.dtype)

    def smoothness(self, trajectory: torch.Tensor) -> torch.Tensor:
        """0.5 * trace(A^T R A) / max(max|A|^2, 1e-6)  (scale-invariant roughness of a knot trajectory)."""
        self._ensure_smoothness(trajectory.shape[0], trajectory)
        scale_sq = torch.clamp(trajectory.abs().max() ** 2, min=1e-6)
        return 0.5 * torch.sum((self._R @ trajectory) * trajectory) / scale_sq

    # ----------------------------------------------------------------- hooks
    def pre_optimization(self, old_times: torch.Tensor, new_times: torch.Tensor) -> None:
        """Resample the optimizer moments if the number of nodes has changed (R, L follow lazily)."""
        if self._m is not None and len(self._m) != self.num_nodes:
            self._m = TorchSpline(old_times, self._m, spline_order=SplineType.LINEAR)(new_times)
            self._v = TorchSpline(old_times, self._v, spline_order=SplineType.LINEAR)(new_times)

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if cfg.action_min is None and cfg.action_max is None:
            return x
        return torch.clamp(x, cfg.action_min, cfg.action_max)

    @torch.no_grad()
    def sample_control_knots(self, nominal_knots: torch.Tensor) -> torch.Tensor:
        """Samples control knots given a nominal control input.

        PCEM adds antithetic, smoothness-correlated noise eps = +-(L @ z) * std, with L = chol(R^-1)
        so that the perturbations are smooth in time.

        Args:
            nominal_knots: The nominal control input to sample from. Shape=(num_nodes, nu).

        Returns:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
        """
        num_nodes, nu = nominal_knots.shape
        num_rollouts = self.num_rollouts
        cfg = self.config
        self._ensure_smoothness(num_nodes, nominal_knots)
        std, _ = self._schedule()

        n_free = num_rollouts - int(cfg.include_nominal)
        half = n_free // 2
        n_extra = n_free - 2 * half  # 1 if an odd number of free samples (one unpaired draw)
        noise = torch.randn(half + n_extra, num_nodes, nu, device=nominal_knots.device, dtype=nominal_knots.dtype)
        corr = torch.einsum("ij,mjd->mid", self._L, noise) * std
        if self.use_noise_ramp:
            # Non-cumulative per-node ramp (judo's CEM applies it to sigma, which is re-fitted every step).
            noise_ramp = self.noise_ramp
            ramp = torch.linspace(noise_ramp / num_nodes, noise_ramp, num_nodes,
                                  device=nominal_knots.device, dtype=nominal_knots.dtype)[:, None]
            corr = corr * ramp[None]

        eps_noise = torch.cat([corr[:half], -corr[:half], corr[half:]], dim=0)
        if cfg.include_nominal:
            eps_noise = torch.cat([torch.zeros_like(nominal_knots)[None], eps_noise], dim=0)

        self._nominal = nominal_knots
        self._eps_noise = eps_noise
        return self._clip(nominal_knots[None] + eps_noise)

    @torch.no_grad()
    def update_nominal_knots(self, sampled_knots: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Update the nominal control knots based on the sampled controls and rewards.

        PCEM keeps the num_elites lowest-cost samples (cost = -reward + smoothness_weight * <eps, R mean>),
        forms importance weights exp(-gamma/temperature * energy - smoothness_weight * reg), and moves the
        nominal toward the weighted elite average with RAdam / momentum.

        Args:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
            rewards: The rewards for each sampled control input. Shape=(num_rollouts,).

        Returns:
            nominal_knots: The updated nominal control input. Shape=(num_nodes, nu).
        """
        if self._eps_noise is None or self._eps_noise.shape != sampled_knots.shape:
            raise RuntimeError("update_nominal_knots must follow sample_control_knots with the same samples.")
        cfg = self.config
        mean = self._nominal
        dtype = mean.dtype
        _, gamma = self._schedule()
        smooth_w = cfg.smoothness_weight

        energy = -torch.nan_to_num(rewards.to(device=mean.device, dtype=dtype), nan=-1e6)
        reg = torch.einsum("mtd,td->m", self._eps_noise, self._R @ mean)
        costs = energy + smooth_w * reg

        _, idx = torch.topk(-costs, self._K, dim=-1)
        elite_samples, elite_energy, elite_reg = sampled_knots[idx], energy[idx], reg[idx]

        exponents = -(gamma / cfg.temperature) * elite_energy - smooth_w * elite_reg
        exponents = torch.clip(exponents - exponents.max(), -20, 20)
        weights = torch.softmax(exponents, dim=-1)
        target = torch.einsum("n,nij->ij", weights, elite_samples)

        g = target - mean
        if self._m is None:
            self._m = torch.zeros_like(mean)
            self._v = torch.full_like(mean, 0.1)
        self._m = cfg.beta1 * self._m + (1 - cfg.beta1) * g
        if cfg.use_adam:
            self._t += 1
            t = self._t
            self._v = cfg.beta2 * self._v + (1 - cfg.beta2) * g ** 2
            m_hat = self._m / (1 - cfg.beta1 ** t)
            v_hat = self._v / (1 - cfg.beta2 ** t)
            rho_inf = 2.0 / (1.0 - cfg.beta2) - 1.0
            rho_t = rho_inf - 2.0 * t * cfg.beta2 ** t / (1.0 - cfg.beta2 ** t)
            if rho_t > 5.0:
                r = math.sqrt(max(0.0, (rho_t - 4.0) * (rho_t - 2.0) * rho_inf /
                                  ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t + 1e-8)))
                update = cfg.lr * r * m_hat / (torch.sqrt(v_hat) + cfg.eps)
            else:
                update = cfg.lr * m_hat
        else:
            update = cfg.lr * self._m

        self._iter += 1
        return self._clip(mean + update)
