"""Custom BoTorch acquisition function with permuted variance."""

from __future__ import annotations

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class _EIPermutedVariance(AnalyticAcquisitionFunction):
    """Expected Improvement with permuted per-point variances.

    Computes EI but shuffles the variance across candidate points while keeping
    means intact. This breaks the natural mean-variance correlation for sensitivity
    analysis.

    Note: This acquisition function is designed for discrete search spaces only.
    In continuous optimization, the candidate set changes at each optimizer step,
    making permutation semantics undefined.

    Args:
        model: A fitted single-outcome model.
        best_f: The best observed objective value.
        seed: Optional seed for reproducible permutation. If ``None``, a random
            permutation is generated each call.
        posterior_transform: A PosteriorTransform for multi-output models.
    """

    def __init__(
        self,
        model: Model,
        best_f: float,
        seed: int | None = None,
        posterior_transform: PosteriorTransform | None = None,
    ) -> None:
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=torch.double))
        self.seed = seed

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate EI with permuted variance on the candidate set X.

        Args:
            X: A tensor of shape ``[batch_shape, 1, d]`` representing the candidates.

        Returns:
            A tensor of shape ``[batch_shape]`` with EI values computed using
            permuted variances.
        """
        mean, sigma = self._mean_and_sigma(X)

        # Permute sigma across the candidate (batch) dimension
        n = sigma.shape[0]
        if self.seed is not None:
            gen = torch.Generator(device=sigma.device).manual_seed(self.seed)
        else:
            gen = None
        perm = torch.randperm(n, generator=gen, device=sigma.device)
        sigma_permuted = sigma[perm]

        # EI formula: sigma * (phi(u) + u * Phi(u))
        normal = torch.distributions.Normal(
            torch.tensor(0.0, device=mean.device, dtype=mean.dtype),
            torch.tensor(1.0, device=mean.device, dtype=mean.dtype),
        )
        u = (mean - self.best_f.expand_as(mean)) / sigma_permuted.clamp_min(1e-12)
        ei = sigma_permuted * (torch.exp(normal.log_prob(u)) + u * normal.cdf(u))
        return ei.clamp_min(0.0).squeeze(-1)
