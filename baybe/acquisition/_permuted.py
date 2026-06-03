"""Custom BoTorch acquisition function with permuted variance (log-space)."""

from __future__ import annotations

import torch
from botorch.acquisition.analytic import (
    AnalyticAcquisitionFunction,
    _log_ei_helper,
    _scaled_improvement,
)
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class _LogExpectedImprovementPermutedVariance(AnalyticAcquisitionFunction):
    """Log Expected Improvement with permuted per-point variances.

    Computes log-EI but shuffles the variance across candidate points while keeping
    means intact. This breaks the natural mean-variance correlation for sensitivity
    analysis. Uses numerically stable log-space computation following BoTorch's
    ``LogExpectedImprovement``.

    Note: This acquisition function is designed for discrete search spaces only.
    In continuous optimization, the candidate set changes at each optimizer step,
    making permutation semantics undefined.

    Args:
        model: A fitted single-outcome model.
        best_f: The best observed objective value.
        seed: Optional seed for reproducible permutation. If ``None``, a random
            permutation is generated each call.
        maximize: If ``True``, consider the problem a maximization problem.
        posterior_transform: A PosteriorTransform for multi-output models.
    """

    def __init__(
        self,
        model: Model,
        best_f: float,
        seed: int | None = None,
        maximize: bool = True,
        posterior_transform: PosteriorTransform | None = None,
    ) -> None:
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=torch.double))
        self.seed = seed
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate log-EI with permuted variance on the candidate set X.

        Args:
            X: A tensor of shape ``[batch_shape, 1, d]`` representing the candidates.

        Returns:
            A tensor of shape ``[batch_shape]`` with log-EI values computed using
            permuted variances.

        Raises:
            ValueError: If the model has batch dimensions (sigma has more than
                2 dimensions).
        """
        mean, sigma = self._mean_and_sigma(X)

        # Permute sigma across the candidate (batch) dimension.
        # This assumes no leading batch dimensions on the model — the permutation
        # operates on dim 0, which must be the candidate count.
        if sigma.ndim > 2:
            raise ValueError(
                f"'{self.__class__.__name__}' does not support models with batch "
                f"dimensions. Expected sigma with at most 2 dimensions, "
                f"got shape {sigma.shape}."
            )
        n = sigma.shape[0]
        if self.seed is not None:
            gen = torch.Generator(device=sigma.device).manual_seed(self.seed)
        else:
            gen = None
        perm = torch.randperm(n, generator=gen, device=sigma.device)
        sigma_permuted = sigma[perm]

        # Log-EI with permuted sigma, following BoTorch's LogExpectedImprovement:
        # u = (mean - best_f) / sigma_permuted (negated if minimizing)
        # log_ei = _log_ei_helper(u) + log(sigma_permuted)
        u = _scaled_improvement(mean, sigma_permuted, self.best_f, self.maximize)
        return (_log_ei_helper(u) + sigma_permuted.log()).squeeze(-1)
