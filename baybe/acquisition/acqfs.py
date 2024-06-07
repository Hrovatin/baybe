"""Available acquisition functions."""

import math
from typing import ClassVar

import pandas as pd
from attr.validators import optional as optional_v
from attrs import define, field
from attrs.validators import ge, gt, instance_of, le

from baybe.acquisition.base import AcquisitionFunction
from baybe.searchspace import SearchSpace
from baybe.utils.sampling_algorithms import (
    SamplingMethod,
    sample_numerical_df,
)


########################################################################################
### Active Learning
@define(frozen=True)
class qNegIntegratedPosteriorVariance(AcquisitionFunction):
    """Monte Carlo based negative integrated posterior variance.

    This is particularly useful for active learning as it is a measure for global model
    uncertainty.
    """

    abbreviation: ClassVar[str] = "qNIPV"

    sampling_fraction: float = field(
        converter=float,
        validator=[gt(0.0), le(1.0)],
        default=1.0,
    )
    """Fraction of data that will be sampled for integrating the posterior.

    The fraction will be ignored if 'sampling_n_points' is not None."""

    sampling_n_points: int | None = field(
        validator=optional_v([gt(0), instance_of(int)]),
        default=None,
    )
    """Number of data points that will be sampled for integrating the posterior."""

    sampling_method: SamplingMethod = field(
        converter=SamplingMethod, default=SamplingMethod.Random
    )
    """Strategy used for sampling data for integrating the posterior."""

    def get_integration_points(self, searchspace: SearchSpace) -> pd.DataFrame:
        """Sample points from a search space for integration purposes.

        Sampling of the discrete part can be controlled via 'sampling_method', but
        sampling of the continuous part will always be random.

        Args:
            searchspace: The searchspace from which to sample integration points.

        Returns:
            The sampled data points.

        Raises:
            ValueError: If searchspace is purely continuous and 'sampling_n_points' was
                not provided.
        """
        sampled_parts = []
        n_candidates = None

        # Discrete part
        if searchspace.discrete is not None:
            candidates_discrete = searchspace.discrete.comp_rep
            n_candidates = self.sampling_n_points or math.ceil(
                self.sampling_fraction * len(candidates_discrete)
            )

            sampled_disc = sample_numerical_df(
                candidates_discrete, n_candidates, method=self.sampling_method
            )

            sampled_parts.append(sampled_disc)

        # Continuous part
        if searchspace.continuous is not None:
            # If a discrete part has resulted in a particular choice for n_candidates,
            # take it. Otherwise, use the user specified number of points.
            n_candidates = n_candidates or self.sampling_n_points
            if n_candidates is None:
                raise ValueError(
                    f"'sampling_n_points' must be provided for '{self.__class__}' when"
                    f"sampling purely continuous search spaces."
                )
            sampled_conti = searchspace.continuous.samples_random(n_candidates)

            # Align indices if discrete part is present
            if len(sampled_parts) > 0:
                sampled_conti.index = sampled_parts[0].index
            sampled_parts.append(sampled_conti)

        # Combine different search space parts
        result = pd.concat(sampled_parts, axis=1)

        return result


########################################################################################
### Posterior Mean
@define(frozen=True)
class PosteriorMean(AcquisitionFunction):
    """Posterior mean."""

    abbreviation: ClassVar[str] = "PM"


########################################################################################
### Simple Regret
@define(frozen=True)
class qSimpleRegret(AcquisitionFunction):
    """Monte Carlo based simple regret."""

    abbreviation: ClassVar[str] = "qSR"


########################################################################################
### Expected Improvement
@define(frozen=True)
class ExpectedImprovement(AcquisitionFunction):
    """Analytical expected improvement."""

    abbreviation: ClassVar[str] = "EI"


@define(frozen=True)
class qExpectedImprovement(AcquisitionFunction):
    """Monte Carlo based expected improvement."""

    abbreviation: ClassVar[str] = "qEI"


@define(frozen=True)
class LogExpectedImprovement(AcquisitionFunction):
    """Logarithmic analytical expected improvement."""

    abbreviation: ClassVar[str] = "LogEI"


@define(frozen=True)
class qLogExpectedImprovement(AcquisitionFunction):
    """Logarithmic Monte Carlo based expected improvement."""

    abbreviation: ClassVar[str] = "qLogEI"


@define(frozen=True)
class qNoisyExpectedImprovement(AcquisitionFunction):
    """Monte Carlo based noisy expected improvement."""

    abbreviation: ClassVar[str] = "qNEI"

    prune_baseline: bool = field(default=True, validator=instance_of(bool))
    """Auto-prune candidates that are unlikely to be the best."""


@define(frozen=True)
class qLogNoisyExpectedImprovement(AcquisitionFunction):
    """Logarithmic Monte Carlo based noisy expected improvement."""

    abbreviation: ClassVar[str] = "qLogNEI"

    prune_baseline: bool = field(default=True, validator=instance_of(bool))
    """Auto-prune candidates that are unlikely to be the best."""


########################################################################################
### Probability of Improvement
@define(frozen=True)
class ProbabilityOfImprovement(AcquisitionFunction):
    """Analytical probability of improvement."""

    abbreviation: ClassVar[str] = "PI"


@define(frozen=True)
class qProbabilityOfImprovement(AcquisitionFunction):
    """Monte Carlo based probability of improvement."""

    abbreviation: ClassVar[str] = "qPI"


########################################################################################
### Upper Confidence Bound
@define(frozen=True)
class UpperConfidenceBound(AcquisitionFunction):
    """Analytical upper confidence bound."""

    abbreviation: ClassVar[str] = "UCB"

    beta: float = field(converter=float, validator=ge(0.0), default=0.2)
    """Trade-off parameter for mean and variance.

    A value of zero makes the acquisition mechanism consider the posterior predictive
    mean only, resulting in pure exploitation. Higher values shift the focus more and
    more toward exploration.
    """


@define(frozen=True)
class qUpperConfidenceBound(AcquisitionFunction):
    """Monte Carlo based upper confidence bound."""

    abbreviation: ClassVar[str] = "qUCB"

    beta: float = field(converter=float, validator=ge(0.0), default=0.2)
    """Trade-off parameter for mean and variance.

    A value of zero makes the acquisition mechanism consider the posterior predictive
    mean only, resulting in pure exploitation. Higher values shift the focus more and
    more toward exploration.
    """
