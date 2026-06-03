"""Tests for the LogEIPermutedVar acquisition function."""

import numpy as np
import pandas as pd
import pytest

from baybe.acquisition import LogEIPermutedVar, LogExpectedImprovement
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.targets import NumericalTarget


@pytest.fixture
def setup():
    """Provide a fitted surrogate, searchspace, objective, and measurements."""
    values = np.linspace(0, 10, 20)
    param = NumericalDiscreteParameter("x", values)
    searchspace = SearchSpace.from_product(parameters=[param])
    target = NumericalTarget("y")
    objective = target.to_objective()

    # Create some training data
    rng = np.random.default_rng(42)
    train_x = rng.choice(values, size=5, replace=False)
    train_y = np.sin(train_x) + rng.normal(0, 0.1, size=5)
    measurements = pd.DataFrame({"x": train_x, "y": train_y})

    # Fit surrogate
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace, objective, measurements)

    # Candidates
    candidates = pd.DataFrame({"x": values})

    return surrogate, searchspace, objective, measurements, candidates


def test_reproducibility_with_seed(setup):
    """Same seed produces identical acquisition values."""
    surrogate, searchspace, objective, measurements, candidates = setup

    acqf = LogEIPermutedVar(seed=123)
    values_1 = acqf.evaluate(
        candidates, surrogate, searchspace, objective, measurements
    )
    values_2 = acqf.evaluate(
        candidates, surrogate, searchspace, objective, measurements
    )

    pd.testing.assert_series_equal(values_1, values_2)


def test_randomness_without_seed(setup):
    """Without seed, different evaluations may produce different values."""
    surrogate, searchspace, objective, measurements, candidates = setup

    acqf = LogEIPermutedVar(seed=None)
    values_1 = acqf.evaluate(
        candidates, surrogate, searchspace, objective, measurements
    )
    values_2 = acqf.evaluate(
        candidates, surrogate, searchspace, objective, measurements
    )

    # With 20 candidates, random permutations should almost certainly differ
    assert not values_1.equals(values_2)


def test_differs_from_standard_log_ei(setup):
    """LogEIPermutedVar produces different values than standard LogEI."""
    surrogate, searchspace, objective, measurements, candidates = setup

    log_ei = LogExpectedImprovement()
    log_ei_values = log_ei.evaluate(
        candidates, surrogate, searchspace, objective, measurements
    )

    perm_log_ei = LogEIPermutedVar(seed=42)
    perm_values = perm_log_ei.evaluate(
        candidates, surrogate, searchspace, objective, measurements
    )

    # Values should differ (permuted variance breaks the natural correlation)
    assert not log_ei_values.equals(perm_values)


def test_pending_experiments_raises(setup):
    """Passing pending_experiments raises IncompatibleAcquisitionFunctionError."""
    from baybe.exceptions import IncompatibleAcquisitionFunctionError

    surrogate, searchspace, objective, measurements, candidates = setup

    acqf = LogEIPermutedVar(seed=42)
    pending = pd.DataFrame({"x": [5.0]})

    with pytest.raises(IncompatibleAcquisitionFunctionError):
        acqf.evaluate(
            candidates,
            surrogate,
            searchspace,
            objective,
            measurements,
            pending_experiments=pending,
        )


def test_campaign_recommend(setup):
    """LogEIPermutedVar works in the normal Campaign.recommend() loop."""
    from baybe import Campaign
    from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender

    surrogate, searchspace, objective, measurements, _ = setup

    # LogEIPermutedVar is analytic (non-MC), so batch_size must be 1
    recommender = BotorchRecommender(acquisition_function=LogEIPermutedVar(seed=7))
    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=recommender,
    )
    campaign.add_measurements(measurements)
    recommendations = campaign.recommend(batch_size=1)

    assert len(recommendations) == 1
    assert all(col in recommendations.columns for col in searchspace.parameter_names)
