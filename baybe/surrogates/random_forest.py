"""Random forest surrogates.

Currently, the documentation for this surrogate is not available. This is due to a bug
in our documentation tool, see https://github.com/sphinx-doc/sphinx/issues/11750.

Since we plan to refactor the surrogates, this part of the documentation will be
available in the future. Thus, please have a look in the source code directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from attr import define, field
from sklearn.ensemble import RandomForestRegressor

from baybe.parameters.base import Parameter
from baybe.surrogates.base import GaussianSurrogate
from baybe.surrogates.utils import batchify, catch_constant_targets
from baybe.surrogates.validation import get_model_params_validator

if TYPE_CHECKING:
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from torch import Tensor


@catch_constant_targets
@define
class RandomForestSurrogate(GaussianSurrogate):
    """A random forest surrogate model."""

    # Class variables
    joint_posterior: ClassVar[bool] = False
    # See base class.

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    # Object variables
    model_params: dict[str, Any] = field(
        factory=dict,
        converter=dict,
        validator=get_model_params_validator(RandomForestRegressor.__init__),
    )
    """Optional model parameter that will be passed to the surrogate constructor."""

    _model: RandomForestRegressor | None = field(init=False, default=None, eq=False)
    """The actual model."""

    @staticmethod
    def _make_parameter_scaler_factory(
        parameter: Parameter,
    ) -> type[InputTransform] | None:
        # See base class.

        # Tree-like models do not require any input scaling
        return None

    @staticmethod
    def _make_target_scaler_factory() -> type[OutcomeTransform] | None:
        # See base class.

        # Tree-like models do not require any output scaling
        return None

    @batchify
    def _estimate_moments(
        self, candidates_comp_scaled: Tensor, /
    ) -> tuple[Tensor, Tensor]:
        # See base class.

        # FIXME[typing]: It seems there is currently no better way to inform the type
        #   checker that the attribute is available at the time of the function call
        assert self._model is not None

        import torch

        # Evaluate all trees
        # NOTE: explicit conversion to ndarray is needed due to a pytorch issue:
        # https://github.com/pytorch/pytorch/pull/51731
        # https://github.com/pytorch/pytorch/issues/13918
        predictions = torch.from_numpy(
            np.asarray(
                [
                    self._model.estimators_[tree].predict(candidates_comp_scaled)
                    for tree in range(self._model.n_estimators)
                ]
            )
        )

        # Compute posterior mean and variance
        mean = predictions.mean(dim=0)
        var = predictions.var(dim=0)

        return mean, var

    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        # See base class.
        self._model = RandomForestRegressor(**(self.model_params))
        self._model.fit(train_x, train_y.ravel())
