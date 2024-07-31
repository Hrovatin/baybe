"""Scaling utilities."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from attrs import define, field
from attrs.validators import deep_iterable, deep_mapping, instance_of

if TYPE_CHECKING:
    from botorch.models.transforms.input import InputTransform
    from torch import Tensor


@define
class ColumnTransformer:
    """Class for applying separate transforms to different column groups of tensors."""

    mapping: dict[tuple[int, ...], InputTransform] = field()
    """A mapping defining what transform to apply to which columns."""

    @mapping.validator
    def _validate_mapping_types_lazily(self, attr, value):
        """Perform transform isinstance check using lazy import."""
        from botorch.models.transforms.input import InputTransform

        validator = deep_mapping(
            mapping_validator=instance_of(dict),
            key_validator=deep_iterable(
                member_validator=instance_of(int), iterable_validator=instance_of(tuple)
            ),
            value_validator=instance_of(InputTransform),
        )
        validator(self, attr, value)

    @mapping.validator
    def _validate_mapping_is_disjoint(self, _, value: dict):
        """Validate that the each column is assigned to at most one transformer."""
        for x, y in itertools.combinations(value.keys(), 2):
            if not set(x).isdisjoint(y):
                raise ValueError(
                    f"The provided column specifications {x} and {y} are not disjoint."
                )

    def fit(self, x: Tensor, /) -> None:
        """Fit the transformer to the given tensor."""
        for cols, transformer in self.mapping.items():
            transformer.train()
            transformer(x[..., cols])

    def transform(self, x: Tensor, /) -> Tensor:
        """Transform the given tensor."""
        out = x.clone()
        for cols, transformer in self.mapping.items():
            transformer.eval()
            out[..., cols] = transformer(out[..., cols])
        return out
