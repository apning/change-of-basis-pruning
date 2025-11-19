from dataclasses import dataclass
from typing import Any, ClassVar
import torch
from torch import nn

from src.modeling.utils import RMSNorm2d


@torch.no_grad()
def structured_prune(
    layer: nn.Linear | nn.Conv2d,
    dims_to_prune: tuple[int] | list[int],
    prune_output: bool = True,
):
    if len(dims_to_prune) == 0:
        return

    """Checks"""

    if not isinstance(layer, (nn.Linear, nn.Conv2d)):
        raise ValueError(f"Expected layer to be a Linear or Conv2d, got {type(layer)}")

    if len(set(dims_to_prune)) != len(dims_to_prune):
        raise ValueError("Dimensions to prune must be unique")

    if min(dims_to_prune) < 0:
        raise ValueError("Dimensions to prune cannot be negative")

    out_f, in_f = layer.weight.shape[0:2]
    num_f = out_f if prune_output else in_f

    if len(dims_to_prune) > num_f:
        raise ValueError(f"Cannot prune more than {num_f} dimensions")

    if max(dims_to_prune) >= num_f:
        raise ValueError(
            f"Cannot prune dimension {max(dims_to_prune)} because it is greater than the number of features ({num_f})"
        )

    dims_to_keep = sorted(list(set(range(num_f)) - set(dims_to_prune)))

    """ Prune """

    if prune_output:
        layer.weight = nn.Parameter(layer.weight.data[dims_to_keep].clone())
        layer.bias = (
            nn.Parameter(layer.bias.data[dims_to_keep].clone())
            if layer.bias is not None
            else None
        )
        if isinstance(layer, nn.Linear):
            layer.out_features = len(dims_to_keep)
        else:
            layer.out_channels = len(dims_to_keep)
    else:
        layer.weight = nn.Parameter(layer.weight.data[:, dims_to_keep].clone())
        if isinstance(layer, nn.Linear):
            layer.in_features = len(dims_to_keep)
        else:
            layer.in_channels = len(dims_to_keep)


@torch.no_grad()
def structured_prune_norm(
    norm: nn.BatchNorm2d | nn.RMSNorm | RMSNorm2d, dims_to_prune: tuple[int] | list[int]
):
    if len(dims_to_prune) == 0:
        return

    """Checks"""

    if len(set(dims_to_prune)) != len(dims_to_prune):
        raise ValueError("Dimensions to prune must be unique")

    if min(dims_to_prune) < 0:
        raise ValueError("Dimensions to prune cannot be negative")

    if isinstance(norm, nn.BatchNorm2d):
        num_f = norm.num_features
    elif isinstance(norm, (nn.RMSNorm, RMSNorm2d)):
        if isinstance(norm, RMSNorm2d):
            norm = norm.rn
        if len(norm.normalized_shape) != 1:
            raise NotImplementedError(
                f"Only 1D normalized shapes are supported for RMSNorm2d. Got: {norm.normalized_shape}"
            )
        num_f = norm.normalized_shape[0]
    else:
        raise ValueError(
            f"Expected norm to be a BatchNorm2d or RMSNorm, got {type(norm)}"
        )

    if len(dims_to_prune) > num_f:
        raise ValueError(f"Cannot prune more than {num_f} dimensions")

    if max(dims_to_prune) >= num_f:
        raise ValueError(
            f"Cannot prune dimension {max(dims_to_prune)} because it is greater than the number of features ({num_f})"
        )

    dims_to_keep = sorted(list(set(range(num_f)) - set(dims_to_prune)))

    """ Prune """

    if getattr(norm, "weight", None) is not None:
        norm.weight = nn.Parameter(norm.weight.data[dims_to_keep].clone())
    if getattr(norm, "bias", None) is not None:
        norm.bias = nn.Parameter(norm.bias.data[dims_to_keep].clone())

    if isinstance(norm, nn.BatchNorm2d):
        norm.register_buffer(
            "running_mean", norm.running_mean.data[dims_to_keep].clone()
        )
        norm.register_buffer("running_var", norm.running_var.data[dims_to_keep].clone())
        norm.num_features = len(dims_to_keep)
    elif isinstance(norm, (nn.RMSNorm, RMSNorm2d)):
        norm.normalized_shape = (len(dims_to_keep),)
    else:
        raise ValueError(
            f"Unexpected norm type which should have been caught by earlier code: {type(norm)}"
        )


@dataclass
class PruningStrategy:
    proportion: float | tuple[float] | None = None
    absolute_num: int | tuple[int] | None = None
    zscore_cutoff: float | tuple[float] | None = None
    prop_of_avg: float | tuple[float] | None = None
    prop_of_med: float | tuple[float] | None = None
    prop_of_max: float | tuple[float] | None = None

    _attr_names: ClassVar[list[str]] = [
        "proportion",
        "absolute_num",
        "zscore_cutoff",
        "prop_of_avg",
        "prop_of_med",
        "prop_of_max",
    ]

    def __post_init__(self):
        self.validate()

    def validate(self):
        # Check that exactly one of the pruning strategy attributes is not None
        not_none_attr_names = [
            attr_name
            for attr_name in self._attr_names
            if getattr(self, attr_name) is not None
        ]

        if len(not_none_attr_names) != 1:
            raise ValueError(
                f"Exactly one of {self._attr_names} must be provided, but not multiple. The following attributes are not None: {not_none_attr_names}"
            )

        not_none_attr_name = not_none_attr_names[0]

        # Set to tuple if list
        if isinstance(getattr(self, not_none_attr_name), list):
            setattr(self, not_none_attr_name, tuple(getattr(self, not_none_attr_name)))

    def get_not_none_kv(self) -> tuple[str, Any]:
        """Returns the key and value of the one non-None pruning strategy attribute"""
        self.validate()

        for attr_name in self._attr_names:
            value = getattr(self, attr_name)
            if value is not None:
                return attr_name, value

        raise ValueError("HOW DID THE CODE GET HERE???")
