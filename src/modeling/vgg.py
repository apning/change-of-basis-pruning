import copy
from dataclasses import dataclass, field
from itertools import chain
import os
from pathlib import Path
import statistics
from typing import Union

from torch import nn

import torch
from torch.nn import AvgPool2d, BatchNorm2d, Dropout, MaxPool2d

from torchvision.models.vgg import cfgs

from src.act_funcy.tsra import TSRA, TSRALogisiticFunc
from src.modeling.pruning import (
    PruningStrategy,
    structured_prune,
    structured_prune_norm,
)
from src.modeling.utils import RMSNorm2d
from src.pruning.rotate import compute_rotation
from src.utils import JsonMixin
from src.modeling.cob import COB_Rotation, absorb_cob_conv2d, absorb_cob_linear

NORM_MAP = {None: None, "batchnorm": BatchNorm2d, "rmsnorm": RMSNorm2d}

DROPOUT_MAP = {None: None, "dropout": Dropout}

INIT_STRATEGIES = {None, "original", "kaiming_uniform"}

ACT_FUNC_MAPPINGS = {
    "elementwise": {
        "relu": nn.ReLU,
    },
    "tsra": {"logistic": TSRALogisiticFunc},
}

POOL_MAP = {"maxpool": MaxPool2d, "avgpool": AvgPool2d}


@dataclass
class COB_VGG_Config(JsonMixin):
    # See torchvision.models.vgg.cfgs. 'D' is for VGG-16
    cfg: str | list[int | str] = "D"
    num_classes: int = 10
    classifier_cfg: list[int, int] = field(default_factory=lambda: [4096, 4096])

    norm: str | None = None
    norm_kwargs: dict | None = None

    dropout_type: str | None = "dropout"
    dropout_p: float | None = 0.5

    init_strategy: str | None = None

    pool_type: str = "maxpool"

    act_func_type: str = "elementwise"
    act_func_name: str = "relu"

    act_func_kwargs: dict | None = None
    tsra_scaling_kwargs: dict | None = None

    smaller_32x32_classifier_in: bool = False

    # Mainly used for loading functionality. If specified, model with be initialized with COB instantiated
    _cob_orthogonal_map: str | None = None

    def __post_init__(self):
        if isinstance(self.cfg, str):
            if self.cfg not in cfgs:
                raise ValueError(
                    f"Invalid cfg: {self.cfg}. Please choose from {list(cfgs.keys())} or specify your own list of ints/letters"
                )
            self.cfg = cfgs[self.cfg].copy()

        self.validate()

    def validate(self):
        if self.norm not in NORM_MAP:
            raise ValueError(
                f"Invalid norm: {self.norm}. Please choose from {list(NORM_MAP.keys())}"
            )
        if not self.norm and self.norm_kwargs:
            raise ValueError(
                f"norm_kwargs must be None if norm is None. Got {self.norm_kwargs}"
            )

        if self.dropout_type not in DROPOUT_MAP:
            raise ValueError(
                f"Invalid dropout type: {self.dropout_type}. Please choose from {list(DROPOUT_MAP.keys())}"
            )

        if (self.dropout_p is None) ^ (self.dropout_type is None):
            raise ValueError(
                "dropout_p and dropout_type must be either both None or both not None"
            )

        if self.init_strategy not in INIT_STRATEGIES:
            raise ValueError(
                f"Invalid init strategy: {self.init_strategy}. Please choose from {INIT_STRATEGIES}"
            )

        self._validate_act_func()

    def _validate_act_func(self):
        if self.act_func_type not in ACT_FUNC_MAPPINGS:
            raise ValueError(
                f"Invalid act func type: {self.act_func_type}. Please choose from {list(ACT_FUNC_MAPPINGS.keys())}"
            )

        if self.act_func_name not in ACT_FUNC_MAPPINGS[self.act_func_type]:
            raise ValueError(
                f"Invalid act func name: {self.act_func_name}. Please choose from {list(ACT_FUNC_MAPPINGS[self.act_func_type].keys())}"
            )

        if self.act_func_type != "tsra" and self.tsra_scaling_kwargs is not None:
            raise ValueError(
                f"tsra_scaling_kwargs must be None for {self.act_func_type} act func type. Got {self.tsra_scaling_kwargs}"
            )

    def get_act_func(self, reduce_dim: int | None = None, num_dims: int | None = None):
        act_func_kwargs = self.act_func_kwargs or {}

        if self.act_func_type in ["elementwise"]:
            ActFuncClass = ACT_FUNC_MAPPINGS[self.act_func_type][self.act_func_name]
            act_func = ActFuncClass(**act_func_kwargs)

        elif self.act_func_type == "tsra":
            if reduce_dim is None:
                raise ValueError("reduce_dim must be provided for tsra act func type")
            if num_dims is None:
                raise ValueError("num_dims must be provided for tsra act func type")

            ScalingFunc = ACT_FUNC_MAPPINGS[self.act_func_type][self.act_func_name]

            tsra_scaling_kwargs = self.tsra_scaling_kwargs or {}

            scaling_func = ScalingFunc(**tsra_scaling_kwargs)

            act_func = TSRA(
                scaling_func=scaling_func,
                subspace2_start=num_dims // 2,
                expected_dims=num_dims,
                reduce_dim=reduce_dim,
                **act_func_kwargs,
            )
        else:
            raise ValueError(
                f"Invalid act func type: {self.act_func_type}. Please choose from {list(ACT_FUNC_MAPPINGS.keys())}"
            )

        return act_func


class COB_Mixin:
    # Instance attributes
    config: COB_VGG_Config
    cob: COB_Rotation | None
    _next_layer: Union[tuple["COB_Mixin"], tuple[nn.Linear], None]

    # Class attributes
    rotation_dim: int = -1

    def _subspace_sizes(self, skip_sum_check: bool = False) -> tuple[int]:
        """Returns the sizes of the subspaces that may be in use by the activation function"""

        if isinstance(self.act_func, TSRA):
            subspace2_start = self.act_func.subspace2_start
            expected_dims = self.act_func.expected_dims
            subspace_sizes = (subspace2_start, expected_dims - subspace2_start)
        else:
            subspace_sizes = (self.out_features,)

        if not skip_sum_check:
            assert sum(subspace_sizes) == self.out_features, (
                f"Expected {self.out_features} dimensions, but subspace sizes summed to {sum(subspace_sizes)}"
            )

        return subspace_sizes

    @property
    def subspace_sizes(self) -> tuple[int]:
        return self._subspace_sizes()

    def adjust_subspace_sizes(self, new_subspace_sizes: tuple[int]):
        if len(new_subspace_sizes) != len(self._subspace_sizes(skip_sum_check=True)):
            raise ValueError(
                f"Expected {len(self._subspace_sizes(skip_sum_check=True))} new subspace sizes, got {len(new_subspace_sizes)}"
            )
        if sum(new_subspace_sizes) != self.out_features:
            raise ValueError(
                f"Expected {self.out_features} dimensions, but new subspace sizes summed to {sum(new_subspace_sizes)}"
            )

        if isinstance(self.act_func, TSRA):
            assert len(new_subspace_sizes) == 2, (
                f"TSRA act func type expects 2 subspaces, got {len(new_subspace_sizes)}"
            )
            self.act_func.subspace2_start = new_subspace_sizes[0]
            self.act_func.expected_dims = sum(new_subspace_sizes)
        else:
            pass

    def instantiate_cob(
        self,
        orthogonal_map: str = "matrix_exp",
        replace: bool = False,
        rot_weights: list[torch.Tensor] | tuple[torch.Tensor] | None = None,
    ):
        if not replace and self.cob is not None:
            raise ValueError(
                "COB already instantiated. If you wish to replace, specify replace=True"
            )

        self.cob = COB_Rotation(
            subspace_sizes=self.subspace_sizes,
            rotation_dim=self.rotation_dim,
            orthogonal_map=orthogonal_map,
            rot_weights=rot_weights,
            device=self.layer.weight.device,
            dtype=self.layer.weight.dtype,
        )

        self.config._cob_orthogonal_map = orthogonal_map

    def _post_init(self):
        self.cob = None
        self._capture_next_hidden_states = False

        if self.config._cob_orthogonal_map is not None:
            if self.cob is not None:
                raise ValueError(
                    "COB should not yet be instantiated. Did you call _post_init() out of context?"
                )

            self.instantiate_cob(orthogonal_map=self.config._cob_orthogonal_map)

    def absorb_cob(self):
        if self.cob is None:
            raise ValueError(
                "COB not instantiated. Cannot absorb IF THERE IS NOTHING TO ABSORB. Or, I guess this would count as the trivial absorption. Either way. NO"
            )

        """ Absorb COB into self """
        self._absorb_cob_self(self.cob, cob_after=True)

        """ Absorb COB into next layer """
        if self.next_layer is not None:
            if isinstance(self.next_layer, nn.Linear):
                absorb_cob_linear(self.next_layer, self.cob, cob_after=False)
            elif isinstance(self.next_layer, COB_Mixin):
                self.next_layer._absorb_cob_self(self.cob, cob_after=False)
            else:
                raise ValueError(
                    f"Expected next layer to be a Linear or COB_Mixin, got {type(self.next_layer)}"
                )
        else:
            raise ValueError("No next layer to absorb COB into")

        """ Delete COB """
        self.cob = None

    @property
    def next_layer(self):
        if self._next_layer is None:
            return None
        return self._next_layer[0]

    # This exists to avoid registering next_layer as a sub-module of this module
    def set_next_layer(self, next_layer: Union["COB_Mixin", nn.Linear, None]):
        if next_layer is None:
            self._next_layer = None
        else:
            self._next_layer = (next_layer,)

    def _capture_hidden_states(self, hidden_states: torch.Tensor):
        if getattr(self, "_captured_hidden_states", None) is not None:
            raise ValueError(
                "Attempted to capture hidden states but already have some captured."
            )

        if hidden_states.requires_grad:
            raise ValueError(
                "Attempted to capture hidden states that require gradients. Make sure to run under torch.no_grad() or torch.inference_mode()"
            )

        self._captured_hidden_states = hidden_states

    def enable_capture_next_hidden_states(self):
        self._capture_next_hidden_states = True

    def disable_capture_next_hidden_states(self):
        self._captured_hidden_states = None
        self._capture_next_hidden_states = False

    def structured_prune(self, pruning_strategy: PruningStrategy) -> dict:
        """Checks"""

        if self.cob is not None:
            raise ValueError(
                "Cannot structured prune a COB_Mixin layer that has a COB instantiated. Absorb it first!"
            )

        if getattr(self, "_importance_scores", None) is None:
            raise ValueError(
                "Importance scores not found. Cannot resolve dimensions to prune"
            )
        if len(self._importance_scores) != self.out_features:
            raise ValueError(
                f"Importance scores have {len(self._importance_scores)} dimensions, but the layer has {self.out_features} output features."
            )

        prun_strat_k, prun_strat_v = pruning_strategy.get_not_none_kv()
        if isinstance(prun_strat_v, tuple):
            if len(prun_strat_v) != len(self.subspace_sizes):
                raise ValueError(
                    f"Expected {len(self.subspace_sizes)} pruning strategy values, got {len(prun_strat_v)}. Pruning strategy: {prun_strat_k} = {prun_strat_v}"
                )

        """ Resolve dimensions to prune """

        return_statistics = {
            "pruning_strategy": copy.deepcopy(pruning_strategy),
            "initial_subspace_sizes": self.subspace_sizes,
            "initial_out_features": self.out_features,
        }

        # Pair importance scores with indices
        importance_scores = list(
            zip(range(len(self._importance_scores)), self._importance_scores)
        )

        # Chunk importance scores across subspaces
        # Also, Add to each (index, score) pair the index of its subspace. So now they are (index, score, subspace_idx) tuples
        importance_scores_chunks = []
        start = 0
        for i, subspace_size in enumerate(self.subspace_sizes):
            end = start + subspace_size
            importance_scores_chunk = importance_scores[start:end]
            importance_scores_chunk = [x + (i,) for x in importance_scores_chunk]
            importance_scores_chunks.append(importance_scores_chunk)
            start = end

        importance_scores = list(chain(*importance_scores_chunks))

        return_statistics["initial_importance_scores"] = tuple(importance_scores)
        return_statistics["initial_importance_scores_chunks"] = tuple(
            importance_scores_chunks
        )

        ### Find dims to prune based on pruning strategy

        if pruning_strategy.proportion is not None:
            dims_to_prune, dims_to_prune_chunks = self._structured_prune_proportion(
                pruning_strategy.proportion,
                importance_scores=importance_scores,
                importance_scores_chunks=importance_scores_chunks,
            )
        elif pruning_strategy.absolute_num is not None:
            dims_to_prune, dims_to_prune_chunks = self._structured_prune_absolute_num(
                pruning_strategy.absolute_num,
                importance_scores=importance_scores,
                importance_scores_chunks=importance_scores_chunks,
            )
        elif pruning_strategy.zscore_cutoff is not None:
            dims_to_prune, dims_to_prune_chunks = self._structured_prune_zscore_cutoff(
                pruning_strategy.zscore_cutoff,
                importance_scores=importance_scores,
                importance_scores_chunks=importance_scores_chunks,
            )
        elif (
            pruning_strategy.prop_of_avg is not None
            or pruning_strategy.prop_of_med is not None
            or pruning_strategy.prop_of_max is not None
        ):
            if pruning_strategy.prop_of_avg is not None:
                prop_of = pruning_strategy.prop_of_avg
                metric = "avg"
            elif pruning_strategy.prop_of_med is not None:
                prop_of = pruning_strategy.prop_of_med
                metric = "med"
            elif pruning_strategy.prop_of_max is not None:
                prop_of = pruning_strategy.prop_of_max
                metric = "max"
            else:
                raise ValueError("Invalid prop_of_* attribute")
            dims_to_prune, dims_to_prune_chunks = self._structured_prune_prop_of(
                prop_of,
                metric=metric,
                importance_scores=importance_scores,
                importance_scores_chunks=importance_scores_chunks,
            )
        else:
            raise ValueError(f"Invalid pruning strategy: {pruning_strategy}")

        return_statistics["dims_to_prune_chunks"] = tuple(dims_to_prune_chunks)
        return_statistics["dims_to_prune"] = tuple(dims_to_prune)

        ## Resolve new subspace sizes
        new_subspace_sizes = []
        for old_subspace_size, dims_to_prune_chunk in zip(
            self.subspace_sizes, dims_to_prune_chunks
        ):
            new_subspace_sizes.append(old_subspace_size - len(dims_to_prune_chunk))
        new_subspace_sizes = tuple(new_subspace_sizes)

        return_statistics["new_subspace_sizes"] = new_subspace_sizes

        """ Prune self """

        _dims_to_prune = tuple(x[0] for x in dims_to_prune)

        structured_prune(self.layer, dims_to_prune=_dims_to_prune, prune_output=True)

        """ Prune next layer """

        if self.next_layer is not None:
            if isinstance(self.next_layer, nn.Linear):
                structured_prune(
                    self.next_layer, dims_to_prune=_dims_to_prune, prune_output=False
                )
            elif isinstance(self.next_layer, COB_Mixin):
                structured_prune(
                    self.next_layer.layer,
                    dims_to_prune=_dims_to_prune,
                    prune_output=False,
                )
            else:
                raise ValueError(
                    f"Expected next layer to be a Linear or COB_Mixin, got {type(self.next_layer)}"
                )
        else:
            raise ValueError("No next layer to prune")

        """ Prune norm """

        if getattr(self, "norm", None) is not None:
            structured_prune_norm(self.norm, dims_to_prune=_dims_to_prune)

        """ Adjust act func internal variables """

        self.adjust_subspace_sizes(new_subspace_sizes)

        """ Set new importance scores """

        new_importance_scores = set(importance_scores) - set(dims_to_prune)
        # sort by index
        new_importance_scores = sorted(new_importance_scores, key=lambda x: x[0])
        if not (
            len(new_importance_scores) == self.out_features == sum(self.subspace_sizes)
        ):
            raise ValueError(
                f"Mismatch between number of importance scores and number of output features and sum of number of subspace sizes. Importance scores: {len(new_importance_scores)}, Output features: {self.out_features}, Subspace sizes: {self.subspace_sizes}"
            )
        return_statistics["new_importance_scores"] = new_importance_scores

        self._importance_scores = tuple(x[1] for x in new_importance_scores)

        ## Note: it is the responsiblity of COB_VGG to modify the dimensions specified in config so that the model can be properly re-instantiated from config and loaded back in

        """ Statistics stuff """

        ## TODO: The below is messy and random. Make it more consistent

        return_statistics["pruned_absolute_num"] = (
            return_statistics["initial_out_features"] - self.out_features
        )
        return_statistics["pruned_prop"] = (
            1 - self.out_features / return_statistics["initial_out_features"]
        )

        pruned_absolute_num_chunks = []
        pruned_prop_chunks = []
        for importance_scores_chunk, dims_to_prune_chunk in zip(
            importance_scores_chunks, return_statistics["dims_to_prune_chunks"]
        ):
            pruned_absolute_num_chunks.append(len(dims_to_prune_chunk))
            pruned_prop_chunks.append(
                len(dims_to_prune_chunk) / len(importance_scores_chunk)
            )

        return_statistics["pruned_absolute_num_chunks"] = tuple(
            pruned_absolute_num_chunks
        )
        return_statistics["pruned_prop_chunks"] = tuple(pruned_prop_chunks)

        return return_statistics

    def _structured_prune_proportion(
        self,
        proportion: float | tuple[float],
        importance_scores: list[tuple[int, float, int]],
        importance_scores_chunks: list[list[tuple[int, float, int]]],
    ) -> tuple[list[tuple[int, float, int]], list[list[tuple[int, float, int]]]]:
        """Convert proportion to absolute num"""

        if isinstance(proportion, tuple):
            absolute_num = []
            for subspace_p, subspace_size in zip(proportion, self.subspace_sizes):
                absolute_num.append(int(subspace_p * subspace_size))
            absolute_num = tuple(absolute_num)
        elif isinstance(proportion, float):
            absolute_num = int(proportion * self.out_features)
        else:
            raise ValueError(f"Invalid proportion type: {type(proportion)}")

        return self._structured_prune_absolute_num(
            absolute_num,
            importance_scores=importance_scores,
            importance_scores_chunks=importance_scores_chunks,
        )

    def _structured_prune_absolute_num(
        self,
        absolute_num: int | tuple[int],
        importance_scores: list[tuple[int, float, int]],
        importance_scores_chunks: list[list[tuple[int, float, int]]],
    ) -> tuple[list[tuple[int, float, int]], list[list[tuple[int, float, int]]]]:
        """Check absolute num"""

        if isinstance(absolute_num, tuple):
            for subspace_num, subspace_size in zip(absolute_num, self.subspace_sizes):
                if subspace_num > subspace_size:
                    raise ValueError(
                        f"Cannot prune {subspace_num} dimensions because there are only {subspace_size} dimensions available in the subspace in question. That's just math, silly"
                    )
        elif isinstance(absolute_num, int):
            if absolute_num > self.out_features:
                raise ValueError(
                    f"Cannot prune {absolute_num} dimensions because there are only {self.out_features} dimensions available. Duh"
                )
        else:
            raise ValueError(f"Invalid absolute num type: {type(absolute_num)}")

        """ Find dims to prune """

        if isinstance(absolute_num, tuple):
            ## Figure out which dims to prune per chunk
            dims_to_prune_chunks = []
            for subspace_num, importance_scores_chunk in zip(
                absolute_num, importance_scores_chunks
            ):
                importance_scores_chunk = sorted(
                    importance_scores_chunk, key=lambda x: x[1]
                )
                dims_to_prune_chunks.append(importance_scores_chunk[:subspace_num])

            dims_to_prune = list(chain(*dims_to_prune_chunks))
        else:
            importance_scores = sorted(importance_scores, key=lambda x: x[1])
            dims_to_prune = importance_scores[:absolute_num]
            dims_to_prune_chunks = []
            for i in range(len(self.subspace_sizes)):
                dims_to_prune_chunks.append([x for x in dims_to_prune if x[2] == i])

        return dims_to_prune, dims_to_prune_chunks

    # TODO: refactor this method into _structured_prune_prop_of. And rename it _structured_prune_cutoff or something. Just calculate the cutoff with the zscore.
    def _structured_prune_zscore_cutoff(
        self,
        zscore_cutoff: float | tuple[float],
        importance_scores: list[tuple[int, float, int]],
        importance_scores_chunks: list[list[tuple[int, float, int]]],
    ) -> tuple[list[tuple[int, float, int]], list[list[tuple[int, float, int]]]]:
        """
        Prunes all dims with an importance that has a z-score below zscore_cutoff.
        """

        if isinstance(zscore_cutoff, tuple):
            absolute_num = []
            for importance_scores_chunk, chunk_cutoff in zip(
                importance_scores_chunks, zscore_cutoff
            ):
                _importance_scores_chunk = [x[1] for x in importance_scores_chunk]
                mean = statistics.mean(_importance_scores_chunk)
                std = statistics.stdev(_importance_scores_chunk)
                absolute_num.append(
                    len(
                        [
                            x
                            for x in _importance_scores_chunk
                            if (x - mean) / std < chunk_cutoff
                        ]
                    )
                )
            absolute_num = tuple(absolute_num)
        else:
            _importance_scores = [x[1] for x in importance_scores]
            mean = statistics.mean(_importance_scores)
            std = statistics.stdev(_importance_scores)
            absolute_num = len(
                [x for x in _importance_scores if (x - mean) / std < zscore_cutoff]
            )

        return self._structured_prune_absolute_num(
            absolute_num,
            importance_scores=importance_scores,
            importance_scores_chunks=importance_scores_chunks,
        )

    def _structured_prune_prop_of(
        self,
        prop_of: float | tuple[float],
        metric: str,
        importance_scores: list[tuple[int, float, int]],
        importance_scores_chunks: list[list[tuple[int, float, int]]],
    ) -> tuple[list[tuple[int, float, int]], list[list[tuple[int, float, int]]]]:
        """
        Prunes all dims with an importance that is less than prop_of * metric (eg. the avg/median/max importance).
        """

        if metric not in ("avg", "med", "max"):
            raise ValueError(
                f"Invalid metric: {metric}. Please choose from ('avg', 'med', 'max')"
            )

        if isinstance(prop_of, tuple):
            absolute_num = []
            for importance_scores_chunk, chunk_prop_of in zip(
                importance_scores_chunks, prop_of
            ):
                _importance_scores_chunk = [x[1] for x in importance_scores_chunk]
                if metric == "avg":
                    cutoff = chunk_prop_of * statistics.mean(_importance_scores_chunk)
                elif metric == "med":
                    cutoff = chunk_prop_of * statistics.median(_importance_scores_chunk)
                elif metric == "max":
                    cutoff = chunk_prop_of * max(_importance_scores_chunk)
                absolute_num.append(
                    len([x for x in _importance_scores_chunk if x < cutoff])
                )
            absolute_num = tuple(absolute_num)
        else:
            _importance_scores = [x[1] for x in importance_scores]
            if metric == "avg":
                cutoff = prop_of * statistics.mean(_importance_scores)
            elif metric == "med":
                cutoff = prop_of * statistics.median(_importance_scores)
            elif metric == "max":
                cutoff = prop_of * max(_importance_scores)
            absolute_num = len([x for x in _importance_scores if x < cutoff])

        return self._structured_prune_absolute_num(
            absolute_num,
            importance_scores=importance_scores,
            importance_scores_chunks=importance_scores_chunks,
        )

    def _captured_hs_as_2d(self) -> torch.Tensor:
        captured_hs = self._captured_hidden_states
        ## If hidden states are from conv, flatten 4d tensor into 2d
        if captured_hs.dim() == 4:
            # Transpose channel dim to last dim. Everything else is flattened into batch dim
            num_channels = captured_hs.shape[1]
            captured_hs = captured_hs.transpose(1, -1).reshape(-1, num_channels)

        if captured_hs.dim() != 2:
            raise ValueError(
                f"Captured hidden states must be a 2D tensor. Got {captured_hs.shape}"
            )

        return captured_hs

    def rotate_for_prune(self, strategy: str = "l2"):
        """Checks"""

        if self.cob is not None:
            raise ValueError(
                "Cannot rotate a COB_Mixin layer that already has a COB instantiated. Absorb it first!"
            )

        """ Compute rotation weights """

        if strategy in ("l2", "l1"):
            if getattr(self, "_captured_hidden_states", None) is None:
                raise ValueError(
                    "Cannot rotate a COB_Mixin layer that has not captured hidden states."
                )

            with torch.no_grad():
                captured_hs = self._captured_hs_as_2d()

                ## Split hidden states into chunks by subspace
                captured_hs_chunks = []
                start_dim = 0
                for subspace_size in self.subspace_sizes:
                    end_dim = start_dim + subspace_size
                    captured_hs_chunks.append(captured_hs[:, start_dim:end_dim])
                    start_dim = end_dim

                rot_weights = []
                for captured_hs_chunk in captured_hs_chunks:
                    rot_weights.append(
                        compute_rotation(captured_hs_chunk, norm=strategy)
                    )
        else:
            raise ValueError(f"Invalid strategy: {strategy}.")

        """ Instantiate COB with rot weights """
        self.instantiate_cob(rot_weights=rot_weights)

        ## Now that a rotation has been added, any previously captured hidden states or importance scores are incorrect. So, wipe them
        self.disable_capture_next_hidden_states()
        self._importance_scores = None

    def compute_importance_scores(self, strategy="l2"):
        if strategy in ("l2", "l1"):
            if getattr(self, "_captured_hidden_states", None) is None:
                raise ValueError(
                    f"Cannot compute importance scores for a COB_Mixin layer that has not captured hidden states with strategy {strategy}."
                )

            captured_hs = self._captured_hs_as_2d()

            with torch.no_grad():
                if strategy == "l2":
                    p = 2
                elif strategy == "l1":
                    p = 1
                importance_scores = tuple(
                    torch.norm(captured_hs, p=p, dim=0).cpu().tolist()
                )

            if len(importance_scores) != self.out_features:
                raise ValueError(
                    f"Mismatch between number of importance scores and number of output features. Importance scores: {len(importance_scores)}, Output features: {self.out_features}"
                )

        else:
            raise ValueError(f"Invalid strategy: {strategy}.")

        self._importance_scores = importance_scores


class COB_VGG_Block(nn.Module, COB_Mixin):
    rotation_dim: int = -3  # C, H, W

    def __init__(
        self,
        config: COB_VGG_Config,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        padding="same",
        next_layer: COB_Mixin | nn.Linear | None = None,
    ):
        super().__init__()

        self.config = config

        """ Get conv """

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )

        """ Get norm """
        self.norm = NORM_MAP[self.config.norm]
        if self.norm is not None:
            norm_kwargs = self.config.norm_kwargs or {}
            self.norm = self.norm(out_channels, **norm_kwargs)

        """ Get act func """

        self.act_func = self.config.get_act_func(
            reduce_dim=self.rotation_dim, num_dims=out_channels
        )

        """ Set up change-of-basis """

        self.set_next_layer(next_layer)

        """ Misc """

        self._post_init()

    def forward(self, x):
        x = self.conv(x)

        if self.cob is not None:
            x = self.cob(x)

        if self.norm is not None:
            if self.cob is not None:
                x = self.cob(x, inverse=True)

            x = self.norm(x)

            if self.cob is not None:
                x = self.cob(x)

        x = self.act_func(x)

        if self._capture_next_hidden_states:
            self._capture_hidden_states(x)

        if self.cob is not None:
            x = self.cob(x, inverse=True)

        return x

    def _absorb_cob_self(self, cob: COB_Rotation, cob_after: bool = True):
        absorb_cob_conv2d(conv=self.conv, cob=cob, cob_after=cob_after)

    @property
    def in_channels(self) -> int:
        return self.conv.in_channels

    @property
    def out_channels(self) -> int:
        return self.conv.out_channels

    # For COB_Mixin compability
    @property
    def in_features(self) -> int:
        return self.in_channels

    # For COB_Mixin compability
    @property
    def out_features(self) -> int:
        return self.out_channels

    # For COB_Mixin compability
    @property
    def layer(self):
        return self.conv


class COB_VGG_Linear(nn.Module, COB_Mixin):
    rotation_dim: int = -1

    def __init__(
        self,
        config: COB_VGG_Config,
        in_features: int,
        out_features: int,
        bias: bool = True,
        next_layer: COB_Mixin | nn.Linear | None = None,
    ):
        super().__init__()

        self.config = config

        """ Get conv """

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        """ Get act func """

        self.act_func = self.config.get_act_func(
            reduce_dim=self.rotation_dim, num_dims=out_features
        )

        """ Get dropout """

        self.dropout = DROPOUT_MAP[self.config.dropout_type]
        if self.dropout is not None:
            self.dropout = self.dropout(self.config.dropout_p)

        """ Set up change-of-basis """

        self.set_next_layer(next_layer)

        """ Misc """

        self._post_init()

    def forward(self, x):
        x = self.linear(x)

        if self.cob is not None:
            x = self.cob(x)

        x = self.act_func(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self._capture_next_hidden_states:
            self._capture_hidden_states(x)

        if self.cob is not None:
            x = self.cob(x, inverse=True)

        return x

    def _absorb_cob_self(self, cob: COB_Rotation, cob_after: bool = True):
        absorb_cob_linear(linear=self.linear, cob=cob, cob_after=cob_after)

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    # For COB_Mixin compability
    @property
    def layer(self):
        return self.linear


def make_layers(config: COB_VGG_Config) -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in config.cfg:
        if v == "M":
            layers.append(POOL_MAP[config.pool_type](kernel_size=2, stride=2))
        else:
            block = COB_VGG_Block(
                config=config,
                in_channels=in_channels,
                out_channels=v,
            )
            layers.append(block)
            in_channels = v

    # Link next_layer attributes for each COB block
    next_layer = None
    for layer in reversed(layers):
        if isinstance(layer, COB_Mixin):
            layer.set_next_layer(next_layer)
            next_layer = layer

    return nn.Sequential(*layers)


class COB_VGG(nn.Module):
    def __init__(self, config: COB_VGG_Config):
        super().__init__()

        config.validate()

        self.config = config

        """ Create components """
        self.features = make_layers(self.config)
        if self.config.smaller_32x32_classifier_in:
            self.avgpool = nn.Identity()
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        last_conv_block = next(
            m for m in reversed(self.features) if isinstance(m, COB_VGG_Block)
        )
        last_block_channels = last_conv_block.out_channels

        if self.config.smaller_32x32_classifier_in:
            classifier_in_dims = last_block_channels
        else:
            classifier_in_dims = last_block_channels * 7 * 7

        linear1_out_dims = self.config.classifier_cfg[0]
        linear2_out_dims = self.config.classifier_cfg[1]

        # Instantiate linear objects here in reverse order so we can link next_layer attributes
        output_linear = nn.Linear(linear2_out_dims, self.config.num_classes)
        linear2 = COB_VGG_Linear(
            self.config, linear1_out_dims, linear2_out_dims, next_layer=output_linear
        )
        linear1 = COB_VGG_Linear(
            self.config, classifier_in_dims, linear1_out_dims, next_layer=linear2
        )
        last_conv_block.set_next_layer(linear1)

        self.classifier = nn.Sequential(
            linear1,
            linear2,
            output_linear,
        )

        """ Weight init """
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        if self.config.init_strategy is None:
            pass
        elif self.config.init_strategy == "kaiming_uniform":

            def _kaiming_uniform(m):
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_uniform_(
                        m.weight, mode="fan_out", nonlinearity=self.config.act_func_name
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            self.apply(_kaiming_uniform)
        elif self.config.init_strategy == "original":
            self._original_weight_init()
        else:
            raise ValueError(
                f"Invalid init strategy: {self.config.init_strategy}. Please choose from {list(INIT_STRATEGIES)}"
            )

    @torch.no_grad()
    def _original_weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, RMSNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        elif x.dim() != 4:
            raise ValueError(f"Input must be a 3D or 4D tensor. Got {x.dim()}D tensor")

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def instantiate_cob(
        self,
        orthogonal_map: str = "matrix_exp",
        replace: bool = False,
        freeze_other_params: bool = True,
    ):
        if not replace:
            for m in self.modules():
                if isinstance(m, (COB_VGG_Block, COB_VGG_Linear)):
                    if m.cob is not None:
                        raise ValueError(
                            "COB already instantiated. If you wish to replace, specify replace=True"
                        )

        if freeze_other_params:
            for p in self.parameters():
                p.requires_grad = False

        for m in self.modules():
            if isinstance(m, COB_Mixin):
                m.instantiate_cob(orthogonal_map=orthogonal_map, replace=replace)

    @classmethod
    def from_pretrained(cls, save_dir: os.PathLike) -> "COB_VGG":
        save_dir = Path(save_dir)

        config = COB_VGG_Config.load_from_json(save_dir / "config.json")

        state_dict = torch.load(
            save_dir / "model_state_dict.pth",
            map_location=torch.device("cpu"),
            weights_only=True,
        )

        model = cls(config)

        model.load_state_dict(state_dict)

        return model

    def save_pretrained(self, save_dir: os.PathLike):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.recalculate_config_cfg()
        self.config.save_as_json(save_dir / "config.json", override_if_exists=True)
        torch.save(self.state_dict(), save_dir / "model_state_dict.pth")

    def absorb_cob(self, unfreeze_params: bool = True):
        for m in self.modules():
            if isinstance(m, COB_Mixin):
                m.absorb_cob()

        if unfreeze_params:
            for p in self.parameters():
                p.requires_grad = True

        self.config._cob_orthogonal_map = None

    def get_ordered_cob_modules(self) -> list[COB_Mixin]:
        cob_modules = []
        for m in chain(self.features, self.classifier):
            if isinstance(m, COB_Mixin):
                cob_modules.append(m)

        return cob_modules

    def recalculate_config_cfg(self):
        ## Update self.config.cfg
        cfg = []
        for m in self.features:
            if isinstance(m, tuple(POOL_MAP.values())):
                cfg.append("M")
            elif isinstance(m, COB_VGG_Block):
                cfg.append(m.out_channels)
            else:
                raise ValueError(f"Unrecognized layer type: {type(m)}")
        self.config.cfg = cfg

        ## Update self.config.classifier_cfg
        classifier_cfg = [
            self.classifier[0].out_features,
            self.classifier[1].out_features,
        ]
        self.config.classifier_cfg = classifier_cfg

    def enable_capture_next_hidden_states(self):
        for m in self.modules():
            if isinstance(m, COB_Mixin):
                m.enable_capture_next_hidden_states()

    def disable_capture_next_hidden_states(self):
        for m in self.modules():
            if isinstance(m, COB_Mixin):
                m.disable_capture_next_hidden_states()

    def rotate_for_prune(
        self, data: torch.Tensor, strategy: str = "l2", absorb: bool = True
    ):
        """Checks"""

        for m in self.modules():
            if isinstance(m, COB_Mixin):
                if m._capture_next_hidden_states:
                    raise ValueError(
                        "Cannot rotate a COB_Mixin layer that already has capture hidden states enabled. Disable capture_next_hidden_states first!"
                    )
                if m.cob is not None:
                    raise ValueError(
                        "Cannot rotate a COB_Mixin layer that already has a COB instantiated. Absorb it first!"
                    )

        """ Capture hidden states """

        self.enable_capture_next_hidden_states()

        was_training = self.training
        self.eval()

        with torch.no_grad():
            self(data)

        self.train(was_training)

        """ Rotate """

        for m in self.modules():
            if isinstance(m, COB_Mixin):
                m.rotate_for_prune(strategy=strategy)

        self.disable_capture_next_hidden_states()

        if absorb:
            self.absorb_cob(unfreeze_params=False)

    def compute_importance_scores(self, data: torch.Tensor, strategy: str = "l2"):
        """Checks"""

        for m in self.modules():
            if isinstance(m, COB_Mixin):
                if m._capture_next_hidden_states:
                    raise ValueError(
                        "Cannot compute a COB_Mixin layer that already has capture hidden states enabled. Disable capture_next_hidden_states first!"
                    )

        """ Capture hidden states """

        self.enable_capture_next_hidden_states()

        was_training = self.training
        self.eval()

        with torch.no_grad():
            self(data)

        self.train(was_training)

        """ Compute importance scores """

        for m in self.modules():
            if isinstance(m, COB_Mixin):
                m.compute_importance_scores(strategy=strategy)

        self.disable_capture_next_hidden_states()
