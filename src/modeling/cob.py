import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal

from src.act_funcy.utils import _resolve_dim


class COB_Rotation(nn.Module):
    def __init__(
        self,
        subspace_sizes: list[int] | tuple[int],
        rotation_dim: int = -1,
        orthogonal_map: str = "matrix_exp",
        rot_weights: list[torch.Tensor] | tuple[torch.Tensor] | None = None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ):
        super().__init__()

        self.subspace_sizes = list(subspace_sizes)
        self.rotation_dim = rotation_dim

        self.rotations = nn.ModuleList()

        if rot_weights is not None and len(rot_weights) != len(self.subspace_sizes):
            raise ValueError(
                f"Expected {len(self.subspace_sizes)} rotation weights, got {len(rot_weights)}"
            )

        for i, size in enumerate(subspace_sizes):
            linear = nn.Linear(size, size, bias=False, device=device, dtype=dtype)
            if rot_weights is not None:
                rot_weight = rot_weights[i]
                if rot_weight.shape != (size, size):
                    raise ValueError(
                        f"Expected rotation weight shape ({size}, {size}) at index {i}, got {rot_weight.shape}"
                    )
                rot_weight = rot_weight.to(device).to(dtype)
                with torch.no_grad():
                    linear.weight.data.copy_(rot_weight)
                
            rotation = orthogonal(
                linear,
                orthogonal_map=orthogonal_map,
            )
            self.rotations.append(rotation)

    @property
    def expected_dims(self) -> int:
        return sum(self.subspace_sizes)

    def forward(self, x: torch.Tensor, inverse: bool = False):
        ## Transpose x if necessary so dim to rotate is last dim

        if self.rotation_dim in (-1, _resolve_dim(x, -1)):
            transpose_dims = False
        else:
            transpose_dims = True

        if transpose_dims:
            x = x.transpose(-1, self.rotation_dim)

        ## Ensure last dim of expected size

        if x.size(-1) != self.expected_dims:
            raise ValueError(
                f"Expected {self.expected_dims} dimensions, got {x.size(-1)}. x shape after possible dim transpose: {x.shape}"
            )

        ## Split x up into its subspace chunks

        subspace_chunks = []
        start_dim = 0
        for size in self.subspace_sizes:
            end_dim = start_dim + size
            subspace_chunks.append(x[..., start_dim:end_dim])
            start_dim = end_dim

        ## Apply respective rotation to each subspace chunk
        for i, (subspace_chunk, rotation) in enumerate(
            zip(subspace_chunks, self.rotations)
        ):
            rot_weight = rotation.weight
            if not inverse:
                rot_weight = rot_weight.T

            subspace_chunk = subspace_chunk @ rot_weight

            subspace_chunks[i] = subspace_chunk

        ## Merge subspace chunks back together
        x = torch.cat(subspace_chunks, dim=-1)

        ## Transpose if necessary
        if transpose_dims:
            x = x.transpose(-1, self.rotation_dim)

        return x


@torch.no_grad()
def absorb_cob_conv2d(conv: nn.Conv2d, cob: COB_Rotation, cob_after: bool = True):
    """Shape checks"""
    cw = conv.weight.data
    cb = conv.bias.data if conv.bias is not None else None
    cw_shape = cw.shape
    out_f, in_f, k1, k2 = cw_shape

    if cob_after:
        if cob.expected_dims != out_f:
            raise ValueError(
                f"COB expected {cob.expected_dims} dimensions but conv layer has {out_f} output channels. conv weight shape: {cw.shape}"
            )
    else:
        if cob.expected_dims != in_f:
            raise ValueError(
                f"COB expected {cob.expected_dims} dimensions but conv layer has {in_f} input channels. conv weight shape: {cw.shape}"
            )

    """ Reshape to match linear shape """

    if cob_after:
        cw = cw.reshape(out_f, -1)
    else:
        cw = cw.transpose(1, -1).reshape(-1, in_f)

    """ Absorb cob """

    cw, cb = _absorb_cob_linear(weight=cw, bias=cb, cob=cob, cob_after=cob_after)

    """ Reshape back into conv shape """

    if cob_after:
        cw = cw.reshape(cw_shape)
    else:
        cw = cw.reshape(out_f, k2, k1, in_f).transpose(1, -1)

    """ Copy absorbed weights into conv """

    conv.weight.data.copy_(cw)
    if conv.bias is not None:
        conv.bias.data.copy_(cb)


@torch.no_grad()
def absorb_cob_linear(linear: nn.Linear, cob: COB_Rotation, cob_after: bool = True):
    weight = linear.weight.data
    bias = linear.bias.data if linear.bias is not None else None

    weight, bias = _absorb_cob_linear(
        weight=weight, bias=bias, cob=cob, cob_after=cob_after
    )

    """ Copy absorbed weights into linear """

    linear.weight.data.copy_(weight)
    if linear.bias is not None:
        linear.bias.data.copy_(bias)


@torch.no_grad()
def _absorb_cob_linear(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cob: COB_Rotation,
    cob_after: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shape checks"""

    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got {weight.dim()}D tensor")
    if bias is not None and bias.dim() != 1:
        raise ValueError(f"Expected 1D bias tensor, got {bias.dim()}D tensor")
    out_f, in_f = weight.shape
    if cob_after:
        if cob.expected_dims != out_f:
            raise ValueError(
                f"COB expected {cob.expected_dims} dimensions but linear layer has {out_f} output channels. linear weight shape: {weight.shape}"
            )
    else:
        if cob.expected_dims != in_f:
            raise ValueError(
                f"COB expected {cob.expected_dims} dimensions but linear layer has {in_f} input channels. linear weight shape: {weight.shape}"
            )

    """ Partition linear weight and bias into subspace chunks """

    weight_chunks = []
    bias_chunks = [] if cob_after and bias is not None else None
    start_dim = 0
    for size in cob.subspace_sizes:
        end_dim = start_dim + size

        if cob_after:
            weight_chunks.append(weight[start_dim:end_dim])
        else:
            weight_chunks.append(weight[:, start_dim:end_dim])

        if cob_after and bias_chunks is not None:
            bias_chunks.append(bias[start_dim:end_dim])

        start_dim = end_dim

    """ Apply rotation to chunks """

    for i in range(len(weight_chunks)):
        rot_weight = cob.rotations[i].weight.data
        weight_chunk = weight_chunks[i]

        if cob_after:
            weight_chunks[i] = rot_weight @ weight_chunk
            if bias_chunks is not None:
                bias_chunks[i] = rot_weight @ bias_chunks[i]
        else:
            weight_chunks[i] = weight_chunk @ rot_weight.T

    """ Concat chunks back together """

    if cob_after:
        absorbed_weight = torch.cat(weight_chunks, dim=0)
        absorbed_bias = torch.cat(bias_chunks, dim=0) if bias is not None else None
    else:
        absorbed_weight = torch.cat(weight_chunks, dim=1)
        absorbed_bias = bias

    return absorbed_weight, absorbed_bias
