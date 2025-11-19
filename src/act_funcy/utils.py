import torch


def _resolve_dim(x: torch.Tensor, reduce_dim: int) -> int:
    """Map possibly-negative reduce_dim to [0, x.dim()-1]."""
    if reduce_dim < 0:
        reduce_dim = x.dim() + reduce_dim
    if not (0 <= reduce_dim < x.dim()):
        raise ValueError(f"reduce_dim={reduce_dim} out of range for x.dim()={x.dim()}")
    return reduce_dim


def _move_to_last(x: torch.Tensor, dim: int):
    """Return y with `dim` moved to last and a callable `undo` to restore."""
    dim = _resolve_dim(x, dim)
    if dim == x.dim() - 1:
        return x, lambda t: t  # already last
    order = [d for d in range(x.dim()) if d != dim] + [dim]
    inv = [0] * x.dim()
    for i, d in enumerate(order):
        inv[d] = i
    return x.permute(order), lambda t: t.permute(inv)
