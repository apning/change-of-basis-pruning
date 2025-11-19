import torch
import torch.nn as nn
from src.act_funcy.utils import _move_to_last


class TSRALogisiticFunc(nn.Module):
    def __init__(
        self,
        a_U: float = 5.0,
        a_V: float = 5.0,
        b_U: float = 0.5,
        b_V: float = 0.7,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.a_U = a_U
        self.a_V = a_V
        self.b_U = b_U
        self.b_V = b_V
        self.eps = eps

    def forward(self, s_U: torch.Tensor, s_V: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_U (torch.Tensor): Tensor indicating the euclidean norm of the x_U subspace
            s_V (torch.Tensor): Tensor indicating the euclidean norm of the x_V subspace
        Returns:
            g_U, g_V (torch.Tensor): Scalar gains of same shape as s_U and s_V

        """

        """ Checks """
        if s_U.shape != s_V.shape:
            raise ValueError(
                f"s_U and s_V must have the same shape. Got shapes {s_U.shape} and {s_V.shape}."
            )

        """ Do the math part """
        # Calculate the euclidean norm of the whole latent state x
        s = torch.sqrt(s_U**2 + s_V**2)

        # Calculate the ratio
        r = s_U / (s + self.eps)

        # Calculate the gains
        g_U = torch.sigmoid(self.a_U * (r - self.b_U))
        g_V = torch.sigmoid(self.a_V * (r - self.b_V))

        return g_U, g_V


class TSRA(nn.Module):
    """
    Two-Subspace Radial Activation (TSRA) with optional norm-setting modes.

    Splits the channel axis into two subspaces and applies gains g_U,g_V that depend
    only on the two subspace norms (s_U, s_V). Two per-subspace modes control how
    the gains are applied:

      - If norm_mode_Z = False (default):  out_Z = g_Z * x_Z
      - If norm_mode_Z = True:             out_Z = (g_Z / (||x_Z|| + eps)) * x_Z
        (so the resulting subspace norm equals g_Z)

    Args:
        scaling_func: Function which takes as input two tensors s_U and s_V representing batched euclidean norms of latent states. Outputs gains g_U, g_V. Defaults to an instance of TSRALogisiticFunc if not specified
        subspace2_start (int): Start index of subspace B
        expected_dims (int): Expected channel count
        reduce_dim (int): Channel-like dimension to split/scale (default 1).
        norm_mode_U (bool): If True, set final L2 norm of subspace A to gA per position.
        norm_mode_V (bool): If True, set final L2 norm of subspace B to gB per position.
        eps (float): Numerical stability term
    """

    def __init__(
        self,
        scaling_func=None,
        *,
        subspace2_start: int,
        expected_dims: int,
        reduce_dim: int = 1,
        norm_mode_U: bool = False,
        norm_mode_V: bool = False,
        eps: float = 1e-12,
    ):
        super().__init__()

        """ Check/Process args """

        if subspace2_start is None or expected_dims is None:
            raise ValueError("TSRA requires subspace2_start and expected_dims.")
        if not (1 <= subspace2_start < expected_dims):
            raise ValueError(
                f"subspace2_start must be in [1, expected_dims-1]; "
                f"got subspace2_start={subspace2_start}, expected_dims={expected_dims}."
            )

        scaling_func = scaling_func or TSRALogisiticFunc()

        """ Assign attributes """

        self.scaling_func = scaling_func
        self.reduce_dim = reduce_dim
        self.subspace2_start = subspace2_start
        self.expected_dims = expected_dims

        self.norm_mode_U = norm_mode_U
        self.norm_mode_V = norm_mode_V
        self.eps = eps

    @property
    def subspace2_start(self) -> int | None:
        if self._subspace2_start is None:
            return None
        return self._subspace2_start.item()

    @subspace2_start.setter
    def subspace2_start(self, value: int | None):
        if value is not None:
            value = torch.tensor(value, dtype=torch.long)
        self.register_buffer("_subspace2_start", value)

    @property
    def expected_dims(self) -> int | None:
        if self._expected_dims is None:
            return None
        return self._expected_dims.item()

    @expected_dims.setter
    def expected_dims(self, value: int | None):
        if value is not None:
            value = torch.tensor(value, dtype=torch.long)
        self.register_buffer("_expected_dims", value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Checks"""
        # Move reduction dim to last dim
        x, undo = _move_to_last(x, self.reduce_dim)
        # check if expected number of dims
        if (dims := x.shape[-1]) != self.expected_dims:
            raise ValueError(
                f"TSRA expected {self.expected_dims} dims, but got {dims}."
            )

        """ Start process """
        ## Split into two subspaces
        x_U = x[..., : self.subspace2_start]
        x_V = x[..., self.subspace2_start :]
        if x_U.shape[-1] == 0 or x_V.shape[-1] == 0:
            raise ValueError("Invalid split: At least one of the subspaces is empty.")

        ## Calculate euclidean norm per subspace
        s_U = x_U.norm(p=2, dim=-1, keepdim=True)
        s_V = x_V.norm(p=2, dim=-1, keepdim=True)

        # Calculate gains
        g_U, g_V = self.scaling_func(s_U, s_V)

        # Apply modes per subspace
        if self.norm_mode_U:
            out_x_U = (g_U / (s_U + self.eps)) * x_U
        else:
            out_x_U = g_U * x_U

        if self.norm_mode_V:
            out_x_V = (g_V / (s_V + self.eps)) * x_V
        else:
            out_x_V = g_V * x_V

        out_X = torch.cat([out_x_U, out_x_V], dim=-1)

        return undo(out_X)
