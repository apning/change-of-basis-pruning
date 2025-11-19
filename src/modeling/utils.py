from typing import Any
import torch
import torch.nn as nn


class RMSNorm2d(nn.Module):
    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        eps: float | None = None,
        elementwise_affine: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
    ):
        """
        A wrapper around RMSNorm which automatically transposes the input from shape [..., C, H, W] to [..., W, H, C] before RMSNorm (and then returns the shape afterwards) so that RMSNorm can be applied channel-wise.

        Args:
            normalized_shape (int|list[int]|torch.Size): Passed directly to 'normalized_shape' argument in nn.RMSNorm
                If list or list-like, must have only 1-dimension as transpose of dimensions only makes sense for channel-wise RMSNorm.
            eps (float | None): Passed directly to 'eps' argument in nn.RMSNorm
            elementwise_affine (bool): Passed directly to 'elementwise_affine' argument in nn.RMSNorm
            device (Any | None): Passed directly to 'device' argument in nn.RMSNorm
            dtype (Any | None): Passed directly to 'dtype' argument in nn.RMSNorm
        """
        super().__init__()

        if not isinstance(normalized_shape, int) and len(normalized_shape) != 1:
            raise ValueError(
                f"{self.__class__.__name__}: normalized_shape must be an int or a list-like with only 1 dimension! But got: {normalized_shape}"
            )

        self.rn = nn.RMSNorm(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    @property
    def weight(self):
        return self.rn.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            raise ValueError(
                f"{self.__class__.__name__}: Expected input tensor with at least 3 dimensions (..., channels, height, width), but got {x.dim()} dimensions."
            )

        x = x.transpose(-1, -3)  # Change shape from [..., C, H, W] to [..., W, H, C]

        x = self.rn(x)

        x = x.transpose(-1, -3)  # Change shape back to [..., C, H, W]

        return x