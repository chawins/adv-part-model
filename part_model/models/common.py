import torch
from torch import nn

_NormVal = list[float] | tuple[float, float, float] | None


class Normalize(nn.Module):
    def __init__(self, mean: _NormVal = None, std: _NormVal = None) -> None:
        super().__init__()
        if mean is None or std is None:
            self.mean, self.std = None, None
        else:
            self.register_buffer(
                "mean", torch.tensor(mean)[None, :, None, None]
            )
            self.register_buffer("std", torch.tensor(std)[None, :, None, None])

    def forward(self, x):
        # TODO(chawins@): place normalization in unified interface.
        if self.mean is None:
            return x
        return (x - self.mean) / self.std
