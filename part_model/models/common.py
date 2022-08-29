import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, mean, std, *args, **kwargs):
        super().__init__()
        if mean is None or std is None:
            self.mean, self.std = None, None
        else:
            self.register_buffer(
                "mean", torch.tensor(mean)[None, :, None, None]
            )
            self.register_buffer("std", torch.tensor(std)[None, :, None, None])

    def forward(self, x):
        if self.mean is None:
            return x
        return (x - self.mean) / self.std
