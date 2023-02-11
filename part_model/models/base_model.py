"""Base classifier interface."""

from __future__ import annotations

from typing import List, Tuple, Union

import torch
from torch import nn

_NormVal = Union[List[float], Tuple[float, float, float]]


class Classifier(nn.Module):
    """Base Classifier interface."""

    def __init__(
        self,
        model: nn.Module,
        normalize: dict[str, _NormVal] | None = None,
    ) -> None:
        """Initialize Classifier.

        Args:
            model: Main PyTorch model.
            normalize: Dictionary containing normalization values; must contain
                "mean" and "std". Defaults to None.
        """
        super().__init__()
        self._model: nn.Module = model
        self._normalize: dict[str, _NormVal] | None = normalize
        if normalize is not None:
            mean = normalize["mean"]
            std = normalize["std"]
            self.register_buffer(
                "mean", torch.tensor(mean)[None, :, None, None]
            )
            self.register_buffer("std", torch.tensor(std)[None, :, None, None])

    def forward(self, inputs: torch.Tensor, **kwargs):
        """Forward pass.

        Args:
            inputs: Input images.

        Returns:
            Output logits.
        """
        _ = kwargs  # Unused
        if self._normalize is None:
            return inputs
        inputs = (inputs - self.mean) / self.std
        return self._model(inputs)


class SegClassifier(Classifier):
    """Base Classifier interface."""

    def forward(
        self, inputs: torch.Tensor, return_mask: bool = False, **kwargs
    ):
        """Forward pass.

        Args:
            inputs: Input images.
            return_mask: If True, returns predicted segmentation mask together
                with the outputs. Defaults to False.

        Returns:
            Output logits.
        """
        _ = kwargs  # Unused
        if self._normalize is None:
            return inputs
        inputs = (inputs - self.mean) / self.std
        return self._model(inputs, return_mask=return_mask)
