"""Base AttackModule class."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn


class AttackModule(nn.Module):
    """Base class of all attack modules."""

    def __init__(
        self,
        core_model: nn.Module | None = None,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        norm: str | None = "Linf",
        eps: float = 8 / 255,
        verbose: bool = False,
        forward_args: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialize AttackModule.

        Args:
            core_model: Target model to attack.
            loss_fn: Loss function to optimize.
            norm: Lp-norm of perturbation (options: "L2", "Linf").
            eps: Epsilon or maximum perturbation norm.
            verbose: If True, progress and messages will be printed.
            forward_args: Additional keyword arguments to pass with any call to
                forward() of core_model.
        """
        super().__init__()
        _ = kwargs  # Unused
        if norm not in (None, "L2", "Linf"):
            raise NotImplementedError(
                f"Norm {norm} is not implemented! Only supports L2 and Linf."
            )
        self._core_model: nn.Module | None = core_model
        self._loss_fn: Callable[..., torch.Tensor] | None = loss_fn
        self._eps: float = eps
        self._norm: str = norm
        self._verbose: bool = verbose
        self._forward_args: dict[str, Any] = (
            forward_args if forward_args is not None else {}
        )
        # Public attributes
        self.use_mask: bool = False
        self.dual_losses: bool = False

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Generates and returns adversarial examples."""
        raise NotImplementedError("Implement forward()!")
