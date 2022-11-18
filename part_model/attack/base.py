"""Base AttackModule class."""

from __future__ import annotations

from typing import Any

from torch import nn


class AttackModule(nn.Module):
    """Base class of all attack modules."""

    def __init__(
        self,
        attack_config: dict[str, Any] | None = None,
        core_model: nn.Module | None = None,
        loss_fn: nn.Module | None = None,
        norm: str = "Linf",
        eps: float = 8 / 255,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Initialize AttackModule."""
        super().__init__()
        if norm not in ("L2", "Linf"):
            raise NotImplementedError(
                f"Norm {norm} is not implemented! Only supports L2 and Linf."
            )
        self.core_model = core_model
        self.loss_fn = loss_fn
        self.eps = eps
        self.norm = norm
        self.verbose = verbose
        # Public attributes
        self.use_mask: bool = False
        self.dual_losses: bool = False
