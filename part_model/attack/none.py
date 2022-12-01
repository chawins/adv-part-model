"""No attack (identity function)."""

import torch

from part_model.attack.base import AttackModule


class NoAttack(AttackModule):
    """No attack."""

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Returns original inputs."""
        _ = targets, kwargs  # Unused
        return inputs
