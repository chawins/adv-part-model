"""Attack used in TRADES adversarial training."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from part_model.attack.pgd import PGDAttack
from part_model.utils.loss import KLDLoss

_Loss = Callable[..., torch.Tensor]


class TRADESAttack(PGDAttack):
    """TRADES Attack."""

    def __init__(
        self,
        attack_config: dict[str, Any],
        core_model: nn.Module,
        loss_fn: Callable[..., torch.Tensor],
        norm: str = "Linf",
        eps: float = 8 / 255,
        **kwargs,
    ) -> None:
        """Initialize TRADES Attack.

        For argument description, see AttackModule.
        """
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        device = next(core_model.parameters()).device
        self._trades_loss_fn: _Loss = KLDLoss(reduction="sum-non-batch").to(
            device
        )

    @torch.enable_grad()
    def _compute_grads(
        self, x_adv: torch.Tensor, cl_logits: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute logits, loss, gradients."""
        x_adv.requires_grad_()
        logits = self._core_model(x_adv, **kwargs, **self._forward_args)
        # pylint: disable=not-callable
        loss = self._trades_loss_fn(cl_logits, logits).mean()
        grads = torch.autograd.grad(loss, x_adv, allow_unused=True)[0]
        grads.detach_()
        return grads

    def _forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: _Loss | None = None,
        **kwargs,
    ) -> torch.Tensor:
        _ = loss_fn  # Unused
        mode = self._core_model.training
        self._core_model.eval()
        inputs.detach_()
        inputs_min = inputs - self._eps
        inputs_max = inputs + self._eps

        # Initialize worst-case inputs
        x_adv_best = inputs.clone()
        inputs.requires_grad_()
        with torch.enable_grad():
            cl_logits = self._core_model(inputs, **kwargs, **self._forward_args)
        loss_best = torch.zeros(len(inputs), 1, 1, 1, device=inputs.device)
        loss_best -= 1e9

        # Repeat PGD for specified number of restarts
        for _ in range(self._num_restarts):
            x_adv = self._init_adv(inputs)

            # Run PGD on inputs for specified number of steps
            for _ in range(self._num_steps):
                grads = self._compute_grads(x_adv, cl_logits, **kwargs)
                x_adv = self._update_and_proj(
                    x_adv,
                    grads,
                    inputs=inputs,
                    inputs_min_max=(inputs_min, inputs_max),
                )

            x_adv_best, loss_best = self._save_best(
                x_adv, targets, x_adv_best, loss_best, **kwargs
            )

        # Return worst-case perturbed input logits
        self._core_model.train(mode)
        return torch.cat([inputs.detach(), x_adv_best.detach()], dim=0)
