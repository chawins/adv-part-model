from __future__ import annotations

import torch
from torch import nn

from part_model.attack.pgd import PGDAttack
from part_model.utils.loss import KLDLoss

EPS = 1e-6


class TRADESAttack(PGDAttack):
    def __init__(self, attack_config, core_model, loss_fn, norm, eps, **kwargs):
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        device = next(core_model.parameters()).device
        self._trades_loss_fn: nn.Module = KLDLoss(reduction="sum-non-batch").to(
            device
        )

    def _forward(self, inputs, targets, **kwargs):
        mode = self._core_model.training
        self._core_model.eval()
        inputs.detach_()

        # Initialize worst-case inputs
        x_adv_best = inputs.clone()
        inputs.requires_grad_()
        with torch.enable_grad():
            cl_logits = self._core_model(inputs, **kwargs, **self.forward_args)
        loss_best = torch.zeros(len(inputs), 1, 1, 1, device=inputs.device)

        # Repeat PGD for specified number of restarts
        for _ in range(self.num_restarts):
            x_adv = self._init_adv(inputs)

            # Run PGD on inputs for specified number of steps
            for _ in range(self.num_steps):
                x_adv.requires_grad_()
                # Compute logits, loss, gradients
                with torch.enable_grad():
                    logits = self._core_model(
                        x_adv, **kwargs, **self.forward_args
                    )
                    loss = self._trades_loss_fn(cl_logits, logits).mean()
                    grads = torch.autograd.grad(loss, x_adv)[0].detach()
                self._update_and_proj(x_adv, grads, inputs)

            x_adv_best, loss_best = self._save_best(
                x_adv, targets, x_adv_best, loss_best, **kwargs
            )

        # Return worst-case perturbed input logits
        self._core_model.train(mode)
        return torch.cat([inputs.detach(), x_adv_best.detach()], dim=0)
