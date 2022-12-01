from typing import Any, Callable

import torch
from torch import nn

from part_model.attack.pgd import PGDAttack

_Loss = Callable[..., torch.Tensor]


class SegPGDAttackModule(PGDAttack):
    def __init__(
        self,
        attack_config: dict[str, Any],
        core_model: nn.Module,
        loss_fn: _Loss | tuple[_Loss, _Loss],
        norm: str = "Linf",
        eps: float = 8 / 255,
        **kwargs,
    ) -> None:
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        self._forward_args["return_mask"] = True
        self.use_mask: bool = True
        self.dual_losses: bool = isinstance(loss_fn, (list, tuple))
        if self.dual_losses:
            self.loss_fn1: _Loss = loss_fn[0]
            self.loss_fn2: _Loss = loss_fn[1]

    def _forward_l2(self, x, y, mask, loss_fn=None):
        loss_fn = self._loss_fn if loss_fn is None else loss_fn
        mode = self._core_model.training
        self._core_model.eval()

        # Initialize worst-case inputs
        x_adv_worst = x.clone().detach()
        worst_losses = torch.zeros(len(x), 1, 1, 1, device=x.device)

        # Repeat PGD for specified number of restarts
        for _ in range(self.num_restarts):
            x_adv = x.clone().detach()

            # Initialize adversarial inputs
            x_adv += self._project_l2(torch.randn_like(x_adv), self._eps)
            x_adv.clamp_(0, 1)

            # Run PGD on inputs for specified number of steps
            for _ in range(self.num_steps):
                x_adv.requires_grad_()

                # Compute logits, loss, gradients
                with torch.enable_grad():
                    logits = self._core_model(x_adv, **self._forward_args)
                    loss = loss_fn(logits, y, mask).mean()
                    grads = torch.autograd.grad(loss, x_adv, allow_unused=True)[
                        0
                    ].detach()

                with torch.no_grad():
                    # Perform gradient update, project to norm ball
                    delta = x_adv - x + self._project_l2(grads, self.step_size)
                    x_adv = x + self._project_l2(delta, self._eps)
                    # Clip perturbed inputs to image domain
                    x_adv.clamp_(0, 1)

            if self.num_restarts == 1:
                x_adv_worst = x_adv
            else:
                # Update worst-case inputs with itemized final losses
                out = self._core_model(x_adv, **self._forward_args)
                fin_losses = loss_fn(out, y, mask).reshape(worst_losses.shape)
                up_mask = (fin_losses >= worst_losses).float()
                x_adv_worst = x_adv * up_mask + x_adv_worst * (1 - up_mask)
                worst_losses = fin_losses * up_mask + worst_losses * (
                    1 - up_mask
                )

        # Return worst-case perturbed input logits
        self._core_model.train(mode)
        return x_adv_worst.detach()

    def _forward_linf(self, x, y, mask, loss_fn=None):
        loss_fn = self._loss_fn if loss_fn is None else loss_fn
        mode = self._core_model.training
        self._core_model.eval()

        # Initialize worst-case inputs
        x_adv_worst = x.clone().detach()
        worst_losses = torch.zeros(len(x), 1, 1, 1, device=x.device)

        # Repeat PGD for specified number of restarts
        for _ in range(self.num_restarts):
            x_adv = x.clone().detach()

            # Initialize adversarial inputs
            x_adv += torch.zeros_like(x_adv).uniform_(-self._eps, self._eps)
            x_adv = torch.clamp(x_adv, 0, 1)

            # Run PGD on inputs for specified number of steps
            for _ in range(self.num_steps):
                x_adv.requires_grad_()

                # Compute logits, loss, gradients
                with torch.enable_grad():
                    logits = self._core_model(x_adv, **self._forward_args)
                    loss = loss_fn(logits, y, mask).mean()
                    grads = torch.autograd.grad(loss, x_adv, allow_unused=True)[
                        0
                    ].detach()

                with torch.no_grad():
                    # Perform gradient update, project to norm ball
                    x_adv = x_adv.detach() + self.step_size * torch.sign(grads)
                    x_adv = torch.min(
                        torch.max(x_adv, x - self._eps), x + self._eps
                    )
                    # Clip perturbed inputs to image domain
                    x_adv = torch.clamp(x_adv, 0, 1)

            if self.num_restarts == 1:
                x_adv_worst = x_adv
            else:
                # Update worst-case inputs with itemized final losses
                out = self._core_model(x_adv, **self._forward_args)
                fin_losses = loss_fn(out, y, mask).reshape(worst_losses.shape)
                up_mask = (fin_losses >= worst_losses).float()
                x_adv_worst = x_adv * up_mask + x_adv_worst * (1 - up_mask)
                worst_losses = fin_losses * up_mask + worst_losses * (
                    1 - up_mask
                )

        # Return worst-case perturbed input logits
        self._core_model.train(mode)
        return x_adv_worst.detach()

    def _forward(self, *args, loss_fn=None):
        if self._norm == "L2":
            return self._forward_l2(*args, loss_fn=loss_fn)
        return self._forward_linf(*args, loss_fn=loss_fn)

    def forward(self, *args):
        if self.dual_losses:
            x1 = self._forward(*args, loss_fn=self.loss_fn1)
            x2 = self._forward(*args, loss_fn=self.loss_fn2)
            return torch.cat([x1, x2], axis=0)
        return self._forward(*args)
