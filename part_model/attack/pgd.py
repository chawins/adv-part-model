"""PGD Attack."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from part_model.attack.base import AttackModule

EPS = 1e-6

_Loss = Callable[..., torch.Tensor]


class PGDAttack(AttackModule):
    """PGD Attack."""

    def __init__(
        self,
        attack_config: dict[str, Any],
        core_model: nn.Module,
        loss_fn: Callable[..., torch.Tensor],
        norm: str = "Linf",
        eps: float = 8 / 255,
        **kwargs,
    ) -> None:
        """Initialize PGD Attack.

        For argument description, see AttackModule.
        """
        super().__init__(core_model, loss_fn, norm, eps, **kwargs)
        assert self._norm in ("L2", "Linf")
        self._num_steps: int = attack_config["pgd_steps"]
        self._step_size: float = attack_config["pgd_step_size"]
        self._num_restarts: int = attack_config["num_restarts"]
        self.dual_losses: bool = isinstance(loss_fn, (list, tuple))
        if self.dual_losses:
            self.loss_fn1: _Loss = loss_fn[0]
            self.loss_fn2: _Loss = loss_fn[1]

    @torch.no_grad()
    def _init_adv(self, inputs: torch.Tensor) -> torch.Tensor:
        """Initialize adversarial inputs."""
        x_adv = inputs.clone()
        if self._norm == "L2":
            x_adv += self._project_l2(torch.randn_like(x_adv), self._eps)
        else:
            x_adv += torch.zeros_like(x_adv).uniform_(-self._eps, self._eps)
        x_adv.clamp_(0, 1)
        return x_adv

    @torch.enable_grad()
    def _compute_grads(
        self, x_adv: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute logits, loss, gradients."""
        logits = self._core_model(x_adv, **kwargs, **self._forward_args)
        loss = self._loss_fn(logits, targets).mean()
        grads = torch.autograd.grad(loss, x_adv, allow_unused=True)[0]
        grads = grads.detach()
        return grads

    @torch.no_grad()
    def _update_and_proj(
        self,
        x_adv: torch.Tensor,
        grads: torch.Tensor,
        inputs: torch.Tensor | None = None,
        inputs_min_max: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Perform gradient update, project to norm ball."""
        if self._norm == "L2":
            delta = x_adv - inputs + self._project_l2(grads, self._step_size)
            x_adv = inputs + self._project_l2(delta, self._eps)
        else:
            x_adv += self._step_size * torch.sign(grads)
            x_adv = torch.min(
                torch.max(x_adv, inputs_min_max[0]), inputs_min_max[1]
            )
        # Clip perturbed inputs to image domain
        x_adv.clamp_(0, 1)
        return x_adv

    @torch.no_grad()
    def _save_best(
        self,
        x_adv: torch.Tensor,
        targets: torch.Tensor,
        x_adv_best: torch.Tensor,
        loss_best: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keeps best attack and loss."""
        if self._num_restarts == 1:
            x_adv_best = x_adv
        else:
            # Update worst-case inputs with itemized final losses
            fin_losses = self._loss_fn(
                self._core_model(x_adv, **kwargs, **self._forward_args),
                targets,
            ).reshape(loss_best.shape)
            up_mask = (fin_losses >= loss_best).float()
            x_adv_best = x_adv * up_mask + x_adv_best * (1 - up_mask)
            loss_best = fin_losses * up_mask + loss_best * (1 - up_mask)
        return x_adv_best, loss_best

    def _project_l2(self, inputs: torch.Tensor, eps: float) -> torch.Tensor:
        dims = [-1] + [1] * (inputs.ndim - 1)
        return (
            inputs
            / (inputs.view(len(inputs), -1).norm(2, 1).view(dims) + EPS)
            * eps
        )

    def _forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: _Loss | None = None,
        **kwargs,
    ) -> torch.Tensor:
        loss_fn: _Loss = self._loss_fn if loss_fn is None else loss_fn
        mode = self._core_model.training
        self._core_model.eval()
        inputs.detach_()
        inputs_min = inputs - self._eps
        inputs_max = inputs + self._eps

        # Initialize worst-case inputs
        x_adv_best = inputs.clone()
        loss_best = torch.zeros(len(inputs), 1, 1, 1, device=inputs.device)
        loss_best -= 1e9

        # Repeat PGD for specified number of restarts
        for _ in range(self._num_restarts):
            x_adv = self._init_adv(inputs)

            # Run PGD on inputs for specified number of steps
            for _ in range(self._num_steps):
                x_adv.requires_grad_()
                grads = self._compute_grads(x_adv, targets, **kwargs)
                x_adv = self._update_and_proj(
                    x_adv,
                    grads,
                    inputs=inputs,
                    inputs_min_max=(inputs_min, inputs_max),
                )

            x_adv_best, loss_best = self._save_best(
                x_adv, targets, x_adv_best, loss_best, **kwargs
            )

        # DEBUG
        # from torchvision.utils import save_image
        # print(y)
        # show_img = [img for img in x.cpu()][:16]
        # orig_mask = self.core_model(x, **self.forward_args)[1].argmax(1)
        # show_img.extend([COLORMAP[m].permute(2, 0, 1) for m in orig_mask][:16])
        # mask = logits[1].argmax(1)
        # show_img.extend([COLORMAP[m].permute(2, 0, 1) for m in mask][:16])

        # save_image(show_img, 'test.png')
        # import pdb
        # pdb.set_trace()

        # Return worst-case perturbed input logits
        self._core_model.train(mode)
        return x_adv_best

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Generates and returns adversarial examples."""
        if self.dual_losses:
            x_adv1 = self._forward(*args, **kwargs, loss_fn=self.loss_fn1)
            x_adv2 = self._forward(*args, **kwargs, loss_fn=self.loss_fn2)
            return torch.cat([x_adv1, x_adv2], axis=0)
        return self._forward(*args, **kwargs)
