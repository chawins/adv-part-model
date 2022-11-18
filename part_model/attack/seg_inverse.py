"""Adaptive attack on part models that focuses on the classifier first."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

import part_model.utils.loss as loss_lib
from part_model.attack.pgd import PGDAttackModule


class SegInverseAttackModule(PGDAttackModule):
    """Attack part models in inverse order.

    This algorithm first attacks the second-stage classifier of the part model
    and uses the generated worst-case mask to attack the segmenter by
    perturbing input to generate mask as close as possible to this one.
    """

    def __init__(
        self,
        attack_config,
        core_model,
        loss_fn,
        norm,
        eps,
        **kwargs,
    ):
        """Initialize SegInverseAttackModule."""
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        self._num_steps = attack_config["pgd_steps"]
        self._step_size = attack_config["pgd_step_size"]
        self._num_restarts = attack_config["num_restarts"]
        self._mask_l2_eps = attack_config["mask_l2_eps"]
        self._mask_l2_step_size = self._mask_l2_eps / self._num_steps * 2
        self._targeted = False
        self._loss_fn = loss_lib.SemiSumLoss(
            seg_const=attack_config["seg_const"],
            reduction="none",
            targeted_seg=True,
            seg_loss_fn="kld",
        )
        self._clf_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.core_model = self.core_model.module

    def _attack_mask(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Perturb input (mask) to second-stage classifier in L2-norm."""
        mode = self.core_model.training
        self.core_model.eval()

        # Perturb in logit space
        x_mask = self.core_model.segmenter(x)
        x_adv = x_mask.clone().detach()
        # Initialize adversarial inputs
        x_adv += self._project_l2(torch.randn_like(x_adv), self._mask_l2_eps)

        # Run PGD on inputs for specified number of steps
        for j in range(self._num_steps):
            x_adv.requires_grad_()

            # Compute logits, loss, gradients
            with torch.enable_grad():
                features = self.core_model.feature_extactor(
                    x_adv, from_logits=True
                )
                logits = self.core_model.core_model(features)
                loss = self._clf_loss_fn(logits, y).mean()
                if self._targeted:
                    loss *= -1
                grads = torch.autograd.grad(loss, x_adv)[0].detach()

            with torch.no_grad():
                # Perform gradient update, project to norm ball
                delta = (
                    x_adv
                    - x_mask
                    + self._project_l2(grads, self._mask_l2_step_size)
                )
                x_adv = x_mask + self._project_l2(delta, self._mask_l2_eps)

            # DEBUG
            # if j % 10 == 0:
            #     from part_model.dataloader.cityscapes_seg import COLORMAP
            #     img = [COLORMAP[m].permute(2, 0, 1) for m in logits[1].argmax(1)]
            #     save_image(img, f'test_{j:03d}.png')

        # DEBUG
        # print(y)
        # show_img = [img for img in x.cpu()][:16]
        # orig_mask = self.core_model(x, forward_mask=True)[1].argmax(1)
        # show_img.extend([COLORMAP[m].permute(2, 0, 1) for m in orig_mask][:16])
        # mask = logits[1].argmax(1)
        # show_img.extend([COLORMAP[m].permute(2, 0, 1) for m in mask][:16])

        # save_image(show_img, 'test.png')
        # import pdb
        # pdb.set_trace()

        # Return worst-case perturbed input logits
        self.core_model.train(mode)
        return F.softmax(x_adv.detach(), dim=1)

    def _forward_l2(self, x, y):
        raise NotImplementedError()

    def _forward_linf(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.core_model.training
        self.core_model.eval()

        # Initialize worst-case inputs
        x_adv_worst = x.clone().detach()
        worst_losses = torch.zeros(len(x), 1, 1, 1, device=x.device) - 1e9

        # Repeat PGD for specified number of restarts
        for _ in range(self._num_restarts):

            x_adv = x + torch.zeros_like(x).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)

            # First find the mask that fools classifier
            adv_mask = self._attack_mask(x, y)

            # Run PGD on inputs for specified number of steps
            for j in range(self._num_steps):
                x_adv.requires_grad_()

                # Compute logits, loss, gradients
                with torch.enable_grad():
                    logits = self.core_model(x_adv, return_mask=True)
                    loss = self._loss_fn(logits, y, adv_mask).mean()
                    if self._targeted:
                        loss *= -1
                    grads = torch.autograd.grad(loss, x_adv)[0].detach()

                with torch.no_grad():
                    # Perform gradient update, project to norm ball
                    x_adv = x_adv.detach() + self._step_size * torch.sign(grads)
                    x_adv = torch.min(
                        torch.max(x_adv, x - self.eps), x + self.eps
                    )
                    # Clip perturbed inputs to image domain
                    x_adv = torch.clamp(x_adv, 0, 1)

                # DEBUG
                # if j % 10 == 0:
                #     from part_model.dataloader.cityscapes_seg import COLORMAP
                #     img = [COLORMAP[m].permute(2, 0, 1) for m in logits[1].argmax(1)]
                #     save_image(img, f'test_{j:03d}.png')

            # import pdb
            # pdb.set_trace()
            if self._num_restarts == 1:
                x_adv_worst = x_adv
            else:
                # Update worst-case inputs with itemized final losses only in
                # 2nd stage.
                fin_losses = self._clf_loss_fn(
                    self.core_model(x_adv, return_mask=False), y
                ).reshape(worst_losses.shape)
                up_mask = (fin_losses >= worst_losses).float()
                x_adv_worst = x_adv * up_mask + x_adv_worst * (1 - up_mask)
                worst_losses = fin_losses * up_mask + worst_losses * (
                    1 - up_mask
                )

        # Return worst-case perturbed input logits
        self.core_model.train(mode)
        return x_adv_worst.detach()

    def _forward(self, *args, **kwargs):
        if self.norm == "L2":
            return self._forward_l2(*args, **kwargs)
        return self._forward_linf(*args, **kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Run attack."""
        return self._forward(x, y)
