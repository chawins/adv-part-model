import torch

from part_model.attack.pgd import PGDAttack

EPS = 1e-6


class MaskedPGDAttack(PGDAttack):
    def __init__(self, attack_config, core_model, loss_fn, norm, eps, **kwargs):
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        self.use_mask = True

    @torch.no_grad()
    def _init_adv(
        self, inputs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Initialize adversarial inputs."""
        x_adv = inputs.clone()
        if self._norm == "L2":
            x_adv += self._project_l2(torch.randn_like(x_adv), self._eps) * mask
        else:
            x_adv += (
                torch.zeros_like(x_adv).uniform_(-self._eps, self._eps) * mask
            )
        x_adv.clamp_(0, 1)
        return x_adv

    def _forward(self, x, y, mask=None):
        # TODO(chawins@): clean up
        mode = self._core_model.training
        self._core_model.eval()

        # Initialize worst-case inputs
        x_adv_worst = x.clone().detach()
        worst_losses = torch.zeros(len(x), 1, 1, 1, device=x.device)
        mask = (mask > 0).float().unsqueeze(1)  # TODO

        # Repeat PGD for specified number of restarts
        for _ in range(self.num_restarts):
            x_adv = self._init_adv(x, mask)

            # Run PGD on inputs for specified number of steps
            for _ in range(self.num_steps):
                x_adv.requires_grad_()

                # Compute logits, loss, gradients
                with torch.enable_grad():
                    logits = self._core_model(x_adv, **self.forward_args)
                    loss = self._loss_fn(logits, y).mean()
                    grads = torch.autograd.grad(loss, x_adv)[0].detach()

                with torch.no_grad():
                    # Perform gradient update, project to norm ball
                    x_adv = (
                        x_adv.detach()
                        + self.step_size * torch.sign(grads) * mask
                    )
                    x_adv = torch.min(
                        torch.max(x_adv, x - self._eps), x + self._eps
                    )
                    # Clip perturbed inputs to image domain
                    x_adv = torch.clamp(x_adv, 0, 1)

            if self.num_restarts == 1:
                x_adv_worst = x_adv
            else:
                # Update worst-case inputs with itemized final losses
                fin_losses = self._loss_fn(self._core_model(x_adv), y).reshape(
                    worst_losses.shape
                )
                up_mask = (fin_losses >= worst_losses).float()
                x_adv_worst = x_adv * up_mask + x_adv_worst * (1 - up_mask)
                worst_losses = fin_losses * up_mask + worst_losses * (
                    1 - up_mask
                )

        # Return worst-case perturbed input logits
        self._core_model.train(mode)
        return x_adv_worst.detach()
