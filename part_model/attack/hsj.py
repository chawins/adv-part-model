import foolbox
from foolbox import PyTorchModel
from foolbox.attacks import HopSkipJumpAttack

from .base import AttackModule


class HopSkipJump(AttackModule):

    def __init__(self, attack_config, core_model, loss_fn, norm, eps,
                 num_classes=10, **kwargs):
        super().__init__(attack_config, core_model, loss_fn, norm, eps, **kwargs)
        self.model = PyTorchModel(core_model, bounds=(0, 1), preprocessing=None)
        # max_iter = 5000
        hsj_init_grad_steps = 100
        hsj_max_grad_steps = 10000
        hsj_gamma = 1
        steps = 32

        self.attack = HopSkipJumpAttack(
            steps=steps,
            initial_gradient_eval_steps=hsj_init_grad_steps,
            max_gradient_eval_steps=hsj_max_grad_steps,
            gamma=hsj_gamma,
            constraint='linf',
            stepsize_search='geometric_progression',
        )

    def forward(self, x, y, target=None):
        if target is None:
            criteria = foolbox.criteria.Misclassification(y)
            starting_points = None
        else:
            criteria = foolbox.criteria.TargetedMisclassification(target[1])
            starting_points = target[0]
        x_adv = self.attack.run(self.model, x,
                                criterion=criteria,
                                starting_points=starting_points)
        delta = x_adv - x
        print(delta.view(x.size(0), -1).max(1))
        delta.clamp_(- self._eps, self._eps)
        x_adv = x + delta
        x_adv.clamp_(0, 1)
        return x_adv
