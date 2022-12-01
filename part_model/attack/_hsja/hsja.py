import foolbox
import numpy as np
from foolbox import PyTorchModel
from foolbox.attacks import PGD

from ..base import AttackModule
from .hop_skip_jump import HopSkipJump


class HopSkipJumpAttack(AttackModule):

    def __init__(self, attack_config, core_model, loss_fn, norm, eps, **kwargs):
        super().__init__(attack_config, core_model, loss_fn, norm, eps, **kwargs)
        self.model = PyTorchModel(core_model, bounds=(0, 1), preprocessing=None)
        # (1) gradient_eval_steps is min([initial_gradient_eval_steps *
        # math.sqrt(step + 1), max_gradient_eval_steps]) (L149)
        # (2) step size search also uses a few more queries. Geometric search
        # has while loop and can't be pre-determined (L166)
        # (3) binary search (L184) also has while loop

        # Approximate (upper bound of) `steps`
        # \sum_{i=1}^{steps} (sqrt(i) * init_grad_steps) <= max_iter
        # max_iter = attack_config['hsj_max_iter'] - 51  # 51 for init attack
        max_iter = 20000
        hsj_init_grad_steps = 100
        iters = np.sqrt(np.arange(100)) * hsj_init_grad_steps
        iters = np.cumsum(iters)
        steps = np.where(iters <= max_iter)[0][-1]

        self.attack = HopSkipJump(
            max_queries=max_iter,
            steps=steps,
            initial_gradient_eval_steps=hsj_init_grad_steps,
            max_gradient_eval_steps=10000,
            gamma=1.0,
            constraint='linf',
            verbose=True,
            # init_attack=PGD(rel_stepsize=0.0333, abs_stepsize=None, steps=40, random_start=True),
        )

    def forward(self, x, y):
        mode = self._core_model.training
        self._core_model.eval()
        criteria = foolbox.criteria.Misclassification(y)
        x_adv = self.attack.run(self.model, x, criterion=criteria)
        delta = x_adv - x
        delta.clamp_(- self._eps, self._eps)
        x_adv = x + delta
        x_adv.clamp_(0, 1)
        self._core_model.train(mode)
        return x_adv
