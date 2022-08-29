
from .base import AttackModule


class NoAttackModule(AttackModule):

    def __init__(self, attack_config, core_model, loss_fn, norm, eps, **kwargs):
        super(NoAttackModule, self).__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs)

    def forward(self, x, y):
        return x
