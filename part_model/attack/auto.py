from autoattack_modified import AutoAttack

from .base import AttackModule


class AutoAttackModule(AttackModule):

    def __init__(self, attack_config, core_model, loss_fn, norm, eps,
                 verbose=False, num_classes=10, **kwargs):
        super(AutoAttackModule, self).__init__(
            attack_config, core_model, loss_fn, norm, eps,
            verbose=verbose, **kwargs)
        self.num_classes = num_classes

    def forward(self, x, y):
        mode = self.core_model.training
        self.core_model.eval()
        # TODO: Try to init adversary only once
        adversary = AutoAttack(
            self.core_model,
            norm=self.norm,
            eps=self.eps,
            version='standard',
            verbose=self.verbose,
            num_classes=self.num_classes
        )
        x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
        self.core_model.train(mode)
        return x_adv
