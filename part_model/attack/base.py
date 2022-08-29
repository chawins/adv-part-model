import torch.nn as nn


class AttackModule(nn.Module):

    def __init__(self, attack_config, core_model, loss_fn, norm, eps,
                 verbose=False, **kwargs):
        super(AttackModule, self).__init__()
        self.core_model = core_model
        self.loss_fn = loss_fn
        self.eps = eps
        self.norm = norm
        self.verbose = verbose
        self.use_mask = False
        self.dual_losses = False
        assert self.norm in ('L1', 'L2', 'Linf')
