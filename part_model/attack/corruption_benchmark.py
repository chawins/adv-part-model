
import numpy as np
import torch
from imagecorruptions import corrupt

from .base import AttackModule


class CorruptionBenchmarkModule(AttackModule):

    def __init__(self, attack_config, core_model, loss_fn, norm, eps,
                 corruption_number, **kwargs):
        super(CorruptionBenchmarkModule, self).__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs)
        self.corruption_number = corruption_number

    def forward(self, x, y):
        x = x.clone().detach()
        batch_size = x.shape[0]
        device = torch.device(f'cuda:{x.get_device()}')
        x = x.permute((0, 2, 3, 1)) * 255
        x = x.cpu().numpy().astype('uint8')
        cor_x = np.empty((batch_size, 5, x.shape[1], x.shape[2], x.shape[3]))
        for i in range(batch_size):
            cor_x[i] = np.asarray([corrupt(x[i], corruption_number=self.corruption_number, severity=j)
                                  for j in range(1, 6)])
        cor_x = torch.from_numpy(cor_x).permute(0, 1, 4, 2, 3).to(device).float() / 255
        cor_x = torch.flatten(cor_x, start_dim=0, end_dim=1)

        return cor_x
