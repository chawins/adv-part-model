# Copyright (c) 2019-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections
import time

import torch

from .fab_base import FABAttack, zero_gradients
from .other_utils import get_pred

# from torch.autograd.gradcheck import zero_gradients


class FABAttack_PT(FABAttack):
    """
    Fast Adaptive Boundary Attack (Linf, L2, L1)
    https://arxiv.org/abs/1907.02044

    :param predict:       forward pass function
    :param norm:          Lp-norm to minimize ('Linf', 'L2', 'L1' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           epsilon for the random restarts
    :param alpha_max:     alpha_max
    :param eta:           overshooting
    :param beta:          backward step
    """

    def __init__(
            self,
            predict,
            norm='Linf',
            n_restarts=1,
            n_iter=100,
            eps=None,
            alpha_max=0.1,
            eta=1.05,
            beta=0.9,
            loss_fn=None,
            verbose=False,
            seed=0,
            targeted=False,
            device=None,
            n_target_classes=9):
        """ FAB-attack implementation in pytorch """

        self.predict = predict
        super().__init__(norm,
                         n_restarts,
                         n_iter,
                         eps,
                         alpha_max,
                         eta,
                         beta,
                         loss_fn,
                         verbose,
                         seed,
                         targeted,
                         device,
                         n_target_classes)

    def _predict_fn(self, x, **kwargs):
        return self.predict(x, **kwargs)

    def _get_predicted_label(self, x, **kwargs):
        with torch.no_grad():
            outputs = self._predict_fn(x, **kwargs)

        # DEBUG
        # _, y = torch.max(outputs, dim=1)
        y = get_pred(outputs)

        return y

    def get_diff_logits_grads_batch(self, imgs, la):
        im = imgs.clone().requires_grad_()
        with torch.enable_grad():
            y = self.predict(im)

        g2 = torch.zeros([y.shape[-1], *imgs.size()]).to(self.device)
        grad_mask = torch.zeros_like(y)
        for counter in range(y.shape[-1]):
            zero_gradients(im)
            grad_mask[:, counter] = 1.0
            y.backward(grad_mask, retain_graph=True)
            grad_mask[:, counter] = 0.0
            g2[counter] = im.grad.data

        g2 = torch.transpose(g2, 0, 1).detach()
        #y2 = self.predict(imgs).detach()
        y2 = y.detach()
        # DEBUG
        if y.size(-1) == 1:
            y2_ = y2.expand(-1, 2, -1, -1, -1)
            y2_[:, 1] *= -1
            df = y2 - y2_[torch.arange(imgs.shape[0]), la].unsqueeze(1)
            g2_ = g2.expand(-1, 2, -1, -1, -1)
            g2_[:, 1] *= -1
            dg = g2 - g2_[torch.arange(imgs.shape[0]), la].unsqueeze(1)
            # FIXME
            df[torch.arange(imgs.shape[0]), la] = 1e10
        else:
            df = y2 - y2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
            dg = g2 - g2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
            df[torch.arange(imgs.shape[0]), la] = 1e10

        return df, dg

    def get_diff_logits_grads_batch_targeted(self, imgs, la, la_target, **kwargs):
        u = torch.arange(imgs.shape[0])
        im = imgs.clone().requires_grad_()
        with torch.enable_grad():

            # print(len(imgs))
            # print(len(kwargs['dino_targets']))
            # import pdb; pdb.set_trace()

            # 244 245 246 247 248 249 250 251 252 253 254 255
            # iterate through parameters of self.predict with indices
            # for i, (name, param) in enumerate(self.predict.named_parameters()): print(i, name, param.size())
            y = self.predict(im, **kwargs)
            diffy = -(y[u, la] - y[u, la_target])
            sumdiffy = diffy.sum()

        zero_gradients(im)
        sumdiffy.backward()
        graddiffy = im.grad.data
        df = diffy.detach().unsqueeze(1)
        dg = graddiffy.unsqueeze(1)

        return df, dg
