import numpy as np
import torch

from ..base import AttackModule
from .general_torch_model import GeneralTorchModel
from .pgbar import progress_bar


class RayS(AttackModule):

    def __init__(self, attack_config, core_model, loss_fn, norm, eps,
                 num_classes=10, **kwargs):
        super().__init__(attack_config, core_model, loss_fn, norm, eps,
                         verbose=False, **kwargs)
        self.model = GeneralTorchModel(
            core_model, n_class=num_classes, im_mean=None, im_std=None)
        self.ord = np.inf
        self.epsilon = eps
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.queries = None
        self.query_limit = 10000
        # DEBUG
        self.verbose = False

    def get_xadv(self, x, v, d, lb=0., ub=1.):
        if isinstance(d, int):
            d = torch.tensor(d).repeat(len(x)).cuda()
        out = x + d.view(len(x), 1, 1, 1) * v
        out = torch.clamp(out, lb, ub)
        return out

    def attack_hard_label(self, x, y, target=None, seed=None):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            (x, y): original image
        """
        shape = list(x.shape)
        dim = np.prod(shape[1:])
        if seed is not None:
            np.random.seed(seed)

        # init variables
        self.queries = torch.zeros_like(y).cuda()
        self.sgn_t = torch.sign(torch.ones(shape)).cuda()
        self.d_t = torch.ones_like(y).float().fill_(float("Inf")).cuda()
        working_ind = (self.d_t > self.epsilon).nonzero().flatten()

        stop_queries = self.queries.clone()
        dist = self.d_t.clone()
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)

        block_level = 0
        block_ind = 0
        for i in range(self.query_limit):
            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            valid_mask = (self.queries < self.query_limit)
            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[valid_mask.nonzero().flatten(), start:end] *= -1.
            attempt = attempt.view(shape)

            self.binary_search(x, y, target, attempt, valid_mask)

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = torch.norm((self.x_final - x).reshape(shape[0], -1), self.ord, 1)
            stop_queries[working_ind] = self.queries[working_ind]
            working_ind = (dist > self.epsilon).nonzero().flatten()

            if torch.sum(self.queries >= self.query_limit) == shape[0]:
                if self.verbose:
                    print('out of queries')
                break

            if self.verbose:
                progress_bar(torch.min(self.queries.float()), self.query_limit,
                             'd_t: %.4f | adbd: %.4f | queries: %.4f | rob acc: %.4f | iter: %d'
                             % (torch.mean(self.d_t), torch.mean(dist), torch.mean(self.queries.float()),
                                len(working_ind) / len(x), i + 1))

        stop_queries = torch.clamp(stop_queries, 0, self.query_limit)
        return self.x_final, stop_queries, dist, (dist <= self.epsilon)

    # check whether solution is found
    def search_succ(self, x, y, target, mask):
        self.queries[mask] += 1
        if target:
            return self.model.predict_label(x[mask]) == target[mask]
        else:
            return self.model.predict_label(x[mask]) != y[mask]

    # binary search for decision boundary along sgn direction
    def binary_search(self, x, y, target, sgn, valid_mask, tol=1e-3):
        sgn_norm = torch.norm(sgn.view(len(x), -1), 2, 1)
        sgn_unit = sgn / sgn_norm.view(len(x), 1, 1, 1)

        d_start = torch.zeros_like(y).float().cuda()
        d_end = self.d_t.clone()

        initial_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target, valid_mask)
        to_search_ind = valid_mask.nonzero().flatten()[initial_succ_mask]
        d_end[to_search_ind] = torch.min(self.d_t, sgn_norm)[to_search_ind]

        while len(to_search_ind) > 0:
            d_mid = (d_start + d_end) / 2.0
            search_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, d_mid), y, target, to_search_ind)
            d_end[to_search_ind[search_succ_mask]] = d_mid[to_search_ind[search_succ_mask]]
            d_start[to_search_ind[~search_succ_mask]] = d_mid[to_search_ind[~search_succ_mask]]
            to_search_ind = to_search_ind[((d_end - d_start)[to_search_ind] > tol)]

        to_update_ind = (d_end < self.d_t).nonzero().flatten()
        if len(to_update_ind) > 0:
            self.d_t[to_update_ind] = d_end[to_update_ind]
            self.x_final[to_update_ind] = self.get_xadv(x, sgn_unit, d_end)[to_update_ind]
            self.sgn_t[to_update_ind] = sgn[to_update_ind]

    def forward(self, x, y, target=None):
        x_adv, _, _, idx_success = self.attack_hard_label(x, y, target=target)
        # x_adv[~idx_success] = x[~idx_success]
        delta = x_adv - x
        delta.clamp_(- self._eps, self._eps)
        x_adv = x + delta
        x_adv.clamp_(0, 1)
        return x_adv
