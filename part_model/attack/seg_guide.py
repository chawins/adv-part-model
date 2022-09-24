import glob
import os

import numpy as np
import torch
from torchvision.utils import save_image
import torch.nn.functional as F

from .base import AttackModule

EPS = 1e-6


class SegGuidedAttackModule(AttackModule):
    def __init__(
        self,
        attack_config,
        core_model,
        loss_fn,
        norm,
        eps,
        dataloader=None,
        num_guides=1000,
        input_dim=None,
        classifier=None,
        no_bg=False,
        num_classes=10,
        **kwargs
    ):
        super(SegGuidedAttackModule, self).__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        assert self.norm in ("L2", "Linf")
        self.classifier = classifier
        self.num_classes = num_classes
        self.num_steps = attack_config["pgd_steps"]
        self.step_size = attack_config["pgd_step_size"]
        self.num_restarts = attack_config["num_restarts"]
        self._load_guides(dataloader, num_guides, input_dim)

    def _load_guides(self, dataloader, num_guides, input_dim):
        """Load guide masks into memory"""
        # TODO: apply to general dataset
        # from part_model.dataloader.cityscapes import seg_file_to_mask

        # dirs = os.listdir(guide_path)
        # classes = sorted(
        #     [d for d in dirs if os.path.isdir(os.path.join(guide_path, d))]
        # )
        # size = input_dim[1:]
        # shape = (
        #     len(classes),
        #     num_guides,
        # ) + size
        # self.guides = torch.zeros(shape, dtype=torch.long)
        # self.guide_masks = torch.zeros(shape, dtype=torch.float32)
        guide_masks = []
        labels = []
        pred_scores = []
        
        # for j, c in enumerate(classes):
        for _, segs, targets in dataloader:
            # search_files = os.path.join(guide_path, c, "*.tif")
            # filenames = sorted(glob.glob(search_files))
            # np.random.seed(0)
            # np.random.shuffle(filenames)
            # for i, filename in enumerate(filenames[:num_guides]):
            #     guide, mask = seg_file_to_mask(filename)
            #     self.guides[j][i] = guide
            #     self.guide_masks[j][i] = mask[:, :, j]

            # Get classifier's score on gt masks
            onehot_segs = F.one_hot(segs, num_classes=self.num_classes)
            # classifier takes logit masks but gt mask is categorical so we
            # scale it with a large constant (like temperature).
            logits = self.classifier(onehot_segs * 1e3)
            pred_scores = F.softmax(logits, dim=-1)

            guide_masks.append(segs)
            labels.append(targets)
            pred_scores.append(pred_scores)

        # Sort scores by class

        self.guide_masks = torch.cat(guide_masks, dim=0)
        self.guide_labels = torch.cat(labels, dim=0)
        self.guide_scores = torch.cat(pred_scores, dim=0)
            

    def _select_guide(self, x, y):
        """
        Select `n` guide masks with the highest similarity to the predicted
        mask for `x` where `n` is `num_restarts`.
        """
        with torch.no_grad():
            masks = self.core_model(x, return_mask=True)[1].detach().cpu()
            # masks = masks.argmax(1).cpu()

        guides = torch.zeros(
            (self.num_restarts,) + masks.shape,
            device=x.device,
            dtype=self.guides.dtype,
        )
        guide_masks = torch.zeros(
            (self.num_restarts,) + masks.shape, device=x.device, dtype=x.dtype
        )
        for i, (mask, label) in enumerate(zip(masks, y)):
            # TODO: handle multi-class and move to targeted attack
            tgt_label = 1 - label
            # match_scores = mask[None, :, :] == self.guides[tgt_label]
            match_scores = ((mask.unsqueeze(0) - self.guides[tgt_label]) ** 2).sum((2, 3))
            match_scores = match_scores.float().sum((1, 2))
            idx = match_scores.topk(self.num_restarts)[1]
            guides[:, i] = self.guides[tgt_label][idx]
            guide_masks[:, i] = self.guide_masks[tgt_label][idx]
            
        return guides.to(x.device), guide_masks.to(x.device)

    def _project_l2(self, x, eps):
        dims = [-1,] + [
            1,
        ] * (x.ndim - 1)
        return x / (x.view(len(x), -1).norm(2, 1).view(dims) + EPS) * eps

    def _forward_l2(self, x, y):
        mode = self.core_model.training
        self.core_model.eval()

        # Initialize worst-case inputs
        x_adv_worst = x.clone().detach()
        worst_losses = torch.zeros(len(x), 1, 1, 1, device=x.device) - 1e9
        guides, guide_masks = self._select_guide(x, y)

        # Repeat PGD for specified number of restarts
        for _ in range(self.num_restarts):
            x_adv = x.clone().detach()

            # Initialize adversarial inputs
            x_adv += self._project_l2(torch.randn_like(x_adv), self.eps)
            x_adv.clamp_(0, 1)

            # Run PGD on inputs for specified number of steps
            for _ in range(self.num_steps):
                x_adv.requires_grad_()

                # Compute logits, loss, gradients
                with torch.enable_grad():
                    logits = self.core_model(x_adv, return_mask=True)
                    loss = self.loss_fn(logits, y).mean()
                    grads = torch.autograd.grad(loss, x_adv)[0].detach()

                with torch.no_grad():
                    # Perform gradient update, project to norm ball
                    delta = x_adv - x + self._project_l2(grads, self.step_size)
                    x_adv = x + self._project_l2(delta, self.eps)
                    # Clip perturbed inputs to image domain
                    x_adv.clamp_(0, 1)

            if self.num_restarts == 1:
                x_adv_worst = x_adv
            else:
                # Update worst-case inputs with itemized final losses
                fin_losses = self.loss_fn(self.core_model(x_adv), y).reshape(
                    worst_losses.shape
                )
                up_mask = (fin_losses >= worst_losses).float()
                x_adv_worst = x_adv * up_mask + x_adv_worst * (1 - up_mask)
                worst_losses = fin_losses * up_mask + worst_losses * (
                    1 - up_mask
                )

        # Return worst-case perturbed input logits
        self.core_model.train(mode)
        return x_adv_worst.detach()

    def _forward_linf(self, x, y):
        mode = self.core_model.training
        self.core_model.eval()

        # Initialize worst-case inputs
        x_adv_worst = x.clone().detach()
        worst_losses = torch.zeros(len(x), 1, 1, 1, device=x.device) - 1e9
        guides, guide_masks = self._select_guide(x, y)

        # Repeat PGD for specified number of restarts
        for i in range(self.num_restarts):
            x_adv = x.clone().detach()

            # Initialize adversarial inputs
            x_adv += torch.zeros_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)
            guide, guide_mask = guides[i], guide_masks[i]

            # Run PGD on inputs for specified number of steps
            for j in range(self.num_steps):
                x_adv.requires_grad_()

                # Compute logits, loss, gradients
                with torch.enable_grad():
                    logits = self.core_model(x_adv, return_mask=True)
                    loss = self.loss_fn(logits, y, guide, guide_mask).mean()
                    grads = torch.autograd.grad(loss, x_adv)[0].detach()

                with torch.no_grad():
                    # Perform gradient update, project to norm ball
                    x_adv = x_adv.detach() + self.step_size * torch.sign(grads)
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

            if self.num_restarts == 1:
                x_adv_worst = x_adv
            else:
                # Update worst-case inputs with itemized final losses
                fin_losses = self.loss_fn(self.core_model(x_adv), y).reshape(
                    worst_losses.shape
                )
                up_mask = (fin_losses >= worst_losses).float()
                x_adv_worst = x_adv * up_mask + x_adv_worst * (1 - up_mask)
                worst_losses = fin_losses * up_mask + worst_losses * (
                    1 - up_mask
                )

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
        return x_adv_worst.detach()

    def forward(self, *args):
        if self.norm == "L2":
            return self._forward_l2(*args)
        return self._forward_linf(*args)
