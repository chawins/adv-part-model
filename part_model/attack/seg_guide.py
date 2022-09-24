from typing import Any, Optional, Tuple

import numpy as np
import part_model.utils.loss as loss_lib
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

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
        dataloader: Optional[Any] = None,
        classifier: Optional[torch.nn.Module] = None,
        no_bg: bool = False,
        seg_labels: int = 40,
        **kwargs,
    ):
        super(SegGuidedAttackModule, self).__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        assert self.norm in ("L2", "Linf")
        if classifier is None:
            raise ValueError("classifier must be torch.Module and not None.")
        self.classifier = classifier
        self.seg_labels = seg_labels
        self.num_steps = attack_config["pgd_steps"]
        self.step_size = attack_config["pgd_step_size"]
        self.num_restarts = attack_config["num_restarts"]

        # "opt": optimizes for worst-case mask.
        # "random": random guide from any of incorrect class.
        # "2nd_pred_random":
        # "2nd_pred_by_scores": Pick guides from 2nd-most confident class and
        #     then sort by score of the guides.
        # "2nd_gt_by_sim": Pick guides from 2nd-most confident class and then
        #     sort by similarity to original mask.
        self.guide_selection = attack_config["guide_selection"]
        self.all_guide_selections = (
            "opt",
            "random",
            "2nd_pred_by_scores",
            "2nd_gt_random",
            "2nd_gt_by_sim",
        )
        if self.guide_selection not in self.all_guide_selections:
            raise NotImplementedError(
                f"guide_selection {self.guide_selection} not implemented! "
                f"(options: {self.all_guide_selections})"
            )
        if self.guide_selection != "opt" and dataloader is None:
            raise ValueError(
                "dataloader cannot be None for guide_selection "
                f"{self.guide_selection}."
            )

        self.use_two_stages = attack_config["use_two_stages"]
        self.loss_fn = loss_lib.SemiSumLoss(
            seg_const=attack_config["seg_const"], reduction="none"
        )
        self.seg_loss_fn = loss_lib.SemiSumLoss(seg_const=1, reduction="none")
        # Classification loss is used only for saving final best attack
        self.clf_loss_fn = loss_lib.SemiSumLoss(seg_const=0, reduction="none")

        if self.guide_selection != "opt":
            with torch.no_grad():
                self._load_guides(dataloader)

    def _load_guides(self, dataloader):
        """Load guide masks, computes predicted scores, and store indices."""
        guide_masks = []
        labels = []
        pred_scores = []
        scale_const = 1e3

        for i, (_, segs, targets) in enumerate(dataloader):
            # Get classifier's score on gt masks
            onehot_segs = F.one_hot(segs, num_classes=self.seg_labels).cuda()
            onehot_segs = onehot_segs.permute(0, 3, 1, 2)
            # classifier takes logit masks but gt mask is categorical so we
            # scale it with a large constant (like temperature).
            logits = (
                self.classifier(onehot_segs * scale_const) - scale_const / 2
            )
            num_classes = logits.shape[-1]
            scores = F.softmax(logits, dim=-1).cpu()

            guide_masks.append(segs)
            labels.append(targets)
            pred_scores.append(scores)
            # if i == 50:
            #     break  # DEBUG

        self.guide_masks = torch.cat(guide_masks, dim=0)
        self.guide_labels = torch.cat(labels, dim=0)
        self.guide_scores = torch.cat(pred_scores, dim=0)

        # Cache index of guides by gt and predicted labels
        self.label_idx_dict = {}
        self.pred_idx_dict = {}
        for i in range(num_classes):
            self.label_idx_dict[i] = torch.where(self.guide_labels == i)[0]
            pred_idx = torch.where(self.guide_scores.argmax(dim=-1) == i)[0]
            # Sort by scores
            sort_idx = torch.argsort(
                self.guide_scores[pred_idx, i], descending=True
            )
            self.pred_idx_dict[i] = pred_idx[sort_idx]
            self._check_num_guides(self.label_idx_dict, i)
            self._check_num_guides(self.pred_idx_dict, i)

        print("Finished loading guides.")

    def _check_num_guides(self, label_idx_dict, class_idx):
        num_guides = len(label_idx_dict[class_idx])
        min_guides = 1
        assert num_guides >= min_guides, (
            f"Not enough guides for class {class_idx}! There are {num_guides} "
            f"guides, but need at least {min_guides}. Consider using "
            f"more guides in dataloader."
        )

    def _select_opt(self, y_2nd: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
        return

    def _select_random(self, y_sort: torch.Tensor) -> torch.Tensor:
        guide_masks = []
        for cur_y_sort in y_sort.cpu().numpy():
            masks = []
            for y in cur_y_sort[: self.num_restarts]:
                # Pick a random mask
                idx = np.random.choice(self.label_idx_dict[y])
                mask = self.guide_masks[idx]
                masks.append(mask.unsqueeze(0))
            masks = torch.cat(masks, dim=0)
            guide_masks.append(masks.unsqueeze(1))
        guide_masks = torch.cat(guide_masks, dim=1)
        return guide_masks

    def _select_2nd_gt_random(self, y_2nd: torch.Tensor) -> torch.Tensor:
        guide_masks = []
        for cur_y_2nd in y_2nd.cpu().numpy():
            all_idx = self.label_idx_dict[cur_y_2nd]
            rand_idx = np.arange(len(all_idx))
            np.random.shuffle(rand_idx)
            idx = all_idx[rand_idx[: self.num_restarts]]
            guide_masks.append(self.guide_masks[idx].unsqueeze(1))
        guide_masks = torch.cat(guide_masks, dim=1)
        return guide_masks

    def _select_2nd_pred_by_scores(self, y_2nd: torch.Tensor) -> torch.Tensor:
        guide_masks = []
        for cur_y_2nd in y_2nd.cpu().numpy():
            idx = self.pred_idx_dict[cur_y_2nd][: self.num_restarts]
            guide_masks.append(self.guide_masks[idx].unsqueeze(1))
        guide_masks = torch.cat(guide_masks, dim=1)
        return guide_masks

    def _select_guide(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            guide_masks: shape [num_restarts, B, H, W]
        """
        with torch.no_grad():
            logits, _ = self.core_model(x, return_mask=True)

        # Get 2nd-most confident class
        batch_size = logits.shape[0]
        copy_logits = logits.clone()
        copy_logits[torch.arange(batch_size), y] -= 1e9
        y_2nd = copy_logits.argmax(-1).to(x.device)
        y_guides = y_2nd.unsqueeze(0).expand(self.num_restarts, -1)

        # TODO
        if self.guide_selection == "2nd_pred_by_scores":
            guide_masks = self._select_2nd_pred_by_scores(y_2nd)
        elif self.guide_selection == "2nd_gt_random":
            guide_masks = self._select_2nd_gt_random(y_2nd)
        elif self.guide_selection == "opt":
            guide_masks = self._select_opt(y_2nd)
        elif self.guide_selection == "random":
            y_sort = torch.argsort(copy_logits, dim=-1, descending=True)
            guide_masks = self._select_random(y_sort)
            y_guides = y_sort.permute(1, 0)[: self.num_restarts]

        return guide_masks.to(x.device), y_guides.to(x.device)

    def _project_l2(self, x, eps):
        dims = [-1,] + [
            1,
        ] * (x.ndim - 1)
        return x / (x.view(len(x), -1).norm(2, 1).view(dims) + EPS) * eps

    def _forward_l2(self, x, y):
        raise NotImplementedError()
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

    def _forward_linf(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        guides: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
    ) -> torch.Tensor:
        mode = self.core_model.training
        self.core_model.eval()
        targeted = True

        if guides is None:
            guide_masks, guide_labels = self._select_guide(x, y)
            x_init = x
            loss_fn = self.seg_loss_fn if self.use_two_stages else self.loss_fn
            in_1st_stage = self.use_two_stages
        else:
            # If guides are given, we are in second stage
            assert self.use_two_stages
            guide_masks, guide_labels, x_init = guides
            loss_fn = self.loss_fn
            in_1st_stage = False

        # Initialize worst-case inputs
        x_adv_worst = [] if in_1st_stage else x.clone().detach()
        worst_losses = torch.zeros(len(x), 1, 1, 1, device=x.device) - 1e9

        # Repeat PGD for specified number of restarts
        for i in range(self.num_restarts):

            num_guides = len(guide_masks)
            x_adv = x_init if x_init.ndim == x.ndim else x_init[i % num_guides]
            x_adv = x_adv.clone().detach()
            guide_mask = guide_masks[i % num_guides]
            if targeted:
                y = guide_labels[i % num_guides]

            # Initialize adversarial inputs
            x_adv += torch.zeros_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)

            # Run PGD on inputs for specified number of steps
            for j in range(self.num_steps):
                x_adv.requires_grad_()

                # Compute logits, loss, gradients
                with torch.enable_grad():
                    logits = self.core_model(x_adv, return_mask=True)
                    loss = loss_fn(logits, y, guide_mask).mean()
                    if targeted:
                        loss *= -1
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

            if in_1st_stage:
                # In 1st stage, save all x_adv to use as restarts in 2nd stage
                x_adv_worst.append(x.unsqueeze(0))
            elif self.num_restarts == 1:
                x_adv_worst = x_adv
            else:
                # Update worst-case inputs with itemized final losses only in
                # 2nd stage.
                fin_losses = self.clf_loss_fn(
                    self.core_model(x_adv), y, None
                ).reshape(worst_losses.shape)
                if targeted:
                    fin_losses *= -1
                up_mask = (fin_losses >= worst_losses).float()
                x_adv_worst = x_adv * up_mask + x_adv_worst * (1 - up_mask)
                worst_losses = fin_losses * up_mask + worst_losses * (
                    1 - up_mask
                )

        if isinstance(x_adv_worst, list):
            x_adv_worst = torch.cat(x_adv_worst, axis=0)

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

        if in_1st_stage:
            return guide_masks, guide_labels, x_adv_worst.detach()
        return x_adv_worst.detach()

    def _forward(self, *args, **kwargs):
        if self.norm == "L2":
            return self._forward_l2(*args, **kwargs)
        return self._forward_linf(*args, **kwargs)

    def forward(self, *args):
        if not self.use_two_stages:
            return self._forward(*args)

        # Two-staged attack
        guides = self._forward(*args)
        return self._forward(*args, guides=guides)
