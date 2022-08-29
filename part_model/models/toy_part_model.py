import torch.nn as nn
import torch.nn.functional as F


class ToyPartModel(nn.Module):
    def __init__(self, args, core_model, rep_dim, total_parts=None):
        super(ToyPartModel, self).__init__()
        self.core_model = core_model
        self.num_parts = args.parts
        self.total_parts = (
            self.num_parts if total_parts is None else total_parts
        )
        self.rep_dim = rep_dim
        self.linear_dim = rep_dim * self.num_parts
        self.linear = nn.Linear(self.linear_dim, args.num_classes)

        # For taking part indices as inputs
        self.use_part_idx = args.use_part_idx
        if self.use_part_idx:
            self.part_idx_to_rep = nn.Linear(total_parts, self.rep_dim)
            self.combine_rep = nn.Linear(self.linear_dim, self.linear_dim)
        self.part_idx_rep = None

    def set_part_idx(self, x):
        if not self.use_part_idx:
            return
        batch_size = x.size(0)
        one_hot = F.one_hot(x, num_classes=self.total_parts).float()
        self.part_idx_rep = self.part_idx_to_rep(one_hot).view(
            batch_size * self.num_parts, -1
        )

    def forward(self, x, part_idx=False):
        if part_idx:
            self.set_part_idx(x)
            return

        batch_size = x.size(0)
        num_channels = int(x.size(1) / self.num_parts)
        out = self.core_model(
            x.view(
                (
                    batch_size * self.num_parts,
                    num_channels,
                )
                + x.shape[2:]
            )
        )

        if self.use_part_idx:
            out = (out * self.part_idx_rep).view(batch_size, self.linear_dim)
            out = F.relu(
                self.combine_rep(F.relu(out, inplace=True)), inplace=True
            )
        else:
            out = out.view(batch_size, self.linear_dim)

        return self.linear(out)
