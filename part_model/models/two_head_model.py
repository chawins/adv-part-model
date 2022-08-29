import torch.nn as nn


class TwoHeadModel(nn.Module):
    def __init__(self, args, segmentor, mode):
        super(TwoHeadModel, self).__init__()
        self.mode = mode
        if self.mode == "d":
            segmentor[1].segmentation_head = Heads(
                segmentor[1], args.num_classes
            )
        else:
            latent_dim = 2048  # TODO: depends on backbone
            pool_size = 4
            fc_dim = 64
            segmentor[1].classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.BatchNorm2d(latent_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(latent_dim, fc_dim, (pool_size, pool_size)),
                nn.BatchNorm2d(fc_dim),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(fc_dim, args.num_classes),
            )
        self.segmentor = segmentor

    def forward(self, images, return_mask=False, **kwargs):
        # Segmentation part
        if self.mode == "d":
            self.segmentor[1].segmentation_head.returnMask = return_mask
            return self.segmentor(images)

        out = self.segmentor(images)
        if return_mask:
            return out[1], out[0]
        return out[1]


class Heads(nn.Module):
    def __init__(self, segmentor, num_classes):
        super().__init__()
        self.returnMask = False
        self.heads = nn.ModuleList(
            [
                segmentor.segmentation_head,
                nn.Sequential(
                    nn.Conv2d(256, 50, (3, 3), (1, 1)),
                    nn.BatchNorm2d(50),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(50, 10, (1, 1), (1, 1)),
                    nn.BatchNorm2d(10),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(1690, 100),
                    nn.ReLU(),
                    nn.Linear(100, num_classes),
                    # nn.Softmax()
                ),
            ]
        )

    def forward(self, x):
        out = [head(x) for head in self.heads]
        if self.returnMask:
            return out[1], out[0]
        return out[1]
