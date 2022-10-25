import torch
import torch.nn as nn
import torch.nn.functional as F
from part_model.dataloader import DATASET_DICT

from DINO.main import build_model_main

class DinoBoundingBoxModel(nn.Module):
    def __init__(self, args):
        print("=> Initializing DinoBoundingBoxModel...")
        super(DinoBoundingBoxModel, self).__init__()
        
        # TODO: load weights if args.load_from_segmenter
        self.object_detector, self.criterion, self.postprocessors = build_model_main(args)        
        # 91 for number of part labels and 4 for bounding box coords
        input_dim = args.num_queries * (91+4)
        # how did we get 50 here
        self.core_model = nn.Sequential(
            nn.Identity(),
            nn.Flatten(),
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, args.num_classes),
        )

    def forward(self, images, return_mask=False, **kwargs):
        # Object Detection part
        dino_outputs = self.object_detector(images)

        # concatenate softmax'd logits and bounding box predictions
        features = torch.cat([F.softmax(dino_outputs['pred_logits'], dim=1), dino_outputs['pred_boxes']], dim=2)
        out = self.core_model(features)
        
        if return_mask:
            return out, dino_outputs
            # return out, outputs['pred_logits'], outputs['pred_boxes']
        return out

