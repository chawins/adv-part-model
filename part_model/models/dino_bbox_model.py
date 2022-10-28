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
        tmp_num_classes = args.num_classes
        
        setattr(args, 'num_classes', args.seg_labels)
        self.object_detector, self.criterion, self.postprocessors = build_model_main(args)        
        
        # TODO: remove from here. only for debugging
        n_seg = sum(p.numel() for p in self.object_detector.parameters()) / 1e6
        nt_seg = (
            sum(p.numel() for p in self.object_detector.parameters() if p.requires_grad)
            / 1e6
        )
        print(f"=> object detector params (train/total): {nt_seg:.2f}M/{n_seg:.2f}M")
        
        setattr(args, 'num_classes', tmp_num_classes)
        # logits for part labels and 4 for bounding box coords
        input_dim = args.num_queries * (args.seg_labels+4)
        print('input_dim', input_dim)

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

    def forward(self, images, dino_targets, need_tgt_for_training, return_mask=False, **kwargs):
        # outputs, dino_outputs = model(images, target_bbox, need_tgt_for_training, return_mask=need_tgt_for_training)

        # Object Detection part
        if need_tgt_for_training:
            dino_outputs = self.object_detector(images, dino_targets)
        else:
            dino_outputs = self.object_detector(images)
        

        # concatenate softmax'd logits and bounding box predictions
        features = torch.cat([F.softmax(dino_outputs['pred_logits'], dim=1), dino_outputs['pred_boxes']], dim=2)
        out = self.core_model(features)
        
        if return_mask:
            return out, dino_outputs
            # return out, outputs['pred_logits'], outputs['pred_boxes']
        return out

