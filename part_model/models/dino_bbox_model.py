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

        
        
        
        # # load pretrained dino backbone
        # import DINO.util.misc as utils
        # from DINO.util.utils import ModelEma


        # if (not args.resume) and args.pretrain_model_path:
        #     checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        # from collections import OrderedDict
        # _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        # ignorelist = []

        # def check_keep(keyname, ignorekeywordlist):
        #     for keyword in ignorekeywordlist:
        #         if keyword in keyname:
        #             ignorelist.append(keyname)
        #             return False
        #     return True

        # # logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        # _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        # # _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        # _load_output = self.object_detector.load_state_dict(_tmp_st, strict=False)
        # # logger.info(str(_load_output))

        # if args.use_ema:
        #     if 'ema_model' in checkpoint:
        #         print('ema in checkpoint')
        #         ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
        #     else:
        #         del ema_m
        #         ema_m = ModelEma(self.object_detector, args.ema_decay) 







        
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

    def forward(self, images, masks, dino_targets, need_tgt_for_training, return_mask=False, **kwargs):
        # Object Detection part
        from DINO.util.misc import NestedTensor
        nested_tensors = NestedTensor(images, masks)
        if need_tgt_for_training:
            dino_outputs = self.object_detector(nested_tensors, dino_targets)
            # dino_outputs = self.object_detector(images, dino_targets)
        else:
            dino_outputs = self.object_detector(nested_tensors)
            # dino_outputs = self.object_detector(images)
        

        # concatenate softmax'd logits and bounding box predictions
        features = torch.cat([F.softmax(dino_outputs['pred_logits'], dim=1), dino_outputs['pred_boxes']], dim=2)
        out = self.core_model(features)
        
        if return_mask:
            return out, dino_outputs
            # return out, outputs['pred_logits'], outputs['pred_boxes']
        return out

