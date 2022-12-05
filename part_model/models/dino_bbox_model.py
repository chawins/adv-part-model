from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from projects.dino.modeling import DINO
from detrex.utils import inverse_sigmoid
from typing import List

class DinoBoundingBoxModel(DINO):
    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        seg_labels: int,
        num_queries: int,
        criterion: nn.Module,
        pixel_mean: List[float] = [0, 0, 0],
        pixel_std: List[float] = [1, 1, 1],
        aux_loss: bool = True,
        select_box_nums_for_evaluation: int = 300,
        device="cuda",
        dn_number: int = 100,
        label_noise_ratio: float = 0.2,
        box_noise_scale: float = 1.0,
    ):
        print("=> Initializing DinoBoundingBoxModel...")
        super().__init__(num_classes=seg_labels)
        
        # # setattr(args, 'num_classes', tmp_num_classes)
        # # logits for part labels and 4 for bounding box coords
        input_dim = num_queries * (seg_labels + 4)
        # print("input_dim", input_dim)


        self.core_model = nn.Sequential(
            # nn.Conv1d((args.seg_labels - 1), 10, 5),
            nn.Identity(),
            nn.Flatten(),
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, args.num_classes),
        )

    # def forward(
    #     self,
    #     images,
    #     masks,
    #     dino_targets,
    #     need_tgt_for_training,
    #     return_mask=False,
    #     **kwargs
    # ):
    def forward(self, batched_inputs, return_detector_outputs=False, **kwargs):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)

        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)

        # original features
        features = self.backbone(images.tensor)  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in DINO
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # denoising preprocessing
        # prepare label query embedding
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.embed_dim,
                label_enc=self.label_enc,
            )
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # feed into transformer
        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,  # [0..1]
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
        )
        # hack implementation for distributed training
        inter_states[0] += self.label_enc.weight[0, 0] * 0.0

        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # denoising postprocessing
        if dn_meta is not None:
            outputs_class, outputs_coord = self.dn_post_process(
                outputs_class, outputs_coord, dn_meta
            )

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        interm_coord = enc_reference
        interm_class = self.transformer.decoder.class_embed[-1](enc_state)
        output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}

        # concatenate softmax'd logits and bounding box predictions
        features = torch.cat(
            [
                F.softmax(output["pred_logits"], dim=1),
                output["pred_boxes"],
            ],
            dim=2,
        )
        out = self.core_model(features)

        if return_detector_outputs:
            return out, output
            # return out, outputs['pred_logits'], outputs['pred_boxes']
        return out
            
        #     if self.training:
        #         loss_dict = self.criterion(output, targets, dn_meta)
        #         weight_dict = self.criterion.weight_dict
        #         for k in loss_dict.keys():
        #             if k in weight_dict:
        #                 loss_dict[k] *= weight_dict[k]
        #         return loss_dict
        #     else:
        #         box_cls = output["pred_logits"]
        #         box_pred = output["pred_boxes"]
        #         results = self.inference(box_cls, box_pred, images.image_sizes)
        #         processed_results = []
        #         for results_per_image, input_per_image, image_size in zip(
        #             results, batched_inputs, images.image_sizes
        #         ):
        #             height = input_per_image.get("height", image_size[0])
        #             width = input_per_image.get("width", image_size[1])
        #             r = detector_postprocess(results_per_image, height, width)
        #             processed_results.append({"instances": r})
        # return processed_results



        # # Object Detection part
        # nested_tensors = NestedTensor(images, masks)

        # # out = self.backbone(nested_tensors)

        # # out[0][-1].tensors
        # # import pdb
        # # pdb.set_trace()

        # if need_tgt_for_training:
        #     dino_outputs = self.object_detector(nested_tensors, dino_targets)
        # else:
        #     dino_outputs = self.object_detector(nested_tensors)

        # # concatenate softmax'd logits and bounding box predictions
        # features = torch.cat(
        #     [
        #         F.softmax(dino_outputs["pred_logits"], dim=1),
        #         dino_outputs["pred_boxes"],
        #     ],
        #     dim=2,
        # )
        # out = self.core_model(features)

        # if return_mask:
        #     return out, dino_outputs
        #     # return out, outputs['pred_logits'], outputs['pred_boxes']
        # return out
