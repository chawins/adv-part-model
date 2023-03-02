from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, draw_bounding_boxes
from torchvision.ops import box_convert, masks_to_boxes
from skimage.measure import label, regionprops

from partimagenet_hparams import partimagenet_id2name

def get_seg_type(args):
    seg_type = args.experiment.split("-")[0]
    kws = ["part", "object", "fg"]
    for kw in kws:
        if kw == seg_type:
            return kw
    return None


def get_box(mask, pad):
    coord = np.where(mask)
    ymin, ymax = coord[0].min(), coord[0].max()
    xmin, xmax = coord[1].min(), coord[1].max()
    width, height = xmax - xmin, ymax - ymin
    size = max(width, height)
    extra_obj_pad = int(pad * size)
    xmin -= extra_obj_pad
    ymin -= extra_obj_pad
    xmax, ymax = xmin + width + extra_obj_pad, ymin + height + extra_obj_pad
    return ymin, ymax, xmin, xmax


def constrain_box(img, box):
    ymin, ymax, xmin, xmax = box
    height, width = img.shape[0], img.shape[1]
    ymin = max(ymin, 0)
    xmin = max(xmin, 0)
    ymax = min(ymax, height - 1)
    xmax = min(xmax, width - 1)
    return (ymin, ymax, xmin, xmax)


def crop(img, mask, pad):
    box = get_box(mask, pad)
    ymin, ymax, xmin, xmax = constrain_box(img, box)
    return img[ymin:ymax, xmin:xmax]


def get_part_box(mask, pad):
    box = get_box(mask, pad)
    ymin, ymax, xmin, xmax = constrain_box(mask, box)
    part_mask = np.zeros_like(mask)
    part_mask[ymin:ymax, xmin:xmax] = 1
    return part_mask


# ============================== Square version ============================= #


def compute_padding(img, box):
    """Compute extra padding in case the bounding box extends beyond image."""
    ymin, ymax, xmin, xmax = box
    height, width = img.shape[0], img.shape[1]
    pad_size = (
        (max(-ymin, 0), max(ymax - height, 0)),
        (max(-xmin, 0), max(xmax - width, 0)),
    )
    ymin += pad_size[0][0]
    ymax += pad_size[0][0]
    xmin += pad_size[1][0]
    xmax += pad_size[1][0]
    if img.ndim == 3:
        pad_size = pad_size + ((0, 0),)
    return pad_size, (ymin, ymax, xmin, xmax)


def get_box_square(mask, pad, rand=False):
    """
    Take a boolean `mask` and return box coordinates with extra padding equal
    to the largest size * `pad`.
    """
    coord = np.where(mask)
    ymin, ymax = coord[0].min(), coord[0].max()
    xmin, xmax = coord[1].min(), coord[1].max()
    # Make sure that bounding box is square
    width, height = xmax - xmin, ymax - ymin
    size = max(width, height)
    xpad, ypad = int((size - width) / 2), int((size - height) / 2)
    extra_obj_pad = int(pad * size)
    size += 2 * extra_obj_pad
    if rand:
        rand_pad = np.random.uniform(0, pad * 2, size=2)
        rand_pad_w = int(rand_pad[0] * size)
        rand_pad_h = int(rand_pad[1] * size)
        xmin = max(0, xmin - xpad - rand_pad_w)
        ymin = max(0, ymin - ypad - rand_pad_h)
    else:
        # Hot fix in the worst case where initial padding is not sufficient
        xmin = max(0, xmin - xpad - extra_obj_pad)
        ymin = max(0, ymin - ypad - extra_obj_pad)
    xmax, ymax = xmin + size, ymin + size
    return ymin, ymax, xmin, xmax


def crop_square(img, mask, pad, return_bbox=False, **pad_kwargs):
    box = get_box_square(mask, pad, **pad_kwargs)
    ymin, ymax, xmin, xmax = box
    if return_bbox:
        return img[ymin:ymax, xmin:xmax], box
    return img[ymin:ymax, xmin:xmax]


def get_part_box_square(mask, pad, **pad_kwargs):
    box = get_box_square(mask, pad, **pad_kwargs)
    ymin, ymax, xmin, xmax = box
    part_mask = np.zeros_like(mask)
    part_mask[ymin:ymax, xmin:xmax] = 1
    return part_mask


def unnormalize_bbox_targets(target_bbox):
    # convert target boxes format from cxcywh to xyxy and resize to image size
    for j, tbox in enumerate(target_bbox):
        shape = tbox["size"]
        boxes = tbox["boxes"]
        boxes = box_convert(
            boxes, in_fmt="cxcywh", out_fmt="xyxy"
        )
        boxes[:, ::2] = boxes[:, ::2] * shape[1]
        boxes[:, 1::2] = boxes[:, 1::2] * shape[0]
        target_bbox[j]["boxes"] = boxes
    return target_bbox


def get_bbox_from_masks(outputs, use_connected_components=True):
    # values, pred_masks = torch.max(outputs, dim=1)
    pred_masks = outputs.argmax(dim=1)
    probs = F.softmax(outputs, dim=1)
    batch_size, _, _ = pred_masks.shape
    bbox_preds = []
    for bi in range(batch_size):
        img_bbox_preds = {}
        img_mask = pred_masks[bi]
        pred_labels = torch.unique(img_mask, sorted=True)
        # remove background
        if 0 in pred_labels: 
            pred_labels = pred_labels[1:]
        
        img_bbox_preds['boxes'] = []
        img_bbox_preds['scores'] = []
        img_bbox_preds['labels'] = []

        if use_connected_components:
            for _, part_label in enumerate(pred_labels):
                part_label_mask = (img_mask == part_label) * 1
                part_label_mask = np.uint8(part_label_mask.cpu())
            
                lbl_0, _ = label(part_label_mask, return_num=True, connectivity=1)
                props = regionprops(lbl_0)

                for _, prop in enumerate(props):
                    y1, x1, y2, x2 = prop.bbox
                    img_bbox_preds['boxes'].append([x1, y1, x2, y2])
                    img_bbox_preds['labels'].append(part_label-1) # subtract one because of background class in seg masks (no background bbox)
                    bbox_prob = probs[bi, part_label, prop.coords[:, 0], prop.coords[:, 1]].mean().item()
                    img_bbox_preds['scores'].append(bbox_prob)
                    
            img_bbox_preds['boxes'] = torch.Tensor(img_bbox_preds['boxes']).cuda()
        else:
            label_mask = img_mask == pred_labels[:, None, None]
            img_boxes = masks_to_boxes(label_mask) 
            img_bbox_preds['boxes'] = img_boxes
            for lbl in pred_labels:
                lbl = lbl.item()-1 # subtract one because of background class in seg masks (no background bbox)
                img_bbox_preds['scores'].append(probs[:, lbl, :, :].mean().item())
                img_bbox_preds['labels'].append(lbl)  

        img_bbox_preds['scores'] = torch.Tensor(img_bbox_preds['scores']).cuda()
        img_bbox_preds['labels'] = torch.LongTensor(img_bbox_preds['labels']).cuda()
        bbox_preds.append(img_bbox_preds)
    return bbox_preds


def get_masks_from_bbox(pred_bbox, target_sizes, seg_labels):
    batch_size = len(pred_bbox)
    batch_seg_masks = []
    for bi in range(batch_size):
        seg_mask = torch.zeros((seg_labels, target_sizes[bi][0], target_sizes[bi][1]))
        for score, label, bbox in zip(pred_bbox[bi]['scores'], pred_bbox[bi]['labels'], pred_bbox[bi]['boxes']):
            if label > 39:
                continue
            xmin, ymin, xmax, ymax = bbox
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(target_sizes[bi][0], int(xmax))
            ymax = min(target_sizes[bi][1], int(ymax))
            seg_mask[label.item()+1, ymin : ymax + 1, xmin : xmax + 1] += score.item() # add one because of background class in seg masks
        batch_seg_masks.append(seg_mask)
    batch_seg_masks = torch.stack(batch_seg_masks, dim=0).cuda()
    batch_seg_masks = F.softmax(batch_seg_masks, dim=1)
    return batch_seg_masks

def plot_img_bbox(images, target_bbox):
    batch_size = images.shape[0]
    images_with_boxes = []
    for bi in range(batch_size):
        img_uint8 = (images[bi].cpu() * 255).byte()
        boxes = target_bbox[bi]["boxes"]
        labels = [partimagenet_id2name[int(l)] for l in target_bbox[bi]["labels"]]
        
        if boxes.shape[0] > 0:
            img_with_boxes = draw_bounding_boxes(
                img_uint8, boxes=boxes, colors="red", labels=labels
            )
        else:
            img_with_boxes = img_uint8
        images_with_boxes.append(img_with_boxes / 255.0)
    save_image(images_with_boxes, "test_box_labels.png")
