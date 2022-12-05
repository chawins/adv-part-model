from __future__ import annotations

import numpy as np


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
