"""
Code is adapted from https://github.com/micco00x/py-pascalpart

Usage examples:
python prepare_cityscapes_v2.py --data-dir ~/data/cityscapes/ --name name
"""
import argparse
import glob
import os
import random

import numpy as np
import PIL
import scipy
from tqdm import tqdm

from panoptic_parts.utils.format import decode_uids
from part_model.utils import (
    crop,
    crop_square,
    get_part_box,
    get_part_box_square,
)

LABELS = ["human", "vehicle"]
LABEL_TO_SIDS = [[24, 25], [26, 27, 28]]  # human, vehicle
# ['head', 'torso', 'arms', 'legs']
# ['windows', 'wheels', 'lights', 'license_plate', 'chassis']
LABEL_TO_PARTS = [[1, 2, 3, 4], [1, 2, 3, 4, 5]]

ALL_SIDS = []
for ids in LABEL_TO_SIDS:
    ALL_SIDS.extend(ids)

SID_TO_LABEL = {}
for i, sids in enumerate(LABEL_TO_SIDS):
    for sid in sids:
        SID_TO_LABEL[sid] = i


def _get_box_from_bin_mask(bin_mask):
    box_mask = np.zeros_like(bin_mask)
    if bin_mask.sum() == 0:
        return box_mask
    y, x = np.where(bin_mask)
    ymin, ymax = y.min(), y.max()
    xmin, xmax = x.min(), x.max()
    box_mask[ymin : ymax + 1, xmin : xmax + 1] = 1
    return box_mask


def _pixelwise_to_box(mask):
    uids = np.unique(mask)
    box_mask = np.zeros_like(mask)
    for y in uids:
        # TODO: Background has "small" uid. This can be made cleaner.
        if y < 2e6:
            continue
        bin_mask = _get_box_from_bin_mask(mask == y)
        box_mask += (box_mask == 0) * bin_mask * y
    return box_mask


def create_parts_hdf5(args, partition, data_dict=None):

    root = args.data_dir
    dirname = args.name
    pad = args.pad

    annotation_path = os.path.join(root, "gtFinePanopticParts", partition)
    image_path = os.path.join(root, "leftImg8bit", partition)
    part_path = os.path.join(root, "PartImages", dirname)
    os.makedirs(part_path, exist_ok=True)

    # Get all filenames in the GT directory
    filenames = [file for file in glob.glob(annotation_path + "/*")]
    filenames.extend([file for file in glob.glob(annotation_path + "/*/*")])
    filenames = list(filter(lambda x: x.endswith(".tif"), filenames))

    print("==> Collecting images and their parts...")
    obj_list, label_list, part_list, gt_list = [], [], [], []
    num_occluded = 0
    for filename in tqdm(filenames):
        gt_filename = os.path.basename(filename)
        # Set names for file_name and image_id
        image_filename = gt_filename.replace(
            "_gtFinePanopticParts.tif", "_leftImg8bit.png"
        )
        city = filename.split("/")[-2]
        img = np.array(
            PIL.Image.open(os.path.join(image_path, city, image_filename))
        )
        gt = np.array(PIL.Image.open(os.path.join(filename)))
        # Pre-pad images and labels to prevent out-of-bound crop
        pad_width = int(max(img.shape) * 0.1)
        pad_width = ((pad_width, pad_width), (pad_width, pad_width))
        img = np.pad(img, pad_width + ((0, 0),))
        gt = np.pad(gt, pad_width)
        # Pad whole image
        sids, iids, _ = decode_uids(gt)

        # Iterate through all instances
        unique_iids = np.unique(iids)
        unique_iids = unique_iids[unique_iids != 0]
        for sid in ALL_SIDS:
            semantic_mask = sids == sid
            for iid in unique_iids:
                instance_mask = (iids == iid) * semantic_mask
                label = SID_TO_LABEL[sid]

                # Skip if sid is not in the desired class or is too small
                if instance_mask.sum() < args.min_area:
                    continue

                if args.square:
                    obj_patch, bbox = crop_square(
                        img,
                        instance_mask,
                        pad,
                        return_bbox=True,
                        rand=args.rand_pad,
                    )
                    ymin, ymax, xmin, xmax = bbox
                    gt_patch = gt[ymin:ymax, xmin:xmax]
                else:
                    obj_patch = crop(img, instance_mask, pad)
                    gt_patch = crop(gt, instance_mask, pad)

                if args.use_box_seg:
                    gt_patch = _pixelwise_to_box(gt_patch)

                # Filter out image that is occluded by objects of the other class
                gt_temp = crop(gt, instance_mask, 0)
                sid_temp, _, _ = decode_uids(gt_temp)
                is_occluded = False
                area_current_sid = np.sum(sid_temp == sid)
                for s in LABEL_TO_SIDS[1 - label]:
                    area_s = np.sum(sid_temp == s)
                    # Consider occlusion when any other object occupies more
                    # than half the size of the main object
                    is_occluded |= area_s > area_current_sid / 2
                if is_occluded:
                    num_occluded += 1
                    continue

                sids_patch, iids_patch, pids_patch = decode_uids(gt_patch)

                if args.part_box or not args.allow_missing_parts:
                    # Find bounding box mask for parts
                    instance_mask_patch = (iids_patch == iid) & (
                        sids_patch == sid
                    )
                    masked_pids = instance_mask_patch * pids_patch
                    unique_pids = np.unique(masked_pids)
                    all_pids = LABEL_TO_PARTS[label]
                    part_is_missing = False
                    part_stack = []
                    for pid in all_pids:
                        if pid not in unique_pids:
                            part_is_missing = True
                            break
                        part_mask = masked_pids == pid
                        labeled, num_blobs = scipy.ndimage.label(
                            part_mask, structure=np.ones((3, 3))
                        )
                        # If there are multiple parts present, select the largest one
                        part_area = 0
                        for l in range(1, num_blobs + 1):
                            subpart_mask = labeled == l
                            subpart_area = subpart_mask.sum()
                            if (
                                subpart_area > args.min_area * 0.1
                                and subpart_area > part_area
                            ):
                                part_area = subpart_area
                                part_mask = subpart_mask

                        part_box = part_mask
                        if part_area == 0 and not args.allow_missing_parts:
                            # Filter out small parts
                            part_is_missing = True
                            break
                        else:
                            # Get box mask from segmentation mask
                            if args.square:
                                part_box = get_part_box_square(part_mask, pad)
                            else:
                                part_box = get_part_box(part_mask, pad)
                        part_stack.append(part_box)

                    # Do not use this object if there's a missing part
                    if part_is_missing:
                        continue

                    # Stack part mask to single channel
                    compressed_part_stack = np.zeros_like(
                        part_stack[0], dtype=np.int64
                    )
                    for i, p in enumerate(part_stack):
                        compressed_part_stack += p * 2**i
                    part_list.append(compressed_part_stack.astype(np.uint8))

                # Crop object and ground truth
                obj_list.append(obj_patch)
                gt_list.append(gt_patch)
                label_list.append(label)

        # DEBUG
        # if len(label_list) > 100:
        #     break

    # Count samples for each class
    print(np.unique(label_list, return_counts=True))
    print(
        f"=> {num_occluded} objects are occluded and so not included in the dataset."
    )

    # DEBUG
    # img = [
    #     torch.from_numpy(obj_list[-1]).permute(2, 0, 1) / 255.,
    #     torch.from_numpy(gt_list[-1]).repeat(3, 1, 1) / gt_list[-1].max(),
    #     # torch.from_numpy(part_list[-1]).repeat(3, 1, 1) / part_list[-1].max()
    # ]
    # save_image(img, 'test.png')
    # import pdb
    # pdb.set_trace()

    if data_dict is None:
        data_dict = {}
        data_dict["labels"] = label_list
        data_dict["images"] = obj_list
        data_dict["panoptic-parts"] = gt_list
        if args.part_box:
            data_dict["part-boxes"] = part_list
    else:
        data_dict["labels"].extend(label_list)
        data_dict["images"].extend(obj_list)
        data_dict["panoptic-parts"].extend(gt_list)
        if args.part_box:
            data_dict["part-boxes"].extend(part_list)

    return data_dict


def save_pil_image(img, path):
    image_path = os.path.join(path)
    pil_img = PIL.Image.fromarray(img)
    pil_img.save(image_path)


def save_images(data_list, path_list):
    ext = ["png", "tif", "png"]
    for j, path in enumerate(path_list):
        for label in LABELS:
            os.makedirs(os.path.join(path, label), exist_ok=True)
        for i, label in enumerate(data_list[0]):
            save_pil_image(
                data_list[j + 1][i],
                os.path.join(path, LABELS[label], f"{i:05d}.{ext[j]}"),
            )


def save_images_partition(partition, data_dict, idx):
    ext = {"images": "png", "panoptic-parts": "tif", "part-boxes": "png"}
    for key in data_dict:
        if key == "labels":
            continue
        path = os.path.join(
            args.data_dir, "PartImages", args.name, key, partition
        )
        for label in LABELS:
            os.makedirs(os.path.join(path, label), exist_ok=True)
        for i in idx:
            label = LABELS[data_dict["labels"][i]]
            save_pil_image(
                data_dict[key][i],
                os.path.join(path, label, f"{i:05d}.{ext[key]}"),
            )


if __name__ == "__main__":
    # Parse arguments from command line:
    parser = argparse.ArgumentParser(
        description="Prepare Cityscapes Panoptic Parts dataset for classification tasks"
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument(
        "--data-dir", default="~/data/", type=str, help="Path to dataset"
    )
    parser.add_argument(
        "--name", default="temp", type=str, help="Name the new part dataset"
    )
    parser.add_argument(
        "--pad",
        default=0.1,
        type=float,
        help="Padding to add from mask segmentation",
    )
    parser.add_argument(
        "--min-area",
        default=1000,
        type=float,
        help="Minimal number of pixels of object to consider",
    )
    parser.add_argument(
        "--square",
        action="store_true",
        help="Crop image and box as squares instead of tighter rectangles",
    )
    parser.add_argument(
        "--rand-pad", action="store_true", help="Use random padding"
    )
    parser.add_argument(
        "--part-box", action="store_true", help="Include bounding box for parts"
    )
    parser.add_argument(
        "--allow-missing-parts",
        action="store_true",
        help="Include samples with missing parts",
    )
    parser.add_argument("--use-box-seg", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    data_dict = create_parts_hdf5(args, "train")
    data_dict = create_parts_hdf5(args, "val", data_dict=data_dict)

    num_samples = len(data_dict["labels"])
    idx = np.arange(num_samples)
    np.random.shuffle(idx)
    num_val, num_test = int(0.1 * num_samples), int(0.1 * num_samples)
    val_idx = idx[:num_val]
    test_idx = idx[num_val : num_val + num_test]
    train_idx = idx[num_val + num_test :]

    for partition, indices in zip(
        ["train", "val", "test"], [train_idx, val_idx, test_idx]
    ):
        save_images_partition(partition, data_dict, indices)
