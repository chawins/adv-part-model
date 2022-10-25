"""
Code is adapted from https://github.com/micco00x/py-pascalpart

Usage examples:
python prepare_pascal_part_v3.py --data-dir ~/data/pascal_part/ --name name
"""
import argparse
import os
import random
from shutil import copyfile

import numpy as np
import PIL
from PIL import Image
import scipy.io
from tqdm import tqdm

CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# NOTE: Order of the "big" parts matters because PASCAL-Part annotation overlaps
# meaning that mask of car windows overlap with car body so whichever part gets
# "painted" first will appear in the final mask. Here, we choose "body" to
# appear last.
PARTS = {
    "aeroplane": {
        "tail": ["tail", "stern"],
        "wing": ["lwing", "rwing", "engine"],
        "body": ["body", "wheel"],
    },
    # 'bicycle': {
    #     'fwheel': ['fwheel'],
    #     'bwheel': ['bwheel', 'chainwheel'],
    #     'handlebar': ['handlebar', 'saddle', 'headlight'],
    # },
    "bird": {
        "head": ["head", "beak", "leye", "reye", "neck"],
        "legs": ["lleg", "rleg", "lfoot", "rfoot"],
        "body": ["torso", "lwing", "rwing", "tail"],
    },
    "car": {
        "wheel": ["wheel"],
        "window": ["window"],
        "body": [
            "bliplate",
            "backside",
            "door",
            "fliplate",
            "frontside",
            "headlight",
            "leftmirror",
            "leftside",
            "rightmirror",
            "rightside",
            "roofside",
        ],
    },
    "cat": {
        "head": ["head", "nose", "leye", "reye", "lear", "rear", "neck"],
        "legs": [
            "lfleg",
            "rfleg",
            "lbleg",
            "rbleg",
            "lfpa",
            "rfpa",
            "lbpa",
            "rbpa",
        ],
        "body": ["torso", "tail"],
    },
    "dog": {
        "head": [
            "head",
            "nose",
            "leye",
            "reye",
            "lear",
            "rear",
            "neck",
            "muzzle",
        ],
        "legs": [
            "lfleg",
            "rfleg",
            "lbleg",
            "rbleg",
            "lfpa",
            "rfpa",
            "lbpa",
            "rbpa",
        ],
        "body": ["torso", "tail"],
    },
}

LABELS = list(PARTS.keys())
LABEL_TO_IDX = {label: i for i, label in enumerate(PARTS.keys())}


def load_annotations(path):

    # Get annotations from the file and relative objects:
    annotations = scipy.io.loadmat(path)["anno"]

    objects = annotations[0, 0]["objects"]

    # List containing information of each object (to add to dictionary):
    objects_list = []

    # Go through the objects and extract info:
    for obj_idx in range(objects.shape[1]):
        obj = objects[0, obj_idx]

        # Get classname and mask of the current object:
        classname = obj["class"][0]
        mask = obj["mask"]

        # List containing information of each body part (to add to dictionary):
        parts_list = []

        parts = obj["parts"]

        # Go through the part of the specific object and extract info:
        for part_idx in range(parts.shape[1]):
            part = parts[0, part_idx]
            # Get part name and mask of the current body part:
            part_name = part["part_name"][0]
            part_mask = part["mask"]

            # Add info to parts_list:
            parts_list.append({"part_name": part_name, "mask": part_mask})

        # Add info to objects_list:
        objects_list.append(
            {"class": classname, "mask": mask, "parts": parts_list}
        )

    return {"objects": objects_list}


def get_part_label_offset(label_idx):
    offset = 0
    for prev_label in LABELS[:label_idx]:
        offset += len(PARTS[prev_label])
    return offset


def relabel_parts(part_label, obj):
    """Label `part_label` with parts of `obj`"""
    label = obj["class"]
    if label not in LABELS:
        return
    all_part_list = [p["part_name"] for p in obj["parts"]]
    label_offset = get_part_label_offset(LABELS.index(label))

    for part_idx, part in enumerate(PARTS[label].keys()):
        part_mask = np.zeros_like(part_label)
        # Aggregate subparts into one part
        for i, subpart in enumerate(all_part_list):
            if any([p in subpart for p in PARTS[label][part]]):
                part_mask += obj["parts"][i]["mask"]
        # Get remaining pixels that still don't have part assigned to
        mask_bg = part_label == 0
        part_label += mask_bg * (part_mask > 0) * (label_offset + part_idx + 1)


def collect_parts(args):
    """
    (1) Sample is the entire image, and its label is determined by that of the
    object that takes up the largest area in the image. Also filter out samples
    that miss at least one of the three part groups.
    (2) Does not filter out objects with missing parts
    """

    root = args.data_dir
    annotation_path = os.path.join(root, "Annotations_Part")
    mat_filenames = os.listdir(annotation_path)
    mat_filenames = sorted(mat_filenames)

    print("==> Collecting images and their parts...")
    file_list, label_list, gt_list = [], [], []
    too_small = 0
    for annotation_filename in tqdm(mat_filenames):
        annotations = load_annotations(
            os.path.join(annotation_path, annotation_filename)
        )
        # PASCAL VOC image have .jpg format
        image_filename = annotation_filename.split(".")[0] + ".jpg"

        # Determine ambiguous label (largest object in the image)
        main_obj_idx, max_area, label = None, 0, None
        for obj_idx, obj in enumerate(annotations["objects"]):
            area = obj["mask"].astype(np.float32).sum()
            if area > max_area:
                main_obj_idx = obj_idx
                label = obj["class"]
                max_area = area

        if main_obj_idx is None or label not in LABELS:
            continue
        image_size = np.prod(obj["mask"].shape)
        if (max_area / image_size) < args.min_area:
            too_small += 1
            continue
        obj = annotations["objects"][main_obj_idx]
        assert obj["class"] in LABELS
        file_list.append(image_filename)
        label_list.append(label)

        # Reassign part labels according to the new grouping
        mask = obj["mask"].astype(np.byte)
        part_label = np.zeros_like(mask)

        # Relabel the main object first
        relabel_parts(part_label, obj)

        # Relabel the rest of the objects in the image in order
        for obj_idx, o in enumerate(annotations["objects"]):
            if obj_idx == main_obj_idx:
                continue
            relabel_parts(part_label, o)

        gt_list.append(part_label)

    # Count samples for each class
    print(np.unique(label_list, return_counts=True))
    print(f"=> {too_small} samples were too small and removed.")

    data_dict = {
        "labels": label_list,
        "images": file_list,
        "panoptic-parts": gt_list,
    }
    return data_dict


def save_pil_image(img, path):
    image_path = os.path.join(path)
    pil_img = PIL.Image.fromarray(img)
    pil_img.save(image_path)


def save_images_partition(partition, data_dict, idx, image_path):
    # Copy images to new directory
    path = os.path.join(
        args.data_dir, "PartImages", args.name, "images", partition
    )
    for label in LABELS:
        os.makedirs(os.path.join(path, label), exist_ok=True)

    for i in idx:
        label = data_dict["labels"][i]
        image_filename = data_dict["images"][i]
        orig_image_path = os.path.join(image_path, image_filename)
        new_image_path = os.path.join(path, label, image_filename)
        if os.path.isfile(orig_image_path):
            copyfile(orig_image_path, new_image_path)

    # Save segmentation labels
    key = "panoptic-parts"
    path = os.path.join(
        args.data_dir, "PartImages", args.name, "panoptic-parts", partition
    )
    for label in LABELS:
        os.makedirs(os.path.join(path, label), exist_ok=True)
    for i in idx:
        label = data_dict["labels"][i]
        filename = data_dict["images"][i].split(".")[0]
        save_pil_image(
            data_dict[key][i], os.path.join(path, label, f"{filename}.tif")
        )


# Load annotations from the annotation folder of PASCAL-Part dataset:
if __name__ == "__main__":
    # Parse arguments from command line:
    parser = argparse.ArgumentParser(
        description="Prepare PASCAL-Part dataset for classification tasks"
    )
    parser.add_argument(
        "--data-dir", default="~/data/", type=str, help="Path to dataset"
    )
    parser.add_argument(
        "--name", default="temp", type=str, help="Name the new part dataset"
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument(
        "--val-ratio",
        default=0.1,
        type=float,
        help="Ratio of validation samples to all samples",
    )
    parser.add_argument(
        "--min-area",
        default=0.0,
        type=float,
        help="Min area of object to consider (relative to image size)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    data_dict = collect_parts(args)

    labels = np.array(data_dict["labels"])
    val_idx, test_idx, train_idx = [], [], []
    for l in np.unique(labels):
        idx = np.where(labels == l)[0]
        num_samples = len(idx)
        np.random.shuffle(idx)
        num_val, num_test = round(0.1 * num_samples), round(0.1 * num_samples)
        val_idx.append(idx[:num_val])
        test_idx.append(idx[num_val : num_val + num_test])
        train_idx.append(idx[num_val + num_test :])
    val_idx = np.concatenate(val_idx, axis=0)
    test_idx = np.concatenate(test_idx, axis=0)
    train_idx = np.concatenate(train_idx, axis=0)

    image_path = os.path.join(args.data_dir, "JPEGImages")
    for partition, indices in zip(
        ["train", "val", "test"], [train_idx, val_idx, test_idx]
    ):
        save_images_partition(partition, data_dict, indices, image_path)
