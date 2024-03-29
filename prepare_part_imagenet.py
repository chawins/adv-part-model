"""
Code is adapted from https://github.com/micco00x/py-pascalpart

Usage examples:
python prepare_pascal_part_v3.py --data-dir ~/data/pascal_part/ --name name
"""
import argparse
import copy
import json
import glob
import os
import random
from itertools import chain, combinations

import numpy as np
import PIL
from tqdm import tqdm

from coco.coco import COCO

CLASSES = {
    "Quadruped": 4,
    "Biped": 5,
    "Fish": 4,
    "Bird": 5,
    "Snake": 2,
    "Reptile": 4,
    "Car": 3,
    "Bicycle": 4,
    "Boat": 2,
    "Aeroplane": 5,
    "Bottle": 2,
}


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_all_image_names(path):
    all_names = glob.glob(f"{path}/JPEGImages/*/*.JPEG")
    all_names = [name.split("/")[-1].split(".")[0] for name in all_names]
    return set(all_names)


def _get_box_from_bin_mask(bin_mask):
    box_mask = np.zeros_like(bin_mask)
    if bin_mask.sum() == 0:
        return box_mask
    y, x = np.where(bin_mask)
    ymin, ymax = y.min(), y.max()
    xmin, xmax = x.min(), x.max()
    box_mask[ymin : ymax + 1, xmin : xmax + 1] = 1
    return box_mask


def get_seg_masks(path, all_image_names, use_box_seg=False):

    coco = COCO(path)

    # Total number of parts including background
    num_parts = 1
    classes = sorted(list(CLASSES.keys()))
    all_part_ids = []
    for k in classes:
        num_parts += CLASSES[k]
        all_part_ids.extend(coco.getCatIds(supNms=k))
    # all_part_ids = coco.getCatIds(supNms=list(CLASSES.keys()))

    assert len(all_part_ids) == num_parts - 1

    data_dict = {
        "seg_masks": [],
        "img_paths": [],
        "labels": [],
    }

    for label in CLASSES:
        print(f"  ==> label: {label}")

        # Get id's of the desired class
        cat_ids = coco.getCatIds(supNms=label)

        # Iterate through all combinations of parts
        img_ids = []
        for ids in powerset(cat_ids):
            if len(ids) == 0:
                continue
            # Select only images from this class
            img_ids.extend(coco.getImgIds(catIds=ids))
        img_ids = set(img_ids)

        # import pdb
        # pdb.set_trace()

        imgs = coco.loadImgs(img_ids)
        seg_masks, img_paths = [], []
        for i, img_id in tqdm(enumerate(img_ids)):
            img = imgs[i]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            if img["file_name"].split(".")[0] not in all_image_names:
                print(f'{img["file_name"].split(".")[0]} file missing!')
                continue
            img_path = (
                f'{img["file_name"].split("_")[0]}/{img["file_name"].split(".")[0]}'
            )
            img_paths.append(img_path)

            # Turn annotation to mask
            seg_mask = np.zeros((img["height"], img["width"]), dtype=np.int8)
            for ann in anns:
                if ann["area"] == 0:
                    continue

                part_mask = coco.annToMask(ann)
                seg_label = all_part_ids.index(ann["category_id"]) + 1
                if use_box_seg:
                    part_mask = _get_box_from_bin_mask(part_mask)
                seg_mask = part_mask * seg_label + (1 - part_mask) * seg_mask
            assert seg_mask.max() <= num_parts
            assert seg_mask.min() >= 0
            seg_masks.append(seg_mask)

        data_dict["seg_masks"].extend(seg_masks)
        data_dict["img_paths"].extend(img_paths)
        data_dict["labels"].extend([list(CLASSES.keys()).index(label)] * len(seg_masks))

    return data_dict


def save_pil_image(img, path):
    image_path = os.path.join(path)
    pil_img = PIL.Image.fromarray(img)
    pil_img.save(image_path)


def save_images_partition(partition, data_dict, idx, label, use_box_seg=False):
    # Copy images to new directory
    if use_box_seg:
        path = os.path.join(args.data_dir, "BoxSegmentations", args.name, partition)
    else:
        path = os.path.join(args.data_dir, "PartSegmentations", args.name, partition)
    label_path = os.path.join(path, label)
    os.makedirs(label_path, exist_ok=True)
    img_paths = data_dict["img_paths"]
    seg_masks = data_dict["seg_masks"]

    # Write image paths to a file
    filenames = []
    for i in idx:
        filenames.append(img_paths[i])
    filenames.sort()
    filenames = [f + "\n" for f in filenames]

    with open(f"{path}/{label}.txt", "w") as path_file:
        path_file.writelines(filenames)
    # Write segmentation as tif file
    for i in idx:
        name = f'{img_paths[i].split("/")[1]}.tif'
        # import pdb
        # pdb.set_trace()
        save_pil_image(seg_masks[i], os.path.join(label_path, name))

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
    parser.add_argument("--use-box-seg", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dict = {
        "seg_masks": [],
        "img_paths": [],
        "labels": [],
    }
    all_names = get_all_image_names(args.data_dir)

    for partition in ["train", "test", "val"]:
        print(f"==> Collecting data from {partition} partition...")
        path = os.path.join(args.data_dir, f"{partition}.json")
        part_dict = get_seg_masks(path, all_names, use_box_seg=args.use_box_seg)
        for k in data_dict:
            data_dict[k].extend(part_dict[k])
    print(f'Total number of samples {len(data_dict["seg_masks"])}.')

    
    all_indices = {}
    all_indices['train'] = []
    all_indices['val'] = []
    all_indices['test'] = []

    # Randomly split data into train/test/val and keep the class ratio
    for l, label in enumerate(CLASSES):
        print(f"==> Writing {label} data...")
        idx = np.where(np.array(data_dict["labels"]) == l)[0]
        num_samples = len(idx)
        np.random.shuffle(idx)
        num_val, num_test = int(0.1 * num_samples), int(0.1 * num_samples)
        val_idx = idx[:num_val]
        test_idx = idx[num_val : num_val + num_test]
        train_idx = idx[num_val + num_test :]

        all_indices['train'].extend(train_idx)
        all_indices['val'].extend(val_idx)
        all_indices['test'].extend(test_idx)
        
        print(
            f"  ==> {num_samples} samples in total",
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )

        for partition, indices in zip(
            ["train", "val", "test"], [train_idx, val_idx, test_idx]
        ):
            print(f"  ==> New {partition} partition.")
            save_images_partition(
                partition,
                data_dict,
                indices,
                label,
                use_box_seg=args.use_box_seg,
            )



    # save bounding box annotations
    ann_data = {}
    ann_data['images'] = []
    ann_data['annotations'] = []
    ann_data['categories'] = []

    # combining all existing bbox annotations
    new_image_id = 0
    new_annotation_id = 0
    for partition in ['train', 'val', 'test']:
        ann_file_path = os.path.join(args.data_dir, f"{partition}.json")

        with open(ann_file_path) as f:
            partition_data = json.load(f)
        
        for image in partition_data['images']:            
            original_image_id = image['id']
            for ann_orig in partition_data['annotations']:
                if ann_orig['image_id'] == original_image_id:
                    ann = copy.deepcopy(ann_orig)
                    ann['image_id'] = new_image_id
                    ann['id'] = new_annotation_id
                    ann_data['annotations'].append(ann)
                    new_annotation_id += 1
            
            # assign image new id
            image['id'] = new_image_id
            ann_data['images'].append(image)
            new_image_id += 1

        if not ann_data['categories']:
            ann_data['categories'].extend(partition_data['categories'])
        print('num images: ', len(partition_data['images']))

    img_paths = data_dict["img_paths"]
    # reshufling the images and saving the new annotations in COCO format
    for partition in ['train', 'val', 'test']:
        partition_ann_data = {}
        partition_ann_data['images'] = []
        partition_ann_data['annotations'] = []
        partition_ann_data['categories'] = ann_data['categories']

        partition_indices = all_indices[partition]

        # Write image paths to a file
        partition_filenames = []
        for i in partition_indices:
            image_filename = img_paths[i].split('/')[-1] + '.JPEG'
            partition_filenames.append(image_filename)

        prev_image_ids_to_new_image_ids = {}
        new_image_id = 0
        new_annotation_id = 0
        for image in ann_data['images']:
            if image['file_name'] in partition_filenames:
                original_image_id = image['id']
                for ann_orig in ann_data['annotations']:
                    if ann_orig['image_id'] == original_image_id:
                        ann = copy.deepcopy(ann_orig)
                        ann['image_id'] = new_image_id
                        ann['id'] = new_annotation_id
                        partition_ann_data['annotations'].append(ann)
                        new_annotation_id += 1
            
                # assign image new id
                image['id'] = new_image_id
                
                imagenet_id, part_imagenet_id = image['file_name'].split('_')
                image['file_name'] = os.path.join(imagenet_id, image['file_name'])
                partition_ann_data['images'].append(image)
                new_image_id += 1
                        
        # save json to file
        ann_file_path = os.path.join(args.data_dir, "PartBoxSegmentations", f"{partition}.json")
        with open(ann_file_path, 'w') as f:
            json.dump(partition_ann_data, f)
