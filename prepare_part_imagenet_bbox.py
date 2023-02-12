import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops

# # reds
# label_dir = '/data/shared/PartImageNet/'
# root = Path('/data/shared/PartImageNet/PartBoxSegmentations')

# reds5k
# label_dir = '/data1/chawins/PartImageNet/'
# root = Path('/data1/chawins/PartImageNet/PartBoxSegmentations')

# # savio
# label_dir = '/global/scratch/users/nabeel126/PartImageNet/'
# root = Path('/global/scratch/users/nabeel126/PartImageNet/PartBoxSegmentations')

def main(split, label_dir):
    root = Path(os.path.join(label_dir, 'PartBoxSegmentations'))

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
    CLASSES = dict(sorted(CLASSES.items()))

    PATHS = {
        "train": (root / "train", root / "image_labels" / 'train.json', root / "annotations" / 'train.json'),
        "val": (root / "val", root / "image_labels" / 'val.json', root / "annotations" / 'val.json'),
        "test": (root / "test", root / "image_labels" / 'test.json', root / "annotations" / 'test.json' ),
        "train_sample": (root / "train", root / "image_labels" / 'train_sample.json', root / "annotations" / 'train_sample.json'),
        "val_sample": (root / "val", root / "image_labels" / 'val_sample.json', root / "annotations" / 'val_sample.json'),
        "test_sample": (root / "test", root / "image_labels" / 'test_sample.json', root / "annotations" / 'test_sample.json' ),
    }

    os.makedirs(root / "train", exist_ok=True)
    os.makedirs(root / "val", exist_ok=True)
    os.makedirs(root / "test", exist_ok=True)
    os.makedirs(root / "image_labels", exist_ok=True)
    os.makedirs(root / "annotations", exist_ok=True)

    partimagenet_id2name = {}
    categories_json = []

    part_id = 0
    categories_json.append(
        {'supercategory': 'background',
        'id': 0,
        'name': 'background'}
    )
    partimagenet_id2name[part_id] = 'background'
    part_id += 1

    for part_imagenet_class in CLASSES:
        for id in range(CLASSES[part_imagenet_class]):
            categories_json.append(
                {'supercategory': part_imagenet_class,
                'id': part_id,
                'name': f'{part_imagenet_class}_{part_id}'}
            )
            partimagenet_id2name[part_id] = f'{part_imagenet_class}_{part_id}'
            part_id += 1


    # saving annotations
    json_object = json.dumps(partimagenet_id2name, indent=4)
    id2name_filename = os.path.join(label_dir, 'id2name.json')
    with open(id2name_filename, "w") as outfile:
        outfile.write(json_object)
        
    sample_proportion = 0.10

    np.random.seed(1234)

    images_json = []
    annotations_json = []

    image_to_label = {}

    global_image_id = 0
    image_part_id = 0
    for class_label, part_imagenet_class in enumerate(CLASSES):
        print('part_imagenet_class', part_imagenet_class)
        # get filenames of all segmentation masks
        if '_sample' in split:
            seg_label_path = os.path.join(label_dir, f"PartSegmentations/All/{split.replace('_sample', '')}/{part_imagenet_class}/*.tif")
        else:
            seg_label_path = os.path.join(label_dir, f'PartSegmentations/All/{split}/{part_imagenet_class}/*.tif')
        seg_filenames = glob.glob(seg_label_path)
        
        if '_sample' in split:
            np.random.shuffle(seg_filenames)
            seg_filenames = seg_filenames[:int(sample_proportion * len(seg_filenames))]

        for filename in seg_filenames:
            # load segmentation
            im = Image.open(filename)
            width, height = im.size
            imarray = np.array(im)

            image_name = filename.split('/')[-1]
            image_id = image_name[:-4]    

            jpeg_image_name = image_id + '.JPEG'
            folder_id = image_id.split('_')[0]

            images_json.append({
                'file_name': f'{folder_id}/{jpeg_image_name}',
                'height': height,
                'width': width,
                'id': global_image_id
                })

            image_to_label[global_image_id] = class_label

            # get unique labels
            image_labels = list(np.unique(imarray))
            
            # remove background class
            if 0 in image_labels:
                image_labels.remove(0)

            assert len(image_labels) <= CLASSES[part_imagenet_class]
            
            for _, part_label in enumerate(image_labels):         
                # get sementation mask for object_id
                mask = (imarray == part_label) * 1
                mask = np.uint8(mask)

                lbl_0, num_regions = label(mask, return_num=True, connectivity=1) 
                props = regionprops(lbl_0)

                for prop_index, prop in enumerate(props):
                    min_row, min_col, max_row, max_col = prop.bbox

                    bbox_width = max_col - min_col
                    bbox_height = max_row - min_row

                    cur_part_bbox = [min_col, min_row, bbox_width, bbox_height]

                    annotations_json.append({
                        'image_id': global_image_id,
                        'bbox': cur_part_bbox,
                        'category_id': int(part_label),
                        'id': image_part_id,
                        'area': bbox_width * bbox_height,
                        'iscrowd': 0
                    })
                    image_part_id += 1
                    
            global_image_id += 1

    # TODO: remove last few images in validation set for debugging. otherwise distributed training does not work
    if 'val' in split:
        images_json = images_json[:2400]

    # saving annotations
    json_object = json.dumps(image_to_label, indent=4)
    with open(PATHS[split][1], "w") as outfile:
        outfile.write(json_object)
        
    part_imagenet_bbox_json = {
        'images': images_json,
        'annotations': annotations_json,
        'categories': categories_json
    }

    json_object = json.dumps(part_imagenet_bbox_json, indent=4)
    with open(PATHS[split][2], "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    # Parse arguments from command line:
    parser = argparse.ArgumentParser(
        description="Prepare Part Image Net dataset for object detection task"
    )
    parser.add_argument(
        "--label-dir", default="~/PartImageNet/", type=str, help="Path to dataset"
    )
    parser.add_argument(
        "--split", default="train", type=str, help="dataset split, one of train, val, test"
    )

    args = parser.parse_args()
    main(split=args.split, label_dir=args.label_dir)
        