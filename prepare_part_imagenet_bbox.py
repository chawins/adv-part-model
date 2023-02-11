import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from partimagenet_hparams import partimagenet_to_grouped_partimagenet

# # reds
# label_dir = '/data/shared/PartImageNet/'
# root = Path('/data/shared/PartImageNet/PartBoxSegmentations')

# reds5k
# label_dir = '/data1/chawins/PartImageNet/'
# root = Path('/data1/chawins/PartImageNet/PartBoxSegmentations')

# # savio
# label_dir = '/global/scratch/users/nabeel126/PartImageNet/'
# root = Path('/global/scratch/users/nabeel126/PartImageNet/PartBoxSegmentations')


def group_parts(array, mapping):
    image_labels = list(np.unique(array))
    for part_id in image_labels:
        groupid = mapping[part_id]
        array[array == part_id] = groupid
    return array

def main(split, label_dir, sample_proportion=0.10, bbox_discard_threshold=0.01):
    root = Path(os.path.join(label_dir, 'PartBoxSegmentations'))
    num_discarded_bbox = 0

    PART_IMAGENET_CLASSES = {
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
    PART_IMAGENET_CLASSES = dict(sorted(PART_IMAGENET_CLASSES.items()))

    if args.use_imagenet_classes:
        if args.group_parts:
            os.makedirs(root / "image_labels" / "imagenet" / "grouped", exist_ok=True)
            os.makedirs(root / "annotations" / "imagenet" / "grouped", exist_ok=True)
            PATHS = {
                "train": (
                    root / "train",
                    root / "image_labels" / "imagenet" / "grouped" / "train.json",
                    root / "annotations" / "imagenet" / "grouped" / "train.json",
                ),
                "val": (
                    root / "val",
                    root / "image_labels" / "imagenet" / "grouped" / "val.json",
                    root / "annotations" / "imagenet" / "grouped" / "val.json",
                ),
                "test": (
                    root / "test",
                    root / "image_labels" / "imagenet" / "grouped" / "test.json",
                    root / "annotations" / "imagenet" / "grouped" / "test.json",
                ),
            }
        else:
            os.makedirs(root / "image_labels" / "imagenet" / "all", exist_ok=True)
            os.makedirs(root / "annotations" / "imagenet" / "all", exist_ok=True)
            PATHS = {
                "train": (
                    root / "train",
                    root / "image_labels" / "imagenet" / "all" / "train.json",
                    root / "annotations" / "imagenet" / "all" / "train.json",
                ),
                "val": (
                    root / "val",
                    root / "image_labels" / "imagenet" / "all" / "val.json",
                    root / "annotations" / "imagenet" / "all" / "val.json",
                ),
                "test": (
                    root / "test",
                    root / "image_labels" / "imagenet" / "all" / "test.json",
                    root / "annotations" / "imagenet" / "all" / "test.json",
                ),
            }
    else:
        if args.group_parts:
            os.makedirs(root / "image_labels" / "partimagenet" / "grouped", exist_ok=True)
            os.makedirs(root / "annotations" / "partimagenet" / "grouped", exist_ok=True)
            PATHS = {
                "train": (
                    root / "train",
                    root / "image_labels" / "partimagenet" / "grouped" / "train.json",
                    root / "annotations" / "partimagenet" / "grouped" / "train.json",
                ),
                "val": (
                    root / "val",
                    root / "image_labels" / "partimagenet" / "grouped" / "val.json",
                    root / "annotations" / "partimagenet" / "grouped" / "val.json",
                ),
                "test": (
                    root / "test",
                    root / "image_labels" / "partimagenet" / "grouped" / "test.json",
                    root / "annotations" / "partimagenet" / "grouped" / "test.json",
                ),
            }
        else:
            os.makedirs(root / "image_labels" / "partimagenet" / "all", exist_ok=True)
            os.makedirs(root / "annotations" / "partimagenet" / "all", exist_ok=True)
            PATHS = {
                "train": (
                    root / "train",
                    root / "image_labels" / "partimagenet" / "all" / "train.json",
                    root / "annotations" / "partimagenet" / "all" / "train.json",
                ),
                "val": (
                    root / "val",
                    root / "image_labels" / "partimagenet" / "all" / "val.json",
                    root / "annotations" / "partimagenet" / "all" / "val.json",
                ),
                "test": (
                    root / "test",
                    root / "image_labels" / "partimagenet" / "all" / "test.json",
                    root / "annotations" / "partimagenet" / "all" / "test.json",
                ),
            }


    # PATHS = {
    #     "train": (root / "train", root / "image_labels" / 'train.json', root / "annotations" / 'train.json'),
    #     "val": (root / "val", root / "image_labels" / 'val.json', root / "annotations" / 'val.json'),
    #     "test": (root / "test", root / "image_labels" / 'test.json', root / "annotations" / 'test.json' ),
    #     "train_sample": (root / "train", root / "image_labels" / 'train_sample.json', root / "annotations" / 'train_sample.json'),
    #     "val_sample": (root / "val", root / "image_labels" / 'val_sample.json', root / "annotations" / 'val_sample.json'),
    #     "test_sample": (root / "test", root / "image_labels" / 'test_sample.json', root / "annotations" / 'test_sample.json' ),
    # }

    os.makedirs(root / "train", exist_ok=True)
    os.makedirs(root / "val", exist_ok=True)
    os.makedirs(root / "test", exist_ok=True)



    
    
    
    # two things that we want to do:
    # 1. predict imagenet labels, not partimagenet labels
    # 2. predict grouped part labels, not individual part labels
    
    # with open(f"{label_dir}/{part_imagenet_class}.txt", "r") as path_file:
    #     path_file.writelines(filenames)

    imagenet_labels_path = os.path.join(label_dir, 'LOC_synset_mapping.txt')
    imagenet_id2name = {}
    with open(imagenet_labels_path, "r") as f:        
        for line in f:
            line = line.strip()
            line_split = line.split()
            imagenet_id2name[line_split[0]] = line_split[1]
    # print(imagenet_id2name)
    # import pdb; pdb.set_trace()

    
    part_segmentations_path = os.path.join(
        label_dir, "PartSegmentations", 'All', args.split
    )

    IMAGENET_IDS = set()
    # create mapping from imagenet id to partimagenet id
    # partimagenetid_to_imagenetid = {}
    for class_label, part_imagenet_class in enumerate(PART_IMAGENET_CLASSES):
        with open(f"{part_segmentations_path}/{part_imagenet_class}.txt", "r") as f:
            filenames = f.readlines()
            for filename in filenames:
                filename = filename.strip()
                imagenet_id = filename.split('/')[0]
                IMAGENET_IDS.add(imagenet_id)

    IMAGENET_IDS = list(IMAGENET_IDS)
    assert len(IMAGENET_IDS) == 158 # https://arxiv.org/pdf/2112.00933.pdf
    IMAGENET_IDS.sort()

    imagenetid_to_imagenetclass = {}
    for imagenet_class, imagenet_id in enumerate(IMAGENET_IDS):
        imagenetid_to_imagenetclass[imagenet_id] = imagenet_class
    # import pdb; pdb.set_trace()




    categories_json = []
    part_id = 0
    # categories_json.append(
    #     {'supercategory': 'background',
    #     'id': 0,
    #     'name': 'background'}
    # )
    # part_id += 1
    if args.group_parts:
        processed_groupids = set()
        for part_imagenet_class in PART_IMAGENET_CLASSES:
            for id in range(PART_IMAGENET_CLASSES[part_imagenet_class]):
                groupid = partimagenet_to_grouped_partimagenet[part_id]
                if groupid not in processed_groupids:                    
                    categories_json.append(
                        {'supercategory': part_imagenet_class,
                        'id': part_id,
                        'name': f'{part_imagenet_class}_{part_id}'}
                    )
                    processed_groupids.add(groupid)
                part_id += 1
    else:
        for part_imagenet_class in PART_IMAGENET_CLASSES:
            for id in range(PART_IMAGENET_CLASSES[part_imagenet_class]):
                categories_json.append(
                    {'supercategory': part_imagenet_class,
                    'id': part_id,
                    'name': f'{part_imagenet_class}_{part_id}'}
                )
                part_id += 1

            
    np.random.seed(1234)

    images_json = []
    annotations_json = []

    image_to_label = {}

    global_image_id = 0
    image_part_id = 0
    for class_label, part_imagenet_class in enumerate(PART_IMAGENET_CLASSES):
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
            image_area = width * height
            imarray = np.array(im)
            
            if args.group_parts:
                imarray = group_parts(imarray, partimagenet_to_grouped_partimagenet)

            image_name = filename.split('/')[-1]
            part_imagenet_image_id = image_name[:-4]    

            jpeg_image_name = part_imagenet_image_id + '.JPEG'
            folder_id = part_imagenet_image_id.split('_')[0]

            if args.use_imagenet_classes:
                class_label = imagenetid_to_imagenetclass[folder_id]

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

            assert len(image_labels) <= PART_IMAGENET_CLASSES[part_imagenet_class]
            
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

                    bbox_area = bbox_width * bbox_height
                    bbox_to_image_area_ratio = bbox_area / image_area
                    
                    small_bbox = bbox_to_image_area_ratio < bbox_discard_threshold
                    if small_bbox:
                        num_discarded_bbox += 1
                        continue

                    cur_part_bbox = [min_col, min_row, bbox_width, bbox_height]

                    # subtract 1 from part label to make it zero-indexed (background class was removed)
                    annotations_json.append({
                        'image_id': global_image_id,
                        'bbox': cur_part_bbox,
                        'category_id': int(part_label)-1,
                        'id': image_part_id,
                        'area': bbox_area,
                        'iscrowd': 0
                    })
                    image_part_id += 1
                    
            global_image_id += 1

    print(f'[INFO] Number of discarded bboxes: {num_discarded_bbox}')

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

    parser.add_argument(
        "--use-imagenet-classes", action="store_true", help="Use imagenet classes instead of part imagenet classes"
    )
    parser.add_argument(
        "--group-parts", action="store_true", help="Group part imagenet classes"
    )

    parser.add_argument(
        "--bbox-discard-threshold", default=0.01, help="bbox with bbox-to-image-area ratio smaller than this threshold will be discarded"
    )

    args = parser.parse_args()
    main(split=args.split, label_dir=args.label_dir, bbox_discard_threshold=args.bbox_discard_threshold)
        