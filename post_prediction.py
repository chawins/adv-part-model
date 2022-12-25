import os
from PIL import Image
import random
import numpy as np
import shutil
import argparse

"""
This script postprocesses the data after generating the pseudo label masks by merging the groundtruth masks used to train the segmentation model with the generated masks.
1. Make the directories for the dataset
2. Get filenames for all of the groundtruth masks used to train the segmentation model and the folder name to metaclass mapping.
3. Get filenames for all of the generated pseudo label masks
4. For each class, partition the generated pseudo label masks
5. Symlink the groundtruth masks over
6. Copy the generated pseudo label masks over
"""

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument(
    "--old-dataset",
    help="path of old dataset",
    default="/data/kornrapatp/PartImageNet/PartSegmentations/All",
    type=str,
)

parser.add_argument(
    "--new-dataset",
    help="path of new dataset",
    default="/data/kornrapatp/PartImageNet/PartSegmentations/All-step3-all-part-new",
    type=str,
)

parser.add_argument(
    "--jpeg-path",
    help="path of jpegs",
    default="/data/shared/train_blurred/Images",
    type=str,
)

parser.add_argument(
    "--num-new-samples",
    help="number of new samples to add to dataset",
    default=24000,
    type=int,
)

parser.add_argument(
    "--prediction-path",
    help="path of predicted masks",
    default="/data/kornrapatp/test",
    type=str,
)

# Read arguments from command line
args = parser.parse_args()

jpeg_path = args.jpeg_path
old_dataset_path = args.old_dataset
new_temp_mask_dataset_path = args.new_dataset + "-mask"
new_dataset_path = args.new_dataset
num_new_samples = args.num_new_samples
prediction_path = args.prediction_path

# relative path is hard-coded
relative_path_old_to_new = "../../../" + old_dataset_path.split("/")[-1]

classes = {
    "Aeroplane": set(),
    "Quadruped": set(),
    "Biped": set(),
    "Fish": set(),
    "Bird": set(),
    "Snake": set(),
    "Reptile": set(),
    "Car": set(),
    "Bicycle": set(),
    "Boat": set(),
    "Bottle": set(),
}

# Step 1
# Make directories
os.mkdir(new_dataset_path)
for partition in ["train", "val", "test"]:
    os.mkdir(new_dataset_path + "/" + partition)
    for c in classes.keys():
        os.mkdir(new_dataset_path + "/" + partition + "/" + c)

# Step 2
# Get all samples from current 1x dataset to get folder-to-class mapping
old_dataset = set()
count = 0
tran_val_count = 0
for path, subdirs, files in os.walk(old_dataset_path):
    for name in files:
        if ".tif" in name:
            if "train" in path or "val" in path:
                tran_val_count += 1
            count += 1
            old_dataset.add(name.split(".")[0])
            metaclass = path.split("/")[-1]
            classes[metaclass].add(name.split("_")[0])


# Create folder to class mapping
folder_to_class = {}
# print(classes)
for k, v in classes.items():
    for folder in v:
        folder_to_class[folder] = k


# Step 3
# Create dataset by merging old dataset and newly generated masks TODO: Dont forget to change target dir
# read generated masks name
mask_names = []
for root, dirs, files in os.walk(prediction_path):
    for file in files:
        if ".tif" in file:
            mask_names.append(file.split(".")[0])

# Step 4
for c in classes.keys():
    # read and partition newly generated masks
    filenames = [
        name for name in mask_names if folder_to_class[name.split("_")[0]] == c
    ]

    # randomly split train/val 8 parts train 1 part val (from original 0.8/0.1/0.1 split)
    random.shuffle(filenames)
    partition_generated = {}
    partition_generated["train"] = filenames[: int(len(filenames) * 8 / 9)]
    temp = filenames[int(len(filenames) * 8 / 9) :]
    partition_generated["val"] = temp[:]
    partition_generated["test"] = []

    for partition in ["train", "val", "test"]:
        print(partition, f"{old_dataset_path}/{partition}/{c}.txt")
        with open(f"{old_dataset_path}/{partition}/{c}.txt") as f:
            lines = f.readlines()
        step_all_list = [name.split("/")[1][:-1] for name in lines]
        new_partition_list = partition_generated[partition] + step_all_list
        # 0 / 0
        print(
            len(step_all_list),
            len(partition_generated[partition]),
            len(new_partition_list),
        )
        new_partition_list = list(set(new_partition_list))
        new_partition_list.sort()

        # Step 5
        # symlink original masks from old dataset
        for fileName in step_all_list:
            os.symlink(
                f"{relative_path_old_to_new}/{partition}/{c}/{fileName}.tif",  # calculate relative path
                f"{new_dataset_path}/{partition}/{c}/{fileName}.tif",
            )

        # Step 6
        # copy newly generated masks over
        for fileName in partition_generated[partition]:
            shutil.copyfile(
                f"{prediction_path}/{fileName}.tif",
                f"{new_dataset_path}/{partition}/{c}/{fileName}.tif",
            )
        # write .txt file
        with open(f"{new_dataset_path}/{partition}/{c}.txt", "w") as f:
            for fileName in new_partition_list:
                f.write(fileName.split("_")[0] + "/" + fileName)
                f.write("\n")
