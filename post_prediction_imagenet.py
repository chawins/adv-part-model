import os
from PIL import Image
import random
import numpy as np
import shutil
import argparse

"""
This script postprocesses the data after generating the pseudo label masks by merging the groundtruth masks used to train the segmentation model with the generated masks.
1. Get filenames for all of the groundtruth masks used to train the segmentation model and the imagenet class.
2. Make the directories for the dataset
3. Get filenames for all of the generated pseudo label masks
4. Randomly partition the generated masks into train/val/test
5. Create training set by symlink the groundtruth masks over, and copy generated masks.
6. Create val/test set by copying over generated masks
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

# Step 1
classes = set()

# Get all samples from current 1x dataset to get folder-to-class mapping
old_dataset = set()
count = 0
tran_val_count = 0
for path, subdirs, files in os.walk(old_dataset_path):
    for name in files:
        if ".png" in name:
            if "train" in path or "val" in path:
                tran_val_count += 1
            count += 1
            old_dataset.add(name.split(".")[0])
            metaclass = path.split("/")[-1]
            classes.add(name.split("_")[0])

print(len(old_dataset))

# Step 2
# Make directories
os.mkdir(new_dataset_path)
for partition in ["train", "val", "test"]:
    os.mkdir(new_dataset_path + "/" + partition)
    for c in classes:
        os.mkdir(new_dataset_path + "/" + partition + "/" + c)


# Create dataset by merging old dataset and newly generated masks TODO: Dont forget to change target dir
# Step 3
# read generated masks name
mask_names = []
for root, dirs, files in os.walk(prediction_path):
    for file in files:
        if ".png" in file:
            mask_names.append(file.split(".")[0])

print(len(mask_names), len(set(mask_names)))

# Step 4
import random

mask_names = list(set(mask_names))

random.shuffle(mask_names)

total_samples = len(mask_names) + len(old_dataset)

# partition generated masks
desired_train_samples = int(0.8 * total_samples) - len(old_dataset)
train_masks = mask_names[:desired_train_samples]
temp = mask_names[desired_train_samples:]
val_masks = temp[: len(temp) // 2]
test_masks = temp[len(temp) // 2 :]

partition_generated = {"val": val_masks, "test": test_masks}

num_train = 0
num_val = 0
num_test = 0

# Step 5
# create training set
for c in classes:
    class_train_masks = [name for name in train_masks if c in name]

    # merge old train and val partitions used to train segmentor to train set
    with open(f"{old_dataset_path}/train/{c}.txt") as f:
        train_lines = [name.split("/")[1][:-1] for name in f.readlines()]
    with open(f"{old_dataset_path}/val/{c}.txt") as f:
        val_lines = [name.split("/")[1][:-1] for name in f.readlines()]

    old_list = train_lines + val_lines
    new_partition_list = list(set(old_list + class_train_masks))
    new_partition_list.sort()

    print(
        c,
        len(set(old_list)),
        len(set(class_train_masks)),
        len(new_partition_list),
    )
    num_train += len(new_partition_list)

    # .txt files
    with open(f"{new_dataset_path}/train/{c}.txt", "w") as f:
        for fileName in new_partition_list:
            f.write(fileName.split("_")[0] + "/" + fileName)
            f.write("\n")

    # symlink
    for fileName in train_lines:
        os.symlink(
            f"{relative_path_old_to_new}/train/{c}/{fileName}.png",  # calculate relative path
            f"{new_dataset_path}/train/{c}/{fileName}.png",
        )
    for fileName in val_lines:
        os.symlink(
            f"{relative_path_old_to_new}/val/{c}/{fileName}.png",  # calculate relative path
            f"{new_dataset_path}/train/{c}/{fileName}.png",
        )

    # copy newly generated masks over
    for fileName in class_train_masks:
        shutil.copyfile(
            f"{prediction_path}/{fileName}.png",
            f"{new_dataset_path}/train/{c}/{fileName}.png",
        )

# Step 6
for c in classes:
    for partition in ["val", "test"]:
        new_partition_list = [
            f for f in partition_generated[partition] if c in f
        ]
        print(
            len(partition_generated[partition]),
            len(new_partition_list),
        )
        new_partition_list = list(set(new_partition_list))
        new_partition_list.sort()

        # copy newly generated masks over
        for fileName in new_partition_list:
            shutil.copyfile(
                f"{prediction_path}/{fileName}.png",
                f"{new_dataset_path}/{partition}/{c}/{fileName}.png",
            )
        # write .txt file
        with open(f"{new_dataset_path}/{partition}/{c}.txt", "w") as f:
            for fileName in new_partition_list:
                f.write(fileName.split("_")[0] + "/" + fileName)
                f.write("\n")
