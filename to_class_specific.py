import os
from PIL import Image
import numpy as np
import pathlib
import random
import shutil

""" 
This script converts metaclass dataset into imagenet class dataset.
1. Fetch the mapping from metaclass to imagenet foldername list (to be used during reading the masks to be relabeled)
2. Fetch the mapping from imagenet foldernames to number of part (to be used to get the part starting index of each class)
3. Make the imagenet dataset directory
4. Get the metaclass mask starting indices and imagenet class mask starting indices (to be used during mask relabel)
5. Relabel each mask, for each non-bg pixel, subtract the metaclass starting index and add the imagenet class starting index.
6. Write the .txt file for the new dataset
"""

metaclass_dataset_dir = (
    "/data/kornrapatp/PartImageNet/PartSegmentations/All-imagenetclass-segtrain"
)
imagenetclass_dataset_dir = "/data/kornrapatp/PartImageNet/PartSegmentations/All-imagenetclass-segtrain-processed"

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

metaclass_to_class = {
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
for path, subdirs, files in os.walk(metaclass_dataset_dir):
    for name in files:
        if ".tif" in name:
            metaclass = path.split("/")[-1]
            imagenet_class = name.split("_")[0]
            metaclass_to_class[metaclass].add(imagenet_class)

# Step 2
numpart = 1
imagenet_classes_part_num = {}
for k, v in metaclass_to_class.items():
    numpart += CLASSES[k] * len(v)
    for imagenet_class in v:
        imagenet_classes_part_num[imagenet_class] = CLASSES[k]
imagenet_classes_part_num = dict(sorted(imagenet_classes_part_num.items()))
print(f"Total part in new dataset: {numpart}")

# Step 3
# make directories
os.mkdir(imagenetclass_dataset_dir)
for partition in ["train", "val", "test"]:
    os.mkdir(imagenetclass_dataset_dir + "/" + partition)
    for c in imagenet_classes_part_num.keys():
        os.mkdir(f"{imagenetclass_dataset_dir}/{partition}/{c}")
        with open(
            f"{imagenetclass_dataset_dir}/{partition}/{c}.txt",
            "w",
        ) as f:
            f.write("")

# Step 4
classes = sorted(CLASSES.keys())
print(classes)
class_starting_index = {}
curid = 1

for c in classes:
    class_starting_index[c] = curid
    curid += CLASSES[c]

print(class_starting_index)


imagenet_class_starting_index = {}
imagenet_indices = {}
curid = 1

for c in imagenet_classes_part_num.keys():
    imagenet_class_starting_index[c] = curid
    imagenet_indices[c] = [
        i for i in range(curid, curid + imagenet_classes_part_num[c])
    ]
    curid += imagenet_classes_part_num[c]

# Step 5
def save_pil_image(img, path):
    image_path = os.path.join(path)
    pil_img = Image.fromarray(img)
    pil_img.save(image_path)


fileList = {}
# Rewrite segmentation labels
for path, subdirs, files in os.walk(metaclass_dataset_dir):
    for name in files:
        className = path.split("/")[-1]
        if ".tif" in name:
            img = np.asarray(Image.open(os.path.join(path, name)))
            imagenet_className = name.split("_")[0]

            new_img = np.where(
                img != 0,
                img
                - (
                    class_starting_index[className]
                    - imagenet_class_starting_index[imagenet_className]
                ),
                np.zeros(img.shape),
            ).astype(np.int32)

            save_pil_image(
                new_img,
                os.path.join(
                    imagenetclass_dataset_dir,
                    path.split("/")[-2],
                    imagenet_className,
                    name,
                ),
            )
            # Save filenames for .txt file
            if path.split("/")[-2] + "/" + imagenet_className not in fileList:
                fileList[path.split("/")[-2] + "/" + imagenet_className] = []
            fileList[path.split("/")[-2] + "/" + imagenet_className].append(
                imagenet_className + "/" + name.split(".")[0] + "\n"
            )

# Step 6
for k, v in fileList.items():
    v = sorted(v)
    with open(
        f"{imagenetclass_dataset_dir}/{k}.txt",
        "w",
    ) as f:
        for name in v:
            f.write(name)
