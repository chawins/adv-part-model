import os
from PIL import Image
import random
import numpy as np
import shutil
import argparse


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


classes = {
    "Aeroplane": set([]),
    "Quadruped": set([]),
    "Biped": set([]),
    "Fish": set([]),
    "Bird": set([]),
    "Snake": set([]),
    "Reptile": set([]),
    "Car": set([]),
    "Bicycle": set([]),
    "Boat": set([]),
    "Bottle": set([]),
}


# Make directories
os.mkdir(new_temp_mask_dataset_path)
for partition in ["train", "val", "test"]:
    os.mkdir(new_temp_mask_dataset_path + "/" + partition)
    for c in classes.keys():
        os.mkdir(new_temp_mask_dataset_path + "/" + partition + "/" + c)

# Get all samples from current 1x dataset to ignore during sample generation
old_dataset = set([])
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

# Get candidate list
count = 0
candidates = []
for folder, className in folder_to_class.items():
    print(folder)
    for path, subdirs, files in os.walk(f"{jpeg_path}/{folder}"):
        for name in files:
            count += 1
            if name.split(".")[0] not in old_dataset:
                candidates.append(className + "-" + name.split(".")[0])
            else:
                pass


# randomly shuffle candidates and pick only num_new_samples samples to include
random.shuffle(candidates)
print(len(candidates))
candidates = candidates[:num_new_samples]

# Put all of our candidates into test set to generate masks in the next step
for c in classes.keys():
    for partition in ["train", "val", "test"]:
        if partition == "test":
            class_candidate = [
                candi.split("-")[1] for candi in candidates if c in candi
            ]
            class_candidate.sort()
            # print(class_candidate)
            # 0 / 0
            with open(
                f"{new_temp_mask_dataset_path}/{partition}/{c}.txt", "w"
            ) as f:
                for candi in class_candidate:
                    f.write(candi.split("_")[0] + "/" + candi)
                    f.write("\n")
        else:
            with open(
                f"{new_temp_mask_dataset_path}/{partition}/{c}.txt", "w"
            ) as f:
                pass


# Create .tif file of correct dimensions inside test dir
def save_pil_image(img, path):
    image_path = os.path.join(path)
    pil_img = Image.fromarray(img)
    pil_img.save(image_path)


for c in classes.keys():
    with open(f"{new_temp_mask_dataset_path}/test/{c}.txt") as f:
        filenames = f.readlines()
    for name in filenames:
        name = name.split("/")[-1]
        img = Image.open(f'{jpeg_path}/{name.split("_")[0]}/{name[:-1]}.JPEG')
        tif = np.zeros(img.size)
        print(img.size)
        save_pil_image(
            tif, f"{new_temp_mask_dataset_path}/test/{c}/{name[:-1]}.tif"
        )

os.mkdir(prediction_path)
