import os
import random
import shutil

# create dataset for segmentor training

dataset_path = "/data/kornrapatp/PartImageNet/PartSegmentations/All"
target_path = "/data/kornrapatp/PartImageNet/PartSegmentations/All-seg"


def copy_files(partition, list_files, c):
    destination_path = f"{target_path}/{partition}/{c}/"
    for file in list_files:
        name = file.split("/")[-1] + ".tif"
        if os.path.exists(f"{dataset_path}/train/{c}/{name}"):
            os.symlink(f"../../../All/train/{c}/{name}", f"{destination_path}{name}")
        elif os.path.exists(f"{dataset_path}/val/{c}/{name}"):
            os.symlink(f"../../../All/val/{c}/{name}", f"{destination_path}{name}")
        elif os.path.exists(f"{dataset_path}/test/{c}/{name}"):
            os.symlink(f"../../../All/test/{c}/{name}", f"{destination_path}{name}")
        else:
            print(file)
            0 / 0


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


os.mkdir(target_path)
os.mkdir(target_path + "/train")
os.mkdir(target_path + "/test")
os.mkdir(target_path + "/val")

for c in classes:
    old_partition = {}
    for p in ["train", "val", "test"]:
        os.mkdir(f"{target_path}/{p}/{c}")
        with open(dataset_path + "/" + p + "/" + c + ".txt") as f:
            l = f.readlines()
        old_partition[p] = [line[:-1] for line in l]
        random.shuffle(old_partition[p])
    total_num = (
        len(old_partition["train"])
        + len(old_partition["val"])
        + len(old_partition["test"])
    )
    target_num_train = int(
        0.8 * 0.09 * total_num
    )  # modify here to change training set size
    target_num_val = int(
        0.8 * 0.01 * total_num
    )  # modify here to change validation set size
    new_train_split = old_partition["train"][:target_num_train]
    new_val_split = old_partition["train"][
        target_num_train : target_num_train + target_num_val
    ]
    new_test_split = (
        old_partition["train"][target_num_train + target_num_val :]
        + old_partition["val"]
        + old_partition["test"]
    )
    new_train_split.sort()
    new_val_split.sort()
    new_test_split.sort()
    print(c, len(new_train_split), len(new_val_split), len(new_test_split))
    with open(f"{target_path}/train/{c}.txt", "w") as f:
        for i in new_train_split:
            f.write(i)
            f.write("\n")
    with open(f"{target_path}/val/{c}.txt", "w") as f:
        for i in new_val_split:
            f.write(i)
            f.write("\n")
    with open(f"{target_path}/test/{c}.txt", "w") as f:
        for i in new_test_split:
            f.write(i)
            f.write("\n")

    copy_files("train", new_train_split, c)
    copy_files("val", new_val_split, c)
    copy_files("test", new_test_split, c)
