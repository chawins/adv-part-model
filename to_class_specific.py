import os
from PIL import Image
import numpy as np
import pathlib
import random
import shutil


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


for path, subdirs, files in os.walk(
    "/data/kornrapatp/PartImageNet/PartSegmentations/All-class-specific"
):
    for name in files:
        if ".tif" in name:
            metaclass = path.split("/")[-1]
            imagenet_class = name.split("_")[0]
            metaclass_to_class[metaclass].add(imagenet_class)

# numclass = 1
# for k, v in metaclass_to_class.items():
#     numclass += CLASSES[k] * len(v)
# print(numclass)
# 0 / 0
# print(metaclass_to_class)
# for k, v in metaclass_to_class.items():
#     print(k, len(v))


classes = sorted(CLASSES.keys())
print(classes)

class_starting_index = {}
curid = 1

for c in classes:
    class_starting_index[c] = curid
    curid += CLASSES[c]

print(class_starting_index)


imagenet_class_starting_index = {}
curid = 1
imagenet_classes_part_num = {}

for c in classes:
    imagenet_classes = metaclass_to_class[c]
    for imagenet_class in imagenet_classes:
        imagenet_class_starting_index[imagenet_class] = curid
        curid += CLASSES[c]
        imagenet_classes_part_num[imagenet_class] = CLASSES[c]

print(imagenet_class_starting_index)


def save_pil_image(img, path):
    image_path = os.path.join(path)
    pil_img = Image.fromarray(img)
    pil_img.save(image_path)


# # Rewrite segmentation labels
# for path, subdirs, files in os.walk(
#     "/data/kornrapatp/PartImageNet/PartSegmentations/All-class-specific"
# ):
#     for name in files:
#         className = path.split("/")[-1]
#         if ".tif" in name:
#             img = np.asarray(Image.open(os.path.join(path, name)))
#             imagenet_className = name.split("_")[0]
#             new_img = np.where(
#                 img != 0,
#                 img
#                 - (
#                     class_starting_index[className]
#                     - imagenet_class_starting_index[imagenet_className]
#                 ),
#                 np.zeros(img.shape),
#             ).astype(np.int32)
#             print(img.dtype, new_img.dtype, np.amax(new_img), className)
#             save_pil_image(
#                 new_img,
#                 os.path.join(
#                     "/data/kornrapatp/PartImageNet/PartSegmentations/All-class-specific-processed",
#                     path.split("/")[-2],
#                     path.split("/")[-1],
#                     name,
#                 ),
#             )


# # Copy over remaining imagenet images to partimagenet
# seen = set([])
# for path, subdirs, files in os.walk(
#     "/data/kornrapatp/PartImageNet/PartSegmentations/All-class-specific/processed"
# ):
#     for name in files:
#         if ".tif" in name:
#             seen.add(name.split(".")[0])

# for k, v in metaclass_to_class.items():
#     classes = list(v)
#     copy_dir = "/data/kornrapatp/PartImageNet/JPEGImagesImageNet"
#     copy_dir = pathlib.Path(copy_dir)
#     copy_dir.mkdir(exist_ok=True)

#     file_list = []
#     print(k)

#     for c in classes:
#         path = pathlib.Path("/data/shared/train_blurred/Images/" + c)
#         images = list(path.glob("*.jpg"))
#         # print(images)
#         # random.shuffle(images)
#         (copy_dir / c).mkdir(exist_ok=True)
#         for img in images:
#             if img.name.split(".")[0] not in seen:
#                 n = img.name.split(".")[0]
#                 file_list.append(n.split("_")[0] + "/" + n)
#                 imgNP = np.asarray(Image.open(img))
#                 seg = np.zeros(imgNP.shape[:2])
#                 save_pil_image(
#                     seg,
#                     "/data/kornrapatp/PartImageNet/PartSegmentations/All-class-specific-processed/test/"
#                     + k
#                     + "/"
#                     + img.name.split(".")[0]
#                     + ".tif",
#                 )
#             shutil.copyfile(img, str(copy_dir / c / img.name))
#     with open(
#         "/data/kornrapatp/PartImageNet/PartSegmentations/All-class-specific-processed/test/"
#         + k
#         + ".txt",
#         "w",
#     ) as f:
#         for line in file_list:
#             f.write(line)
#             f.write("\n")
