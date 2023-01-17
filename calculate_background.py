import os
from PIL import Image
import numpy as np

num = 0
cumPer = 0
for path, subdirs, files in os.walk(
    "/data/kornrapatp/PartImageNet/PartSegmentations/All-class-specific-processed/val"
):
    for name in files:
        className = path.split("/")[-1]
        if ".png" in name:
            img = np.asarray(Image.open(os.path.join(path, name)))
            cumPer += np.sum(img == 0) / (img.shape[0] * img.shape[1])
            num += 1
print(cumPer / num)
