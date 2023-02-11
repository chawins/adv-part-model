import pathlib
from PIL import Image
import argparse

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument(
    "--seg-dataset",
    help="path of dataset to convert to png",
    default="/data/kornrapatp/PartImageNet/PartSegmentations/All",
    type=str,
)

# Read arguments from command line
args = parser.parse_args()

path_to_replace = args.jpeg_path

current_dir = pathlib.Path(path_to_replace)

all_tif_files = current_dir.rglob("*.tif")

for tif_file in all_tif_files:
    if tif_file.is_symlink():
        # get the symlink target
        target = tif_file.readlink()
        print(target)
        # create a new symlink with the same name with suffix .png
        tif_file.with_suffix(".png").symlink_to(target.with_suffix(".png"))
    else:
        Image.open(tif_file).save(tif_file.with_suffix(".png"))
    # remove
    tif_file.unlink(missing_ok=True)
