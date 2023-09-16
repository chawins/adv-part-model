# Part-Based Models Improve Adversarial Robustness

Chawin Sitawarin<sup>1</sup>, Kornrapat Pongmala<sup>1</sup>, Yizheng Chen<sup>1</sup>, Nicholas Carlini<sup>2</sup>, David Wagner<sup>1</sup>  
<sup>1</sup>UC Berkeley <sup>2</sup>Google

Published in ICLR 2023 (poster): [paper](https://openreview.net/forum?id=bAMTaeqluh4)

## Abstract

We show that combining human prior knowledge with end-to-end learning can improve the robustness of deep neural networks by introducing a part-based model for object classification.
We believe that the richer form of annotation helps guide neural networks to learn more robust features without requiring more samples or larger models.
Our model combines a part segmentation model with a tiny classifier and is trained end-to-end to simultaneously segment objects into parts and then classify the segmented object.
Empirically, our part-based models achieve both higher accuracy and higher adversarial robustness than a ResNet-50 baseline on all three datasets.
For instance, the clean accuracy of our part models is up to 15 percentage points higher than the baseline's, given the same level of robustness.
Our experiments indicate that these models also reduce texture bias and yield better robustness against common corruptions and spurious correlations.

---

## Setup

### Dependency Installation

- Tested environment can be installed with either `environment.yml` (for `conda`) or `requirements.txt` (for `pip`).
- `python 3.8`
- [`timm`](https://github.com/rwightman/pytorch-image-models)
- [`segmentation-models-pytorch`](https://github.com/qubvel/segmentation_models.pytorch)
- `imagecorruptions` for testing with common corruptions
- `foolbox` for testing with black-box attacks
- `torchmetrics` for computing IoU
- `kornia` for custom transformations

```bash
# Install dependencies with pip
pip install -r requirements.txt

# OR install dependencies with conda
conda env create -f environment.yml

# OR install dependencies manually with latest packages
# Install pytorch 1.10 (or later)
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -y h5py scipy pandas scikit-learn pip seaborn
conda install -y -c anaconda cython
pip install -U timm kornia albumentations tensorboard wandb cmake onnx \
  ruamel.yaml segmentation-models-pytorch imagecorruptions foolbox matplotlib \
  scikit-image torchmetrics termcolor jaxtyping addict yapf \
  opencv-python==4.5.5.64
pip install -U 'git+https://github.com/fra31/auto-attack'

# These instructions are needed for setting up DINO
mkdir ~/temp/ && cd ~/temp/ \
    && git clone https://github.com/IDEA-Research/DINO.git \
    && cd DINO/models/dino/ops \
    && python setup.py build install
cd ~/temp/ && git clone https://github.com/thomasbrandon/mish-cuda \
    && cd mish-cuda \
    && rm -rf build \
    && mv external/CUDAApplyUtils.cuh csrc/ \
    && python setup.py build install
```

### Prepare Part-ImageNet Dataset

```bash
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

# Download data from https://github.com/tacju/partimagenet manually or via gdown
pip install gdown
mkdir ~/data/PartImageNet
gdown https://drive.google.com/drive/folders/1_sKq9g34dsKwfsgLo6j7WCe8Nfceq0Zo -O ~/data/PartImageNet --folder

# Organize the files
cd PartImageNet
unzip train.zip 
unzip test.zip 
unzip val.zip
mkdir JPEGImages
mv train/* test/* val/* JPEGImages
rm -rf train test val train.zip test.zip val.zip
```

### Prepare Cityscapes Panoptic Parts Dataset

To download Cityscapes with `wget` on a server, see [link](https://github.com/cemsaz/city-scapes-script) or use the following script with your username and password from the [Cityscapes website](https://www.cityscapes-dataset.com/downloads/).

```bash
sh download_cityscapes.sh yourUsername yourPassword
```

- Get ground truth decoder from [panoptic_parts](https://github.com/pmeletis/panoptic_parts).
- For their API, see [link](https://panoptic-parts.readthedocs.io/en/stable/api_and_code.html).
- For id's meaning, see [link](https://github.com/pmeletis/panoptic_parts/tree/master/panoptic_parts/cityscapes_panoptic_parts/dataset_v2.0).

```[bash]
git clone https://github.com/pmeletis/panoptic_parts.git
mv panoptic_parts/panoptic_parts part-model/
rm -rf panoptic_parts
```

### Prepare PASCAL-Part Dataset

```bash
mkdir pascal_part && cd pascal_part
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
wget http://roozbehm.info/pascal-parts/trainval.tar.gz
tar -xvf VOCtrainval_03-May-2010.tar
tar -xvf trainval.tar.gz
mv VOCdevkit/VOC2010/* .
rm -rf VOCdevkit/ trainval.tar.gz VOCtrainval_03-May-2010.tar
```

---

## Usage

- Download and prepare the datasets according to the instructions above.
- Run scripts to prepare the datasets for our classification task: `sh scripts/prepare_data.sh`.
- Download weight files from this [link](https://zenodo.org/record/8342614).

### Model Naming

We lazily use `--experiment` argument to define the types of model and data to use.
The naming convention is explained below:

```
--experiment <SEGTYPE>-<MODELTYPE>[-<MODELARGS>][-no_bg][-semi]
```

- `SEGTYPE`: types of segmentation labels to use
  - `part`: object parts, e.g., dog's legs, cat's legs, dog's head, etc.
  - `object`: objects, e.g., dog, cat, etc.
  - `fg`: background or foreground.
- `MODELTYPE`: types of normal/part models to train/test
  - `wbbox`: Bounding-Box part-based model. Backbone architecture and segmenter architecture are defined by args like `--seg-backbone resnet50 --seg-arch deeplabv3plus`.
  - `pooling`: Downsampled part-based model.
  - `normal`: Normal model (defined by `--arch`, e.g., `--arch resnet50`).

TODO(chawins): Add more explanation

### Example

```bash
# Create dir for saving results, logs, and checkpoints
mkdir results
bash scripts/run_example.sh
```

Other tips and tricks

- Set visible GPUs to pytorch (e.g., `CUDA_VISIBLE_DEVICES=0,1,2,3`) if needed.
- Some parameters that might help if NCCL is crashing (maybe for old version of python/pytorch)

```[bash]
export TORCHELASTIC_MAX_RESTARTS=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
```

### Generalized Robustness

- Download datasets from `https://drive.google.com/file/d/1-CQ4MINuzJ88DJxkKMnY2vpzMLBphDkU/view?usp=sharing` for the shape-vs-texture benchmark
- `https://drive.google.com/file/d/1jnwDFWn03YMhsXhtjSt87JIO83VNZ4iJ/view?usp=sharing` for the spurious correlation benchmark.
- `https://drive.google.com/file/d/199axe3QKp-1n1h_rnczKeyaG_sz4g_VG/view?usp=sharing` for the common corruption benchmark.
- Set dataset flag to `--dataset part-imagenet-geirhos` to test on the shape-vs-texture benchmark.
- `--dataset part-imagenet-mixed` for the spurious correlation benchmark.
- `--dataset part-imagenet-corrupt` for the common corruption benchmark.

```bash
cd ~/data/PartImageNet
gdown https://drive.google.com/u/0/uc?id=1-CQ4MINuzJ88DJxkKMnY2vpzMLBphDkU&export=download
gdown https://drive.google.com/u/0/uc?id=1jnwDFWn03YMhsXhtjSt87JIO83VNZ4iJ&export=download
gdown https://drive.google.com/u/0/uc?id=199axe3QKp-1n1h_rnczKeyaG_sz4g_VG&export=download
tar -xvf corrupt.tar.gz
tar -xvf spurious.tar.gz
tar -xvf shapeVStexture.tar.gz
```

These datasets are pre-generated for the ease of testing. Note that the pregenerated datasets are based on the validation and test partition of Part-ImageNet dataset.

- Shape-vs-texture benchmark are generated according to the steps in [https://github.com/rgeirhos/Stylized-ImageNet#usage](https://github.com/rgeirhos/Stylized-ImageNet#usage). 
- Spurious correlation benchmark are generated from [ImageNet-9](https://github.com/MadryLab/backgrounds_challenge), specifically, filtering out ImageNet-9 to contain only classes belonging to Fish, Bird, Reptile, and Quadruped as those classes also exist in Part-ImageNet. Then, the foreground (Only-FG) of each class is matched to a random background (Only-BG-T) from a randomly chosen other class.
- Common corruption benchmark is generated by taking the original images in Part-ImageNet and generate the corrupted version with 5 severity levels based on [this library](https://github.com/bethgelab/imagecorruptions).

Validation and Test set generation in Shape-vs-Texture and Common Corruptions benchmarks are generated from Validation and Test set of Part-Imagenet. Validation and Test set generation in Spurious Correlation benchmark dataset is from foreground and background images not in the training set, then randomly assign each generated image to either test or validation set.

---

## TODOs

- Upload model weights on a permanent link.
- Improve `argparse` help message and documentation.
- Add result highlight and figures to `README`.

---

## Contributions

- For questions and bug reports, please feel free to create an issue or reach out to us directly.
- Contributions are absolutely welcome.
- Contact: `chawins@berkely.edu`, `kornrapatp@berkeley.edu`.

---

## Acknowledgement

We would like to thank Vikash Sehwag and Jacob Steinhardt for their feedback on the paper.
We also thank the anonymous ICLR reviewers for their helpful comments and suggestions.
This research was supported by the Hewlett Foundation through the Center for Long-Term Cybersecurity (CLTC), by the Berkeley Deep Drive project, and by generous gifts from Open Philanthropy and Google Cloud Research Credits program under Award GCP19980904.
Jacob Steinhardt also generously lent us the computing resources used in this research.

This software and/or data was deposited in the BAIR Open Research Commons repository on Feb 28, 2023.
