# Part-Based Models Improve Adversarial Robustness

## Abstract

We show that combining human prior knowledge with end-to-end learning can improve the robustness of deep neural networks by introducing a part-based model for object classification.
We believe that the richer form of annotation helps guide neural networks to learn more robust features without requiring more samples or larger models.
Our model combines a part segmentation model with a tiny classifier and is trained end-to-end to simultaneously segment objects into parts and then classify the segmented object.
Empirically, our part-based models achieve both higher accuracy and higher adversarial robustness than a ResNet-50 baseline on all three datasets.
For instance, the clean accuracy of our part models is up to 15 percentage points higher than the baseline's, given the same level of robustness.
Our experiments indicate that these models also reduce texture bias and yield better robustness against common corruptions and spurious correlations.

## Setup

### Dependency Installation

- `python 3.8`
- [`timm`](https://github.com/rwightman/pytorch-image-models)
- [`segmentation-models-pytorch`](https://github.com/qubvel/segmentation_models.pytorch)
- `imagecorruptions` for testing with common corruptions
- `foolbox` for testing with black-box attacks
- `torchmetrics` for computing IoU
- `kornia` for custom transformations

```bash
# Install pytorch 1.10 (or the latest)
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -y h5py 
conda install -y -c conda-forge ruamel.yaml scikit-image timm torchmetrics
conda install -y -c anaconda cython
pip install -U kornia wandb segmentation-models-pytorch imagecorruptions foolbox
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

```[bash]
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

```[bash]
mkdir pascal_part && cd pascal_part
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
wget http://roozbehm.info/pascal-parts/trainval.tar.gz
tar -xvf VOCtrainval_03-May-2010.tar
tar -xvf trainval.tar.gz
mv VOCdevkit/VOC2010/* .
rm -rf VOCdevkit/ trainval.tar.gz VOCtrainval_03-May-2010.tar
```

## Usage

- Download and prepare the datasets according to the instructions above.
- Run scripts to prepare the datasets for our classification task: `sh scripts/prepare_data.sh`.
- Download weight files from [https://anonfiles.com/1aiaP056y6](https://anonfiles.com/1aiaP056y6).

### Experiment Naming

```[bash]
<SEGTYPE>-<MODELTYPE>[-<MODELARGS>][-nobg][-semi]
```

- `SEGTYPE`: types of segmentation labels to use
  - `part`: object parts, e.g., dog's legs, cat's legs, dog's head, etc.
  - `object`: objects, e.g., dog, cat, etc.
  - `fg`: background or foreground.

### Example

TODO(chawins): Add usage example

```[bash]
```

### Generalized Robustness

- Download datasets from `https://anonfiles.com/X9a0g2k7ye/shapeVStexture.tar_gz` for the shape-vs-texture benchmark
- `https://anonfiles.com/dfYef1k9yf/spurious.tar_gz` for the spurious correlation benchmark.
- `-` for the common corruption benchmark.
- Set dataset flag to `--dataset part-imagenet-geirhos` to test on the shape-vs-texture benchmark.
- `--dataset part-imagenet-mixed` for the spurious correlation benchmark.
- `--dataset part-imagenet-corrupt` for the common corruption benchmark.

These datasets are pre-generated for the ease of testing.

- Shape-vs-texture benchmark are generated according to the steps in [https://github.com/rgeirhos/Stylized-ImageNet#usage](https://github.com/rgeirhos/Stylized-ImageNet#usage). 
- Spurious correlation benchmark are generated from [ImageNet-9](https://github.com/MadryLab/backgrounds_challenge), specifically, filtering out ImageNet-9 to contain only classes belonging to Fish, Bird, Reptile, and Quadruped as those classes also exist in Part-ImageNet. Then, the foreground (Only-FG) of each class is matched to a random background (Only-BG-T) from a randomly chosen other class.
- Common corruption benchmark is generated by taking the original images in Part-ImageNet and generate the corrupted version with 5 severity levels based on [this library](https://github.com/bethgelab/imagecorruptions).
