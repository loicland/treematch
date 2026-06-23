# Tree counting with satellite imagery

[![python](https://img.shields.io/badge/-Python_3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](#license)

This repository contains the code and data for:

- **TinyTree** — a multi-sensor tree counting benchmark with point-level annotations across three geographic regions and satellite sensors.
- **TreeMatch** — a training method for tree density estimation that leverages optimal transport to learn from both clean and noisy annotations.

## TinyTree Dataset

TinyTree provides georeferenced satellite imagery with per-tree point annotations across three regions, sensors, and resolutions:

| Region | Sensor | GSD | Train (strong) | Train (weak) | Test | Total trees | Total area |
|--------|--------|-----|----------------|--------------|------|-------------|------------|
| Rwanda | PlanetScope | 3.0 m | 231 tiles / 309k trees | 73 tiles / 3.4M trees | 734 tiles / 237k trees | 3.9M | 283 km² |
| China | Gaofen-2 | 0.8 m | 446 tiles / 55k trees | 16,364 tiles / 7.7M trees | 2,565 tiles / 70k trees | 7.8M | 2,344 km² |
| France | SPOT-6 | 1.5 m | 492 tiles / 11k trees | — | 493 tiles / 11k trees | 22k | 0.7 km² |

Each tile is a 5-band GeoTIFF (4 spectral bands + 1 binary validity mask). Point annotations are stored in a single GeoPackage per split with a `tile` column linking each point to its image.

### Download

The dataset is hosted at: *[link TBA]*

```
tinytrees/
├── ps/          # PlanetScope (Rwanda)
│   ├── train/   # *.tif + points.gpkg
│   ├── test/
│   └── unlabeled/
├── gf/          # Gaofen-2 (China)
│   ├── train/
│   ├── test/
│   └── unlabeled/
└── spot/        # SPOT-6 (France)
    ├── train/
    └── test/
```

### Loading data

```python
from data.ps import PlanetScopeStrong
from data.gf import GaofenStrong
from data.spot import SPOTCountingDataset

# Each dataset returns (image, valid_mask, count_map)
# image: (C+1, H, W) float tensor — C normalized bands + 1 validity channel
# valid_mask: (1, H, W) binary tensor
# count_map: (1, H, W) sparse 0/1 tensor with tree locations

ds = PlanetScopeStrong(imsize=64, split="train", root="/path/to/tinytrees/ps")
ds = GaofenStrong(imsize=64, split="train", root="/path/to/tinytrees/gf")
ds = SPOTCountingDataset(imsize=64, split="train", root="/path/to/tinytrees/spot")
```

## TreeMatch

TreeMatch is an optimal-transport-based training method for tree density estimation that supports mixed supervision from clean (expert) and noisy (e.g. CHM-derived) point annotations. It uses unbalanced optimal transport to match predicted density maps to point annotations, with a slack mechanism to down-weight unreliable labels.

### Training

Training is configured via [Hydra](https://hydra.cc/). The main entry point is `train.py`:

```bash
# Train TreeMatch (UDM) on PlanetScope with clean labels only
python train.py dataset=ps model=udm train.clean_ratio=1.0 model.lr=8e-05

# Train with 80% clean + 20% noisy labels
python train.py dataset=ps model=udm train.clean_ratio=0.8 model.lr=8e-05

# Train density regression baseline
python train.py dataset=ps model=density_regressor

# Train DM-Count baseline
python train.py dataset=ps model=dm_count
```

Override dataset paths for your machine in `conf/local/local.yaml`:
```yaml
# @package _global_
dataset:
  root: ${dataset_roots.${hydra:runtime.choices.dataset}}
dataset_noisy:
  root: ${dataset_roots.${hydra:runtime.choices.dataset_noisy}}
dataset_roots:
  ps: /path/to/tinytrees/ps
  gf: /path/to/tinytrees/gf
  spot: /path/to/tinytrees/spot
  ps_noisy: /path/to/tinytrees/ps/unlabeled
  gf_noisy: /path/to/tinytrees/gf/unlabeled
```

### Available models

| Model | Config | Description |
|-------|--------|-------------|
| TreeMatch (UDM) | `model=udm` | Unbalanced OT density matching with slack for noisy labels |
| Density Regression | `model=density_regressor` | Gaussian density map regression (MSE loss) |
| DM-Count | `model=dm_count` | Balanced OT with total variation loss |
| CenterNet | `model=centernet` | Keypoint detection with heatmap regression |
| P2PNet | `model=p2p` | Point-to-point matching |

### Available backbones

| Backbone | Config | Architecture |
|----------|--------|-------------|
| ResNet-50 U-Net | `backbone=resnet50` | `segmentation_models_pytorch` U-Net with ResNet-50 encoder |
| Swin Transformer | `backbone=swint` | SwinV2-Small with FPN decoder |
| ViT | `backbone=vit` | Vision Transformer with FPN decoder |

## Rwanda-Tanzania application

We trained a PlanetScope+Sentinel-1+Sentinel-2 (20-band composite) model to count trees in Rwanda and Tanzania.

## Reference

```bibtex
@article{tinytree2025,
  title={TinyTree: A Multi-Sensor Benchmark for Tree Counting from Satellite Imagery},
  author={TBA},
  year={2025}
}
```

```bibtex
@article{treematch2025,
  title={TreeMatch: Density Estimation with Optimal Transport for Tree Counting},
  author={TBA},
  year={2025}
}
```
