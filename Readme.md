# Transformer-Based Automatic Rigging Evaluation

A systematic benchmark comparing two state-of-the-art transformer models for automatic 3D skeleton generation and skinning weight prediction: **RigAnything** and **MagicArticulate**. Both are included as Git submodules.

The evaluation targets two questions:
1. Which model produces more accurate skeletons and skinning weights overall?
2. How does each model generalize across object categories (humanoids, animals, inanimate objects, fantasy creatures, etc.)?

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Cloning with Submodules](#cloning-with-submodules)
  - [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [Running Evaluation](#running-evaluation)
  - [RigAnything](#riganything)
  - [MagicArticulate](#magicarticulate)
- [Metrics](#metrics)
- [Results](#results)
- [Project Phases](#project-phases)
- [References](#references)

---

## Overview

Rigging a 3D mesh (placing a skeleton inside it and painting skinning weights that control how each bone deforms the surface) is one of the most technically demanding steps in character animation. Traditionally it requires significant manual effort and anatomical expertise.

Recent transformer-based models can infer both skeleton topology and skinning weights automatically from mesh geometry alone. This project benchmarks two such models against ground-truth artist-created rigs from the **Articulation-XL 2.0** dataset, and deliberately probes zero-shot generalization by evaluating on object categories that were held out during training.

---

## Repository Structure

```
.
├── RigAnything/              # Submodule (autoregressive diffusion-based rigging model)
│   ├── data/ (NOT INCLUDED IN REPO DUE TO SIZE)
│   │   ├── articulation_xlv2_test.npz      # Packed test-set ground truth
│   │   └── meta_Articulation_XL_2.0.csv    # Metadata (uuid, category, vertex/joint counts, …)
│   ├── evaluate_riganything.py   # End-to-end evaluation script for RigAnything
|   └── results/                  # Per-sample CSV outputs and aggregate summaries
├── MagicArticulate/          # Submodule (autoregressive transformer rigging model)
│   ├── data/ (NOT INCLUDED IN REPO DUE TO SIZE)
│   │   ├── articulation_xlv2_test.npz      # Packed test-set ground truth
│   │   └── meta_Articulation_XL_2.0.csv    # Metadata (uuid, category, vertex/joint counts, …)
│   ├── skeleton_ckpt/ (NOT INCLUDED IN REPO DUE TO SIZE)
│   │   ├── checkpoint_trainonv2_hier.pth    # Weights for hierarchical-order model
│   │   └── checkpoint_trainonv2_spatial.pth # Weights for spatial-order model
│   ├── evaluate_magicarticulate.py          # End-to-end evaluation script for MagicArticulate
│   └── outputs/                             # Per-sample outputs and aggregate summaries
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (≥ 16 GB VRAM recommended)
- [Blender 3.x](https://www.blender.org/download/) with Python bindings (`bpy`) (required by RigAnything's mesh pipeline)

### Cloning with Submodules

Both models are included as Git submodules. Clone everything in one step:

```bash
git clone --recurse-submodules https://github.com/<your-org>/<this-repo>.git
cd <this-repo>
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### Environment Setup

Each submodule ships its own dependency list. We recommend separate conda environments to avoid conflicts.

**RigAnything:**
```bash
conda create -n riganything python=3.11
conda activate riganything
pip install -r RigAnything/requirements.txt
```

**MagicArticulate:**
```bash
conda create -n magicarticulate python=3.10 -y
conda activate magicarticulate
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r MagicArticulate/requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
```

Then download the Michelangelo encoder weights and MagicArticulate skeleton-generation checkpoint:
```bash
cd MagicArticulate && python download.py && cd ..
```

> **Note:** `evaluate_magicarticulate.py` must be run from inside the `MagicArticulate/` directory (or with `PYTHONPATH` set accordingly) so that the submodule's local imports resolve correctly.

---

## Dataset

Both evaluation scripts operate on the **Articulation-XL 2.0** dataset, which is a subset of Objaverse-XL containing ~59.4k expert-rigged 3D models.

You can download the dataset **Here**: https://huggingface.co/datasets/Seed3D/Articulation-XL2.0/tree/main.

The test split used here (`articulation_xlv2_test.npz`) is a single packed NumPy archive. Each sample stores:

| Field | Description |
|---|---|
| `vertices` | Mesh vertex positions `[V, 3]` |
| `faces` | Triangle face indices `[F, 3]` |
| `normals` | Per-vertex normals `[V, 3]` |
| `joints` | Ground-truth joint positions `[J, 3]` |
| `bones` | Joint adjacency matrix `[J, J]` |
| `root_index` | Index of the root joint |
| `joint_names` | Human-readable joint labels |
| `skinning_weights_value/row/col/shape` | Sparse COO skinning weights `[V, J]` |
| `pc_w_norm` | Pre-sampled 1024-pt surface cloud with normals `[1024, 6]` (optional) |

Category metadata (category label, vertex/joint/bone counts, source) is stored separately in `meta_Articulation_XL_2.0.csv`.

To inspect the exact keys in your local copy of the NPZ:

```python
from evaluate_riganything import inspect_dataset_npz
inspect_dataset_npz("data/articulation_xlv2_test.npz")
```

---

## Running Evaluation

### RigAnything

```bash
conda activate riganything
python evaluate_riganything.py \
    --config  RigAnything/config.yaml \
    --ckpt    RigAnything/ckpt/riganything_ckpt.pt \
    --dataset_npz  data/articulation_xlv2_test.npz \
    --metadata_csv data/meta_Articulation_XL_2.0.csv \
    --output_csv   results/riganything_results.csv
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--max_meshes N` | None | Stop after N successfully evaluated meshes (good for smoke-testing) |
| `--category NAME` | None | Restrict evaluation to a single category label |
| `--device` | `cuda:0` | CUDA device |
| `--amp_dtype` | `bf16` | Mixed precision dtype (`fp16`, `bf16`, `fp32`) |

### MagicArticulate

```bash
conda activate magicarticulate
cd MagicArticulate
python evaluate_magicarticulate.py \
    --dataset_path       data/articulation_xlv2_test.npz \
    --pretrained_weights skeleton_ckpt/checkpoint_trainonv2_hier.pth \
    --metadata_path      data/meta_Articulation_XL_2.0.csv \
    --output_dir         outputs \
    --save_name          magicarticulate_results \
    --input_pc_num       8192 \
    --hier_order
```

> Omit `--hier_order` if you are using the spatial-order checkpoint (`checkpoint_trainonv2_spatial.pth`).

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--max_samples N` | None | Stop after N samples (good for smoke-testing) |
| `--hier_order` | False | Use hierarchical sequence ordering; omit for spatial-order checkpoint |
| `--precision` | `fp16` | Mixed precision dtype (`fp16`, `fp32`) |
| `--metadata_path PATH` | None | CSV with `uuid,category_label` for per-category breakdown |
| `--use_medial_axis` | False | Snap predicted joints to medial axis after generation |
| `--input_pc_num N` | 8192 | Number of point-cloud samples fed to the encoder |
| `--n_max_bones N` | 100 | Hard cap on number of generated bones |

---

## Metrics

All metrics are computed against artist-created ground-truth rigs.

### Skeleton Quality

| Metric | Description |
|---|---|
| **J2J** (Joint-to-Joint) | Mean distance between predicted and ground-truth joints after optimal Hungarian matching. Lower is better. |
| **J2B** (Joint-to-Bone) | Mean distance from each predicted joint to its nearest ground-truth bone segment. Penalizes joints placed far from any plausible bone. Lower is better. |
| **B2B** (Bone-to-Bone) | Symmetric Chamfer distance between predicted and ground-truth bone midpoints. Lower is better. |

### Skinning Quality

Per-vertex hard bone assignments (argmax of skinning weights) are compared after remapping predicted joint indices to ground-truth indices via the J2J Hungarian matching. Metrics are macro-averaged across all ground-truth bone classes present in each mesh.

| Metric | Description |
|---|---|
| **IoU** | Intersection-over-Union per bone, macro-averaged |
| **Precision** | Fraction of vertices assigned to a bone that are correctly assigned |
| **Recall** | Fraction of ground-truth bone vertices that are correctly recovered |

---

## Results

Output CSVs contain one row per mesh with columns:

```
uuid, category_label, joint_count, bone_count, pred_joints,
j2j, j2b, b2b, iou, precision, recall
```

The evaluation scripts print aggregate and per-category summaries to stdout at the end of each run. Example:

```
===== Aggregate Results =====
  j2j         : 0.0842
  j2b         : 0.0531
  b2b         : 0.0614
  iou         : 0.6123
  precision   : 0.7041
  recall      : 0.6387

  Evaluated : 1850
  Skipped   : 147

===== Per-Category Results =====
                    j2j    j2b    b2b    iou  precision  recall
category_label
animal           0.0901  0.057  0.064  0.589      0.681   0.612
humanoid         0.0712  0.044  0.053  0.671      0.748   0.691
inanimate        0.1031  0.067  0.078  0.543      0.631   0.571
```

---

## Project Phases

This repository documents a four-phase study:

| Phase | Goal |
|---|---|
| **1 – Baseline Evaluation** | Run both models on the full test set and establish aggregate performance numbers |
| **2 – Model Selection** | Identify the stronger foundational architecture based on inference speed, skeletal quality, and codebase robustness |
| **3 – Category Analysis** | Break down results by object category and isolate which categories each model handles well or poorly |
| **4 – Hyperparameter Tuning** | Sweep learning rate, epoch count, weight decay, softmax temperature, max joints, and point-cloud sample size |

---

## References

1. Deitke, M. et al. (2023). *Objaverse-XL: A Universe of 10M+ 3D Objects.* NeurIPS 2023.
2. Blender Foundation. *Automatic Weights.* [docs.blender.org](https://docs.blender.org)
3. RigAnything. [GitHub Repository](https://github.com/RigAnything/RigAnything)
4. Articulation-XL 2.0 Dataset.
5. MagicArticulate. [GitHub Repository](https://github.com/MagicArticulate/MagicArticulate)
6. UniRig. [GitHub Repository](https://github.com/UniRig/UniRig)
