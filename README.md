# WikiArt CRNN-Mamba

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/EfficientNet-B0-brightgreen" />
  <img src="https://img.shields.io/badge/Mamba_SSM-optional-yellow" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
</p>

<p align="center">
  Multi-task artwork classification and hyperbolic retrieval on the WikiArt dataset.<br/>
  <strong>EfficientNet-B0 → Bi-GRU → Mamba SSM → Hierarchical Heads → Poincaré Retrieval</strong>
</p>

---

## Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Installation](#-installation)
- [Dataset & CSV Format](#-dataset--csv-format)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Model Outputs](#-model-outputs)
- [Loss Function](#-loss-function)
- [Project Structure](#-project-structure)

---

## 🔍 Overview

WikiArt CRNN-Mamba is a multi-task deep learning pipeline that simultaneously classifies artworks at three hierarchical levels — **style** (27 classes), **meta-genre** (9 classes), and **artist** (1114 classes) — while producing **Poincaré-ball embeddings** for artwork retrieval.

It combines:
- A CNN backbone for spatial visual features
- A bidirectional GRU for local sequential patterns (brushstrokes, textures)
- A Mamba state-space model (or Multi-Head Attention fallback) for global composition context
- Adversarial gradient reversal for style-invariant representations
- Hyperbolic geometry for hierarchical retrieval

---

## Architecture

```
Input Image (224×224)
        │
  ArtBackbone               EfficientNet-B0  →  (B, 1280, H', W')
        │
  SequentialArtEncoder
    ├─ Linear Projection     1280 → 256
    ├─ Bi-GRU                local brushstroke patterns
    └─ Mamba SSM             global composition context
       (fallback: Multi-Head Self-Attention if mamba-ssm unavailable)
        │
        z  (B, 256)  ──────────────────────────────┐
        │                                           │
  HierarchicalHead                          StyleAdversary
    Style → Genre → Artist                  (gradient reversal)
        │
  HyperbolicProjection
    Poincaré-ball embedding for retrieval
```

### Components

| Module | File | Description |
|---|---|---|
| `ArtBackbone` | `models/backbone.py` | EfficientNet-B0 spatial feature extractor |
| `SequentialArtEncoder` | `models/encoder.py` | Linear proj + Bi-GRU + Mamba/MHA |
| `HierarchicalHead` | `models/heads.py` | Coarse-to-fine Style → Genre → Artist heads |
| `StyleAdversary` | `models/adversary.py` | Gradient reversal debiasing |
| `HyperbolicProjection` | `models/hyperbolic.py` | Poincaré-ball projection via `geoopt` |
| `WikiArtCRNNMamba` | `models/wikiart_model.py` | Full assembled model |

---

## Features

- **Hierarchical multi-task classification** — Style, meta-genre, and artist heads share a single backbone; each head conditions on the coarser level above it
- **Mamba SSM encoder** — State-space global context with automatic fallback to Multi-Head Attention on CPU / non-CUDA environments
- **Uncertainty-weighted loss** — Learnable per-task log-variance parameters (Kendall et al.) automatically balance the three classification losses
- **Supervised contrastive learning** — SupCon loss tightens intra-style clusters in embedding space
- **Style adversary** — Gradient reversal forces style-invariant features for less biased genre/artist predictions
- **Hyperbolic retrieval** — Poincaré-ball embeddings natively encode the art hierarchy for nearest-neighbour search
- **Weighted random sampling** — Inverse-frequency class weights handle severe style class imbalance

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/wikiart-crnn-mamba.git
cd wikiart-crnn-mamba
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Mamba SSM — GPU only

```bash
pip install mamba-ssm
```

> If `mamba-ssm` is not installed, the encoder automatically falls back to Multi-Head Self-Attention. You'll see this warning at startup — it's expected:
> ```
> [WARN] mamba-ssm not found — falling back to multi-head attention
> ```

---

## Dataset & CSV Format

The dataset is driven by a CSV file. Required columns:

| Column | Description | Example |
|---|---|---|
| `filename` | Relative path from data root | `Impressionism/monet/img.jpg` |
| `subset` | Split label | `train` or `val` |
| `genre` | Genre string or list-string | `"['portrait']"` |
| `artist` | Artist name | `claude_monet` |

> **Style labels are inferred from the folder prefix** of `filename`.  
> `Impressionism/monet/img.jpg` → style `Impressionism`

### Style → Meta-Genre Mapping

The 27 fine-grained styles are automatically clustered into 9 meta-genres:

| Meta-Genre | Styles |
|---|---|
| `Renaissance` | Early, High, Northern, Mannerism Late Renaissance |
| `Baroque_Rococo` | Baroque, Rococo |
| `Romantic_Realist` | Romanticism, Realism, Contemporary Realism |
| `Impressionist` | Impressionism, Post-Impressionism, Pointillism |
| `Expressionist` | Expressionism, Abstract Expressionism, Action Painting, Color Field |
| `Cubist` | Cubism, Analytical Cubism, Synthetic Cubism |
| `Modern_AvantGarde` | Art Nouveau, Fauvism, Symbolism, Naive Art Primitivism |
| `Contemporary` | Minimalism, New Realism, Pop Art |
| `NonWestern` | Ukiyo-e |

---

## Usage

### Smoke test (no data required)

Verifies model assembly and forward-pass shapes:

```bash
python smoke_test.py
```

### Full training

Edit `DATA_ROOT` and `CSV_PATH` at the top of `train.py`, then:

```bash
python train.py
```

### Inference / encoding only

```python
from wikiart_crnn import WikiArtCRNNMamba
import torch

model = WikiArtCRNNMamba(pretrained=True).eval()
img = torch.randn(1, 3, 224, 224)

# Full forward pass — returns all logits + embeddings
outputs = model(img)
# keys: embedding, hyp_embedding, style, genre, artist, adversarial

# Hyperbolic embedding only (no_grad)
hyp_emb = model.encode(img)  # (1, 256) on Poincaré ball
```

### Retrieval evaluation

```python
from wikiart_crnn.retrieval.evaluate import compute_map

# embeddings: (N, 256) tensor  |  labels: (N,) tensor
map_at_10 = compute_map(embeddings, labels, k=10)
```

---

## Configuration

All global constants live in `wikiart_crnn/config.py`:

| Constant | Default | Description |
|---|---|---|
| `NUM_STYLES` | `27` | Number of fine-grained art styles |
| `NUM_GENRES` | `9` | Number of meta-genre clusters |
| `NUM_ARTISTS` | `1114` | Number of artist classes |
| `EMBED_DIM` | `256` | Shared embedding dimensionality |
| `USE_MAMBA` | auto | `True` if `mamba-ssm` is importable |
| `DEVICE` | auto | `cuda` if available, else `cpu` |

---

## Model Outputs

| Key | Shape | Description |
|---|---|---|
| `embedding` | `(B, 256)` | Euclidean embedding `z` |
| `hyp_embedding` | `(B, 256)` | Poincaré-ball embedding for retrieval |
| `style` | `(B, 27)` | Style classification logits |
| `genre` | `(B, 9)` | Meta-genre classification logits |
| `artist` | `(B, 1114)` | Artist classification logits |
| `adversarial` | `(B, 27)` | Style-adversary logits (loss only) |

---

## Loss Function

```
L = uncertainty_weighted(L_style, L_genre, L_artist)
  + 0.1  × L_contrastive   (Supervised Contrastive / SupCon)
  + 0.05 × L_adversarial   (style-confusion penalty)
```

| Loss | File | Description |
|---|---|---|
| Uncertainty-weighted multi-task | `losses/multitask.py` | Kendall et al. homoscedastic weighting; task weights are learnable |
| Supervised Contrastive (SupCon) | `losses/contrastive.py` | Pulls same-style embeddings together |
| Adversarial | `losses/classification.py` | Penalises style-predictability in `z` via gradient reversal |

---

## Project Structure

```
wikiart_crnn/
├── train.py                          Main training entry point
├── smoke_test.py                     Quick shape / forward-pass check
├── requirements.txt
├── setup.py
└── wikiart_crnn/
    ├── config.py                     Global constants + Mamba toggle
    ├── models/
    │   ├── backbone.py               EfficientNet-B0 feature extractor
    │   ├── encoder.py                Bi-GRU + Mamba/MHA sequential encoder
    │   ├── heads.py                  Hierarchical Style → Genre → Artist heads
    │   ├── adversary.py              Gradient reversal + style adversary
    │   ├── hyperbolic.py             Poincaré-ball projection
    │   └── wikiart_model.py          Full assembled model
    ├── losses/
    │   ├── classification.py         Bias-aware cross-entropy
    │   ├── contrastive.py            Supervised contrastive loss (SupCon)
    │   └── multitask.py              Uncertainty-weighted multi-task loss
    ├── data/
    │   ├── dataset.py                WikiArtDataset + DataLoader factory
    │   ├── transforms.py             Train / val augmentation pipelines
    │   └── samplers.py               Weighted random sampler
    ├── training/
    │   └── trainer.py                train_one_epoch + validate loop
    └── retrieval/
        ├── hyperbolic_retrieval.py   Poincaré nearest-neighbour search
        └── evaluate.py               mAP@k evaluation
```
Made with ❤️ by NiceGuy
