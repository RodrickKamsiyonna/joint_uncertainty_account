# 🔁 Joint Uncertainty Accounting for JEPA

> A heuristics-free self-supervised representation learning framework — no EMA, no explicit regularization, no negative samples.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 📄 Overview

This repository implements two JEPA-inspired self-supervised objectives for visual representation learning, both grounded in a single, shared online encoder without any momentum (EMA) teachers.

The central question: *how should a predictor account for the irreducible uncertainty between two stochastically augmented views of the same image?*

Two answers are explored:

| Method | Uncertainty Mechanism | Regularization |
|---|---|---|
| **VAE-Style** | Explicit KL-regularized latent bottleneck (latent actions) | Requires β-KL penalty |
| **Flow-Matching** | Continuous noise transport via a velocity field | **None — collapse-free by design** |

The flow-matching variant is of particular theoretical interest: by transporting from an isotropic Gaussian base distribution, the objective *implicitly* regularizes learned embeddings toward the same isotropic optimum that SIGReg enforces explicitly — with no added penalty term.

---

## 🏗️ Architecture

```
Image x
  ├── Augmentation t₁  →  view v₁  →  Encoder fθ  →  h₁
  └── Augmentation t₂  →  view v₂  →  Encoder fθ  →  h₂ (stop-grad)
                                            │
                              Predictor (VAE or Flow)
                                            │
                                       Prediction ĥ₂
                                            │
                                          Loss L
```

- **Backbone**: ResNet-18 (with optional small-image adaptation for CIFAR)
- **VAE Predictor**: Inverse Dynamics Model (IDM) → reparameterized latent `z` → forward predictor
- **Flow Predictor**: Lightweight velocity network conditioned on sinusoidal time embeddings and a projected context `c₁ = π(h₁)`

---

## 📦 Installation

```bash
git clone https://github.com/RodrickKamsiyonna/joint_uncertainty_account.git
cd joint_uncertainty_account

pip install torch torchvision scikit-learn tqdm wandb
```

> Python 3.8+ and PyTorch 2.0+ are recommended.

---

## 🗂️ Dataset Format

The training script expects datasets in **ImageFolder** format:

```
data/
  train/
    class_a/
      img1.jpg
      img2.jpg
    class_b/
      ...
  val/
    class_a/
      ...
    class_b/
      ...
```

CIFAR-10 and Imagenette can be downloaded and restructured into this format using standard `torchvision` utilities.

---

## 🚀 Training

### Flow-Matching (recommended — no explicit regularization)

```bash
python train.py \
  --train_dir ./data/train \
  --val_dir   ./data/val \
  --loss      flow \
  --epochs    200 \
  --batch_size 512 \
  --lr        5e-4 \
  --img_size  32 \
  --small_conv \
  --project_name my-ssl-run
```

### VAE-Style

```bash
python train.py \
  --train_dir ./data/train \
  --val_dir   ./data/val \
  --loss      vae \
  --epochs    200 \
  --batch_size 512 \
  --lr        5e-4 \
  --img_size  32 \
  --small_conv \
  --project_name my-ssl-run
```

---

## ⚙️ Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--train_dir` | *(required)* | Path to training data (ImageFolder) |
| `--val_dir` | `None` | Validation data path. Falls back to train set if not provided. |
| `--loss` | `vae` | Predictive objective: `vae` or `flow` |
| `--epochs` | `200` | Number of training epochs |
| `--batch_size` | `512` | Batch size |
| `--lr` | `5e-4` | Learning rate (AdamW) |
| `--embedding_dim` | `512` | Feature embedding dimension |
| `--img_size` | `32` | Input resolution |
| `--small_conv` | `False` | Use 3×3 conv1 for small images (CIFAR-10 etc.) |
| `--eval_every` | `10` | Run k-NN + linear probe every N epochs |
| `--knn_k` | `20` | k for k-NN evaluation |
| `--linear_probe_epochs` | `100` | Epochs for linear probe training |
| `--wandb_key` | `None` | WandB API key for logging |
| `--project_name` | `ssl-generic` | WandB project name |

---

## 📊 Evaluation

Evaluation runs automatically every `--eval_every` epochs (and at the final epoch). Two protocols are used on **frozen** backbone features:

- **k-NN** (k=20): ℓ₂-normalized features, no training.
- **Linear Probe**: Single linear layer trained for `--linear_probe_epochs` epochs via AdamW.

### Preliminary Results

| Dataset | Method | k-NN Acc (%) | Linear Acc (%) |
|---|---|---|---|
| CIFAR-10 | VAE-Style | 76.43 | 77.64 |
| CIFAR-10 | Flow-Matching | 75.51 | **78.48** |
| Imagenette | VAE-Style | 76.76 | 78.62 |
| Imagenette | Flow-Matching | 74.04 | **79.49** |

The flow-matching approach achieves higher linear probing accuracy with a considerably lighter predictor.

---

## 🔬 Theoretical Highlights

**Why does flow-matching avoid collapse without regularization?**

In an Energy-Based Model framework, contrastive methods like InfoNCE approximate the intractable partition function by sampling discrete negatives. Flow-matching instead learns a velocity field over a continuous Gaussian base distribution — providing an *infinite* set of implicit negatives and bounding the partition function dynamically.

**Connection to optimal JEPA representations (SIGReg):**

Recent theory (Balestriero & LeCun, 2026) shows that the integrated squared bias of a downstream probe scales with the Fisher Information of the embedding distribution, and that the unique minimizer under variance constraints is the isotropic Gaussian N(0, I). Flow-matching — by design — transports features back to this base distribution, achieving the same theoretical optimum as SIGReg *without* an explicit distributional penalty.

---

## 🛣️ Roadmap

- [ ] Scale to ImageNet-1K (pending compute)
- [ ] Tighter theoretical characterization of EBM-diffusion link
- [ ] Hierarchical / multi-scale predictive objectives

---

## 📬 Contact & Collaboration

This is a work-in-progress circulated to solicit technical feedback. If you're interested in collaborating — especially on scaling strategies or theoretical positioning — please feel free to open an issue or reach out directly.

**Kamsiyonna Rodrick**

---

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@techreport{rodrick2026jepa,
  title     = {A JEPA-Inspired Self-Supervised Representation Learner with VAE-Style and Flow-Matching Objectives},
  author    = {Rodrick, Kamsiyonna},
  year      = {2026},
  note      = {Progress Report / Preprint}
}
```
