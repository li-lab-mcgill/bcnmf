# bcNMF: Background Contrastive Nonnegative Matrix Factorization

**bcNMF** extracts target-enriched latent components from high-dimensional data by jointly factorizing a target dataset and a matched background using shared nonnegative bases under a contrastive objective that suppresses shared variation.

> Yixuan Li, Archer Y. Yang, Yue Li
> *bcNMF: Background Contrastive Nonnegative Matrix Factorization Identifies Target-Specific Features in High-Dimensional Data*
> [arXiv:2602.22387](https://arxiv.org/html/2602.22387v1)

---

## Overview

Standard dimensionality reduction (PCA, NMF) is variance-driven and often highlights batch effects, cell-type composition, or other background signals that dominate the data. bcNMF addresses this by minimising:

$$\min_{W, H_X, H_Y \geq 0} \mathcal{L}(X \mid WH_X) - \alpha \mathcal{L}(Y \mid WH_Y)$$

where $X$ is the target, $Y$ is the background, $W$ is a shared nonnegative basis, and $\alpha$ controls background suppression strength. Supported likelihoods: **Gaussian (SSE)**, **Poisson**, **Negative Binomial**, **Zero-Inflated Negative Binomial**.

---

## Installation

```bash
git clone https://github.com/li-lab-mcgill/bcnmf.git
cd bcnmf
pip install -e .
```

---

## Quick Start

```python
import numpy as np
from bcNMF import contrastive_nmf_poisson

# X: (n_features, n_target_samples)  — target data (nonnegative)
# Y: (n_features, n_background_samples) — background data (nonnegative)
X = np.random.poisson(5, size=(500, 200)).astype(float)
Y = np.random.poisson(5, size=(500, 100)).astype(float)

W, H_X, H_Y, perf = contrastive_nmf_poisson(X, Y, K=10, alpha=1.0, niter=200)
# W   : (500, 10)  shared nonneg basis
# H_X : (10, 200)  target coefficients  ← use for downstream analysis
# H_Y : (10, 100)  background coefficients
```

For continuous/image data use `contrastive_nmf_sse` (squared-error loss). Mini-batch training is available via `contrastive_nmf_poisson_minibatch`.

---

## Repository Structure

```
bcNMF/
├── bcNMF/                  # Python package
│   ├── __init__.py
│   └── bcnmf.py            # Core multiplicative-update algorithms
├── experiments/
│   ├── simulation/         # Sec 2.2 — MNIST + ImageNet
│   ├── mice_protein/       # Sec 2.3 — Down syndrome protein expression
│   ├── leukemia/           # Sec 2.4 — Leukemia scRNA-seq (pre/post transplant)
│   ├── cancer_cell_lines/  # Sec 2.5 — MIX-seq idasanutlin / TP53
│   └── mdd/                # Sec 2.6 — MDD snRNA-seq (postmortem brain)
├── setup.py
├── requirements.txt
└── README.md
```

---

## Available Functions

| Function | Loss | Use case |
|---|---|---|
| `nmf_sse` | Squared error | Standard NMF baseline |
| `nmf_poisson` | Poisson | Standard NMF, count data |
| `nmf_poisson_minibatch` | Poisson | Large-scale standard NMF |
| `contrastive_nmf_sse` | Squared error | bcNMF for continuous / image data |
| `contrastive_nmf_poisson` | Poisson | bcNMF for scRNA-seq / count data |
| `contrastive_nmf_poisson_minibatch` | Poisson | bcNMF, large-scale |
| `contrastive_nmf_sse_multi` | Squared error | bcNMF for two-modality data |

---

## Experiments

Each subdirectory under `experiments/` contains a Jupyter notebook that reproduces the corresponding paper figure. Data files are not included; see the notebook headers for download instructions and expected directory layout.

---

## Citation

```bibtex
@article{li2025bcnmf,
  title  = {bcNMF: Background Contrastive Nonnegative Matrix Factorization
            Identifies Target-Specific Features in High-Dimensional Data},
  author = {Li, Yixuan and Yang, Archer Y. and Li, Yue},
  year   = {2025},
  eprint = {2602.22387},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2602.22387}
}
```
