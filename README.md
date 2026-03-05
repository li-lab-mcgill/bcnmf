# bcNMF: Background Contrastive Nonnegative Matrix Factorization

**bcNMF** extracts target-enriched latent components from high-dimensional data by jointly factorizing a target dataset and a matched background using shared nonnegative bases under a contrastive objective that suppresses shared variation.

> Yixuan Li, Archer Y. Yang, Yue Li
> *bcNMF: Background Contrastive Nonnegative Matrix Factorization Identifies Target-Specific Features in High-Dimensional Data*
> [arXiv:2602.22387](https://arxiv.org/html/2602.22387v1)

---

## System Requirements

### Software dependencies

- **Python** >= 3.9
- **PyTorch** >= 2.0
- **numpy** >= 1.22
- **scipy**
- **scikit-learn**
- **tqdm**
- **umap-learn**
- **matplotlib**
- **scanpy** (required for scRNA-seq experiments)

All dependencies are listed in `requirements.txt`.

### Operating systems tested

- macOS 13+ (Ventura / Sonoma)
- Linux (Ubuntu 20.04+)
- Windows 10/11

### Hardware

No non-standard hardware is required. A CUDA-capable GPU is optional and will be used automatically if available; all functions fall back to CPU otherwise.

---

## Installation Guide

### Instructions

```bash
git clone https://github.com/li-lab-mcgill/bcnmf.git
cd bcnmf
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

### Typical install time

On a standard desktop computer with a normal internet connection, installation takes **< 5 minutes**.

---

## Demo

### Dataset

A self-contained demo is provided in the `demo/` directory. It reproduces the MNIST + natural image background experiment (Section 2.2 of the paper) using pre-generated data included in the repository:

```
demo/
├── simulation.ipynb              # Demo notebook
├── demo_data/
│   ├── target_images.csv         # 784 × 200  (MNIST digits 0/1 on flower backgrounds)
│   ├── background_images.csv     # 784 × 150  (flower patches only)
│   └── target_labels.csv         # Ground-truth digit labels (0 or 1)
└── results/                      # Output figures and matrices written here
```

### Instructions to run

```bash
cd demo
jupyter notebook simulation.ipynb
```

Run all cells in order. The notebook will:
1. Display example target images (digits on backgrounds) and background-only images
2. Fit standard NMF with K=2
3. Fit bcNMF with K=2 and α=1
4. Produce scatter plots of NMF vs. bcNMF factor scores coloured by digit label
5. Report ARI (Adjusted Rand Index) for each method

### Expected output

- Two scatter plots saved to `demo/results/nmf_vs_bcnmf.png`
- Factor matrices saved to `demo/results/H_nmf.csv`, `demo/results/H_bcnmf.csv`
- ARI summary saved to `demo/results/ari_summary.csv`
- bcNMF achieves substantially higher ARI than standard NMF, showing that the contrastive objective suppresses the background texture signal and recovers the digit structure.

### Expected run time

**< 2 minutes** on a standard desktop CPU (no GPU required).

---

## Instructions for Use

### Running bcNMF on your own data

```python
import numpy as np
from bcNMF import contrastive_nmf_poisson

# X: (n_features, n_target_samples)     — target data (nonnegative counts)
# Y: (n_features, n_background_samples) — background data (nonnegative counts)
X = np.random.poisson(5, size=(500, 200)).astype(float)
Y = np.random.poisson(5, size=(500, 100)).astype(float)

W, H_X, H_Y, perf = contrastive_nmf_poisson(X, Y, K=10, alpha=1.0, niter=200)
# W   : (n_features, K)  shared nonneg basis
# H_X : (K, n_target_samples)  target coefficients  ← use for downstream analysis
# H_Y : (K, n_background_samples)  background coefficients
```

For continuous/image data use `contrastive_nmf_sse` (squared-error loss). Mini-batch training is available via `contrastive_nmf_poisson_minibatch` for large datasets.

### Available functions

| Function | Loss | Use case |
|---|---|---|
| `nmf_sse` | Squared error | Standard NMF baseline |
| `nmf_poisson` | Poisson | Standard NMF, count data |
| `nmf_poisson_minibatch` | Poisson | Large-scale standard NMF |
| `contrastive_nmf_sse` | Squared error | bcNMF for continuous / image data |
| `contrastive_nmf_poisson` | Poisson | bcNMF for scRNA-seq / count data |
| `contrastive_nmf_poisson_minibatch` | Poisson | bcNMF, large-scale |
| `contrastive_nmf_sse_multi` | Squared error | bcNMF for two-modality data |

### Key parameters

| Parameter | Description | Default |
|---|---|---|
| `K` | Number of factors | required |
| `alpha` | Background suppression strength (higher = more contrastive) | `1.0` |
| `niter` | Number of multiplicative-update iterations | `200` |
| `tol` | Convergence tolerance | `1e-4` |

---

## Reproduction Instructions

Each experiment in the paper has a corresponding Jupyter notebook under `experiments/`. Pre-computed result matrices are provided in `result/` so figures can be reproduced without re-running the full fitting.

| Section | Notebook | Data |
|---|---|---|
| Sec 2.2 — Simulation (MNIST + ImageNet) | `experiments/simulation/mnist_imagenet.ipynb` | `dataset/simulation/n11939491/` (ImageNet grass images) |
| Sec 2.3 — Mice protein | `experiments/mice_protein/mice_protein.ipynb` | `dataset/mice_protein/Data_Cortex_Nuclear.csv` |
| Sec 2.4 — Leukemia scRNA-seq | `experiments/leukemia/leukemia.ipynb` | `dataset/leukemia/` (see note below) |
| Sec 2.5 — Cancer cell lines | `experiments/cancer_cell_lines/mcfarland.ipynb` | `dataset/cancer_cell_lines/` (see note below) |
| Sec 2.6 — MDD snRNA-seq | `experiments/mdd/mdd_final.ipynb` | `dataset/mdd/` (see note below) |

**Note on large datasets:** The following raw data files exceed GitHub's file size limit and are not included in this repository. Each notebook contains instructions at the top for downloading the data.

- `dataset/leukemia/adata_X_hvg_3000.h5ad` (183 MB) — generated by running the preprocessing cells in `leukemia.ipynb` on the raw 10x Genomics BMMC data
- `dataset/cancer_cell_lines/**/matrix.mtx` (~105 MB each) — downloaded automatically by running Cell 1 in `mcfarland.ipynb`
- MDD full dataset (`mdd_full.h5ad`, ~9 GB) — available from the original data source; see `mdd_final.ipynb` for details

---

## Repository Structure

```
bcNMF/
├── bcNMF/                  # Python package
│   ├── __init__.py
│   └── bcnmf.py            # Core multiplicative-update algorithms
├── demo/                   # Self-contained demo (runs without downloading data)
│   ├── simulation.ipynb
│   ├── demo_data/
│   └── results/
├── experiments/
│   ├── simulation/         # Sec 2.2 — MNIST + ImageNet
│   ├── mice_protein/       # Sec 2.3 — Down syndrome protein expression
│   ├── leukemia/           # Sec 2.4 — Leukemia scRNA-seq (pre/post transplant)
│   ├── cancer_cell_lines/  # Sec 2.5 — MIX-seq idasanutlin / TP53
│   └── mdd/                # Sec 2.6 — MDD snRNA-seq (postmortem brain)
├── dataset/                # Data files (large files excluded, see .gitignore)
├── result/                 # Pre-computed result matrices (.npy)
├── LICENSE
├── setup.py
├── requirements.txt
└── README.md
```

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

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
