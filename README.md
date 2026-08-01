# bcNMF: Background Contrastive Nonnegative Matrix Factorization

bcNMF identifies target-enriched nonnegative programs by jointly factorizing a target dataset and a matched background dataset. It is designed for settings in which the target contains both shared background variation and variation of scientific interest. The contrastive parameter `alpha` downweights programs that are also active in the background.

This release contains the installable Python package, compact processed inputs, reproducible notebooks for the manuscript analyses, a self-contained count-data simulation, and the numerical source data underlying the reported figures and tables.

## Installation

bcNMF requires Python 3.9 or later. A CUDA-capable GPU is used automatically when available; CPU execution is also supported.

```bash
git clone https://github.com/li-lab-mcgill/bcnmf.git
cd bcnmf
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Using bcNMF

bcNMF takes a target matrix `X` and background matrix `Y`, both with shape **features x cells**. The feature order must be identical in `X` and `Y`.

For raw nonnegative counts, use the Poisson likelihood:

```python
import numpy as np
from sklearn.cluster import KMeans

from bcnmf import run_bcnmf

# X: genes x target cells; Y: the same genes x background cells.
X = np.load("target_counts.npy")
Y = np.load("background_counts.npy")

W, H_target, H_background, trace = run_bcnmf(
    X,
    Y,
    k=10,
    alpha=2.0,
    likelihood="poisson",
    n_iter=200,
    seed=0,
    n_starts=3,
    damping=0.5,
)

# H_target has shape K x target cells and is used for downstream analyses.
clusters = KMeans(n_clusters=2, random_state=42, n_init=20).fit_predict(H_target.T)
```

`W` contains nonnegative feature loadings, and `H_target` and `H_background` contain topic usage in target and background cells. Labels are not used while fitting; they may be used afterwards for evaluation or for associating topics with an external phenotype.

For log-normalized nonnegative data, set `likelihood="sse"`. The number of topics `k` and contrastive strength `alpha` should be selected jointly. The manuscript recommends held-out reconstruction and topic stability as label-free selection diagnostics.

### Main functions

| Function | Objective | Typical use |
|---|---|---|
| `contrastive_nmf_poisson` | Poisson contrastive objective | Raw count data, including scRNA-seq counts |
| `contrastive_nmf_sse` | Squared-error contrastive objective | Log-normalized, continuous, or image data |
| `contrastive_nmf_sse_combined_basis_reg` | Combined shared and target-specific squared-error model | The MNIST two-digit analysis |
| `nmf_poisson`, `nmf_sse` | Standard non-contrastive NMF | Baseline analyses |
| `run_bcnmf` | Wrapper for Poisson or squared-error bcNMF | Recommended entry point for new analyses |
| `ari_from_target_coefficients` | K-means and adjusted Rand index | Optional post-fit evaluation when labels are available |

The lower-level fitting functions return `W`, `H_target`, `H_background`, and an optimization trace. `run_bcnmf` provides a single interface for the Poisson and squared-error implementations.

## Reproducing the manuscript analyses

Start Jupyter from the repository root so that the notebooks can find the released inputs:

```bash
jupyter lab notebooks/
```

Each notebook loads its stored processed input, fits bcNMF, and calculates the reported adjusted Rand index (ARI) endpoint. The released matrices and all fitting functions use the features-by-cells convention.

| Notebook | Input | Target/background construction | Endpoint |
|---|---|---|---|
| `notebooks/mdd.ipynb` | 1,147 nuclei x 3,000 genes | MDD and control nuclei in the target matrix; control nuclei form the background matrix | MDD/control ARI |
| `notebooks/mnist.ipynb` | Blended MNIST K=2 and K=16 matrices | Saved target and background pixel matrices | Digit ARI |
| `notebooks/mouse_ds.ipynb` | Mice Protein Expression data | S/C saline control and Ts65Dn target; C/S saline control background | DS-status ARI |
| `notebooks/leukemia.ipynb` | Saved 3,000-HVG scRNA-seq matrices | AML027 pre/post-transplant target; healthy donor background | Transplant-status ARI |
| `notebooks/mcfarland.ipynb` | Stored 10,000-HVG MIX-seq matrix | Idasanutlin target; DMSO background | TP53-status ARI |
| `demo/simulation.ipynb` | Generated 3,000-gene count matrices | Two target subtypes with five background programs | Target-subtype ARI |


## Self-contained simulation

`demo/data/` contains labelled count matrices generated at beta = 0, 1, 2, 4, 8, and 16. Each `.npz` file contains `X` and `Y` (genes x cells), target subtype labels, background classes, true programs, beta, and seed. Open `demo/simulation.ipynb` from the repository root to fit the beta = 8 data and inspect the beta-series benchmark without downloading external data.

## Data and source data

The processed matrices in `data/` are the released paper-analysis inputs and contain no direct identifiers. Please cite the original MNIST, Mouse Protein Expression, leukemia, McFarland MIX-seq, and MDD studies when using their source data. Source Data are provided with this paper in `Source_Data.xlsx`, with a separate worksheet for the numerical values underlying each reported display item.

Two processed inputs exceed GitHub's 100 MB per-file limit and are therefore distributed through the project data archive rather than this repository: `data/leukemia/target_prepost_3000hvg.h5ad` and `data/mcfarland/mcfarland_preprocessed.h5ad`. Download each file from the archive and place it at the stated path before running the corresponding notebook. The archive link and DOI will be added here upon deposition.

`docs/DATA_CODE_AVAILABILITY.md` contains manuscript-ready Data Availability and Code Availability statements. 

## Citation

```bibtex
@article{li2026bcnmf,
  title   = {bcNMF: Background Contrastive Nonnegative Matrix Factorization Identifies Target-Specific Features in High-Dimensional Data},
  author  = {Li, Yixuan and Yang, Archer Y. and Li, Yue},
  year    = {2026},
  eprint  = {2602.22387},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2602.22387}
}
```

## License

The code is distributed under the MIT License. Dataset redistribution remains subject to the terms of the original data providers.
