"""Small helpers used by the paper-result notebooks.

Matrices passed to the factorization routines follow the package convention:
features by cells.  AnnData stores cells by features, so the conversion is
kept here rather than reimplemented in each notebook.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from .bcnmf import contrastive_nmf_poisson, contrastive_nmf_sse


def _dense_float32(matrix) -> np.ndarray:
    """Return a C-contiguous float32 matrix without changing its values."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))


def load_h5ad_counts(path: str | Path) -> ad.AnnData:
    """Load a stored paper input and ensure that its count matrix is numeric."""
    adata = ad.read_h5ad(path)
    adata.X = _dense_float32(adata.X)
    return adata


def as_features_by_cells(adata: ad.AnnData) -> np.ndarray:
    """Convert an AnnData count matrix from cells x genes to genes x cells."""
    return np.ascontiguousarray(_dense_float32(adata.X).T)


def ari_from_target_coefficients(H_x: np.ndarray, labels: np.ndarray, n_clusters: int, seed: int = 42) -> float:
    """Cluster target coefficients and calculate ARI against withheld labels."""
    predicted = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(H_x.T)
    return float(adjusted_rand_score(np.asarray(labels), predicted))


def run_bcnmf(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    k: int,
    alpha: float,
    likelihood: str = "poisson",
    n_iter: int = 100,
    seed: int = 0,
    n_starts: int = 1,
    damping: float = 0.5,
):
    """Fit the recorded bcNMF model and return W, target H, background H, and trace."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    likelihood = likelihood.lower()
    if likelihood == "poisson":
        return contrastive_nmf_poisson(
            X, Y, k, alpha, niter=n_iter, seed=seed, n_starts=n_starts, damping=damping, verbose=False
        )
    if likelihood in {"sse", "gaussian"}:
        return contrastive_nmf_sse(X, Y, k, alpha, niter=n_iter, seed=seed, verbose=False)
    raise ValueError("likelihood must be 'poisson' or 'sse'")
