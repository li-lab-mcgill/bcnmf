"""Generate a labelled beta-series count dataset used by ``simulation.ipynb``.

The model has five shared background programs and two target-specific programs.
Increasing beta multiplies the background contribution in both the target and
background matrices. The output format matches the supplied ``beta_*`` files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_programs(rng, n_genes=3000, n_background=5, n_target=2):
    shared = np.zeros((n_genes, n_background), dtype=np.float32)
    target = np.zeros((n_genes, n_target), dtype=np.float32)
    cursor = 0
    for k in range(n_background):
        genes = np.arange(cursor, cursor + 250)
        cursor += 250
        shared[genes, k] = rng.gamma(2.0, 0.6, size=genes.size)
    for k in range(n_target):
        genes = np.arange(cursor, cursor + 120)
        cursor += 120
        target[genes, k] = rng.gamma(2.0, 0.9, size=genes.size)
    shared += rng.gamma(0.5, 0.01, size=shared.shape).astype(np.float32)
    target += rng.gamma(0.5, 0.005, size=target.shape).astype(np.float32)
    return shared, target


def generate(beta: float, seed: int):
    rng = np.random.default_rng(seed)
    n_genes, n_cells, n_background, n_target = 3000, 1000, 5, 2
    shared, target = make_programs(rng, n_genes, n_background, n_target)
    labels = np.repeat(np.arange(n_target), n_cells // n_target)
    rng.shuffle(labels)
    target_classes = rng.integers(0, n_background, n_cells)
    background_classes = rng.integers(0, n_background, n_cells)
    h_target = np.eye(n_target, dtype=np.float32)[:, labels]
    h_shared_x = np.eye(n_background, dtype=np.float32)[:, target_classes]
    h_shared_y = np.eye(n_background, dtype=np.float32)[:, background_classes]
    lib_x = rng.lognormal(0.0, 0.35, n_cells).astype(np.float32)
    lib_y = rng.lognormal(0.0, 0.35, n_cells).astype(np.float32)
    noise_gene = rng.gamma(1.0, 1.0, size=(n_genes, 1)).astype(np.float32)
    noise_x = rng.gamma(1.0, 1.0, size=(1, n_cells)).astype(np.float32)
    noise_y = rng.gamma(1.0, 1.0, size=(1, n_cells)).astype(np.float32)
    mu_x = 0.15 * (target @ h_target) + beta * 10.0 * (shared @ h_shared_x) + 0.02 + 0.01 * (noise_gene @ noise_x)
    mu_y = beta * 10.0 * (shared @ h_shared_y) + 0.02 + 0.01 * (noise_gene @ noise_y)
    return {
        "X": rng.poisson(mu_x * lib_x[None, :]).astype(np.float32),
        "Y": rng.poisson(mu_y * lib_y[None, :]).astype(np.float32),
        "labels": labels.astype(np.int64),
        "target_cell_types": target_classes.astype(np.int64),
        "background_cell_types": background_classes.astype(np.int64),
        "W_shared_true": shared,
        "W_target_true": target,
        "beta": np.asarray(beta),
        "seed": np.asarray(seed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"beta_{args.beta:g}_seed_{args.seed}.npz"
    np.savez_compressed(output, **generate(args.beta, args.seed))
    print(output)


if __name__ == "__main__":
    main()
