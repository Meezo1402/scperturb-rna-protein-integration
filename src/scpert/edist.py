"""Energy distance between perturbations and control (Peidli et al., Nat Methods 2024).

The E-distance between two groups of cells X and Y in PCA space is

    E(X, Y) = 2 * mean d(x, y)  -  mean d(x, x')  -  mean d(y, y')

i.e. twice the mean between-group distance minus the within-group spreads. It is zero
when the two distributions coincide and grows with the shift between them, which makes
it a distribution-level effect size for a perturbation: "how far did this knockout move
the cells, relative to how spread out the cells already are". A permutation test
(the E-test) asks whether the observed E-distance exceeds what random relabelling of
the same cells produces.

Implementation notes
- Within-group means exclude the diagonal (an unbiased estimate of the expected pairwise
  distance); scPerturb's reference implementation includes the zero diagonal, which
  biases the within term down slightly. The difference is small at n = 300.
- Groups are subsampled to `max_cells` cells so the permutation test costs the same for
  a 100-cell and a 20,000-cell perturbation, and the pooled distance matrix is computed
  once per perturbation and merely re-indexed for every permutation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import multipletests

log = logging.getLogger("scpert")


def _offdiag_mean(D: np.ndarray) -> float:
    n = D.shape[0]
    if n < 2:
        return float("nan")
    return float((D.sum() - np.trace(D)) / (n * (n - 1)))


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """E-distance between two point clouds (rows = observations)."""
    dxy = cdist(X, Y).mean()
    dxx = _offdiag_mean(cdist(X, X))
    dyy = _offdiag_mean(cdist(Y, Y))
    return float(2 * dxy - dxx - dyy)


def _edist_from_pooled(D: np.ndarray, idx_x: np.ndarray, idx_y: np.ndarray) -> float:
    dxy = D[np.ix_(idx_x, idx_y)].mean()
    dxx = _offdiag_mean(D[np.ix_(idx_x, idx_x)])
    dyy = _offdiag_mean(D[np.ix_(idx_y, idx_y)])
    return float(2 * dxy - dxx - dyy)


def edistance_to_control(emb: np.ndarray, labels: pd.Series, control: str, max_cells: int | None = 300,
                         n_perm: int = 1000, seed: int = 0) -> pd.DataFrame:
    """E-distance of every perturbation to the control group with a permutation E-test.

    emb     : cells x dims embedding (e.g. the first 50 PCs)
    labels  : perturbation label per cell (same order as emb)
    Returns a DataFrame sorted by E-distance, with BH-adjusted p-values.
    """
    rng = np.random.default_rng(seed)
    labels = pd.Series(np.asarray(labels).astype(str), index=range(len(labels)))
    ctrl_all = labels.index[labels == control].to_numpy()
    if len(ctrl_all) < 2:
        raise ValueError("need at least 2 control cells")
    rows = []
    for pert in sorted(set(labels) - {control}):
        p_all = labels.index[labels == pert].to_numpy()
        p_idx = rng.choice(p_all, min(len(p_all), max_cells or len(p_all)), replace=False)
        c_idx = rng.choice(ctrl_all, min(len(ctrl_all), max_cells or len(ctrl_all)), replace=False)
        pooled = np.concatenate([p_idx, c_idx])
        D = cdist(emb[pooled], emb[pooled])
        n_p = len(p_idx)
        ix_p, ix_c = np.arange(n_p), np.arange(n_p, len(pooled))
        observed = _edist_from_pooled(D, ix_p, ix_c)
        null = np.empty(n_perm)
        for i in range(n_perm):
            perm = rng.permutation(len(pooled))
            null[i] = _edist_from_pooled(D, perm[:n_p], perm[n_p:])
        pval = (1 + np.sum(null >= observed)) / (1 + n_perm)
        rows.append({"perturbation": pert, "n_cells": int(len(p_all)), "n_cells_used": int(n_p),
                     "edistance": observed, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
                     "p_value": pval})
    res = pd.DataFrame(rows)
    if len(res):
        res["p_adj_bh"] = multipletests(res["p_value"], method="fdr_bh")[1]
        res = res.sort_values("edistance", ascending=False).reset_index(drop=True)
    log.info("E-distance computed for %d perturbations (%d permutations each, <= %s cells per group)",
             len(res), n_perm, max_cells)
    return res
