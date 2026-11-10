"""Pseudobulk aggregation.

Single cells from one perturbation are not independent replicates of that perturbation;
treating them as such (the "pseudoreplication" problem) inflates significance by orders
of magnitude (Squair et al. 2021; Crowell et al. 2020). Summing counts into pseudobulk
samples and testing those restores the correct unit of replication.

Which cells form a replicate:
- If the dataset carries a guide identifier (several sgRNAs per target gene), each guide
  is a genuine biological replicate of the target's knock-down and is used as such.
- Otherwise the cells of each perturbation are split at random into `n_pseudo_reps`
  pseudo-replicates. These capture sampling noise but NOT guide- or batch-level
  variability, so p-values from pseudo-replicates are anti-conservative and must be
  described as such. The pipeline records which mode was used in the output.
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

log = logging.getLogger("scpert")


def make_pseudobulk(adata: ad.AnnData, group_col: str, replicate_col: str | None, n_pseudo_reps: int,
                    min_cells_per_rep: int, layer: str = "counts", seed: int = 0,
                    aggregate: str = "sum", splits_per_replicate: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (matrix, meta): matrix is samples x features (summed counts or mean values),
    meta has columns group, replicate, n_cells, mode.

    splits_per_replicate > 1 divides each replicate (real or pseudo) into that many random
    subsets. Use it only to give a sample-level method such as DIABLO enough samples per
    class; never for inference, where it would multiply the replicate count artificially."""
    rng = np.random.default_rng(seed)
    X = adata.layers[layer] if layer in adata.layers else adata.X
    groups = adata.obs[group_col].astype(str).to_numpy()
    if replicate_col and replicate_col not in adata.obs:
        log.warning("replicate_col=%r is not an obs column; falling back to %d pseudo-replicates per group",
                    replicate_col, n_pseudo_reps)
    if replicate_col and replicate_col in adata.obs:
        reps = adata.obs[replicate_col].astype(str).to_numpy().copy()
        mode = f"replicate_col={replicate_col}"
    else:
        reps = np.empty(adata.n_obs, dtype=object)
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            assign = np.resize(np.arange(n_pseudo_reps), len(idx))
            rng.shuffle(assign)
            reps[idx] = [f"rep{a + 1}" for a in assign]
        mode = f"pseudo_replicates(n={n_pseudo_reps})"
    if splits_per_replicate > 1:
        reps = reps.astype(object)
        for g in np.unique(groups):
            for r in np.unique(reps[groups == g]):
                idx = np.where((groups == g) & (reps == r))[0]
                assign = np.resize(np.arange(splits_per_replicate), len(idx))
                rng.shuffle(assign)
                reps[idx] = [f"{r}_s{a + 1}" for a in assign]
        mode += f" x {splits_per_replicate} random splits (sample-level method input only)"

    rows, meta = [], []
    for g in np.unique(groups):
        for r in np.unique(reps[groups == g]):
            idx = np.where((groups == g) & (reps == r))[0]
            if len(idx) < min_cells_per_rep:
                continue
            sub = X[idx]
            vec = np.asarray(sub.sum(axis=0)).ravel() if aggregate == "sum" else np.asarray(sub.mean(axis=0)).ravel()
            rows.append(vec)
            meta.append({"sample": f"{g}__{r}", "group": g, "replicate": r, "n_cells": int(len(idx)), "mode": mode})
    if not rows:
        raise ValueError("no pseudobulk samples met min_cells_per_rep")
    mat = pd.DataFrame(np.vstack(rows), index=[m["sample"] for m in meta], columns=adata.var_names)
    meta = pd.DataFrame(meta).set_index("sample")
    log.info("pseudobulk (%s): %d samples across %d groups, %s", mode, len(meta), meta["group"].nunique(), aggregate)
    return mat, meta
