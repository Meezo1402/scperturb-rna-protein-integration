"""Normalisation and embedding.

RNA: the shifted logarithm, log1p(counts / size_factor * median_count). Ahlmann-Eltze &
Huber (2023) showed it performs as well as or better than more elaborate transforms for
downstream analysis, and it is what scPerturb itself used before PCA, so E-distances
computed here are on the same footing as the published ones. Highly variable genes are
chosen with the seurat_v3 flavour, which works on raw counts and is batch-aware when a
batch column exists.

Protein: centred log-ratio (CLR) per protein across cells (Seurat's margin = 2), the
standard for ADT counts when empty droplets are unavailable for dsb-style background
subtraction. The scPerturb files do not carry empty droplets, so CLR is the right tool.
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

log = logging.getLogger("scpert")


def normalize_rna(rna: ad.AnnData, n_hvg: int, hvg_flavor: str, n_pcs: int, n_neighbors: int,
                  umap: bool, batch_col: str | None, seed: int) -> ad.AnnData:
    if "counts" not in rna.layers:
        rna.layers["counts"] = rna.X.copy()
    rna.X = rna.layers["counts"].copy()
    sc.pp.normalize_total(rna, target_sum=None)        # size factor = total / median(total): shifted log
    sc.pp.log1p(rna)
    rna.layers["lognorm"] = rna.X.copy()

    batch_key = batch_col if (batch_col and batch_col in rna.obs and rna.obs[batch_col].nunique() > 1) else None
    n_hvg = min(n_hvg, rna.n_vars)
    if hvg_flavor == "seurat_v3":
        sc.pp.highly_variable_genes(rna, n_top_genes=n_hvg, flavor="seurat_v3", layer="counts", batch_key=batch_key)
    else:
        sc.pp.highly_variable_genes(rna, n_top_genes=n_hvg, flavor=hvg_flavor, batch_key=batch_key)

    n_pcs = int(min(n_pcs, rna.n_obs - 1, int(rna.var["highly_variable"].sum()) - 1))
    sc.pp.pca(rna, n_comps=n_pcs, mask_var="highly_variable", random_state=seed)
    sc.pp.neighbors(rna, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=seed)
    if umap:
        sc.tl.umap(rna, random_state=seed)
    log.info("RNA normalised: %d HVGs, %d PCs%s", n_hvg, n_pcs, ", UMAP" if umap else "")
    return rna


def clr(x: np.ndarray, axis: int) -> np.ndarray:
    """Centred log-ratio of (counts + 1) along `axis` (0 = across cells per protein,
    1 = across proteins per cell). Returns a dense float array."""
    lx = np.log1p(np.asarray(x, dtype=float))
    return lx - lx.mean(axis=axis, keepdims=True)


def normalize_protein(prot: ad.AnnData, margin: str = "cells") -> ad.AnnData:
    if "counts" not in prot.layers:
        prot.layers["counts"] = prot.X.copy()
    X = prot.layers["counts"]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    axis = 0 if margin == "cells" else 1
    prot.X = clr(X, axis=axis)
    prot.layers["clr"] = prot.X.copy()
    log.info("protein CLR-normalised across %s (%d proteins)", margin, prot.n_vars)
    return prot
