"""Mixscape-style perturbation signatures and KO / NP (non-perturbed) classification.

In a pooled CRISPR screen a cell can carry a guide and still show no effect: the edit
was in-frame, the protein is stable, the cell escaped. Papalexi et al. (2021) showed
that these "escaping" cells dilute every downstream estimate and introduced mixscape to
remove them. Two steps, re-implemented here from the published description:

1. Perturbation signature. For every cell, subtract the mean expression of its k nearest
   control cells (k = 20, in PCA space, within the same replicate/batch). This removes
   the variation shared with control cells (cell cycle, replicate, library size) and
   leaves the component attributable to the perturbation.

2. Classification. For each target gene, the perturbation vector is the difference
   between the mean signature of the target's cells and of control cells over the genes
   that are differentially expressed between them. Every cell's signature is projected
   onto that vector, and the projections of the target's cells are modelled as a mixture
   of two Gaussians: a non-perturbed component pinned to the control cells' projection
   (mean and SD fixed) and a free KO component. Cells with posterior P(KO) > 0.5 are KO.
   The vector is then recomputed from KO cells only and the fit repeated until the labels
   stop changing (at most `iter_num` rounds). A target with fewer than `min_de_genes`
   differentially expressed genes has no detectable perturbation and all of its cells are
   labelled NP.

Defaults follow the Seurat implementation (num.neighbors = 20, ndims = 40,
min.de.genes = 5, iter.num = 10, posterior > 0.5).
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

log = logging.getLogger("scpert")


# --------------------------------------------------------------------------- signature
def perturbation_signature(rna: ad.AnnData, pert_col: str, control: str, batch_col: str | None = None,
                           n_neighbors: int = 20, n_dims: int = 40, layer: str = "lognorm",
                           use_hvg: bool = True) -> ad.AnnData:
    """Store the perturbation signature (expression minus mean of k nearest control cells)
    in `rna.layers["X_pert"]` for the genes in `rna.uns["mixscape_genes"]` (HVGs by default;
    other columns of the layer are left at zero)."""
    X = rna.layers[layer] if layer in rna.layers else rna.X
    genes = rna.var["highly_variable"].to_numpy() if (use_hvg and "highly_variable" in rna.var) else np.ones(rna.n_vars, bool)
    gidx = np.where(genes)[0]
    Xg = X[:, gidx]
    Xg = Xg.toarray() if sp.issparse(Xg) else np.asarray(Xg)
    Xg = Xg.astype(np.float32)
    pcs = rna.obsm["X_pca"][:, : min(n_dims, rna.obsm["X_pca"].shape[1])]
    labels = rna.obs[pert_col].astype(str).to_numpy()
    batches = rna.obs[batch_col].astype(str).to_numpy() if (batch_col and batch_col in rna.obs) else np.repeat("all", rna.n_obs)

    sig = np.empty_like(Xg)
    for b in np.unique(batches):
        in_b = np.where(batches == b)[0]
        ctrl = in_b[labels[in_b] == control]
        if len(ctrl) < n_neighbors:
            raise ValueError(f"batch {b!r} has {len(ctrl)} control cells, fewer than n_neighbors={n_neighbors}")
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(pcs[ctrl])
        _, nbr = nn.kneighbors(pcs[in_b])                      # (n_b, k) indices into ctrl
        # mean over the k control neighbours: sparse averaging matrix (n_b x n_ctrl) @ X_ctrl
        rows = np.repeat(np.arange(len(in_b)), n_neighbors)
        W = sp.csr_matrix((np.full(rows.size, 1.0 / n_neighbors, dtype=np.float32), (rows, nbr.ravel())),
                          shape=(len(in_b), len(ctrl)))
        sig[in_b] = Xg[in_b] - (W @ Xg[ctrl])
    full = np.zeros((rna.n_obs, rna.n_vars), dtype=np.float32)
    full[:, gidx] = sig
    rna.layers["X_pert"] = full
    rna.uns["mixscape_genes"] = rna.var_names[gidx].tolist()
    log.info("perturbation signature: %d genes, k=%d control neighbours in %d PCs, %d batch(es)",
             len(gidx), n_neighbors, pcs.shape[1], len(np.unique(batches)))
    return rna


# --------------------------------------------------------------------------- helpers
def _welch_rows(A: np.ndarray, B: np.ndarray):
    """Vectorised Welch t-test between the columns of A and B (cells x genes). Returns p-values."""
    from scipy import stats
    n1, n2 = A.shape[0], B.shape[0]
    m1, m2 = A.mean(0), B.mean(0)
    v1, v2 = A.var(0, ddof=1), B.var(0, ddof=1)
    se2 = v1 / n1 + v2 / n2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (m1 - m2) / np.sqrt(se2)
        df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    p = 2 * stats.t.sf(np.abs(t), df)
    p[~np.isfinite(p)] = 1.0
    return p, m1 - m2


def _fit_mixture_fixed_control(x: np.ndarray, ctrl_mean: float, ctrl_sd: float, init_ko_mean: float,
                               n_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """EM for a two-component Gaussian mixture in which component 0 (NP) has its mean and SD
    fixed to the control cells' projection; component 1 (KO) has free mean, SD and weight.
    Returns the posterior probability of KO for every value of x."""
    ctrl_sd = max(ctrl_sd, 1e-6)
    mu, sd, w = init_ko_mean, max(np.std(x), 1e-6), 0.5
    def dens(v, m, s):
        return np.exp(-0.5 * ((v - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    post = np.full_like(x, 0.5, dtype=float)
    for _ in range(n_iter):
        p_np = (1 - w) * dens(x, ctrl_mean, ctrl_sd)
        p_ko = w * dens(x, mu, sd)
        new_post = p_ko / np.maximum(p_np + p_ko, 1e-300)
        w_new = float(np.clip(new_post.mean(), 1e-3, 1 - 1e-3))
        mu_new = float(np.sum(new_post * x) / max(new_post.sum(), 1e-12))
        sd_new = float(np.sqrt(np.sum(new_post * (x - mu_new) ** 2) / max(new_post.sum(), 1e-12)))
        sd_new = max(sd_new, 1e-6)
        converged = np.max(np.abs(new_post - post)) < tol
        post, w, mu, sd = new_post, w_new, mu_new, sd_new
        if converged:
            break
    return post


# --------------------------------------------------------------------------- classification
def mixscape_classify(rna: ad.AnnData, pert_col: str, control: str, layer: str = "X_pert",
                      min_de_genes: int = 5, iter_num: int = 10, lfc_threshold: float = 0.25,
                      de_padj: float = 0.05, posterior_threshold: float = 0.5,
                      min_cells: int = 10) -> pd.DataFrame:
    """Label every cell KO / NP / control and return a per-perturbation summary.
    Adds obs columns mixscape_class, mixscape_class_global, mixscape_p_ko."""
    if layer not in rna.layers:
        raise KeyError("run perturbation_signature() first")
    genes = rna.uns.get("mixscape_genes", rna.var_names.tolist())
    gidx = rna.var_names.get_indexer(genes)
    S = np.asarray(rna.layers[layer][:, gidx], dtype=np.float32)
    labels = rna.obs[pert_col].astype(str).to_numpy()
    ctrl_idx = np.where(labels == control)[0]
    S_ctrl = S[ctrl_idx]

    cls = np.array([f"{l} NP" if l != control else control for l in labels], dtype=object)
    glob = np.where(labels == control, "control", "NP").astype(object)
    p_ko = np.zeros(rna.n_obs, dtype=float)
    rows = []
    for pert in sorted(set(labels) - {control}):
        idx = np.where(labels == pert)[0]
        if len(idx) < min_cells:
            rows.append({"perturbation": pert, "n_cells": len(idx), "n_ko": 0, "frac_ko": 0.0, "n_de_genes": 0,
                         "iterations": 0, "status": f"skipped (< {min_cells} cells)"})
            continue
        ko = np.ones(len(idx), dtype=bool)
        n_de, it, status = 0, 0, "no detectable perturbation"
        for it in range(1, iter_num + 1):
            use = idx[ko] if ko.sum() >= min_cells else idx
            pvals, diff = _welch_rows(S[use], S_ctrl)
            padj = multipletests(pvals, method="fdr_bh")[1]
            de = np.where((padj < de_padj) & (np.abs(diff) > lfc_threshold))[0]
            n_de = len(de)
            if n_de < min_de_genes:
                ko[:] = False
                break
            vec = S[use][:, de].mean(0) - S_ctrl[:, de].mean(0)
            vec = vec / max(np.linalg.norm(vec), 1e-12)
            proj_ctrl = S_ctrl[:, de] @ vec
            proj = S[idx][:, de] @ vec
            post = _fit_mixture_fixed_control(proj, float(proj_ctrl.mean()), float(proj_ctrl.std(ddof=1)),
                                              init_ko_mean=float(proj[ko].mean() if ko.any() else proj.mean()))
            new_ko = post > posterior_threshold
            p_ko[idx] = post
            if np.array_equal(new_ko, ko):
                ko = new_ko
                status = "KO detected"
                break
            ko = new_ko
            status = "KO detected"
        if ko.sum() == 0:
            status = "no detectable perturbation"
            p_ko[idx] = 0.0
        cls[idx[ko]] = f"{pert} KO"
        glob[idx[ko]] = "KO"
        rows.append({"perturbation": pert, "n_cells": int(len(idx)), "n_ko": int(ko.sum()),
                     "frac_ko": float(ko.mean()), "n_de_genes": int(n_de), "iterations": int(it), "status": status})
    rna.obs["mixscape_class"] = pd.Categorical(cls)
    rna.obs["mixscape_class_global"] = pd.Categorical(glob, categories=["control", "KO", "NP"])
    rna.obs["mixscape_p_ko"] = p_ko
    summary = pd.DataFrame(rows).sort_values("frac_ko", ascending=False).reset_index(drop=True)
    n_detect = int((summary["status"] == "KO detected").sum())
    log.info("mixscape: KO detected in %d of %d perturbations; %d KO cells, %d NP cells removed from downstream analysis",
             n_detect, len(summary), int((glob == "KO").sum()), int((glob == "NP").sum()))
    return summary
