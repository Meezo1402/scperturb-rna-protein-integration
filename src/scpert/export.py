"""Export pseudobulk matrices for DIABLO (mixOmics, R).

DIABLO (Singh et al. 2019) is a supervised multi-block sPLS-DA that works on a modest
number of samples with a class label each. Single cells are too many, too noisy and not
independent, so the integration is run on pseudobulk samples: for every
perturbation x replicate, RNA counts are summed, converted to log-CPM and restricted to
the most variable HVGs; protein CLR values are averaged. The class label is the
perturbation. Only the `n_top_perturbations` strongest perturbations (by E-distance)
plus control are exported, so that DIABLO has a tractable number of classes with
several samples each.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .pseudobulk import make_pseudobulk

log = logging.getLogger("scpert")


def log_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    lib = counts.sum(axis=1).to_numpy()[:, None]
    return pd.DataFrame(np.log1p(counts.to_numpy() / np.maximum(lib, 1) * 1e6), index=counts.index, columns=counts.columns)


def export_for_diablo(rna: ad.AnnData, prot: ad.AnnData, edist_table: pd.DataFrame, group_col: str, control: str,
                      replicate_col: str | None, n_pseudo_reps: int, min_cells_per_rep: int,
                      n_rna_features: int, n_top_perturbations: int, out_dir: str | Path, seed: int = 0,
                      splits_per_replicate: int = 1) -> dict:
    """splits_per_replicate > 1 further divides every replicate (e.g. each guide) into random
    subsets, so that DIABLO's cross-validation has several samples per class. These are
    pseudo-replicates nested in real ones; the labels file records the mode."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top = edist_table.sort_values("edistance", ascending=False)["perturbation"].head(n_top_perturbations).tolist()
    keep_groups = set(top) | {control}
    mask = rna.obs[group_col].astype(str).isin(keep_groups).to_numpy()
    rna_sub, prot_sub = rna[mask], prot[mask]

    hv = rna_sub.var_names[rna_sub.var["highly_variable"].to_numpy()] if "highly_variable" in rna_sub.var else rna_sub.var_names
    rna_counts, meta = make_pseudobulk(rna_sub[:, hv].copy(), group_col, replicate_col, n_pseudo_reps,
                                       min_cells_per_rep, layer="counts", seed=seed, aggregate="sum",
                                       splits_per_replicate=splits_per_replicate)
    rna_mat = log_cpm(rna_counts)
    var = rna_mat.var(axis=0).sort_values(ascending=False)
    rna_mat = rna_mat[var.index[: min(n_rna_features, len(var))]]

    prot_mat, meta_p = make_pseudobulk(prot_sub.copy(), group_col, replicate_col, n_pseudo_reps,
                                       min_cells_per_rep, layer="clr", seed=seed, aggregate="mean",
                                       splits_per_replicate=splits_per_replicate)
    shared = rna_mat.index.intersection(prot_mat.index)
    rna_mat, prot_mat, meta = rna_mat.loc[shared], prot_mat.loc[shared], meta.loc[shared]

    rna_mat.to_csv(out_dir / "diablo_rna_logcpm.csv")
    prot_mat.to_csv(out_dir / "diablo_protein_clr.csv")
    meta[["group", "replicate", "n_cells", "mode"]].to_csv(out_dir / "diablo_labels.csv")
    log.info("DIABLO export: %d pseudobulk samples, %d classes (incl. control), %d RNA features, %d proteins -> %s",
             len(shared), meta["group"].nunique(), rna_mat.shape[1], prot_mat.shape[1], out_dir)
    return {"n_samples": len(shared), "n_classes": int(meta["group"].nunique()),
            "n_rna_features": int(rna_mat.shape[1]), "n_proteins": int(prot_mat.shape[1]), "classes": sorted(keep_groups)}
