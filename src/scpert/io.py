"""Loading and aligning a scPerturb RNA / protein AnnData pair."""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

log = logging.getLogger("scpert")


def _is_integer_counts(X, n_check: int = 2000) -> bool:
    """True if a sample of the matrix looks like raw counts (non-negative integers)."""
    if sp.issparse(X):
        data = X.data[:n_check] if X.nnz else np.array([0.0])
    else:
        flat = np.asarray(X).ravel()
        data = flat[: min(n_check, flat.size)]
    data = np.asarray(data, dtype=float)
    return bool(np.all(data >= 0) and np.allclose(data, np.round(data)))


def load_pair(rna_path: str | Path, protein_path: str | Path,
              perturbation_col: str = "perturbation") -> tuple[ad.AnnData, ad.AnnData]:
    """Read the RNA and protein h5ad files and restrict both to shared cells, in the
    same order. Raises if either file lacks the perturbation column or if X does not
    look like raw counts (the pipeline normalises itself; pre-normalised input would be
    normalised twice)."""
    rna = ad.read_h5ad(rna_path)
    prot = ad.read_h5ad(protein_path)
    for name, a in (("RNA", rna), ("protein", prot)):
        if perturbation_col not in a.obs:
            raise KeyError(f"{name} AnnData has no obs column '{perturbation_col}'")
        if not _is_integer_counts(a.X):
            raise ValueError(f"{name} .X does not look like raw counts; this pipeline expects counts")
    shared = rna.obs_names.intersection(prot.obs_names)
    if len(shared) == 0:
        raise ValueError("RNA and protein files share no cell barcodes")
    lost_rna, lost_prot = rna.n_obs - len(shared), prot.n_obs - len(shared)
    rna, prot = rna[shared].copy(), prot[shared].copy()
    rna.var_names_make_unique()
    prot.var_names_make_unique()
    log.info("loaded %d shared cells (%d RNA-only and %d protein-only cells dropped); "
             "%d genes, %d proteins", len(shared), lost_rna, lost_prot, rna.n_vars, prot.n_vars)
    # keep a copy of the raw counts before anything touches X
    rna.layers["counts"] = rna.X.copy()
    prot.layers["counts"] = prot.X.copy()
    return rna, prot


def restrict_perturbations(rna: ad.AnnData, prot: ad.AnnData, perturbation_col: str,
                           control_label: str, drop_multi: bool, nperts_col: str | None):
    """Drop cells with no perturbation call and, optionally, cells carrying more than one
    perturbation. Confirms the control label exists."""
    keep = rna.obs[perturbation_col].notna().to_numpy().copy()
    if drop_multi and nperts_col and nperts_col in rna.obs:
        keep &= (rna.obs[nperts_col].fillna(1).astype(int) <= 1).to_numpy()
    n_drop = int((~keep).sum())
    rna, prot = rna[keep].copy(), prot[keep].copy()
    labels = rna.obs[perturbation_col].astype(str)
    if control_label not in set(labels):
        raise ValueError(f"control label '{control_label}' not found in obs['{perturbation_col}']")
    rna.obs[perturbation_col] = labels.values
    prot.obs[perturbation_col] = labels.values
    log.info("dropped %d cells without a single perturbation call; %d perturbations + control remain",
             n_drop, labels.nunique() - 1)
    return rna, prot
