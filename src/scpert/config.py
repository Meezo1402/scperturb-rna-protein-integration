"""Configuration loading. Everything the pipeline does is driven by one YAML file;
no paths or thresholds live in code."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "dataset": {
        "name": "dataset",
        "rna_h5ad": None,
        "protein_h5ad": None,
        "perturbation_col": "perturbation",   # scPerturb standard
        "control_label": "control",           # scPerturb standard
        "replicate_col": None,                # e.g. "guide_id" when several guides target one gene
        "batch_col": None,                    # e.g. "lane" / "batch" if present
        "drop_multi_perturbation": True,      # scPerturb: nperts > 1 -> remove
        "nperts_col": "nperts",
    },
    "qc": {
        "mito_prefix": "MT-",
        "ribo_prefixes": ["RPS", "RPL"],
        "nmads_counts": 5,                    # log1p(total counts)
        "nmads_genes": 5,                     # log1p(n genes)
        "nmads_top20": 5,                     # pct counts in top-20 genes
        "nmads_mito": 3,                      # pct mito, upper tail only ...
        "mito_pct_min_flag": 8.0,             # ... and only if also above this absolute level
        "min_genes_per_cell": 200,
        "min_cells_per_gene": 20,
        "min_cells_per_perturbation": 30,     # perturbations with fewer cells after QC are dropped
        "doublets": {"run": True, "expected_rate": 0.06, "threshold": None},
        "protein_min_total_counts": 1,
    },
    "normalize": {
        "n_hvg": 2000,
        "hvg_flavor": "seurat_v3",            # operates on counts
        "n_pcs": 50,
        "n_neighbors": 15,
        "umap": True,
        "clr_margin": "cells",                # CLR across cells per protein (Seurat margin = 2)
    },
    "mixscape": {
        "run": True,
        "n_neighbors": 20,                    # control neighbours per cell for the perturbation signature
        "n_dims": 40,                         # PCs used to find them
        "min_de_genes": 5,                    # fewer -> "no detectable perturbation"
        "iter_num": 10,
        "lfc_threshold": 0.25,
        "posterior_threshold": 0.5,
        "use_ko_only_downstream": True,       # E-distance, DE and DIABLO use control + KO cells only
    },
    "edist": {
        "obsm_key": "X_pca",
        "n_pcs": 50,
        "max_cells_per_group": 300,
        "n_perm": 1000,
        "seed": 0,
    },
    "pseudobulk": {
        "n_pseudo_reps": 3,                   # used only when replicate_col is None
        "min_cells_per_rep": 20,
    },
    "de": {
        "run_pseudobulk_deseq2": True,
        "run_cell_level_wilcoxon": True,
        "min_counts_per_gene": 10,
        "n_jobs": 1,
    },
    "concordance": {
        "protein_gene_map": {},               # ADT name -> gene symbol, e.g. {"PDL1": "CD274"}
    },
    "export": {
        "n_rna_features": 1000,               # most variable HVGs kept for DIABLO
        "n_top_perturbations": 10,            # by E-distance; control is always kept
        "splits_per_replicate": 3,            # each replicate split into this many pseudobulk samples (DIABLO CV needs several per class)
        "min_cells_per_sample": 20,
    },
    "output_dir": "results/dataset",
    "seed": 0,
    "figures": True,
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, layered over DEFAULTS. Relative paths are resolved
    against the config file's directory's parent (the repository root)."""
    path = Path(path)
    with open(path) as fh:
        user = yaml.safe_load(fh) or {}
    cfg = _merge(DEFAULTS, user)
    root = path.resolve().parent.parent
    for key in ("rna_h5ad", "protein_h5ad"):
        p = cfg["dataset"].get(key)
        if p is not None and not Path(p).is_absolute():
            cfg["dataset"][key] = str((root / p).resolve())
    if not Path(cfg["output_dir"]).is_absolute():
        cfg["output_dir"] = str((root / cfg["output_dir"]).resolve())
    cfg["_root"] = str(root)
    return cfg
