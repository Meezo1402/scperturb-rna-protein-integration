"""Cell and gene quality control.

Thresholds are data-driven (median absolute deviations) rather than fixed numbers, as
recommended by Germain et al. (2020) and the single-cell best-practices book: a fixed
"min 500 genes" cut-off means something different in a deeply sequenced melanoma line
than in THP-1 monocytes, whereas "5 MADs from the median" adapts to each dataset.
Mitochondrial fraction is treated as an upper-tail problem only and a cell has to be
BOTH a MAD outlier and above an absolute floor to be removed, so that a healthy dataset
with uniformly low mito fractions does not lose its upper tail for nothing.
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

log = logging.getLogger("scpert")


def mad_outlier(x: pd.Series | np.ndarray, nmads: float, side: str = "both") -> np.ndarray:
    """Boolean mask of values more than `nmads` MADs from the median.
    side: 'both' | 'upper' | 'lower'. MAD is scaled by 1.4826 for normal consistency."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if mad == 0:                      # degenerate (e.g. all identical) -> nothing is an outlier
        return np.zeros_like(x, dtype=bool)
    z = (x - med) / mad
    if side == "upper":
        return z > nmads
    if side == "lower":
        return z < -nmads
    return np.abs(z) > nmads


def compute_qc_metrics(rna: ad.AnnData, mito_prefix: str, ribo_prefixes: list[str]) -> ad.AnnData:
    rna.var["mt"] = rna.var_names.str.upper().str.startswith(mito_prefix.upper())
    rna.var["ribo"] = rna.var_names.str.upper().str.startswith(tuple(p.upper() for p in ribo_prefixes))
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mt", "ribo"], percent_top=[20], log1p=True, inplace=True)
    return rna


def flag_cells(rna: ad.AnnData, cfg: dict) -> pd.DataFrame:
    """Return a per-cell boolean DataFrame of QC flags (True = fails that criterion)."""
    flags = pd.DataFrame(index=rna.obs_names)
    flags["low_or_high_counts"] = mad_outlier(rna.obs["log1p_total_counts"], cfg["nmads_counts"])
    flags["low_or_high_genes"] = mad_outlier(rna.obs["log1p_n_genes_by_counts"], cfg["nmads_genes"])
    flags["top20_outlier"] = mad_outlier(rna.obs["pct_counts_in_top_20_genes"], cfg["nmads_top20"])
    flags["high_mito"] = (mad_outlier(rna.obs["pct_counts_mt"], cfg["nmads_mito"], side="upper")
                          & (rna.obs["pct_counts_mt"].to_numpy() > cfg["mito_pct_min_flag"]))
    flags["too_few_genes"] = rna.obs["n_genes_by_counts"].to_numpy() < cfg["min_genes_per_cell"]
    return flags


def run_scrublet(rna: ad.AnnData, batch_col: str | None, expected_rate: float,
                 threshold: float | None, seed: int) -> pd.Series:
    """Doublet calls with Scrublet, run per batch when a batch column exists (doublets
    form within a lane, not across lanes). Returns a boolean Series."""
    calls = pd.Series(False, index=rna.obs_names)
    groups = rna.obs[batch_col].astype(str) if batch_col and batch_col in rna.obs else pd.Series("all", index=rna.obs_names)
    for g, idx in groups.groupby(groups).groups.items():
        sub = rna[idx].copy()
        if sub.n_obs < 100:
            log.warning("batch %s has %d cells; too few for Scrublet, skipping", g, sub.n_obs)
            continue
        sc.pp.scrublet(sub, expected_doublet_rate=expected_rate, threshold=threshold,
                       random_state=seed, verbose=False)
        calls.loc[sub.obs_names] = sub.obs["predicted_doublet"].astype(bool).to_numpy()
        log.info("Scrublet batch %s: %d / %d predicted doublets", g, int(calls.loc[sub.obs_names].sum()), sub.n_obs)
    return calls


def apply_cell_qc(rna: ad.AnnData, prot: ad.AnnData, cfg_qc: dict, cfg_ds: dict,
                  seed: int) -> tuple[ad.AnnData, ad.AnnData, pd.DataFrame, pd.DataFrame]:
    """Run all cell-level QC and return filtered (rna, prot), a summary table, and the
    pre-filter obs table (with per-criterion flags) for plotting."""
    rna = compute_qc_metrics(rna, cfg_qc["mito_prefix"], cfg_qc["ribo_prefixes"])
    flags = flag_cells(rna, cfg_qc)

    # protein-side: cells with (near) zero ADT counts carry no protein information
    prot_total = np.asarray(prot.X.sum(axis=1)).ravel()
    flags["no_protein_counts"] = prot_total < cfg_qc["protein_min_total_counts"]

    if cfg_qc["doublets"]["run"]:
        flags["doublet"] = run_scrublet(rna, cfg_ds.get("batch_col"), cfg_qc["doublets"]["expected_rate"],
                                        cfg_qc["doublets"]["threshold"], seed).to_numpy()

    fail_any = flags.any(axis=1).to_numpy()
    summary = pd.DataFrame({
        "criterion": list(flags.columns) + ["any_criterion", "cells_in", "cells_out"],
        "cells_flagged": [int(flags[c].sum()) for c in flags.columns] + [int(fail_any.sum()), rna.n_obs, int((~fail_any).sum())],
    })
    rna.obs = rna.obs.join(flags.add_prefix("qc_fail_"))
    obs_before = rna.obs.copy()                      # all cells, with flags, for the QC figure
    rna, prot = rna[~fail_any].copy(), prot[~fail_any].copy()
    log.info("cell QC: %d -> %d cells", len(fail_any), rna.n_obs)
    return rna, prot, summary, obs_before


def apply_gene_and_group_filters(rna: ad.AnnData, prot: ad.AnnData, cfg_qc: dict, cfg_ds: dict):
    """Drop genes seen in too few cells and perturbations left with too few cells."""
    n_before = rna.n_vars
    sc.pp.filter_genes(rna, min_cells=cfg_qc["min_cells_per_gene"])
    log.info("gene filter: %d -> %d genes (>= %d cells)", n_before, rna.n_vars, cfg_qc["min_cells_per_gene"])

    col = cfg_ds["perturbation_col"]
    counts = rna.obs[col].value_counts()
    keep_groups = counts[counts >= cfg_qc["min_cells_per_perturbation"]].index
    keep = rna.obs[col].isin(keep_groups).to_numpy()
    dropped = sorted(set(counts.index) - set(keep_groups))
    if dropped:
        log.info("dropping %d perturbations with < %d cells: %s%s", len(dropped), cfg_qc["min_cells_per_perturbation"],
                 ", ".join(dropped[:8]), " ..." if len(dropped) > 8 else "")
    rna, prot = rna[keep].copy(), prot[keep].copy()
    if cfg_ds["control_label"] not in set(rna.obs[col]):
        raise ValueError("control cells were removed by QC; relax the thresholds")
    for a in (rna, prot):
        a.obs[col] = pd.Categorical(a.obs[col].astype(str))
    return rna, prot
