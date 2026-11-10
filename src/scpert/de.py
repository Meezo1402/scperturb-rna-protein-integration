"""Differential expression: pseudobulk DESeq2 (primary), cell-level Wilcoxon (secondary),
and per-protein Wilcoxon on CLR values.

Why two RNA tests: the pseudobulk DESeq2 test is the one whose p-values mean what they
say (the replicate is the unit of inference). The cell-level Wilcoxon test is what most
single-cell papers report; it is kept because it is useful for ranking genes within a
perturbation and for comparison with published analyses, but its p-values are
pseudoreplicated and are labelled as such in the output.
"""

from __future__ import annotations

import logging
import re

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

log = logging.getLogger("scpert")


def _safe_level(label: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(label)).strip("_")
    return f"g_{s}" if not s or not s[0].isalpha() else s


def pseudobulk_deseq2(counts: pd.DataFrame, meta: pd.DataFrame, control: str, min_counts_per_gene: int = 10,
                      n_jobs: int = 1) -> pd.DataFrame:
    """One DESeq2 model (~ condition) over all pseudobulk samples, then a Wald contrast of
    every perturbation against control. counts: samples x genes (integers)."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    counts = counts.loc[meta.index]
    keep_genes = counts.columns[counts.sum(axis=0) >= min_counts_per_gene]
    counts = counts[keep_genes].round().astype(int)
    level_map = {g: _safe_level(g) for g in meta["group"].unique()}
    md = pd.DataFrame({"condition": meta["group"].map(level_map).to_numpy()}, index=meta.index)
    groups = [g for g in meta["group"].unique() if g != control]
    if len(groups) == 0:
        raise ValueError("no non-control groups for DESeq2")
    counts_per_group = meta["group"].value_counts()
    if (counts_per_group < 2).any():
        bad = counts_per_group[counts_per_group < 2].index.tolist()
        raise ValueError(f"groups with a single pseudobulk sample cannot be tested: {bad}")

    inference = DefaultInference(n_cpus=n_jobs)
    dds = DeseqDataSet(counts=counts, metadata=md, design="~condition", refit_cooks=True,
                       inference=inference, quiet=True)
    dds.deseq2()
    out = []
    for g in groups:
        stat = DeseqStats(dds, contrast=["condition", level_map[g], level_map[control]], inference=inference, quiet=True)
        stat.summary()
        r = stat.results_df.copy()
        r.insert(0, "gene", r.index)
        r.insert(0, "perturbation", g)
        out.append(r.reset_index(drop=True))
    res = pd.concat(out, ignore_index=True)
    res["test"] = "pseudobulk_deseq2"
    res["replicate_mode"] = meta["mode"].iloc[0]
    log.info("pseudobulk DESeq2: %d perturbations x %d genes; %d gene-perturbation pairs at padj < 0.05",
             len(groups), len(keep_genes), int((res["padj"] < 0.05).sum()))
    return res


def cell_level_wilcoxon(rna: ad.AnnData, group_col: str, control: str, layer: str = "lognorm") -> pd.DataFrame:
    """scanpy rank_genes_groups (Wilcoxon, BH) of every perturbation vs control on
    log-normalised data. Pseudoreplicated by construction; see module docstring."""
    sc.tl.rank_genes_groups(rna, groupby=group_col, reference=control, method="wilcoxon",
                            layer=layer, use_raw=False, pts=True, key_added="rgg_wilcoxon")
    res = sc.get.rank_genes_groups_df(rna, group=None, key="rgg_wilcoxon")
    res = res.rename(columns={"group": "perturbation", "names": "gene", "logfoldchanges": "log2fc",
                              "pvals": "p_value", "pvals_adj": "p_adj_bh", "scores": "wilcoxon_z"})
    res["test"] = "cell_level_wilcoxon (pseudoreplicated)"
    log.info("cell-level Wilcoxon: %d perturbations; %d gene-perturbation pairs at BH < 0.05",
             res["perturbation"].nunique(), int((res["p_adj_bh"] < 0.05).sum()))
    return res


def protein_wilcoxon(prot: ad.AnnData, group_col: str, control: str, layer: str = "clr") -> pd.DataFrame:
    """Per-protein Mann-Whitney U of each perturbation vs control on CLR values, with the
    difference in mean CLR as the effect size and BH correction across all tests."""
    X = prot.layers[layer] if layer in prot.layers else prot.X
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X)
    labels = prot.obs[group_col].astype(str).to_numpy()
    ctrl = X[labels == control]
    rows = []
    for g in sorted(set(labels) - {control}):
        sub = X[labels == g]
        for j, p in enumerate(prot.var_names):
            if np.all(sub[:, j] == sub[0, j]) and np.all(ctrl[:, j] == ctrl[0, j]):
                pval = 1.0
            else:
                pval = mannwhitneyu(sub[:, j], ctrl[:, j], alternative="two-sided").pvalue
            rows.append({"perturbation": g, "protein": p, "n_cells": int(sub.shape[0]),
                         "delta_mean_clr": float(sub[:, j].mean() - ctrl[:, j].mean()), "p_value": float(pval)})
    res = pd.DataFrame(rows)
    if len(res):
        res["p_adj_bh"] = multipletests(res["p_value"], method="fdr_bh")[1]
    res["test"] = "protein_wilcoxon_clr (pseudoreplicated)"
    log.info("protein Wilcoxon: %d perturbations x %d proteins; %d at BH < 0.05",
             res["perturbation"].nunique() if len(res) else 0, prot.n_vars, int((res["p_adj_bh"] < 0.05).sum()) if len(res) else 0)
    return res


def rna_protein_concordance(de_rna: pd.DataFrame, de_prot: pd.DataFrame, protein_gene_map: dict[str, str]) -> pd.DataFrame:
    """Join, per perturbation, the pseudobulk log2FC of a protein's coding gene with the
    protein's CLR shift. `protein_gene_map` maps ADT names to gene symbols
    (e.g. {"PDL1": "CD274"}). Proteins or genes absent from the tables are skipped with a
    log message rather than an error, since ADT naming differs between releases."""
    rows = []
    for prot, gene in protein_gene_map.items():
        r = de_rna[de_rna["gene"] == gene][["perturbation", "log2FoldChange", "padj"]]
        p = de_prot[de_prot["protein"] == prot][["perturbation", "delta_mean_clr", "p_adj_bh"]]
        if r.empty or p.empty:
            log.warning("concordance: %s -> %s skipped (%s not found)", prot, gene,
                        "gene" if r.empty else "protein")
            continue
        m = r.merge(p, on="perturbation", how="inner")
        m.insert(0, "gene", gene)
        m.insert(0, "protein", prot)
        rows.append(m.rename(columns={"log2FoldChange": "rna_log2fc", "padj": "rna_padj",
                                      "delta_mean_clr": "protein_delta_clr", "p_adj_bh": "protein_padj"}))
    if not rows:
        return pd.DataFrame(columns=["protein", "gene", "perturbation", "rna_log2fc", "rna_padj", "protein_delta_clr", "protein_padj"])
    conc = pd.concat(rows, ignore_index=True)
    conc["pattern"] = np.select(
        [(conc["rna_padj"] < 0.05) & (conc["protein_padj"] < 0.05) & (np.sign(conc["rna_log2fc"]) == np.sign(conc["protein_delta_clr"])),
         (conc["rna_padj"] < 0.05) & (conc["protein_padj"] < 0.05),
         (conc["rna_padj"] >= 0.05) & (conc["protein_padj"] < 0.05),
         (conc["rna_padj"] < 0.05) & (conc["protein_padj"] >= 0.05)],
        ["concordant", "opposite direction", "protein only", "mRNA only"], default="neither")
    log.info("RNA/protein concordance: %d perturbation x protein pairs; patterns: %s",
             len(conc), ", ".join(f"{k}={v}" for k, v in conc["pattern"].value_counts().items()))
    return conc
