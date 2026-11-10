"""Figures. Matplotlib only; every function writes a PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def qc_figure(obs: pd.DataFrame, flags_prefix: str, out: Path) -> Path:
    fail = obs[[c for c in obs.columns if c.startswith(flags_prefix)]].any(axis=1).to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(obs["total_counts"], obs["n_genes_by_counts"], s=3, c=np.where(fail, "#c0392b", "#7f8c8d"), alpha=0.5)
    axes[0].set(xscale="log", yscale="log", xlabel="total counts", ylabel="genes detected", title="cells (red = failed QC)")
    axes[1].hist(obs["pct_counts_mt"], bins=60, color="#7f8c8d")
    axes[1].set(xlabel="% mitochondrial counts", ylabel="cells", title="mito fraction")
    axes[2].hist(obs["pct_counts_in_top_20_genes"], bins=60, color="#7f8c8d")
    axes[2].set(xlabel="% counts in top-20 genes", ylabel="cells", title="complexity")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def umap_figure(rna, group_col: str, control: str, top: list[str], out: Path) -> Path:
    if "X_umap" not in rna.obsm:
        return out
    xy = rna.obsm["X_umap"]
    labels = rna.obs[group_col].astype(str).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#dddddd", label="other perturbations")
    m = labels == control
    ax.scatter(xy[m, 0], xy[m, 1], s=3, c="#2c3e50", label=control)
    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(top[:9]):
        m = labels == g
        ax.scatter(xy[m, 0], xy[m, 1], s=4, color=cmap(i), label=g)
    ax.set(xticks=[], yticks=[], title="UMAP: control and strongest perturbations (by E-distance)")
    ax.legend(markerscale=4, fontsize=8, loc="best", frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def edist_figure(tab: pd.DataFrame, out: Path, n: int = 30) -> Path:
    t = tab.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(t))))
    colors = np.where(t["p_adj_bh"] < 0.05, "#2980b9", "#95a5a6")
    ax.barh(t["perturbation"], t["edistance"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="E-distance to control (PCA space)", title="Perturbation strength (blue: E-test BH < 0.05)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def volcano_figure(de: pd.DataFrame, perturbation: str, out: Path, lfc_col="log2FoldChange", p_col="padj", n_label=12) -> Path:
    d = de[de["perturbation"] == perturbation].dropna(subset=[lfc_col, p_col]).copy()
    if d.empty:
        return out
    d["nlp"] = -np.log10(np.clip(d[p_col].to_numpy(), 1e-300, 1))
    sig = d[p_col] < 0.05
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(d[lfc_col], d["nlp"], s=6, c=np.where(sig, "#c0392b", "#95a5a6"), alpha=0.7)
    for _, r in d[sig].sort_values("nlp", ascending=False).head(n_label).iterrows():
        ax.annotate(r["gene"], (r[lfc_col], r["nlp"]), fontsize=7, xytext=(2, 2), textcoords="offset points")
    ax.axhline(-np.log10(0.05), ls="--", lw=0.8, color="k")
    ax.set(xlabel="log2 fold change vs control", ylabel="-log10 adjusted p", title=f"{perturbation}: pseudobulk DESeq2")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def protein_heatmap(ptab: pd.DataFrame, out: Path, order: list[str] | None = None) -> Path:
    piv = ptab.pivot(index="perturbation", columns="protein", values="delta_mean_clr")
    if order:
        piv = piv.loc[[o for o in order if o in piv.index]]
    sig = ptab.pivot(index="perturbation", columns="protein", values="p_adj_bh").loc[piv.index, piv.columns] < 0.05
    v = np.nanmax(np.abs(piv.to_numpy())) or 1.0
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * piv.shape[1] + 2), max(3, 0.3 * piv.shape[0] + 1.5)))
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if sig.iloc[i, j]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean CLR difference vs control")
    ax.set_title("Surface protein shifts (*: Wilcoxon BH < 0.05)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def mixscape_figure(summary: pd.DataFrame, out: Path) -> Path:
    t = summary.sort_values("frac_ko").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(t))))
    colors = np.where(t["status"] == "KO detected", "#16a085", "#95a5a6")
    ax.barh(t["perturbation"], 100 * t["frac_ko"], color=colors)
    for i, r in t.iterrows():
        ax.text(100 * r["frac_ko"] + 1, i, f"{r['n_ko']}/{r['n_cells']}", va="center", fontsize=7)
    ax.set(xlabel="% of cells classified KO (mixscape)", xlim=(0, 115),
           title="Escaping cells: KO fraction per perturbation (grey: no detectable perturbation)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def concordance_figure(conc: pd.DataFrame, out: Path) -> Path:
    """RNA log2FC of the protein's coding gene vs the protein's CLR shift, one point per
    perturbation x protein. Discordant perturbations (protein moves, mRNA does not, or the
    reverse) are the ones a multimodal readout exists to find."""
    if conc.empty:
        return out
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for prot, d in conc.groupby("protein"):
        ax.scatter(d["rna_log2fc"], d["protein_delta_clr"], s=28, label=f"{prot} / {d['gene'].iloc[0]}", alpha=0.85)
    for _, r in conc.iterrows():
        if (r["rna_padj"] < 0.05) or (r["protein_padj"] < 0.05):
            ax.annotate(r["perturbation"], (r["rna_log2fc"], r["protein_delta_clr"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set(xlabel="mRNA log2 fold change vs control (pseudobulk DESeq2)",
           ylabel="surface protein, mean CLR difference vs control",
           title="RNA vs protein concordance per perturbation")
    ax.legend(fontsize=7, frameon=False, title="protein / gene")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out
