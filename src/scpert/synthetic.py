"""Synthetic paired RNA + protein perturbation data in the scPerturb layout.

The real datasets are public but several gigabytes, so the smoke test and CI run on a
small simulated pair that reproduces the structure the pipeline relies on: raw integer
counts in .X, a `perturbation` column with a `control` label, a guide column with two
guides per target, negative-binomial counts with a realistic library-size spread,
mitochondrial genes, a few damaged cells and doublets, and planted effects:

- each non-null perturbation knocks its own target gene down and shifts a random set of
  downstream genes (log2 fold changes ~ N(0, 1.2));
- a subset of surface proteins is coupled to a target gene, so knocking the gene down
  lowers the protein, which gives the RNA / protein integration something real to find;
- some perturbations are null (no effect) so the E-test and the DE tests have true
  negatives to be calibrated against;
- a fraction of the cells in every real perturbation carry the label but no effect
  ("escaping" cells, as after an in-frame edit), so the mixscape stage has something to
  find; they are recorded in obs["truth_escaped"].

A truth file records all of this so the smoke test can check recovery.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _nb(rng, mu, phi):
    """Negative binomial via Gamma-Poisson; phi = dispersion (var = mu + phi mu^2)."""
    shape = 1.0 / phi
    lam = rng.gamma(shape, mu / shape)
    return rng.poisson(lam)


def make_synthetic_pair(out_dir: str | Path, n_genes: int = 1500, n_proteins: int = 12, n_perturbations: int = 12,
                        n_null_perturbations: int = 3, cells_per_perturbation: int = 80, n_control: int = 400,
                        guides_per_target: int = 2, n_targets_per_pert: int = 40, doublet_rate: float = 0.03,
                        damaged_rate: float = 0.03, escape_rate: float = 0.2, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genes = [f"GENE{i:04d}" for i in range(n_genes)]
    n_mito = 13
    for i in range(n_mito):
        genes[i] = f"MT-G{i + 1}"
    for i in range(n_mito, n_mito + 20):
        genes[i] = f"RPL{i}"
    pert_names = [f"GENE{i:04d}" for i in range(100, 100 + n_perturbations)]          # targets are real genes
    null_perts = set(pert_names[-n_null_perturbations:]) if n_null_perturbations else set()
    proteins = [f"PROT{i:02d}" for i in range(n_proteins)]
    coupled = {pert_names[i]: proteins[i] for i in range(min(n_proteins // 2, n_perturbations - n_null_perturbations))}

    # gene baseline abundance (log-normal), mito genes abundant
    base = rng.lognormal(mean=-1.0, sigma=1.6, size=n_genes)
    base[:n_mito] *= 25
    base /= base.sum()
    phi_gene = np.clip(rng.gamma(2.0, 0.05, size=n_genes), 0.01, 1.0)

    # planted effects
    effects = {}
    for p in pert_names:
        if p in null_perts:
            effects[p] = {}
            continue
        targets = rng.choice([g for g in genes[n_mito + 20:] if g != p], n_targets_per_pert, replace=False)
        lfc = dict(zip(targets.tolist(), rng.normal(0, 1.2, size=n_targets_per_pert).round(3).tolist()))
        lfc[p] = -2.0                                                                    # knock-down of the target itself
        effects[p] = lfc

    labels, guides = [], []
    for p in pert_names:
        for c in range(cells_per_perturbation):
            labels.append(p); guides.append(f"{p}_g{c % guides_per_target + 1}")
    labels += ["control"] * n_control
    guides += [f"control_g{c % guides_per_target + 1}" for c in range(n_control)]
    n = len(labels)
    lib = rng.lognormal(mean=np.log(6000), sigma=0.35, size=n)
    labels_arr = np.array(labels)
    escaped = (rng.random(n) < escape_rate) & (labels_arr != "control") & ~np.isin(labels_arr, list(null_perts))

    gene_idx = {g: i for i, g in enumerate(genes)}
    X = np.zeros((n, n_genes), dtype=np.int32)
    P = np.zeros((n, n_proteins), dtype=np.int32)
    prot_base = rng.lognormal(mean=np.log(60), sigma=0.6, size=n_proteins)
    prot_idx = {p: i for i, p in enumerate(proteins)}
    for i in range(n):
        mu = base.copy()
        fold_prot = np.ones(n_proteins)
        if labels[i] != "control" and not escaped[i]:
            for g, l in effects[labels[i]].items():
                mu[gene_idx[g]] *= 2.0 ** l
            if labels[i] in coupled:
                fold_prot[prot_idx[coupled[labels[i]]]] = 2.0 ** -1.5
        mu = mu / mu.sum() * lib[i]
        X[i] = _nb(rng, mu, phi_gene)
        P[i] = _nb(rng, prot_base * fold_prot * rng.lognormal(0, 0.25), 0.15)

    # damaged cells: inflate mito fraction
    damaged = rng.random(n) < damaged_rate
    X[damaged, :n_mito] = (X[damaged, :n_mito] * 6).astype(np.int32)
    # doublets: add a random other cell's counts
    is_doublet = rng.random(n) < doublet_rate
    partners = rng.integers(0, n, size=n)
    X[is_doublet] += X[partners[is_doublet]]
    P[is_doublet] += P[partners[is_doublet]]

    cells = [f"cell_{i:06d}" for i in range(n)]
    obs = pd.DataFrame({"perturbation": labels, "guide_id": guides, "nperts": 1, "batch": "lane1",
                        "truth_doublet": is_doublet, "truth_damaged": damaged, "truth_escaped": escaped}, index=cells)
    rna = ad.AnnData(X=sp.csr_matrix(X), obs=obs.copy(), var=pd.DataFrame({"ensembl_id": [f"ENSG{i:011d}" for i in range(n_genes)]}, index=genes))
    prot = ad.AnnData(X=sp.csr_matrix(P), obs=obs.copy(), var=pd.DataFrame(index=proteins))
    rna.write_h5ad(out_dir / "synthetic_RNA.h5ad")
    prot.write_h5ad(out_dir / "synthetic_protein.h5ad")
    truth = {"effects": effects, "null_perturbations": sorted(null_perts), "coupled_proteins": coupled,
             "n_cells": n, "n_doublets": int(is_doublet.sum()), "n_damaged": int(damaged.sum()),
             "n_escaped": int(escaped.sum()), "escape_rate": escape_rate}
    with open(out_dir / "truth.json", "w") as fh:
        json.dump(truth, fh, indent=1)
    return truth
