"""Unit tests for the pure functions plus one small end-to-end recovery test.
Run with:  pytest -q
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpert.edist import edistance_to_control, energy_distance  # noqa: E402
from scpert.normalize import clr  # noqa: E402
from scpert.pseudobulk import make_pseudobulk  # noqa: E402
from scpert.qc import mad_outlier  # noqa: E402
from scpert.synthetic import make_synthetic_pair  # noqa: E402


# ----------------------------------------------------------------------------- E-distance
def test_energy_distance_matches_brute_force():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(30, 5)), rng.normal(size=(20, 5)) + 0.5
    dxy = np.mean([np.linalg.norm(x - y) for x in X for y in Y])
    dxx = np.mean([np.linalg.norm(a - b) for i, a in enumerate(X) for j, b in enumerate(X) if i != j])
    dyy = np.mean([np.linalg.norm(a - b) for i, a in enumerate(Y) for j, b in enumerate(Y) if i != j])
    assert energy_distance(X, Y) == pytest.approx(2 * dxy - dxx - dyy, rel=1e-10)


def test_energy_distance_zero_for_same_distribution_and_positive_for_shift():
    rng = np.random.default_rng(2)
    X, Y = rng.normal(size=(400, 10)), rng.normal(size=(400, 10))
    assert abs(energy_distance(X, Y)) < 0.05
    assert energy_distance(X, Y + 1.0) > 1.0


def test_etest_calibration_and_power():
    rng = np.random.default_rng(3)
    emb = np.vstack([rng.normal(size=(300, 10)),            # control
                     rng.normal(size=(100, 10)),            # null perturbation
                     rng.normal(size=(100, 10)) + 0.8])     # real shift
    labels = pd.Series(["control"] * 300 + ["nullP"] * 100 + ["shiftP"] * 100)
    res = edistance_to_control(emb, labels, "control", max_cells=150, n_perm=300, seed=0).set_index("perturbation")
    assert res.loc["shiftP", "p_value"] < 0.01
    assert res.loc["nullP", "p_value"] > 0.05
    assert res.loc["shiftP", "edistance"] > res.loc["nullP", "edistance"]
    assert res.loc["shiftP", "n_cells_used"] == 100


# ----------------------------------------------------------------------------- normalisation / QC
def test_clr_is_centred_along_the_requested_axis():
    rng = np.random.default_rng(4)
    X = rng.poisson(20, size=(50, 8))
    assert np.allclose(clr(X, axis=0).mean(axis=0), 0)      # per protein across cells
    assert np.allclose(clr(X, axis=1).mean(axis=1), 0)      # per cell across proteins


def test_mad_outlier_flags_tails_and_ignores_constant_input():
    x = np.r_[np.zeros(100), 50.0, -50.0] + np.linspace(0, 1, 102)
    flags = mad_outlier(x, 5)
    assert flags[-2] and flags[-1] and flags[:100].sum() == 0
    assert mad_outlier(np.ones(20), 3).sum() == 0          # MAD = 0 -> nothing flagged, by design
    upper = mad_outlier(x, 5, side="upper")
    assert upper[-2] and not upper[-1]


# ----------------------------------------------------------------------------- pseudobulk
def _toy_adata():
    X = sp.csr_matrix(np.arange(60, dtype=np.int32).reshape(12, 5))
    obs = pd.DataFrame({"pert": ["A"] * 6 + ["control"] * 6, "guide": ["g1", "g1", "g1", "g2", "g2", "g2"] * 2},
                       index=[f"c{i}" for i in range(12)])
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"gene{j}" for j in range(5)]))
    a.layers["counts"] = a.X.copy()
    return a


def test_pseudobulk_sums_by_replicate():
    a = _toy_adata()
    mat, meta = make_pseudobulk(a, "pert", "guide", n_pseudo_reps=3, min_cells_per_rep=1)
    assert mat.shape == (4, 5)
    manual = np.asarray(a.X[[0, 1, 2]].sum(axis=0)).ravel()
    assert np.array_equal(mat.loc["A__g1"].to_numpy(), manual)
    assert (meta["n_cells"] == 3).all() and meta.loc["A__g1", "mode"].startswith("replicate_col")


def test_pseudobulk_random_splits_and_min_cells():
    a = _toy_adata()
    mat, meta = make_pseudobulk(a, "pert", "guide", 3, min_cells_per_rep=1, splits_per_replicate=3)
    assert mat.shape[0] == 12 and "random splits" in meta["mode"].iloc[0]
    mat2, _ = make_pseudobulk(a, "pert", None, n_pseudo_reps=2, min_cells_per_rep=3)
    assert mat2.shape[0] == 4
    with pytest.raises(ValueError):
        make_pseudobulk(a, "pert", None, n_pseudo_reps=2, min_cells_per_rep=100)


# ----------------------------------------------------------------------------- end to end (small)
def test_pipeline_recovers_planted_structure(tmp_path):
    # no doublets / damaged cells here: this test skips QC, and a chance excess of doublets in a
    # 60-cell group is a real distribution shift the E-test would (correctly) pick up
    truth = make_synthetic_pair(tmp_path, n_genes=600, n_proteins=8, n_perturbations=6, n_null_perturbations=2,
                                cells_per_perturbation=60, n_control=200, doublet_rate=0.0, damaged_rate=0.0, seed=5)
    from scpert.de import protein_wilcoxon, pseudobulk_deseq2
    from scpert.io import load_pair
    from scpert.normalize import normalize_protein, normalize_rna
    rna, prot = load_pair(tmp_path / "synthetic_RNA.h5ad", tmp_path / "synthetic_protein.h5ad")
    assert rna.n_obs == prot.n_obs == truth["n_cells"]
    rna = normalize_rna(rna, n_hvg=300, hvg_flavor="seurat_v3", n_pcs=20, n_neighbors=10, umap=False, batch_col=None, seed=0)
    prot = normalize_protein(prot)
    etab = edistance_to_control(rna.obsm["X_pca"], rna.obs["perturbation"], "control", 100, 500, 0).set_index("perturbation")
    real = [p for p in etab.index if p not in truth["null_perturbations"]]
    for p in truth["null_perturbations"]:
        assert etab.loc[p, "p_value"] > 0.002                       # never the most extreme of 500 permutations
        assert etab.loc[p, "edistance"] < etab.loc[real, "edistance"].min()
    assert (etab.loc[real, "p_adj_bh"] < 0.05).mean() >= 0.75
    counts, meta = make_pseudobulk(rna, "perturbation", "guide_id", 3, 10)
    de = pseudobulk_deseq2(counts, meta, "control")
    for p in truth["null_perturbations"]:
        assert (de.loc[de["perturbation"] == p, "padj"] < 0.05).sum() <= 2
    ptab = protein_wilcoxon(prot, "perturbation", "control")
    sig = ptab[ptab["p_adj_bh"] < 0.01]
    for gene, protein in truth["coupled_proteins"].items():
        assert ((sig["perturbation"] == gene) & (sig["protein"] == protein)).any()


# ----------------------------------------------------------------------------- mixscape
def test_mixture_with_fixed_control_component_separates_two_populations():
    from scpert.mixscape import _fit_mixture_fixed_control
    rng = np.random.default_rng(6)
    x = np.r_[rng.normal(0, 1, 60), rng.normal(5, 1, 140)]       # 60 NP-like, 140 KO-like
    post = _fit_mixture_fixed_control(x, ctrl_mean=0.0, ctrl_sd=1.0, init_ko_mean=4.0)
    assert (post[:60] > 0.5).mean() < 0.1
    assert (post[60:] > 0.5).mean() > 0.95


def test_mixscape_recovers_escaping_cells_and_flags_null_perturbations(tmp_path):
    from scpert.io import load_pair
    from scpert.mixscape import mixscape_classify, perturbation_signature
    from scpert.normalize import normalize_rna
    truth = make_synthetic_pair(tmp_path, n_genes=600, n_proteins=6, n_perturbations=6, n_null_perturbations=2,
                                cells_per_perturbation=80, n_control=250, escape_rate=0.25, seed=8)
    rna, _ = load_pair(tmp_path / "synthetic_RNA.h5ad", tmp_path / "synthetic_protein.h5ad")
    rna = normalize_rna(rna, n_hvg=300, hvg_flavor="seurat_v3", n_pcs=20, n_neighbors=10, umap=False, batch_col=None, seed=0)
    rna = perturbation_signature(rna, "perturbation", "control", None, n_neighbors=20, n_dims=20)
    summ = mixscape_classify(rna, "perturbation", "control").set_index("perturbation")
    for p in truth["null_perturbations"]:
        assert summ.loc[p, "status"] == "no detectable perturbation" and summ.loc[p, "n_ko"] == 0
    o = rna.obs
    real = o[(o["perturbation"] != "control") & ~o["perturbation"].isin(truth["null_perturbations"])]
    pred_np = (real["mixscape_class_global"] == "NP").to_numpy()
    esc = real["truth_escaped"].to_numpy()
    sensitivity = (pred_np & esc).sum() / esc.sum()
    specificity = (~pred_np & ~esc).sum() / (~esc).sum()
    assert sensitivity > 0.8 and specificity > 0.9
