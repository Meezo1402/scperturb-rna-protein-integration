#!/usr/bin/env python3
"""End-to-end driver for the Python stages.

    python scripts/run_pipeline.py --config config/smoke.yaml
    python scripts/run_pipeline.py --config config/papalexi2021_eccite_seq.yaml --stages qc,normalize,mixscape
    python scripts/run_pipeline.py --config config/papalexi2021_eccite_seq.yaml --stages edist,de,export

Stages (in order): qc, normalize, mixscape, edist, de, export. Later stages read the
processed AnnData written by `normalize` (and updated by `mixscape`), so they can be
re-run on their own. When mixscape.use_ko_only_downstream is set, edist / de / export
run on control + KO cells only. The R stage (scripts/06_diablo_integration.R) consumes
the `export` output.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scpert import __version__  # noqa: E402
from scpert.config import load_config  # noqa: E402
from scpert.de import cell_level_wilcoxon, protein_wilcoxon, pseudobulk_deseq2  # noqa: E402
from scpert.edist import edistance_to_control  # noqa: E402
from scpert.export import export_for_diablo  # noqa: E402
from scpert.de import rna_protein_concordance  # noqa: E402
from scpert.io import load_pair, restrict_perturbations  # noqa: E402
from scpert.mixscape import mixscape_classify, perturbation_signature  # noqa: E402
from scpert.normalize import normalize_protein, normalize_rna  # noqa: E402
from scpert.plotting import (concordance_figure, edist_figure, mixscape_figure, protein_heatmap, qc_figure,  # noqa: E402
                             umap_figure, volcano_figure)
from scpert.pseudobulk import make_pseudobulk  # noqa: E402
from scpert.qc import apply_cell_qc, apply_gene_and_group_filters  # noqa: E402

ALL_STAGES = ["qc", "normalize", "mixscape", "edist", "de", "export"]


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("scpert")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "pipeline.log", mode="a")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    args = ap.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        sys.exit(f"unknown stages: {sorted(unknown)}; choose from {ALL_STAGES}")

    cfg = load_config(args.config)
    ds, out_dir = cfg["dataset"], Path(cfg["output_dir"])
    log = setup_logging(out_dir)
    log.info("scpert %s | dataset %s | stages %s | output %s", __version__, ds["name"], stages, out_dir)
    t0 = time.time()
    processed_rna, processed_prot = out_dir / "processed_rna.h5ad", out_dir / "processed_protein.h5ad"
    rna = prot = None
    summary: dict = {"dataset": ds["name"], "stages": stages}

    # ---------------------------------------------------------------- qc + normalize
    if "qc" in stages or "normalize" in stages:
        rna, prot = load_pair(ds["rna_h5ad"], ds["protein_h5ad"], ds["perturbation_col"])
        rna, prot = restrict_perturbations(rna, prot, ds["perturbation_col"], ds["control_label"],
                                           ds["drop_multi_perturbation"], ds.get("nperts_col"))
        summary["cells_loaded"] = int(rna.n_obs)
        summary["genes_loaded"] = int(rna.n_vars)
        summary["proteins"] = int(prot.n_vars)
    if "qc" in stages:
        rna, prot, qc_summary, obs_before = apply_cell_qc(rna, prot, cfg["qc"], ds, cfg["seed"])
        qc_summary.to_csv(out_dir / "qc_summary.csv", index=False)
        obs_before.to_csv(out_dir / "qc_per_cell.csv.gz")
        if cfg["figures"]:
            qc_figure(obs_before, "qc_fail_", out_dir / "qc_cells.png")
        rna, prot = apply_gene_and_group_filters(rna, prot, cfg["qc"], ds)
        summary["qc"] = dict(zip(qc_summary["criterion"], map(int, qc_summary["cells_flagged"])))
        summary["cells_after_qc"], summary["genes_after_qc"] = int(rna.n_obs), int(rna.n_vars)
        summary["perturbations_after_qc"] = int(rna.obs[ds["perturbation_col"]].nunique() - 1)
    if "normalize" in stages:
        if "qc" not in stages:
            log.warning("running normalize without qc on the raw loaded data")
        n = cfg["normalize"]
        rna = normalize_rna(rna, n["n_hvg"], n["hvg_flavor"], n["n_pcs"], n["n_neighbors"], n["umap"],
                            ds.get("batch_col"), cfg["seed"])
        prot = normalize_protein(prot, n["clr_margin"])
        rna.write_h5ad(processed_rna)
        prot.write_h5ad(processed_prot)
        log.info("wrote %s and %s", processed_rna.name, processed_prot.name)

    # ---------------------------------------------------------------- downstream stages
    if any(s in stages for s in ("mixscape", "edist", "de", "export")) and rna is None:
        if not processed_rna.exists():
            sys.exit(f"{processed_rna} not found; run the qc,normalize stages first")
        rna, prot = ad.read_h5ad(processed_rna), ad.read_h5ad(processed_prot)
        log.info("loaded processed data: %d cells, %d genes, %d proteins", rna.n_obs, rna.n_vars, prot.n_vars)
    col, control = ds["perturbation_col"], ds["control_label"]

    # ---------------------------------------------------------------- mixscape
    mx = cfg["mixscape"]
    if "mixscape" in stages and mx["run"]:
        rna = perturbation_signature(rna, col, control, ds.get("batch_col"), mx["n_neighbors"], mx["n_dims"])
        ms = mixscape_classify(rna, col, control, min_de_genes=mx["min_de_genes"], iter_num=mx["iter_num"],
                               lfc_threshold=mx["lfc_threshold"], posterior_threshold=mx["posterior_threshold"])
        ms.to_csv(out_dir / "mixscape_summary.csv", index=False)
        del rna.layers["X_pert"]                      # dense; the obs labels are what downstream needs
        prot.obs["mixscape_class_global"] = rna.obs["mixscape_class_global"].to_numpy()
        rna.write_h5ad(processed_rna)
        prot.write_h5ad(processed_prot)
        summary["mixscape_ko_detected"] = int((ms["status"] == "KO detected").sum())
        summary["mixscape_tested"] = int(len(ms))
        summary["mixscape_np_cells_removed"] = int((rna.obs["mixscape_class_global"] == "NP").sum())
        if cfg["figures"]:
            mixscape_figure(ms, out_dir / "mixscape_ko_fraction.png")

    # downstream stages use control + KO cells when mixscape ran and the config asks for it
    rna_ds, prot_ds = rna, prot
    if mx["run"] and mx["use_ko_only_downstream"] and "mixscape_class_global" in rna.obs:
        keep = rna.obs["mixscape_class_global"].isin(["control", "KO"]).to_numpy()
        rna_ds, prot_ds = rna[keep].copy(), prot[keep].copy()
        dropped = rna_ds.obs[col].value_counts()
        dropped = sorted(set(rna.obs[col].astype(str)) - set(dropped.index.astype(str)))
        log.info("downstream analyses use %d control + KO cells (%d NP cells excluded%s)", rna_ds.n_obs,
                 rna.n_obs - rna_ds.n_obs, f"; perturbations with no KO cells dropped: {', '.join(dropped)}" if dropped else "")
        summary["downstream_cells"] = int(rna_ds.n_obs)
        for a in (rna_ds, prot_ds):
            a.obs[col] = pd.Categorical(a.obs[col].astype(str))

    edist_path = out_dir / "edistance.csv"
    if "edist" in stages:
        e = cfg["edist"]
        emb = rna_ds.obsm[e["obsm_key"]][:, : e["n_pcs"]]
        etab = edistance_to_control(emb, rna_ds.obs[col], control, e["max_cells_per_group"], e["n_perm"], e["seed"])
        etab.to_csv(edist_path, index=False)
        summary["edist_significant"] = int((etab["p_adj_bh"] < 0.05).sum())
        summary["edist_tested"] = int(len(etab))
        if cfg["figures"]:
            edist_figure(etab, out_dir / "edistance.png")
            umap_figure(rna_ds, col, control, etab["perturbation"].head(9).tolist(), out_dir / "umap_top_perturbations.png")

    if "de" in stages:
        de_dir = out_dir / "de"
        de_dir.mkdir(exist_ok=True)
        pb = cfg["pseudobulk"]
        counts, meta = make_pseudobulk(rna_ds, col, ds.get("replicate_col"), pb["n_pseudo_reps"], pb["min_cells_per_rep"],
                                       layer="counts", seed=cfg["seed"], aggregate="sum")
        counts.to_csv(de_dir / "pseudobulk_counts.csv")
        meta.to_csv(de_dir / "pseudobulk_meta.csv")
        if cfg["de"]["run_pseudobulk_deseq2"]:
            res = pseudobulk_deseq2(counts, meta, control, cfg["de"]["min_counts_per_gene"], cfg["de"]["n_jobs"])
            res.to_csv(de_dir / "de_pseudobulk_deseq2.csv", index=False)
            summary["de_pseudobulk_pairs_padj05"] = int((res["padj"] < 0.05).sum())
            summary["de_pseudobulk_perturbations_with_hits"] = int(res.loc[res["padj"] < 0.05, "perturbation"].nunique())
            if cfg["figures"]:
                top = (pd.read_csv(edist_path)["perturbation"].iloc[0] if edist_path.exists()
                       else res.loc[res["padj"] < 0.05, "perturbation"].value_counts().index[0] if (res["padj"] < 0.05).any()
                       else res["perturbation"].iloc[0])
                volcano_figure(res, top, de_dir / f"volcano_{top}.png")
        if cfg["de"]["run_cell_level_wilcoxon"]:
            wres = cell_level_wilcoxon(rna_ds, col, control)
            wres.to_csv(de_dir / "de_cell_level_wilcoxon.csv", index=False)
            summary["de_wilcoxon_pairs_bh05"] = int((wres["p_adj_bh"] < 0.05).sum())
        ptab = protein_wilcoxon(prot_ds, col, control)
        ptab.to_csv(de_dir / "de_protein_wilcoxon.csv", index=False)
        summary["protein_pairs_bh05"] = int((ptab["p_adj_bh"] < 0.05).sum())
        if cfg["figures"] and len(ptab):
            order = pd.read_csv(edist_path)["perturbation"].tolist() if edist_path.exists() else None
            protein_heatmap(ptab, de_dir / "protein_heatmap.png", order)
        pgm = cfg["concordance"]["protein_gene_map"]
        if pgm and cfg["de"]["run_pseudobulk_deseq2"]:
            conc = rna_protein_concordance(res, ptab, pgm)
            conc.to_csv(de_dir / "rna_protein_concordance.csv", index=False)
            summary["concordance_patterns"] = conc["pattern"].value_counts().to_dict() if len(conc) else {}
            if cfg["figures"]:
                concordance_figure(conc, de_dir / "rna_protein_concordance.png")

    if "export" in stages:
        if not edist_path.exists():
            sys.exit("export needs edistance.csv; run the edist stage first")
        etab = pd.read_csv(edist_path)
        x = cfg["export"]
        pb = cfg["pseudobulk"]
        info = export_for_diablo(rna_ds, prot_ds, etab, col, control, ds.get("replicate_col"), pb["n_pseudo_reps"],
                                 x.get("min_cells_per_sample", pb["min_cells_per_rep"]), x["n_rna_features"],
                                 x["n_top_perturbations"], out_dir / "diablo_input", cfg["seed"],
                                 splits_per_replicate=x.get("splits_per_replicate", 1))
        summary["diablo_export"] = info

    # ---------------------------------------------------------------- summary
    summary["runtime_min"] = round((time.time() - t0) / 60, 2)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    write_summary_md(summary, out_dir, cfg)
    log.info("done in %.1f min -> %s", summary["runtime_min"], out_dir)


def write_summary_md(s: dict, out_dir: Path, cfg: dict) -> None:
    lines = [f"# {s['dataset']}: pipeline summary", ""]
    if "cells_loaded" in s:
        lines += ["## Input", f"- cells with a single perturbation call: {s['cells_loaded']:,}",
                  f"- genes: {s['genes_loaded']:,}; proteins: {s['proteins']}", ""]
    if "qc" in s:
        lines += ["## QC (cells flagged per criterion; a cell can fail several)", "", "| criterion | cells |", "|---|---|"]
        lines += [f"| {k} | {v:,} |" for k, v in s["qc"].items()]
        lines += ["", f"After QC: {s['cells_after_qc']:,} cells, {s['genes_after_qc']:,} genes, "
                  f"{s['perturbations_after_qc']} perturbations with >= {cfg['qc']['min_cells_per_perturbation']} cells.", ""]
    if "mixscape_tested" in s:
        lines += ["## Escaping cells (mixscape)",
                  f"- KO detected in {s['mixscape_ko_detected']} of {s['mixscape_tested']} perturbations; "
                  f"{s['mixscape_np_cells_removed']:,} non-perturbed (NP) cells excluded from downstream analyses"
                  + (f"; {s['downstream_cells']:,} control + KO cells carried forward" if "downstream_cells" in s else ""), ""]
    if "edist_tested" in s:
        lines += ["## Perturbation strength (E-distance)",
                  f"- {s['edist_significant']} of {s['edist_tested']} perturbations differ from control (E-test, BH < 0.05, "
                  f"{cfg['edist']['n_perm']} permutations, <= {cfg['edist']['max_cells_per_group']} cells per group)", ""]
    if "de_pseudobulk_pairs_padj05" in s or "protein_pairs_bh05" in s:
        lines += ["## Differential expression"]
        if "de_pseudobulk_pairs_padj05" in s:
            lines += [f"- pseudobulk DESeq2: {s['de_pseudobulk_pairs_padj05']:,} gene x perturbation pairs at padj < 0.05 "
                      f"across {s['de_pseudobulk_perturbations_with_hits']} perturbations"]
        if "de_wilcoxon_pairs_bh05" in s:
            lines += [f"- cell-level Wilcoxon (pseudoreplicated, for ranking only): {s['de_wilcoxon_pairs_bh05']:,} pairs at BH < 0.05"]
        if "protein_pairs_bh05" in s:
            lines += [f"- surface proteins: {s['protein_pairs_bh05']} protein x perturbation pairs at BH < 0.05"]
        if s.get("concordance_patterns"):
            lines += ["- RNA vs protein (coding gene vs ADT, per perturbation): "
                      + ", ".join(f"{k} = {v}" for k, v in s["concordance_patterns"].items())]
        lines += [""]
    if "diablo_export" in s:
        d = s["diablo_export"]
        lines += ["## DIABLO input", f"- {d['n_samples']} pseudobulk samples, {d['n_classes']} classes, "
                  f"{d['n_rna_features']} RNA features, {d['n_proteins']} proteins -> `diablo_input/`",
                  "- next: `Rscript scripts/06_diablo_integration.R --input <this dir>/diablo_input --out <this dir>/diablo`", ""]
    lines += [f"Runtime: {s['runtime_min']} min."]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
