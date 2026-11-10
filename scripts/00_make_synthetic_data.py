#!/usr/bin/env python3
"""Generate a small synthetic RNA + protein perturbation dataset in the scPerturb layout.

    python scripts/00_make_synthetic_data.py [out_dir] [--cells-per-perturbation N] [--seed S]

Defaults write to data/synthetic/. See scpert.synthetic for what is planted.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpert.synthetic import make_synthetic_pair  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out_dir", nargs="?", default=str(ROOT / "data" / "synthetic"))
ap.add_argument("--n-genes", type=int, default=1500)
ap.add_argument("--n-proteins", type=int, default=12)
ap.add_argument("--n-perturbations", type=int, default=12)
ap.add_argument("--n-null", type=int, default=3)
ap.add_argument("--cells-per-perturbation", type=int, default=80)
ap.add_argument("--n-control", type=int, default=400)
ap.add_argument("--escape-rate", type=float, default=0.2, help="fraction of cells per real perturbation with no effect")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

truth = make_synthetic_pair(a.out_dir, n_genes=a.n_genes, n_proteins=a.n_proteins, n_perturbations=a.n_perturbations,
                            n_null_perturbations=a.n_null, cells_per_perturbation=a.cells_per_perturbation,
                            n_control=a.n_control, escape_rate=a.escape_rate, seed=a.seed)
print(f"wrote synthetic pair to {a.out_dir}: {truth['n_cells']} cells, {a.n_perturbations} perturbations "
      f"({len(truth['null_perturbations'])} null), {truth['n_escaped']} escaping cells, "
      f"{truth['n_doublets']} doublets, {truth['n_damaged']} damaged cells")
