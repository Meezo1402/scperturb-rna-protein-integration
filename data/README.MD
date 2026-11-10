# Data

Nothing in this folder is committed. Two kinds of input live here.

## `data/scperturb/` (download)

### Primary dataset: Papalexi et al. 2021, ECCITE-seq (THP-1, PD-L1 regulators)

A pooled CRISPR screen read out with the transcriptome, the guide, and a small panel of
surface proteins including PD-L1, in IFN-gamma-stimulated THP-1 cells (111 gRNAs
targeting PD-L1 regulators plus non-targeting controls; ~20.7k cells). The screen is
about the regulation of a protein, which is what makes the paired RNA + protein readout
the point rather than a bonus. Two ways to obtain it:

**A. scPerturb release** (harmonised, raw counts, `obs.perturbation` with `control`):
download `PapalexiSatija2021_eccite_RNA.h5ad` and `PapalexiSatija2021_eccite_protein.h5ad`
from the Zenodo record linked at https://projects.sanderlab.org/scperturb/ into this
folder. Config: `config/papalexi2021_eccite_seq.yaml`.

**B. pertpy loader**: `pertpy.data.papalexi_2021()` returns a MuData with `rna` and `adt`
modalities (20,729 cells; 18,649 genes; 4 proteins) and obs columns `perturbation`,
`gene_target`, `NT`, `replicate`, `guide_ID`. Write the two modalities to h5ad files
and set `perturbation_col: gene_target`, `control_label: NT`, `replicate_col: guide_ID`,
`batch_col: replicate` in the config.

In either case, open the files once and check `adata.obs.columns` and the protein
`var_names` before running; the config marks every name that has to be verified.

### Secondary dataset: Frangieh et al. 2021, Perturb-CITE-seq (melanoma)

`FrangiehIzar2021_RNA.h5ad` + `FrangiehIzar2021_protein.h5ad` from the same Zenodo
record; ~218k cells, 248 perturbations, 20 surface proteins, three culture conditions.
Config: `config/frangieh2021_perturb_cite_seq.yaml`. Set `batch_col` to the condition
column, expect the E-test and DE stages to take much longer, and budget 32 GB of RAM.

## `data/synthetic/` (generated)

`python scripts/00_make_synthetic_data.py` writes a small simulated RNA + protein pair in
the same layout, with planted effects recorded in `truth.json`: differentially expressed
genes per perturbation, null perturbations, escaping cells, gene-protein couplings,
doublets and damaged cells. It exists so the whole pipeline can be run and checked for
recovery without a large download; see `src/scpert/synthetic.py` for what is simulated.
