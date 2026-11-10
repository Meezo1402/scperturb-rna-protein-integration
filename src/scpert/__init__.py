"""scpert: QC, normalisation, perturbation-effect sizing, differential expression and
pseudobulk export for paired RNA + surface-protein single-cell perturbation data
(scPerturb-harmonised AnnData files).

Stages map one-to-one onto the scripts in ``scripts/``:

    io          -> load and align the RNA and protein AnnData pair
    qc          -> MAD-based cell filtering, optional doublet detection, gene filtering
    normalize   -> shifted-log RNA, HVGs, PCA, neighbours/UMAP; CLR protein
    mixscape    -> perturbation signature and KO / NP classification (Papalexi et al. 2021)
    edist       -> energy distance of each perturbation to control (Peidli et al. 2024) + E-test
    pseudobulk  -> sum counts per perturbation x replicate
    de          -> pseudobulk DESeq2 (PyDESeq2), cell-level Wilcoxon, protein Wilcoxon
    export      -> pseudobulk RNA / protein matrices + labels for DIABLO (mixOmics, R)
"""

__version__ = "0.1.0"
