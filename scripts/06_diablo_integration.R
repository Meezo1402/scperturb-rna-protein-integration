#!/usr/bin/env Rscript
# 06_diablo_integration.R -- supervised RNA + protein integration with DIABLO (mixOmics)
#
#   Rscript scripts/06_diablo_integration.R --input results/<dataset>/diablo_input \
#           --out results/<dataset>/diablo [--design 0.1] [--ncomp-max 5] [--folds 5] \
#           [--nrepeat 10] [--seed 0] [--test-mode guide|random|none]
#
# Input: the three CSVs written by the Python `export` stage (pseudobulk RNA log-CPM,
# pseudobulk mean protein CLR, labels). Output: tuned DIABLO model, selected features
# per block and component, held-out predictions, and the standard mixOmics plots.
#
# Why DIABLO here: it is a supervised multi-block sparse PLS-DA (Singh et al. 2019). It
# asks which RNA features and which surface proteins co-vary with each other AND with
# the perturbation label, and it selects a sparse signature per block. It works at the
# level of samples with a class label, which is why the Python side aggregates cells
# into pseudobulk samples first. Cell-level alternatives (WNN, totalVI, MOFA+) answer a
# different question (a joint embedding) and are not run here.
#
# Design matrix: 0.1 between blocks is the mixOmics recommendation when discrimination
# matters more than block-to-block correlation (full design = 1 maximises correlation at
# the expense of classification). It is a parameter, not a constant.
#
# Held-out evaluation: when the pseudobulk samples are nested in real replicates (e.g.
# guides), one whole replicate per class is held out, so the test set never shares a
# guide with the training set. With pure pseudo-replicates no clean holdout exists and
# the random split is labelled optimistic in the output.

suppressPackageStartupMessages({
  ok <- requireNamespace("mixOmics", quietly = TRUE)
})
if (!ok) stop("mixOmics is required: BiocManager::install('mixOmics')")
suppressPackageStartupMessages(library(mixOmics))

# ------------------------------------------------------------------ arguments
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) return(default)
  args[i + 1]
}
in_dir    <- get_arg("--input", NULL)
out_dir   <- get_arg("--out", NULL)
design_v  <- as.numeric(get_arg("--design", "0.1"))
ncomp_max <- as.integer(get_arg("--ncomp-max", "5"))
folds_cfg <- as.integer(get_arg("--folds", "5"))
nrepeat   <- as.integer(get_arg("--nrepeat", "10"))
seed      <- as.integer(get_arg("--seed", "0"))
test_mode <- get_arg("--test-mode", "auto")           # auto | guide | random | none
if (is.null(in_dir) || is.null(out_dir)) stop("--input and --out are required")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)
logm <- function(...) cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), sprintf(...)))

# ------------------------------------------------------------------ data
rna    <- as.matrix(read.csv(file.path(in_dir, "diablo_rna_logcpm.csv"), row.names = 1, check.names = FALSE))
prot   <- as.matrix(read.csv(file.path(in_dir, "diablo_protein_clr.csv"), row.names = 1, check.names = FALSE))
labels <- read.csv(file.path(in_dir, "diablo_labels.csv"), row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
stopifnot(identical(rownames(rna), rownames(prot)), identical(rownames(rna), rownames(labels)))
Y_all <- factor(labels$group)
logm("loaded %d pseudobulk samples, %d classes, %d RNA features, %d proteins", nrow(rna), nlevels(Y_all), ncol(rna), ncol(prot))
logm("replicate mode: %s", labels$mode[1])

# drop zero-variance features (DIABLO scales each block)
keep_rna  <- apply(rna, 2, stats::sd) > 0
keep_prot <- apply(prot, 2, stats::sd) > 0
rna <- rna[, keep_rna, drop = FALSE]; prot <- prot[, keep_prot, drop = FALSE]

# ------------------------------------------------------------------ train / test split
nested_in_real_reps <- grepl("replicate_col=", labels$mode[1]) && grepl("random splits", labels$mode[1])
if (test_mode == "auto") test_mode <- if (nested_in_real_reps) "guide" else if (grepl("replicate_col=", labels$mode[1])) "none" else "random"
base_rep <- if (nested_in_real_reps) sub("_s[0-9]+$", "", labels$replicate) else labels$replicate

test_idx <- integer(0)
if (test_mode == "guide") {
  for (cl in levels(Y_all)) {
    reps_cl <- unique(base_rep[Y_all == cl])
    if (length(reps_cl) >= 2) {
      held <- sample(reps_cl, 1)
      test_idx <- c(test_idx, which(Y_all == cl & base_rep == held))
    }
  }
  logm("holding out one whole replicate per class: %d test samples", length(test_idx))
} else if (test_mode == "random") {
  for (cl in levels(Y_all)) {
    idx <- which(Y_all == cl)
    n_test <- max(1, floor(0.25 * length(idx)))
    if (length(idx) - n_test >= 3) test_idx <- c(test_idx, sample(idx, n_test))
  }
  logm("random stratified holdout: %d test samples (OPTIMISTIC: test samples share cells' replicate structure with training)", length(test_idx))
} else {
  logm("no held-out evaluation (test-mode = none)")
}
train_idx <- setdiff(seq_len(nrow(rna)), test_idx)
X_train <- list(rna = rna[train_idx, , drop = FALSE], protein = prot[train_idx, , drop = FALSE])
Y_train <- droplevels(Y_all[train_idx])
if (nlevels(Y_train) < 2) stop("fewer than two classes in the training set")

design <- matrix(design_v, ncol = 2, nrow = 2, dimnames = list(names(X_train), names(X_train)))
diag(design) <- 0

# CV folds cannot exceed the smallest training class
min_class <- min(table(Y_train))
folds <- max(2L, min(folds_cfg, min_class))
ncomp_max <- max(1L, min(ncomp_max, nlevels(Y_train) - 1L, min(sapply(X_train, ncol)) - 1L))
logm("training on %d samples (%d classes, smallest class %d); %d-fold CV x %d repeats; up to %d components",
     length(train_idx), nlevels(Y_train), min_class, folds, nrepeat, ncomp_max)

# ------------------------------------------------------------------ number of components
ncomp <- ncomp_max
if (ncomp_max > 1 && min_class >= 2) {
  full <- block.splsda(X = X_train, Y = Y_train, ncomp = ncomp_max, design = design)
  perf_full <- perf(full, validation = "Mfold", folds = folds, nrepeat = nrepeat)
  wv  <- perf_full$WeightedVote.error.rate$centroids.dist      # rows: classes, Overall.ER, Overall.BER; cols: comp1..
  err <- wv["Overall.BER", ]
  write.csv(data.frame(ncomp = seq_along(err), BER_weighted_vote = as.numeric(err), ER_weighted_vote = as.numeric(wv["Overall.ER", ])),
            file.path(out_dir, "diablo_perf_ncomp.csv"), row.names = FALSE)
  ch <- tryCatch(perf_full$choice.ncomp$WeightedVote["Overall.BER", "centroids.dist"], error = function(e) NA)
  ncomp <- if (is.na(ch) || is.null(ch)) which.min(err) else as.integer(ch)
  ncomp <- max(1L, min(ncomp, ncomp_max))
  logm("chosen ncomp = %d (balanced error rate by component: %s)", ncomp, paste(round(err, 3), collapse = ", "))
}

# ------------------------------------------------------------------ keepX tuning
grid_rna  <- unique(pmin(c(5, 10, 20, 50, 100), ncol(X_train$rna)))
grid_prot <- unique(pmin(c(2, 3, 5, 8, ncol(X_train$protein)), ncol(X_train$protein)))
keepX <- list(rna = rep(min(20, ncol(X_train$rna)), ncomp), protein = rep(min(5, ncol(X_train$protein)), ncomp))
if (min_class >= 2) {
  tuned <- tryCatch(
    tune.block.splsda(X = X_train, Y = Y_train, ncomp = ncomp, test.keepX = list(rna = grid_rna, protein = grid_prot),
                      design = design, validation = "Mfold", folds = folds, nrepeat = max(1L, nrepeat %/% 2L),
                      dist = "centroids.dist"),
    error = function(e) { logm("tune.block.splsda failed (%s); using default keepX", conditionMessage(e)); NULL })
  if (!is.null(tuned)) {
    keepX <- tuned$choice.keepX
    logm("tuned keepX: rna = %s | protein = %s", paste(keepX$rna, collapse = ","), paste(keepX$protein, collapse = ","))
  }
}
json_txt <- paste0('{"ncomp": ', ncomp, ', "keepX_rna": [', paste(keepX$rna, collapse = ","),
                   '], "keepX_protein": [', paste(keepX$protein, collapse = ","), '], "design": ', design_v, "}")
writeLines(json_txt, file.path(out_dir, "diablo_tuned_parameters.json"))

# ------------------------------------------------------------------ final model
final <- block.splsda(X = X_train, Y = Y_train, ncomp = ncomp, keepX = keepX, design = design)
saveRDS(final, file.path(out_dir, "diablo_model.rds"))
for (k in seq_len(ncomp)) {
  for (b in names(X_train)) {
    sv <- selectVar(final, block = b, comp = k)[[b]]
    write.csv(data.frame(feature = sv$name, loading = sv$value[, 1]),
              file.path(out_dir, sprintf("diablo_selected_%s_comp%d.csv", b, k)), row.names = FALSE)
  }
}
cv_ber <- NA_real_
if (min_class >= 2) {
  perf_final <- perf(final, validation = "Mfold", folds = folds, nrepeat = nrepeat)
  cv_ber <- perf_final$WeightedVote.error.rate$centroids.dist["Overall.BER", ncomp]
}
logm("final model: %d components; repeated-CV weighted-vote BER = %s", ncomp, ifelse(is.na(cv_ber), "n/a", sprintf("%.3f", cv_ber)))

# ------------------------------------------------------------------ held-out evaluation
metrics <- data.frame(metric = c("cv_weighted_vote_BER", "n_train", "n_test", "test_mode"),
                      value = c(round(cv_ber, 4), length(train_idx), length(test_idx), test_mode))
if (length(test_idx) > 0) {
  X_test <- list(rna = rna[test_idx, , drop = FALSE], protein = prot[test_idx, , drop = FALSE])
  pr <- predict(final, newdata = X_test)
  pred <- pr$WeightedVote$centroids.dist[, ncomp]
  truth <- as.character(Y_all[test_idx])
  acc <- mean(pred == truth)
  per_class <- sapply(unique(truth), function(cl) mean(pred[truth == cl] == cl))
  ber <- 1 - mean(per_class)
  write.csv(data.frame(sample = rownames(rna)[test_idx], truth = truth, predicted = pred),
            file.path(out_dir, "diablo_test_predictions.csv"), row.names = FALSE)
  metrics <- rbind(metrics, data.frame(metric = c("test_accuracy", "test_BER"), value = c(round(acc, 4), round(ber, 4))))
  logm("held-out: accuracy %.3f, balanced error rate %.3f on %d samples (%s)", acc, ber, length(test_idx), test_mode)
}
write.csv(metrics, file.path(out_dir, "diablo_metrics.csv"), row.names = FALSE)

# ------------------------------------------------------------------ plots
# A cosmetic failure (e.g. no correlations above the circos cutoff) must not abort the run.
save_plot <- function(file, width, height, expr) {
  png(file.path(out_dir, file), width = width, height = height, res = 150)
  ok <- tryCatch({ force(expr); TRUE }, error = function(e) { logm("plot %s skipped: %s", file, conditionMessage(e)); FALSE })
  dev.off()
  if (!ok) unlink(file.path(out_dir, file))
}
save_plot("diablo_sample_plot.png", 1600, 800,
          plotIndiv(final, ind.names = FALSE, legend = TRUE, title = "DIABLO: pseudobulk samples by block"))
if (ncomp >= 2) {
  save_plot("diablo_variable_plot.png", 1000, 1000,
            plotVar(final, var.names = FALSE, legend = TRUE, title = "Correlation circle: RNA and protein features"))
  save_plot("diablo_circos.png", 1400, 1400,
            circosPlot(final, cutoff = 0.7, line = TRUE, size.labels = 0.9, size.variables = 0.6))
}
save_plot("diablo_block_correlation_comp1.png", 900, 900, plotDiablo(final, ncomp = 1))
save_plot("diablo_loadings_comp1.png", 1600, 900,
          plotLoadings(final, comp = 1, contrib = "max", method = "median", title = "Component 1 loadings (class with max median)"))

writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
logm("done -> %s", out_dir)
