# =============================================================================
# utils.R — Logging, seeding, and common helpers
# =============================================================================

#' Set random seed for reproducibility and optionally report it.
#' @param seed Integer. Default from config or 348.
#' @param verbose If TRUE, print the seed used.
set_pipeline_seed <- function(seed = 348L, verbose = TRUE) {
  set.seed(seed)
  if (verbose) {
    log_msg("Random seed set to ", seed, " for reproducibility.")
  }
  invisible(seed)
}

#' Log a message with optional timestamp and prefix.
#' @param ... Passed to paste0().
#' @param timestamp If TRUE, prepend ISO timestamp.
log_msg <- function(..., timestamp = TRUE) {
  msg <- paste0(...)
  if (timestamp) {
    msg <- paste0("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ", msg)
  }
  message(msg)
}

#' Load pipeline configuration (sources config if not already in environment).
#' Returns the config list from config.R or a default list.
get_config <- function(config_path = "config.R") {
  def <- default_config()
  if (!file.exists(config_path)) return(def)
  env <- new.env()
  source(config_path, local = env)
  if (!exists("config", envir = env)) return(def)
  for (n in names(env$config)) def[[n]] <- env$config[[n]]
  def
}

#' Default configuration when no config.R is present.
default_config <- function() {
  list(
    seed = 348L,
    data_dir = "data",
    train_file = "train.csv",
    test_file = "test.csv",
    results_dir = "results",
    submission_dir = "output/submission",
    n_folds = 10L,
    cv_repeats = 1L,
    tune_method = "bayes",  # "bayes" or "grid"
    bayes_initial = 10L,
    bayes_iter = 30L,
    use_smote = FALSE,
    smote_over_ratio = 1,
    pca_threshold = 0.95,
    rare_threshold = 0.01,
    # Model-specific (overridden per model where needed)
    logreg_penalty_range = c(-10, 0),
    logreg_mixture_range = c(0, 1),
    rf_mtry_range = c(1L, 50L),
    rf_min_n_range = c(2L, 20L),
    rf_trees = 500L,
    knn_neighbors_range = c(1L, 20L),
    nb_laplace_range = c(0, 5),
    mlp_hidden_units_range = c(1L, 10L),
    mlp_penalty_range = c(-5, 0),
    mlp_epochs_range = c(50L, 200L),
    svm_cost_range = c(-6, 4)
  )
}

#' Null coalescing for optional config keys.
'%||%' <- function(x, y) if (is.null(x)) y else x
