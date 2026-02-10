#!/usr/bin/env Rscript
# =============================================================================
# run_all_models.R — Run a set of models and collect metrics/submissions
# =============================================================================
# Usage: Rscript run_all_models.R [--config config.R] [--output_dir results]
# =============================================================================

suppressPackageStartupMessages({
  library(tidymodels)
  library(embed)
  library(vroom)
  library(tune)
  library(dials)
  library(workflows)
  library(dplyr)
})

if (file.exists("R/utils.R")) {
  source("R/utils.R")
  source("R/data_loading.R")
  source("R/preprocessing.R")
  source("R/models.R")
  source("R/tuning.R")
  source("R/predict_submission.R")
} else {
  stop("Run this script from the project root.")
}

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(config = "config.R", output_dir = "output/submission")
  i <- 1L
  while (i <= length(args)) {
    if (args[i] == "--config" && i < length(args)) {
      out$config <- args[i + 1L]
      i <- i + 2L
    } else if (args[i] == "--output_dir" && i < length(args)) {
      out$output_dir <- args[i + 1L]
      i <- i + 2L
    } else {
      i <- i + 1L
    }
  }
  out
}

# Subset of models to run (can be overridden; full set is slow)
DEFAULT_MODELS <- c("penalized_logreg", "rf", "knn", "svm_linear")

main <- function() {
  cli <- parse_args()
  config <- get_config(cli$config)
  set_pipeline_seed(config$seed)

  if (!dir.exists(cli$output_dir)) dir.create(cli$output_dir, recursive = TRUE)

  # Optional: pass models via env or use default
  models_to_run <- Sys.getenv("AMAZON_MODELS", "")
  if (nchar(models_to_run) > 0L) {
    models_to_run <- strsplit(models_to_run, ",")[[1L]]
  } else {
    models_to_run <- DEFAULT_MODELS
  }

  log_msg("Running models: ", paste(models_to_run, collapse = ", "))
  metrics_list <- list()

  for (model_name in models_to_run) {
    log_msg("=== Model: ", model_name, " ===")
    use_freq <- model_name %in% models_using_freq_encoding()
    dat <- load_data(config, add_freq_encoding = use_freq)
    recipe <- get_recipe_for_model(model_name, dat$train, config)
    wf <- get_workflow(model_name, recipe, config)
    param_info <- get_tune_params(model_name, config)
    folds <- make_resamples(dat$train, config)
    tune_results <- tune_model(wf, folds, config, param_info = param_info)
    best_params <- select_best_params(tune_results, metric = "roc_auc")
    summary_list <- summarize_tune_results(tune_results, metric = "roc_auc", n = 1L)
    metrics_list[[model_name]] <- summary_list$best_metrics %>%
      dplyr::mutate(model = model_name)
    final_fit <- fit_final_model(wf, best_params, dat$train)
    pred_df <- predict_test(final_fit, dat$test, id_col = "id")
    write_kaggle_submission(pred_df, file.path(cli$output_dir, paste0(model_name, "_submission.csv")))
  }

  # Write combined metrics
  all_metrics <- dplyr::bind_rows(metrics_list)
  metrics_path <- file.path(cli$output_dir, "all_models_metrics.csv")
  vroom::vroom_write(all_metrics, metrics_path, delim = ",")
  log_msg("Metrics written: ", metrics_path)
  log_msg("=== Done ===")
}

main()
