#!/usr/bin/env Rscript
# =============================================================================
# run_model.R — CLI entry point: train/tune a model and write a submission file
# =============================================================================
# Usage (from project root):
#   Rscript run_model.R --model penalized_logreg
#   Rscript run_model.R --model rf --config config.R --output_dir results
# =============================================================================

# Load packages (required for pipeline)
suppressPackageStartupMessages({
  library(tidymodels)
  library(embed)
  library(vroom)
  library(tune)
  library(dials)
  library(workflows)
  library(parsnip)
  library(recipes)
  library(rsample)
  library(yardstick)
  library(dplyr)
})

# Source pipeline modules (assume run from project root)
if (file.exists("R/utils.R")) {
  source("R/utils.R")
  source("R/data_loading.R")
  source("R/preprocessing.R")
  source("R/models.R")
  source("R/tuning.R")
  source("R/predict_submission.R")
} else {
  stop("Run this script from the project root (where R/ and data/ live).")
}

# Parse command-line arguments
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(
    model = "penalized_logreg",
    config = "config.R",
    output_dir = "results"
  )
  i <- 1L
  while (i <= length(args)) {
    if (args[i] == "--model" && i < length(args)) {
      out$model <- args[i + 1L]
      i <- i + 2L
    } else if (args[i] == "--config" && i < length(args)) {
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

# Valid model names
VALID_MODELS <- c("penalized_logreg", "logreg", "rf", "knn", "nb", "svm_linear", "mlp")

main <- function() {
  cli <- parse_args()
  model_name <- cli$model
  if (!model_name %in% VALID_MODELS) {
    stop("Invalid --model. Choose one of: ", paste(VALID_MODELS, collapse = ", "))
  }

  log_msg("=== Amazon Employee Access — run_model.R ===")
  log_msg("Model: ", model_name)
  log_msg("Config: ", cli$config)
  log_msg("Output dir: ", cli$output_dir)

  config <- get_config(cli$config)
  set_pipeline_seed(config$seed)

  # Load data (with frequency encoding for logreg-type models)
  use_freq <- model_name %in% models_using_freq_encoding()
  dat <- load_data(config, add_freq_encoding = use_freq)
  train_data <- dat$train
  test_data <- dat$test

  # Recipe and workflow
  recipe <- get_recipe_for_model(model_name, train_data, config)
  wf <- get_workflow(model_name, recipe, config)
  param_info <- get_tune_params(model_name, config)

  # Resampling and tuning
  folds <- make_resamples(train_data, config)
  log_msg("Tuning (method: ", config$tune_method, ")...")
  tune_results <- tune_model(wf, folds, config, param_info = param_info)

  summary_list <- summarize_tune_results(tune_results, metric = "roc_auc", n = 10L)
  log_msg("Best parameters:")
  print(summary_list$best_params)
  log_msg("Best ROC AUC:")
  print(summary_list$best_metrics)

  # Optionally save tune results to results/
  if (nchar(cli$output_dir) > 0L) {
    if (!dir.exists(cli$output_dir)) dir.create(cli$output_dir, recursive = TRUE)
    res_path <- file.path(cli$output_dir, paste0(model_name, "_tune_results.rds"))
    saveRDS(tune_results, res_path)
    log_msg("Tune results saved: ", res_path)
  }

  # Final fit and submission
  best_params <- select_best_params(tune_results, metric = "roc_auc")
  final_fit <- fit_final_model(wf, best_params, train_data)
  pred_df <- predict_test(final_fit, test_data, id_col = "id")
  submission_path <- file.path(cli$output_dir, paste0(model_name, "_submission.csv"))
  write_kaggle_submission(pred_df, submission_path)

  log_msg("=== Done ===")
}

main()
