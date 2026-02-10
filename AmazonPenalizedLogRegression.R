# =============================================================================
# Amazon Employee Access — Penalized Logistic Regression (grid + SMOTE variant)
# =============================================================================
# Thin wrapper around the unified pipeline. Uses config (e.g. use_smote, tune_method).
# Output: results/penalized_logreg_submission.csv and PenLogRegModelSubmission.csv.
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

model_name <- "penalized_logreg"
config <- get_config()
# Uses same pipeline as run_model.R -- penalized_logreg (logit recipe). Set use_smote in config if desired.
set_pipeline_seed(config$seed)

dat <- load_data(config, add_freq_encoding = TRUE)
recipe <- get_recipe_for_model(model_name, dat$train, config)
wf <- get_workflow(model_name, recipe, config)
param_info <- get_tune_params(model_name, config)
folds <- make_resamples(dat$train, config)
tune_results <- tune_model(wf, folds, config, param_info = param_info)
best_params <- select_best_params(tune_results, metric = "roc_auc")
final_fit <- fit_final_model(wf, best_params, dat$train)
pred_df <- predict_test(final_fit, dat$test, id_col = "id")
if (!dir.exists("results")) dir.create("results", recursive = TRUE)
write_kaggle_submission(pred_df, file.path("results", paste0(model_name, "_submission.csv")))
write_kaggle_submission(pred_df, "PenLogRegModelSubmission.csv")
log_msg("Done. Submission also written to PenLogRegModelSubmission.csv (legacy).")
