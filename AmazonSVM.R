# =============================================================================
# Amazon Employee Access — SVM Linear (unified pipeline)
# =============================================================================
# Thin wrapper. For full SVM comparison (linear/poly/RBF) use run_model.R or
# extend R/models.R. Output: output/submission/svm_linear_submission.csv and SVMSubmission.csv (legacy).
# =============================================================================

suppressPackageStartupMessages({
  library(tidymodels)
  library(embed)
  library(vroom)
  library(kernlab)
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

model_name <- "svm_linear"
config <- get_config()
set_pipeline_seed(config$seed)

dat <- load_data(config, add_freq_encoding = FALSE)
recipe <- get_recipe_for_model(model_name, dat$train, config)
wf <- get_workflow(model_name, recipe, config)
param_info <- get_tune_params(model_name, config)
folds <- make_resamples(dat$train, config)
tune_results <- tune_model(wf, folds, config, param_info = param_info)
best_params <- select_best_params(tune_results, metric = "roc_auc")
final_fit <- fit_final_model(wf, best_params, dat$train)
pred_df <- predict_test(final_fit, dat$test, id_col = "id")
sub_dir <- config$submission_dir %||% "output/submission"
if (!dir.exists(sub_dir)) dir.create(sub_dir, recursive = TRUE)
write_kaggle_submission(pred_df, file.path(sub_dir, "svm_linear_submission.csv"))
write_kaggle_submission(pred_df, "SVMSubmission.csv")
log_msg("Done. Submission also written to SVMSubmission.csv (legacy).")
