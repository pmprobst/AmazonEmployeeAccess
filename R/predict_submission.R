# =============================================================================
# predict_submission.R — Fit final model, predict, write Kaggle submission
# =============================================================================

#' Fit the workflow with best parameters on full training data.
#'
#' @param workflow A workflows::workflow.
#' @param best_params One-row tibble from select_best_params().
#' @param train_data Training tibble.
#' @return Fitted workflow.
fit_final_model <- function(workflow, best_params, train_data) {
  wf_final <- tune::finalize_workflow(workflow, best_params)
  fit(wf_final, data = train_data)
}

#' Predict probabilities for test data (probability of class 1).
#'
#' @param fitted_workflow Fitted workflow from fit_final_model().
#' @param test_data Test tibble (must have same predictor structure).
#' @param id_col Name of ID column in test_data (default "id").
#' @return Tibble with id_col and Action (probability of ACTION == 1).
predict_test <- function(fitted_workflow, test_data, id_col = "id") {
  pred <- predict(fitted_workflow, new_data = test_data, type = "prob")
  pred <- pred %>%
    dplyr::select(-.pred_0) %>%
    dplyr::rename(Action = .pred_1)
  ids <- test_data %>% dplyr::select(dplyr::all_of(id_col))
  out <- dplyr::bind_cols(ids, pred)
  # Kaggle sample format uses "Id" (capital I)
  if (id_col == "id") {
    out <- out %>% dplyr::rename(Id = id)
  }
  out
}

#' Write submission tibble to CSV (Id, Action).
#'
#' @param pred_df Tibble with Id and Action (from predict_test).
#' @param path File path for CSV.
write_kaggle_submission <- function(pred_df, path) {
  dir <- dirname(path)
  if (nchar(dir) > 0L && !dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
  vroom::vroom_write(pred_df, path, delim = ",")
  log_msg("Submission written: ", path)
  invisible(path)
}
