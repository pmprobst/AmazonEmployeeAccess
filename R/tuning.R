# =============================================================================
# tuning.R — Resampling, tuning, and result summarization
# =============================================================================

#' Create cross-validation resamples.
#'
#' @param train_data Training tibble.
#' @param config List with \code{n_folds}, \code{cv_repeats}, \code{seed}.
#' @return An rsample::rset (vfold_cv).
make_resamples <- function(train_data, config) {
  n_folds <- as.integer(config$n_folds %||% 10L)
  repeats <- as.integer(config$cv_repeats %||% 1L)
  folds <- rsample::vfold_cv(train_data, v = n_folds, repeats = repeats)
  log_msg("Created ", length(folds$splits), " CV folds (v = ", n_folds, ", repeats = ", repeats, ").")
  folds
}

#' Run tuning (Bayesian or grid) and return tune result.
#'
#' @param workflow A workflows::workflow (with recipe and model).
#' @param resamples Result of make_resamples().
#' @param config List with \code{tune_method}, \code{bayes_initial}, \code{bayes_iter}, etc.
#' @param param_info From get_tune_params(); used when tune_method == "bayes".
#' @param metric Primary metric (default roc_auc).
#' @return tune_results object.
tune_model <- function(workflow, resamples, config,
                       param_info = NULL,
                       metric = yardstick::metric_set(yardstick::roc_auc)) {
  method <- config$tune_method %||% "bayes"
  if (method == "bayes") {
    initial <- as.integer(config$bayes_initial %||% 10L)
    iter <- as.integer(config$bayes_iter %||% 30L)
    if (is.null(param_info)) {
      param_info <- dials::parameters()
    }
    workflow %>%
      tune::tune_bayes(
        resamples = resamples,
        initial = initial,
        iter = iter,
        param_info = param_info,
        metrics = metric,
        control = tune::control_bayes(verbose = TRUE)
      )
  } else {
    # Grid: extract param_info and build grid_regular
    if (is.null(param_info)) {
      grid <- dials::grid_regular(dials::parameters(), levels = 3)
    } else {
      levels <- as.integer(config$grid_levels %||% 5L)
      grid <- dials::grid_regular(param_info, levels = levels)
    }
    workflow %>%
      tune::tune_grid(
        resamples = resamples,
        grid = grid,
        metrics = metric
      )
  }
}

#' Select best tuning parameters by metric.
#'
#' @param tune_results Result of tune_model().
#' @param metric Character, e.g. "roc_auc" or "accuracy".
#' @return One-row tibble of best parameters.
select_best_params <- function(tune_results, metric = "roc_auc") {
  tune::select_best(tune_results, metric = metric)
}

#' Summarize tuning results (best and top N).
#'
#' @param tune_results tune_results object.
#' @param metric Character.
#' @param n Top N combinations to show.
#' @return List with best_params, best_metrics, top_metrics.
summarize_tune_results <- function(tune_results, metric = "roc_auc", n = 10L) {
  best_params <- tune::select_best(tune_results, metric = metric)
  best_metrics <- tune::show_best(tune_results, metric = metric, n = 1L)
  top_metrics <- tune_results %>%
    tune::collect_metrics() %>%
    dplyr::arrange(dplyr::desc(mean)) %>%
    head(n)
  list(
    best_params = best_params,
    best_metrics = best_metrics,
    top_metrics = top_metrics
  )
}
