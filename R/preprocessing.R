# =============================================================================
# preprocessing.R — Shared feature engineering and recipes
# =============================================================================
# Recipe types:
# - "logit": For penalized/logistic regression: freq encoding (done in data load),
#   step_other, target encoding, dummy, normalize. No PCA.
# - "standard": For SVM/KNN/NB/MLP: step_other, dummy, normalize, optional PCA.
# - "tree": For random forest: step_other, dummy only (no normalization/PCA).
# =============================================================================

#' Build a recipe for linear/penalized models (logistic regression).
#'
#' Expects train_data to already have frequency encoding if desired.
#' Steps: numeric->factor, step_other, target encoding, dummy, normalize.
#'
#' @param train_data Training tibble with ACTION and predictors.
#' @param config List with \code{rare_threshold} (default 0.01).
#' @return A recipes::recipe object.
make_logit_recipe <- function(train_data, config = list()) {
  thresh <- config$rare_threshold %||% 0.01
  recipes::recipe(ACTION ~ ., data = train_data) %>%
    recipes::step_mutate_at(recipes::all_numeric_predictors(), fn = factor) %>%
    recipes::step_other(recipes::all_nominal_predictors(), threshold = thresh) %>%
    embed::step_lencode_mixed(recipes::all_nominal_predictors(), outcome = dplyr::vars(ACTION)) %>%
    recipes::step_dummy(recipes::all_nominal_predictors(), one_hot = FALSE) %>%
    recipes::step_zv(recipes::all_predictors()) %>%
    recipes::step_normalize(recipes::all_predictors())
}

#' Build a standard recipe (dummy, normalize, optional PCA and SMOTE).
#'
#' Used by SVM, KNN, Naive Bayes, MLP. Optionally applies PCA (e.g. 0.95 variance)
#' and/or SMOTE for class balancing.
#'
#' @param train_data Training tibble.
#' @param config List with \code{rare_threshold}, \code{use_smote}, \code{pca_threshold} (NULL = no PCA).
#' @return A recipes::recipe object.
make_standard_recipe <- function(train_data, config = list()) {
  thresh <- config$rare_threshold %||% 0.01
  r <- recipes::recipe(ACTION ~ ., data = train_data) %>%
    recipes::step_mutate_at(recipes::all_numeric_predictors(), fn = factor) %>%
    recipes::step_other(recipes::all_nominal_predictors(), threshold = thresh) %>%
    recipes::step_dummy(recipes::all_nominal_predictors()) %>%
    recipes::step_normalize(recipes::all_predictors())

  if (!is.null(config$pca_threshold)) {
    r <- r %>% recipes::step_pca(recipes::all_numeric_predictors(), threshold = config$pca_threshold)
  }
  if (isTRUE(config$use_smote)) {
    r <- r %>% themis::step_smote(ACTION, over_ratio = config$smote_over_ratio %||% 1)
  }
  r
}

#' Build a tree-friendly recipe (no normalization, no PCA).
#'
#' Used by random forest. Steps: numeric->factor, step_other, dummy.
#'
#' @param train_data Training tibble.
#' @param config List with \code{rare_threshold}. \code{use_smote} optional.
make_tree_recipe <- function(train_data, config = list()) {
  thresh <- config$rare_threshold %||% 0.01
  r <- recipes::recipe(ACTION ~ ., data = train_data) %>%
    recipes::step_mutate_at(recipes::all_numeric_predictors(), fn = factor) %>%
    recipes::step_other(recipes::all_nominal_predictors(), threshold = thresh) %>%
    recipes::step_dummy(recipes::all_nominal_predictors())
  if (isTRUE(config$use_smote)) {
    r <- r %>% themis::step_smote(ACTION, over_ratio = config$smote_over_ratio %||% 1)
  }
  r
}

#' Return the appropriate recipe for a model type.
#'
#' @param model_name One of "logreg", "penalized_logreg", "rf", "knn", "nb", "svm", "mlp".
#' @param train_data Training data.
#' @param config Config list.
get_recipe_for_model <- function(model_name, train_data, config) {
  switch(
    model_name,
    logreg = ,
    penalized_logreg = make_logit_recipe(train_data, config),
    rf = make_tree_recipe(train_data, config),
    knn = ,
    nb = ,
    svm_linear = ,
    svm = ,
    mlp = make_standard_recipe(train_data, config),
    stop("Unknown model: ", model_name)
  )
}
