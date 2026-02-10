# =============================================================================
# models.R — Model specs and workflows (tidymodels)
# =============================================================================
# Supported: penalized_logreg, logreg, rf, knn, nb, svm_linear, mlp
# =============================================================================

#' Get a parsnip model spec for the given model name.
#'
#' @param model_name One of penalized_logreg, logreg, rf, knn, nb, svm_linear, mlp.
#' @param config Config list for hyperparameter ranges.
#' @return A parsnip model spec (not yet fitted).
get_model_spec <- function(model_name, config) {
  switch(
    model_name,
    penalized_logreg = {
      parsnip::logistic_reg(
        penalty = tune::tune(),
        mixture = tune::tune()
      ) %>%
        parsnip::set_engine("glmnet") %>%
        parsnip::set_mode("classification")
    },
    logreg = {
      parsnip::logistic_reg(
        penalty = tune::tune(),
        mixture = tune::tune()
      ) %>%
        parsnip::set_engine("glmnet") %>%
        parsnip::set_mode("classification")
    },
    rf = {
      parsnip::rand_forest(
        mtry = tune::tune(),
        min_n = tune::tune(),
        trees = as.integer(config$rf_trees %||% 500L)
      ) %>%
        parsnip::set_engine("ranger") %>%
        parsnip::set_mode("classification")
    },
    knn = {
      parsnip::nearest_neighbor(
        neighbors = tune::tune(),
        weight_func = "rectangular",
        dist_power = 2
      ) %>%
        parsnip::set_engine("kknn") %>%
        parsnip::set_mode("classification")
    },
    nb = {
      parsnip::naive_Bayes(Laplace = tune::tune()) %>%
        parsnip::set_engine("naivebayes") %>%
        parsnip::set_mode("classification")
    },
    svm_linear = {
      parsnip::svm_linear(cost = tune::tune()) %>%
        parsnip::set_engine("kernlab", cache = 500, maxiter = 2000, tol = 0.001) %>%
        parsnip::set_mode("classification")
    },
    mlp = {
      parsnip::mlp(
        hidden_units = tune::tune(),
        penalty = tune::tune(),
        epochs = tune::tune()
      ) %>%
        parsnip::set_engine("nnet", MaxNWts = 5000) %>%
        parsnip::set_mode("classification")
    },
    stop("Unknown model: ", model_name)
  )
}

#' Get tuning parameters (dials) for Bayesian or grid search.
#'
#' @param model_name Model name.
#' @param config Config list.
#' @return A dials::parameters object or NULL for grid (use default ranges).
get_tune_params <- function(model_name, config) {
  switch(
    model_name,
    penalized_logreg = ,
    logreg = dials::parameters(
      dials::penalty(range = config$logreg_penalty_range %||% c(-10, 0)),
      dials::mixture(range = config$logreg_mixture_range %||% c(0, 1))
    ),
    rf = dials::parameters(
      dials::mtry(range = config$rf_mtry_range %||% c(1L, 50L)),
      dials::min_n(range = config$rf_min_n_range %||% c(2L, 20L))
    ),
    knn = dials::parameters(
      dials::neighbors(range = config$knn_neighbors_range %||% c(1L, 20L))
    ),
    nb = dials::parameters(
      dials::Laplace(range = config$nb_laplace_range %||% c(0, 5))
    ),
    svm_linear = dials::parameters(
      dials::cost(range = config$svm_cost_range %||% c(-6, 4))
    ),
    mlp = dials::parameters(
      dials::hidden_units(range = config$mlp_hidden_units_range %||% c(1L, 10L)),
      dials::penalty(range = config$mlp_penalty_range %||% c(-5, 0)),
      dials::epochs(range = config$mlp_epochs_range %||% c(50L, 200L))
    ),
    NULL
  )
}

#' Build a workflow for the given model and recipe.
#'
#' @param model_name Model name.
#' @param recipe A recipes::recipe (e.g. from get_recipe_for_model).
#' @param config Config list.
#' @return A workflows::workflow with recipe and model attached.
get_workflow <- function(model_name, recipe, config) {
  spec <- get_model_spec(model_name, config)
  workflows::workflow() %>%
    workflows::add_recipe(recipe) %>%
    workflows::add_model(spec)
}

#' List of model names that use frequency encoding in the data step.
models_using_freq_encoding <- function() {
  c("logreg", "penalized_logreg")
}
