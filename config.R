# =============================================================================
# config.R — Central configuration for the ML pipeline
# =============================================================================
# Override any defaults here. Omit entries to use defaults from R/utils.R.
# =============================================================================

config <- list(
  seed = 348L,
  data_dir = "data",
  train_file = "train.csv",
  test_file = "test.csv",
  results_dir = "results",
  n_folds = 10L,
  cv_repeats = 1L,
  tune_method = "bayes",
  bayes_initial = 10L,
  bayes_iter = 30L,
  use_smote = FALSE,
  pca_threshold = 0.95,
  rare_threshold = 0.01,
  # Model-specific
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
