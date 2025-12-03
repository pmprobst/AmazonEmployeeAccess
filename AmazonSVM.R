#AmazonAnalysis
print("=== Starting Amazon Employee Access SVM Classification ===")
print(paste("Start time:", Sys.time()))

library(tidymodels)
library(embed)
library(vroom)
library(kernlab)
library(doParallel)
library(parallel)
library(tune)  # For control_bayes()

print("Libraries loaded successfully")

###############################################################################
# 0. Set up parallel processing
###############################################################################

# Detect number of available cores
num_cores <- parallel::detectCores()
print(paste("Detected", num_cores, "CPU cores"))

# Use all cores except 1 (leave one for system processes)
# For dedicated servers, you can use all cores by setting: cores_to_use <- num_cores
cores_to_use <- num_cores
print(paste("Setting up parallel processing with", cores_to_use, "cores"))
print(paste("Expected speedup: ~", cores_to_use, "x faster for CV tuning"))

# Create and register parallel backend
cl <- makePSOCKcluster(cores_to_use)
registerDoParallel(cl)

print("Parallel processing backend registered successfully")
print("Note: tune_bayes() will automatically use parallel processing")

set.seed(348)  # for reproducibility

###############################################################################
# 1. Read and prepare data
###############################################################################

# Read data in, set ACTION as a factor (binary classification outcome)
print("Loading training data...")
train_data <- vroom("data/train.csv") %>%
  mutate(ACTION = factor(ACTION))
print(paste("Training data loaded:", nrow(train_data), "rows,", ncol(train_data), "columns"))

print("Loading test data...")
test_data  <- vroom("data/test.csv")
print(paste("Test data loaded:", nrow(test_data), "rows,", ncol(test_data), "columns"))

###############################################################################
# 2. Preprocessing recipe with PCA
#
#    - Convert rare factor levels to "other"
#    - Dummy encode all nominal predictors
#    - Normalize numeric predictors
#    - Apply PCA with 95% variance threshold to speed up SVM fitting
#      by reducing dimensionality while retaining most information
###############################################################################

svm_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  # Use variance threshold to reduce dimension for speed
  # Retains enough components to explain 95% of variance
  step_pca(all_numeric_predictors(), threshold = 0.95)

###############################################################################
# 3. Resampling for model tuning
###############################################################################

print("Creating cross-validation folds...")
# Using 3-fold CV for faster Bayesian optimization
# Bayesian optimization is more efficient than grid search, so fewer folds are needed
folds <- vfold_cv(train_data, v = 3, repeats = 1)  
print(paste("Created", length(folds$splits), "folds (3-fold CV with 1 repeat)"))

###############################################################################
# 4. Define SVM model specifications
#
#    We will tune three kernels using Bayesian optimization:
#      - Linear
#      - Polynomial
#      - Radial (RBF)
#
#    Bayesian optimization intelligently explores the hyperparameter space,
#    focusing on promising regions, which is much faster than grid search.
#    For each kernel, we use 25 iterations to find optimal hyperparameters.
###############################################################################

# 4.1 Linear SVM
print("=== Starting Linear SVM Tuning ===")
print(paste("Linear SVM start time:", Sys.time()))

svm_linear_spec <- svm_linear(
  cost = tune()              # regularization strength
) %>%
  set_engine("kernlab", 
             cache = 500,      # maximum cache size for server execution
             maxiter = 2000,   # increase max iterations to prevent convergence warnings
             tol = 0.001) %>%  # tolerance for convergence
  set_mode("classification")

print("Creating linear SVM workflow...")
svm_linear_wf <- workflow() %>%
  add_recipe(svm_recipe) %>%
  add_model(svm_linear_spec)

print("Running linear SVM Bayesian optimization...")
print("Using 25 iterations to efficiently explore cost parameter space")
print("Cost range: log2 scale from -6 to 4 (0.015625 to 16)")
print("Parallel processing will be used automatically by tune_bayes()")
svm_linear_res <- svm_linear_wf %>%
  tune_bayes(
    resamples = folds,
    initial   = 5,              # Start with 5 random initial points
    iter      = 25,             # 25 Bayesian optimization iterations
    param_info = parameters(
      cost(range = c(-6, 4))    # cost range on log2 scale
    ),
    metrics   = metric_set(roc_auc),
    control   = control_bayes(
      parallel_over = "resamples",  # Parallelize over CV folds
      verbose = TRUE                 # Show progress
    )
  )

print("Linear SVM tuning complete")
print(paste("Linear SVM end time:", Sys.time()))

best_linear <- svm_linear_res %>%
  select_best(metric = "roc_auc")

best_linear_metrics <- svm_linear_res %>%
  show_best(metric = "roc_auc", n = 1)

print("Best linear SVM parameters:")
print(best_linear)
print("Best linear SVM ROC AUC:")
print(best_linear_metrics)

###############################################################################
# 4.2 Polynomial SVM
###############################################################################

print("=== Starting Polynomial SVM Tuning ===")
print(paste("Polynomial SVM start time:", Sys.time()))

svm_poly_spec <- svm_poly(
  cost          = tune(),
  degree        = tune(),      # tune degree (2, 3, 4, or 5)
  scale_factor  = tune()       # scaling of input features
) %>%
  set_engine("kernlab",
             cache = 500,      # maximum cache size for server execution
             maxiter = 2000,   # increase max iterations to prevent convergence warnings
             tol = 0.001) %>%  # tolerance for convergence
  set_mode("classification")

print("Creating polynomial SVM workflow...")
svm_poly_wf <- workflow() %>%
  add_recipe(svm_recipe) %>%
  add_model(svm_poly_spec)

print("Running polynomial SVM Bayesian optimization...")
print("Using 25 iterations to efficiently explore 3D parameter space")
print("Parameters: cost, degree (2-5), scale_factor")
print("Parallel processing will be used automatically by tune_bayes()")
svm_poly_res <- svm_poly_wf %>%
  tune_bayes(
    resamples = folds,
    initial   = 5,              # Start with 5 random initial points
    iter      = 25,             # 25 Bayesian optimization iterations
    param_info = parameters(
      cost(range = c(-5, 3)),           # cost range on log2 scale
      degree(range = c(2L, 5L)),        # degree: integer from 2 to 5
      scale_factor(range = c(-5, 1))    # scale_factor range on log2 scale
    ),
    metrics   = metric_set(roc_auc),
    control   = control_bayes(
      parallel_over = "resamples",  # Parallelize over CV folds
      verbose = TRUE                 # Show progress
    )
  )

print("Polynomial SVM tuning complete")
print(paste("Polynomial SVM end time:", Sys.time()))

best_poly <- svm_poly_res %>%
  select_best(metric = "roc_auc")

best_poly_metrics <- svm_poly_res %>%
  show_best(metric = "roc_auc", n = 1)

print("Best polynomial SVM parameters:")
print(best_poly)
print("Best polynomial SVM ROC AUC:")
print(best_poly_metrics)

###############################################################################
# 4.3 Radial (RBF) SVM
###############################################################################

print("=== Starting Radial (RBF) SVM Tuning ===")
print(paste("RBF SVM start time:", Sys.time()))

svm_rbf_spec <- svm_rbf(
  cost       = tune(),
  rbf_sigma  = tune()        # kernel width parameter
) %>%
  set_engine("kernlab",
             cache = 500,      # maximum cache size for server execution
             maxiter = 2000,   # increase max iterations to prevent convergence warnings
             tol = 0.001) %>%  # tolerance for convergence
  set_mode("classification")

print("Creating RBF SVM workflow...")
svm_rbf_wf <- workflow() %>%
  add_recipe(svm_recipe) %>%
  add_model(svm_rbf_spec)

print("Running RBF SVM Bayesian optimization...")
print("Using 25 iterations to efficiently explore 2D parameter space")
print("Parameters: cost and rbf_sigma (kernel width)")
print("Parallel processing will be used automatically by tune_bayes()")
svm_rbf_res <- svm_rbf_wf %>%
  tune_bayes(
    resamples = folds,
    initial   = 5,              # Start with 5 random initial points
    iter      = 25,             # 25 Bayesian optimization iterations
    param_info = parameters(
      cost(range = c(-6, 4)),        # cost range on log2 scale
      rbf_sigma(range = c(-6, 2))    # rbf_sigma range on log2 scale
    ),
    metrics   = metric_set(roc_auc),
    control   = control_bayes(
      parallel_over = "resamples",  # Parallelize over CV folds
      verbose = TRUE                 # Show progress
    )
  )

print("RBF SVM tuning complete")
print(paste("RBF SVM end time:", Sys.time()))

best_rbf <- svm_rbf_res %>%
  select_best(metric = "roc_auc")

best_rbf_metrics <- svm_rbf_res %>%
  show_best(metric = "roc_auc", n = 1)

print("Best RBF SVM parameters:")
print(best_rbf)
print("Best RBF SVM ROC AUC:")
print(best_rbf_metrics)

###############################################################################
# 5. Compare kernels and select the best-performing SVM
###############################################################################

print("=== Comparing Kernel Performance ===")

linear_auc <- best_linear_metrics$mean[1]
poly_auc   <- best_poly_metrics$mean[1]
rbf_auc    <- best_rbf_metrics$mean[1]

kernel_performance <- tibble(
  kernel = c("linear", "polynomial", "radial"),
  roc_auc = c(linear_auc, poly_auc, rbf_auc)
) %>%
  arrange(desc(roc_auc))

print("Kernel performance comparison:")
print(kernel_performance)

best_kernel <- kernel_performance$kernel[1]
print(paste("Best kernel selected:", best_kernel))

###############################################################################
# 6. Finalize the best SVM workflow and fit on full training data
###############################################################################

print("=== Finalizing Best SVM Model ===")
print(paste("Using", best_kernel, "kernel"))

if (best_kernel == "linear") {
  print("Finalizing linear SVM workflow...")
  final_svm_wf <- svm_linear_wf %>%
    finalize_workflow(best_linear)
} else if (best_kernel == "polynomial") {
  print("Finalizing polynomial SVM workflow...")
  final_svm_wf <- svm_poly_wf %>%
    finalize_workflow(best_poly)
} else {
  print("Finalizing RBF SVM workflow...")
  final_svm_wf <- svm_rbf_wf %>%
    finalize_workflow(best_rbf)
}

print("Fitting final model on full training data...")
print(paste("Final model fitting start time:", Sys.time()))
final_svm_fit <- final_svm_wf %>%
  fit(data = train_data)
print("Final model fitted successfully")
print(paste("Final model fitting end time:", Sys.time()))

###############################################################################
# 7. Generate predictions for Amazon test data and write submission file
#
#    - We output the probability of ACTION == 1 (".pred_1")
#    - Format matches Kaggle's expected "id" and "Action" columns
###############################################################################

print("=== Generating Predictions ===")
print("Generating predictions on test data...")
svm_predictions <- predict(
  final_svm_fit,
  new_data = test_data,
  type     = "prob"
)
print(paste("Predictions generated for", nrow(svm_predictions), "test samples"))

svm_predictions <- svm_predictions %>%
  select(-.pred_0) %>%
  rename(Action = .pred_1)

print("Preparing submission file...")
kaggle_submission <- bind_cols(
  test_data %>% select(id),
  svm_predictions
) %>%
  rename(Id = id)  # Match sample submission format: "Id" (capital I) and "Action"

print("Writing submission file...")
vroom_write(
  kaggle_submission,
  "SVMSubmission.csv",
  delim = ","
)
print("Submission file written: SVMSubmission.csv")

###############################################################################
# 8. Clean up parallel processing
###############################################################################

print("Stopping parallel processing workers...")
stopCluster(cl)
print("Parallel processing workers stopped")

print("=== Script Complete ===")
print(paste("End time:", Sys.time()))
