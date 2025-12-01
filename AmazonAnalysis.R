#AmazonAnalysis
library(tidymodels)
library(embed)
library(vroom)
library(kernlab)
install.packages("kernlab") # Required for SVM models with tidymodels

set.seed(348)  # for reproducibility

###############################################################################
# 1. Read and prepare data
###############################################################################

# Read data in, set ACTION as a factor (binary classification outcome)
train_data <- vroom("data/train.csv") %>%
  mutate(ACTION = factor(ACTION))
test_data  <- vroom("data/test.csv")

###############################################################################
# 2. Preprocessing recipe with PCA
#
#    - Convert rare factor levels to "other"
#    - Dummy encode all nominal predictors
#    - Normalize numeric predictors
#    - Apply PCA to speed up SVM fitting by reducing dimensionality
###############################################################################

svm_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  # Use a fixed number of principal components to reduce dimension.
  # You can adjust num_comp if you want more/less information retained.
  step_pca(all_numeric_predictors(), num_comp = 50)

###############################################################################
# 3. Resampling for model tuning
###############################################################################

folds <- vfold_cv(train_data, v = 3, repeats = 1)

###############################################################################
# 4. Define SVM model specifications
#
#    We will tune three kernels:
#      - Linear
#      - Polynomial
#      - Radial (RBF)
#
#    For each, we tune the primary hyperparameters and evaluate ROC AUC.
###############################################################################

# 4.1 Linear SVM
svm_linear_spec <- svm_linear(
  cost = tune()              # regularization strength
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_linear_wf <- workflow() %>%
  add_recipe(svm_recipe) %>%
  add_model(svm_linear_spec)

linear_grid <- grid_regular(
  cost(range = c(-5, 5)),    # cost on log2 scale (kernlab uses 2^cost)
  levels = 7
)

svm_linear_res <- svm_linear_wf %>%
  tune_grid(
    resamples = folds,
    grid      = linear_grid,
    metrics   = metric_set(roc_auc)
  )

best_linear <- svm_linear_res %>%
  select_best(metric = "roc_auc")

best_linear_metrics <- svm_linear_res %>%
  show_best(metric = "roc_auc", n = 1)

###############################################################################
# 4.2 Polynomial SVM
###############################################################################

svm_poly_spec <- svm_poly(
  cost          = tune(),
  degree        = tune(),    # polynomial degree
  scale_factor  = tune()     # scaling of input features
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_poly_wf <- workflow() %>%
  add_recipe(svm_recipe) %>%
  add_model(svm_poly_spec)

poly_grid <- grid_regular(
  cost(range = c(-5, 5)),
  degree(range = c(2L, 4L)),
  scale_factor(range = c(-5, 5)),
  levels = 3
)

svm_poly_res <- svm_poly_wf %>%
  tune_grid(
    resamples = folds,
    grid      = poly_grid,
    metrics   = metric_set(roc_auc)
  )

best_poly <- svm_poly_res %>%
  select_best(metric = "roc_auc")

best_poly_metrics <- svm_poly_res %>%
  show_best(metric = "roc_auc", n = 1)

###############################################################################
# 4.3 Radial (RBF) SVM
###############################################################################

svm_rbf_spec <- svm_rbf(
  cost       = tune(),
  rbf_sigma  = tune()        # kernel width parameter
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_rbf_wf <- workflow() %>%
  add_recipe(svm_recipe) %>%
  add_model(svm_rbf_spec)

rbf_grid <- grid_regular(
  cost(range = c(-5, 5)),
  rbf_sigma(range = c(-5, 5)),
  levels = 5
)

svm_rbf_res <- svm_rbf_wf %>%
  tune_grid(
    resamples = folds,
    grid      = rbf_grid,
    metrics   = metric_set(roc_auc)
  )

best_rbf <- svm_rbf_res %>%
  select_best(metric = "roc_auc")

best_rbf_metrics <- svm_rbf_res %>%
  show_best(metric = "roc_auc", n = 1)

###############################################################################
# 5. Compare kernels and select the best-performing SVM
###############################################################################

linear_auc <- best_linear_metrics$mean[1]
poly_auc   <- best_poly_metrics$mean[1]
rbf_auc    <- best_rbf_metrics$mean[1]

kernel_performance <- tibble(
  kernel = c("linear", "polynomial", "radial"),
  roc_auc = c(linear_auc, poly_auc, rbf_auc)
) %>%
  arrange(desc(roc_auc))

kernel_performance

best_kernel <- kernel_performance$kernel[1]

###############################################################################
# 6. Finalize the best SVM workflow and fit on full training data
###############################################################################

if (best_kernel == "linear") {
  final_svm_wf <- svm_linear_wf %>%
    finalize_workflow(best_linear)
} else if (best_kernel == "polynomial") {
  final_svm_wf <- svm_poly_wf %>%
    finalize_workflow(best_poly)
} else {
  final_svm_wf <- svm_rbf_wf %>%
    finalize_workflow(best_rbf)
}

final_svm_fit <- final_svm_wf %>%
  fit(data = train_data)

###############################################################################
# 7. Generate predictions for Amazon test data and write submission file
#
#    - We output the probability of ACTION == 1 (".pred_1")
#    - Format matches Kaggle's expected "id" and "Action" columns
###############################################################################

svm_predictions <- predict(
  final_svm_fit,
  new_data = test_data,
  type     = "prob"
)

svm_predictions <- svm_predictions %>%
  select(-.pred_0) %>%
  rename(Action = .pred_1)

kaggle_submission <- bind_cols(
  test_data %>% select(id),
  svm_predictions
)

vroom_write(
  kaggle_submission,
  "SVMSubmission.csv",
  delim = ","
)
