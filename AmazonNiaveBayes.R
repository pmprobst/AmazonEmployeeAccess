#AmazonAnalysis
#=============================================================================
# SERVER STATUS: Starting script execution
#=============================================================================
print("=== SERVER: Loading required libraries ===")
library(tidymodels)
library(embed)
library(vroom)
library(discrim)  # Required for naive_Bayes model specification
library(themis)  # Required for SMOTE balancing
# Install naivebayes package if not already installed (required for naive_Bayes engine)
print("=== SERVER: Libraries loaded successfully ===")

#=============================================================================
# SERVER STATUS: Loading training and test data
#=============================================================================
print("=== SERVER: Reading training data ===")
#Read data in, set ACTION feature as factor
train_data <- vroom("data/train.csv") %>%
  mutate(ACTION = factor(ACTION))
print(paste("=== SERVER: Training data loaded -", nrow(train_data), "rows,", ncol(train_data), "columns ==="))
print(paste("=== SERVER: Class distribution:", table(train_data$ACTION), "==="))

print("=== SERVER: Reading test data ===")
test_data <- vroom("data/test.csv")
print(paste("=== SERVER: Test data loaded -", nrow(test_data), "rows,", ncol(test_data), "columns ==="))

#=============================================================================
# SERVER STATUS: Creating preprocessing recipe
#=============================================================================
print("=== SERVER: Creating preprocessing recipe ===")
#Create Recipe
my_recipe <- recipe(ACTION ~ . ,data = train_data) %>%
  step_mutate_at(all_numeric_predictors() ,fn = factor) %>%
  #not as necessary for peanalized regression
  step_other(all_nominal_predictors() ,threshold = .01) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_smote(ACTION, over_ratio = 1)  # Balance classes using SMOTE
  ##normalize features
print("=== SERVER: Recipe created successfully ===")

#=============================================================================
# SERVER STATUS: Preparing and baking recipe (this may take time with SMOTE)
#=============================================================================
print("=== SERVER: Preparing recipe (includes SMOTE balancing) - this may take several minutes ===")
#Prep & Bake Recipe
prep <- prep(my_recipe)
print("=== SERVER: Recipe prepared successfully ===")

print("=== SERVER: Baking recipe on training data ===")
baked <- bake(prep ,new_data = train_data)
print(paste("=== SERVER: Baked data dimensions -", nrow(baked), "rows,", ncol(baked), "columns ==="))
print(paste("=== SERVER: Post-SMOTE class distribution:", table(baked$ACTION), "==="))

#=============================================================================
# NAIVE BAYES CLASSIFIER
#=============================================================================
# Naive Bayes is a probabilistic classifier based on applying Bayes' theorem
# with strong (naive) independence assumptions between features. It works well
# for categorical data and is computationally efficient.
#
# Key assumptions:
# - Features are conditionally independent given the class
# - All features contribute equally to the classification
#
# The Laplace smoothing parameter helps handle zero probabilities when
# a feature value doesn't appear in the training data for a given class.

#------------------------------------------------------------------------------
# Step 1: Set Up Naive Bayes Model
#------------------------------------------------------------------------------
# SERVER STATUS: Configuring Naive Bayes model specification
print("=== SERVER: Setting up Naive Bayes model ===")
# Configure the Naive Bayes model with Laplace smoothing parameter to tune
# Laplace smoothing (also called additive smoothing) prevents zero probabilities
# by adding a small constant to all counts. This is especially important when
# dealing with categorical features that may have rare combinations.
nb_model <- naive_Bayes(
  Laplace = tune()  # Smoothing parameter to prevent zero probabilities
) %>%
  set_engine("naivebayes") %>%  # Uses the naivebayes package
  set_mode("classification")
print("=== SERVER: Naive Bayes model configured ===")

#------------------------------------------------------------------------------
# Step 2: Create Workflow
#------------------------------------------------------------------------------
# SERVER STATUS: Creating workflow combining recipe and model
print("=== SERVER: Creating workflow ===")
# Combine the preprocessing recipe with the Naive Bayes model
nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)
print("=== SERVER: Workflow created successfully ===")

#------------------------------------------------------------------------------
# Step 3: Set Up Tuning Grid
#------------------------------------------------------------------------------
# SERVER STATUS: Creating hyperparameter tuning grid
print("=== SERVER: Setting up tuning grid (40 Laplace values) ===")
# Create a grid of Laplace smoothing values to test
# Laplace values typically range from 0 (no smoothing) to 1 (strong smoothing)
# We'll test a range to find the optimal smoothing parameter
tuning_grid <- grid_regular(
  Laplace(range = c(0, 1)),  # Test values from 0 to 1
  levels = 40  # Test 10 different values across the range
)
print(paste("=== SERVER: Tuning grid created -", nrow(tuning_grid), "combinations to test ==="))

#------------------------------------------------------------------------------
# Step 4: Cross-Validation Setup
#------------------------------------------------------------------------------
# SERVER STATUS: Creating cross-validation folds
print("=== SERVER: Creating 10-fold cross-validation splits ===")
# Create 3-fold cross-validation splits for tuning
# This helps us evaluate model performance across different data subsets
folds <- vfold_cv(train_data, v = 10, repeats = 1)
print("=== SERVER: Cross-validation folds created successfully ===")

#------------------------------------------------------------------------------
# Step 5: Tune Hyperparameters
#------------------------------------------------------------------------------
# SERVER STATUS: Starting hyperparameter tuning (THIS IS THE LONGEST STEP)
# This will test 40 parameter values across 10 CV folds = 400 model fits
# With SMOTE applied in each fold, this may take 30-60+ minutes depending on server
print("=== SERVER: Starting hyperparameter tuning ===")
print("=== SERVER: This will test 40 Laplace values across 10 CV folds ===")
print("=== SERVER: Estimated time: 30-60+ minutes (varies by server) ===")
print("=== SERVER: Tuning in progress... ===")
# Perform grid search with cross-validation to find the best Laplace parameter
# We use ROC-AUC as our evaluation metric since this is a binary classification
# problem and we want to optimize for ranking performance
start_time <- Sys.time()
CV_results <- nb_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc)
  )
end_time <- Sys.time()
elapsed_time <- difftime(end_time, start_time, units = "mins")
print(paste("=== SERVER: Hyperparameter tuning completed in", round(elapsed_time, 2), "minutes ==="))

#------------------------------------------------------------------------------
# Step 6: Select Best Model
#------------------------------------------------------------------------------
# SERVER STATUS: Selecting best hyperparameters
print("=== SERVER: Selecting best hyperparameters ===")
# Extract the best hyperparameters based on highest ROC-AUC
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# Display the best tuning parameters
print("=== SERVER: Best Laplace parameter found ===")
print("Best Laplace parameter:")
print(bestTune)

#------------------------------------------------------------------------------
# Step 7: Evaluate Tuning Results
#------------------------------------------------------------------------------
# SERVER STATUS: Collecting and displaying tuning metrics
print("=== SERVER: Collecting tuning metrics ===")
# View all tuning results sorted by performance
# This helps understand how sensitive the model is to the Laplace parameter
tuning_metrics <- CV_results %>%
  collect_metrics() %>%
  arrange(desc(mean))
print("=== SERVER: Top 5 tuning results ===")
print(head(tuning_metrics, 5))

# Visualize the tuning results
# The plot shows how ROC-AUC varies with different Laplace values
# SERVER NOTE: Plot may not display on headless servers, but metrics are saved
print("=== SERVER: Generating tuning plot (may not display on headless server) ===")
autoplot(CV_results)

#------------------------------------------------------------------------------
# Step 8: Finalize and Fit Model
#------------------------------------------------------------------------------
# SERVER STATUS: Fitting final model with best hyperparameters
print("=== SERVER: Finalizing workflow with best hyperparameters ===")
# Apply the best hyperparameters and fit the final model on all training data
print("=== SERVER: Fitting final model on full training data (includes SMOTE) ===")
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)
print("=== SERVER: Final model fitted successfully ===")

#=============================================================================
# PREDICTIONS
#=============================================================================

#------------------------------------------------------------------------------
# Step 9: Generate Predictions on Test Data
#------------------------------------------------------------------------------
# SERVER STATUS: Generating predictions on test set
print("=== SERVER: Generating predictions on test data ===")
# Use the fitted Naive Bayes model to predict probabilities for test data
# type = "prob" returns class probabilities (P(ACTION=0) and P(ACTION=1))
predictions <- predict(
  final_wf,
  new_data = test_data,
  type = "prob"
)
print(paste("=== SERVER: Predictions generated for", nrow(predictions), "test samples ==="))

#------------------------------------------------------------------------------
# Step 10: Format Predictions for Submission
#------------------------------------------------------------------------------
# SERVER STATUS: Formatting predictions for Kaggle submission
print("=== SERVER: Formatting predictions for submission ===")
# Remove the probability of class 0 (ACTION=0) and keep only P(ACTION=1)
# Rename the remaining column to "Action" as required by Kaggle submission format
predictions <- predictions %>% 
  select(-.pred_0) %>%
  rename(Action = .pred_1)
print("=== SERVER: Predictions formatted successfully ===")

#------------------------------------------------------------------------------
# Step 11: Create Submission File
#------------------------------------------------------------------------------
# SERVER STATUS: Creating and saving submission file
print("=== SERVER: Creating submission file ===")
# Combine test data IDs with predictions
kaggle_submission <- bind_cols(
  test_data %>% select(id),
  predictions
)

# Write submission file to CSV
# This file can be directly submitted to Kaggle
vroom_write(kaggle_submission, "NaiveBayesSubmission.csv", delim = ",")

print("=== SERVER: Naive Bayes predictions saved to NaiveBayesSubmission.csv ===")
print(paste("=== SERVER: Submission file contains", nrow(kaggle_submission), "predictions ==="))
print("=== SERVER: Script execution completed successfully ===")
