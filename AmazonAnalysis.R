#AmazonAnalysis
library(tidymodels)
library(embed)
library(vroom)
library(discrim)  # Required for naive_Bayes model specification
# Install naivebayes package if not already installed (required for naive_Bayes engine)


#Read data in, set ACTION feature as factor
train_data <- vroom("data/train.csv") %>%
  mutate(ACTION = factor(ACTION))
test_data <- vroom("data/test.csv")

#Create Recipe
my_recipe <- recipe(ACTION ~ . ,data = train_data) %>%
  step_mutate_at(all_numeric_predictors() ,fn = factor) %>%
  #not as necessary for peanalized regression
  step_other(all_nominal_predictors() ,threshold = .01) %>% 
  step_dummy(all_nominal_predictors())
  ##normalize features

#Prep & Bake Recipe
prep <- prep(my_recipe)
baked <- bake(prep ,new_data = train_data)

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
# Configure the Naive Bayes model with Laplace smoothing parameter to tune
# Laplace smoothing (also called additive smoothing) prevents zero probabilities
# by adding a small constant to all counts. This is especially important when
# dealing with categorical features that may have rare combinations.
nb_model <- naive_Bayes(
  Laplace = tune()  # Smoothing parameter to prevent zero probabilities
) %>%
  set_engine("naivebayes") %>%  # Uses the naivebayes package
  set_mode("classification")

#------------------------------------------------------------------------------
# Step 2: Create Workflow
#------------------------------------------------------------------------------
# Combine the preprocessing recipe with the Naive Bayes model
nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

#------------------------------------------------------------------------------
# Step 3: Set Up Tuning Grid
#------------------------------------------------------------------------------
# Create a grid of Laplace smoothing values to test
# Laplace values typically range from 0 (no smoothing) to 1 (strong smoothing)
# We'll test a range to find the optimal smoothing parameter
tuning_grid <- grid_regular(
  Laplace(range = c(0, 1)),  # Test values from 0 to 1
  levels = 40  # Test 10 different values across the range
)

#------------------------------------------------------------------------------
# Step 4: Cross-Validation Setup
#------------------------------------------------------------------------------
# Create 3-fold cross-validation splits for tuning
# This helps us evaluate model performance across different data subsets
folds <- vfold_cv(train_data, v = 10, repeats = 1)

#------------------------------------------------------------------------------
# Step 5: Tune Hyperparameters
#------------------------------------------------------------------------------
# Perform grid search with cross-validation to find the best Laplace parameter
# We use ROC-AUC as our evaluation metric since this is a binary classification
# problem and we want to optimize for ranking performance
CV_results <- nb_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc)
  )

#------------------------------------------------------------------------------
# Step 6: Select Best Model
#------------------------------------------------------------------------------
# Extract the best hyperparameters based on highest ROC-AUC
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# Display the best tuning parameters
print("Best Laplace parameter:")
print(bestTune)

#------------------------------------------------------------------------------
# Step 7: Evaluate Tuning Results
#------------------------------------------------------------------------------
# View all tuning results sorted by performance
# This helps understand how sensitive the model is to the Laplace parameter
CV_results %>%
  collect_metrics() %>%
  arrange(desc(mean))

# Visualize the tuning results
# The plot shows how ROC-AUC varies with different Laplace values
autoplot(CV_results)

#------------------------------------------------------------------------------
# Step 8: Finalize and Fit Model
#------------------------------------------------------------------------------
# Apply the best hyperparameters and fit the final model on all training data
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

#=============================================================================
# PREDICTIONS
#=============================================================================

#------------------------------------------------------------------------------
# Step 9: Generate Predictions on Test Data
#------------------------------------------------------------------------------
# Use the fitted Naive Bayes model to predict probabilities for test data
# type = "prob" returns class probabilities (P(ACTION=0) and P(ACTION=1))
predictions <- predict(
  final_wf,
  new_data = test_data,
  type = "prob"
)

#------------------------------------------------------------------------------
# Step 10: Format Predictions for Submission
#------------------------------------------------------------------------------
# Remove the probability of class 0 (ACTION=0) and keep only P(ACTION=1)
# Rename the remaining column to "Action" as required by Kaggle submission format
predictions <- predictions %>% 
  select(-.pred_0) %>%
  rename(Action = .pred_1)

#------------------------------------------------------------------------------
# Step 11: Create Submission File
#------------------------------------------------------------------------------
# Combine test data IDs with predictions
kaggle_submission <- bind_cols(
  test_data %>% select(id),
  predictions
)

# Write submission file to CSV
# This file can be directly submitted to Kaggle
vroom_write(kaggle_submission, "NaiveBayesSubmission.csv", delim = ",")

print("Naive Bayes predictions saved to NaiveBayesSubmission.csv")
