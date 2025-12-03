#AmazonAnalysis
print("=== Starting Amazon Employee Access Random Forest Classification ===")
print(paste("Start time:", Sys.time()))

library(tidymodels)
library(embed)
library(vroom)
library(themis)  # For SMOTE resampling

print("Libraries loaded successfully")

#Read data in, set ACTION feature as factor
print("Loading training data...")
train_data <- vroom("data/train.csv") %>%
  mutate(ACTION = factor(ACTION))
print(paste("Training data loaded:", nrow(train_data), "rows,", ncol(train_data), "columns"))
print("Class distribution:")
print(table(train_data$ACTION))

print("Loading test data...")
test_data <- vroom("data/test.csv")
print(paste("Test data loaded:", nrow(test_data), "rows,", ncol(test_data), "columns"))

#Create Recipe
print("Creating recipe with preprocessing steps...")
my_recipe <- recipe(ACTION ~ . ,data = train_data) %>%
  step_mutate_at(all_numeric_predictors() ,fn = factor) %>%
  #not as necessary for penalized regression
  step_other(all_nominal_predictors() ,threshold = .01) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_pca(all_numeric_predictors(), threshold = 0.95) %>%  # PCA: retain 95% variance for speed
  step_smote(ACTION)  # Apply SMOTE to balance classes
  ##normalize features
print("Recipe created successfully.")

#Prep & Bake Recipe
print("Preparing recipe (fitting preprocessing steps)...")
prep <- prep(my_recipe)
print("Recipe prepared successfully.")
print("Baking recipe (applying preprocessing to training data)...")
baked <- bake(prep ,new_data = train_data)
print(paste("Baked data: ", nrow(baked), "rows,", ncol(baked), "columns"))

#Set Up Random Forest Model
print("Setting up Random Forest model...")
rf_model <- rand_forest(
  mtry  = tune(),      # number of variables randomly sampled at each split
  min_n = tune(),      # minimum number of data points in a node
  trees = 500          # number of trees in the forest
) %>%
  set_engine("ranger") %>%
  set_mode("classification")
print("Model specification created.")

#Set Workflow
print("Creating workflow...")
wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_model)
print("Workflow created successfully.")

#set up grid of tuning values
print("Creating tuning grid...")
tuning_grid <- grid_regular(
  mtry(range = c(1L, 50L)),   # range of variables to sample
  min_n(),                     # default range for min_n
  levels = 5                   # 5x5 = 25 combinations
)
print(paste("Tuning grid created with", nrow(tuning_grid), "combinations"))

print("Creating cross-validation folds...")
folds <- vfold_cv(train_data ,v = 5 ,repeats = 1)
print(paste("Created", length(folds$splits), "CV folds"))

#CV
print("=== Starting Cross-Validation (this may take a while) ===")
print(paste("Tuning", nrow(tuning_grid), "parameter combinations across", length(folds$splits), "folds..."))
print(paste("CV start time:", Sys.time()))
CV_results <- wf %>%
  tune_grid(resamples = folds
              ,grid = tuning_grid
              ,metrics = metric_set(roc_auc))
print("=== Cross-Validation completed ===")
print(paste("CV end time:", Sys.time()))

print("Selecting best tuning parameters...")
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")
print("Best tuning parameters:")
print(bestTune)

#finalize and fit workflow
print("Finalizing workflow with best parameters...")
final_wf <-
  wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)
print("Final model fitted successfully.")

#Identify the best levels of mtry and min_n (highest mean)
print("Top 10 parameter combinations by ROC AUC:")
top_results <- CV_results %>%
  collect_metrics() %>%
  arrange(desc(mean))
print(head(top_results, 10))

#plot levels of mtry and min_n
print("Generating tuning plot...")
autoplot(CV_results)
print("Plot generated.")

#Get Predictions
print("Generating predictions on test data...")
predictions <-predict(final_wf,
                      new_data = test_data
                      ,type = "prob")
print(paste("Predictions generated for", nrow(predictions), "test samples"))

#Remove p(0) column from df
print("Formatting predictions for submission...")
predictions <- predictions %>% 
  select(-.pred_0) %>%
  #rename .pred_1 as "Action" for kaggle submission
  rename (Action = .pred_1)

print("Summary of prediction probabilities:")
print(summary(predictions$Action))

# Combine with test_data ID
print("Combining predictions with test IDs...")
kaggle_submission <- bind_cols(
  test_data %>% select(id),
  predictions
)
print(paste("Submission dataframe created:", nrow(kaggle_submission), "rows"))

#write submission df to CSV for submission
print("Writing submission file...")
vroom_write(kaggle_submission, "RandomForestSubmission.csv" ,delim = ",")
print("=== Analysis Complete ===")
print("Submission file saved as: RandomForestSubmission.csv")
print(paste("End time:", Sys.time()))
