#AmazonAnalysis
print("=== Starting Amazon Employee Access KNN Classification ===")
print(paste("Start time:", Sys.time()))

library(tidymodels)
library(embed)
library(vroom)
library(themis)

print("Libraries loaded successfully")

#Read data in, set ACTION feature as factor
print("Loading training data...")
train_data <- vroom("data/train.csv") %>%
  mutate(ACTION = factor(ACTION))
print(paste("Training data loaded:", nrow(train_data), "rows,", ncol(train_data), "columns"))

print("Loading test data...")
test_data <- vroom("data/test.csv")
print(paste("Test data loaded:", nrow(test_data), "rows,", ncol(test_data), "columns"))

#Create Recipe
my_recipe <- recipe(ACTION ~ . ,data = train_data) %>%
  step_mutate_at(all_numeric_predictors() ,fn = factor) %>%
  #not as necessary for peanalized regression
  step_other(all_nominal_predictors() ,threshold = .01) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_pca(all_numeric_predictors(), threshold = 0.95) %>%  # PCA: retain 95% variance for speed
  step_smote(ACTION)  # Apply SMOTE to balance classes
  ##normalize features

#Prep & Bake Recipe
print("Preparing recipe...")
prep <- prep(my_recipe)
print("Recipe prepared successfully")

print("Baking recipe on training data...")
baked <- bake(prep ,new_data = train_data)
print(paste("Baked data dimensions:", nrow(baked), "rows,", ncol(baked), "columns"))

#Set Up K Nearest Neighbors
print("Setting up KNN model...")
knn_model <- nearest_neighbor(
  mode      = "classification",
  neighbors = tune(),
  weight_func = "rectangular",  # standard unweighted KNN
  dist_power = 2
  ) %>% set_engine("kknn")

#Set Workflow
print("Creating workflow...")
wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

#set up grid of tuning values
print("Setting up tuning grid...")
tuning_grid <- grid_regular(
                  neighbors(range = c(1L, 20L)),
                  levels = 3)
print(paste("Tuning grid size:", nrow(tuning_grid), "combinations"))

print("Creating cross-validation folds...")
folds <- vfold_cv(train_data ,v = 3 ,repeats = 1)
print(paste("Created", length(folds$splits), "folds"))

#CV
print("=== Starting Cross-Validation ===")
print(paste("CV start time:", Sys.time()))
CV_results <- wf %>%
  tune_grid(resamples = folds
              ,grid = tuning_grid
              ,metrics = metric_set(roc_auc))
print("=== Cross-Validation Complete ===")
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
print("Final model fitted successfully")

#Identify the best levels of penalty and mixture (highest mean)
print("Top CV results:")
CV_results %>%
  collect_metrics() %>%
  arrange(desc(mean)) %>%
  print()

#plot levels of penalty and mixture
autoplot(CV_results)

#Get Predictions
print("Generating predictions on test data...")
predictions <-predict(final_wf,
                      new_data = test_data
                      ,type = "prob")
print(paste("Predictions generated for", nrow(predictions), "test samples"))

#Remove p(0) column from df
predictions <- predictions %>% 
  select(-.pred_0) %>%
  #rename .pred_1 as "action" for kaggle submission
  rename (Action = .pred_1)


# Combine with test_data ID
kaggle_submission <- bind_cols(
  test_data %>% select(id),
  predictions
)

#write submission df to CSV for submission
print("Writing submission file...")
vroom_write(kaggle_submission, "KNNSubmission.csv" ,delim = ",")
print("Submission file written: KNNSubmission.csv")
print("=== Script Complete ===")
print(paste("End time:", Sys.time()))
