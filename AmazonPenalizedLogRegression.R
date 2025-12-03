#AmazonAnalysis
print("=== Starting Amazon Employee Access Analysis ===")
print("Loading required libraries...")
library(tidymodels)
library(embed)
library(vroom)
library(themis)  # For SMOTE resampling
print("Libraries loaded successfully.")

#Read data in, set ACTION feature as factor
print("Reading training data...")
train_data <- vroom("data/train.csv") %>%
  mutate(ACTION = factor(ACTION))
print(paste("Training data loaded: ", nrow(train_data), "rows,", ncol(train_data), "columns"))
print("Class distribution:")
print(table(train_data$ACTION))

print("Reading test data...")
test_data <- vroom("data/test.csv")
print(paste("Test data loaded: ", nrow(test_data), "rows,", ncol(test_data), "columns"))

#Create Recipe
print("Creating recipe with preprocessing steps...")
my_recipe <- recipe(ACTION ~ . ,data = train_data) %>%
  step_mutate_at(all_numeric_predictors() ,fn = factor) %>%
  #not as necessary for peanalized regression
  step_other(all_nominal_predictors() ,threshold = .01) %>% 
  step_dummy(all_nominal_predictors()) %>%
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

#Set Up Logistic Regression Model
print("Setting up penalized logistic regression model...")
PenLogRegModel <- logistic_reg(mixture = tune() ,penalty = tune()) %>%
  set_engine("glmnet")
print("Model specification created.")

#Set Workflow
print("Creating workflow...")
wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(PenLogRegModel)
print("Workflow created successfully.")

#set up grid of tuning values
print("Creating tuning grid...")
tuning_grid <- grid_regular(penalty()
                             ,mixture()
                             ,levels = 10)
print(paste("Tuning grid created with", nrow(tuning_grid), "combinations"))

print("Creating cross-validation folds...")
folds <- vfold_cv(train_data ,v = 10 ,repeats = 1)
print(paste("Created", length(folds$splits), "CV folds"))

#CV
print("=== Starting Cross-Validation (this may take a while) ===")
print(paste("Tuning", nrow(tuning_grid), "parameter combinations across", length(folds$splits), "folds..."))
CV_results <- wf %>%
  tune_grid(resamples = folds
              ,grid = tuning_grid
              ,metrics = metric_set(roc_auc))
print("=== Cross-Validation completed ===")

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

#Identify the best levels of penalty and mixture (highest mean)
print("Top 10 parameter combinations by ROC AUC:")
top_results <- CV_results %>%
  collect_metrics() %>%
  arrange(desc(mean))
print(head(top_results, 10))

#plot levels of penalty and mixture
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
  #rename .pred_1 as "action" for kaggle submission
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
vroom_write(kaggle_submission, "PenLogRegModelSubmission.csv" ,delim = ",")
print("=== Analysis Complete ===")
print("Submission file saved as: PenLogRegModelSubmission.csv")
