#AmazonAnalysis
library(tidymodels)
library(embed)
library(vroom)

#Read data in, set ACTION feature as factor
train_data <- vroom("data/train.csv") %>%
  mutate(ACTION = factor(ACTION))
test_data <- vroom("data/test.csv")

#Create Recipe for MLP (with normalization for neural networks)
mlp_recipe <- recipe(ACTION ~ . ,data = train_data) %>%
  step_mutate_at(all_numeric_predictors() ,fn = factor) %>%
  step_other(all_nominal_predictors() ,threshold = .01) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>%  # Normalize features for neural network
  step_pca(all_numeric_predictors(), threshold = 0.95)  # PCA: retain 95% variance for speed

#Set Up MLP Model
mlp_model <- mlp(
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune()
) %>%
  set_engine("nnet", MaxNWts = 5000) %>%  # Increase weight limit
  set_mode("classification")

#Set MLP Workflow
mlp_wf <- workflow() %>%
  add_recipe(mlp_recipe) %>%
  add_model(mlp_model)

#Set up grid of tuning values for MLP
mlp_tuning_grid <- grid_regular(
  hidden_units(range = c(1L, 10L)),  # Reduced range to avoid too many weights
  penalty(range = c(-5, 0)),  # log10 scale
  epochs(range = c(50L, 200L)),
  levels = 3  # 7^3 = 343 combinations
)

#Create cross-validation folds
folds <- vfold_cv(train_data ,v = 3 ,repeats = 1)

#Cross Validation for MLP (using accuracy metric)
mlp_CV_results <- mlp_wf %>%
  tune_grid(
    resamples = folds,
    grid = mlp_tuning_grid,
    metrics = metric_set(accuracy)
  )

#Select best tune based on accuracy
mlp_bestTune <- mlp_CV_results %>%
  select_best(metric = "accuracy")

#View best tuning parameters
print("Best MLP Parameters:")
print(mlp_bestTune)

# View metrics
mlp_metrics <- mlp_CV_results %>%
  collect_metrics() %>%
  arrange(desc(mean))

print(mlp_metrics)

# Create summary of mean(accuracy) by hidden_units
mlp_hidden_units_acc <- mlp_metrics %>%
  dplyr::filter(.metric == "accuracy") %>%
  dplyr::group_by(hidden_units) %>%
  dplyr::summarise(
    mean_accuracy = mean(mean),
    .groups = "drop"
  )

print("Mean accuracy by hidden_units:")
print(mlp_hidden_units_acc)

# Plot: hidden_units (x) vs mean(accuracy) (y)
library(ggplot2)

ggplot(mlp_hidden_units_acc, aes(x = hidden_units, y = mean_accuracy)) +
  geom_point() +
  geom_line() +
  labs(
    title = "Cross-Validation Accuracy vs Hidden Units",
    x = "Hidden Units",
    y = "Mean Accuracy (CV)"
  ) +
  theme_minimal()

#View metrics
mlp_CV_results %>%
  collect_metrics() %>%
  arrange(desc(mean))

#Plot tuning results
autoplot(mlp_CV_results)

#Finalize and fit MLP workflow
mlp_final_wf <-
  mlp_wf %>%
  finalize_workflow(mlp_bestTune) %>%
  fit(data = train_data)

#Get MLP Predictions
mlp_predictions <- predict(mlp_final_wf,
                          new_data = test_data,
                          type = "prob")

#Remove p(0) column from df
mlp_predictions <- mlp_predictions %>% 
  select(-.pred_0) %>%
  #rename .pred_1 as "Action" for kaggle submission
  rename(Action = .pred_1)

# Combine with test_data ID
mlp_kaggle_submission <- bind_cols(
  test_data %>% select(id),
  mlp_predictions
)

#write MLP submission df to CSV for submission
vroom_write(mlp_kaggle_submission, "MLPSubmission.csv" ,delim = ",")
