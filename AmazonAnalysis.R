#AmazonAnalysis
library(tidymodels)
library(embed)
library(vroom)

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

#Set Up K Nearest Neighbors
knn_model <- nearest_neighbor(
  mode      = "classification",
  neighbors = tune(),
  weight_func = "rectangular",  # standard unweighted KNN
  dist_power = 2
  ) %>% set_engine("kknn")

#Set Workflow
wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

#set up grid of tuning values
tuning_grid <- grid_regular(
                  neighbors(range = c(1L, 51L)),
                  levels = 20)

folds <- vfold_cv(train_data ,v = 10 ,repeats = 1)

#CV
CV_results <- wf %>%
  tune_grid(resamples = folds
              ,grid = tuning_grid
              ,metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

#finalize and fit workflow
final_wf <-
  wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

#Identify the best levels of penalty and mixture (highest mean)
CV_results %>%
  collect_metrics() %>%
  arrange(desc(mean))

#plot levels of penalty and mixture
autoplot(CV_results)

#Get Predictions
predictions <-predict(final_wf,
                      new_data = test_data
                      ,type = "prob")

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
vroom_write(kaggle_submission, "PenLogRegModelSubmission.csv" ,delim = ",")
