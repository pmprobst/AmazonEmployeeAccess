#AmazonAnalysis
library(tidymodels)
library(embed)

#Read data in, set ACTION feature as factor
train_data <- vroom("data/train.csv") %>%
  mutate(ACTION = factor(ACTION))
test_data <- vroom("data/test.csv")

#Create Recipe
my_recipe <- recipe(ACTION ~ . ,data = train_data) %>%
  step_mutate_at(all_numeric_predictors() ,fn = factor) %>%
  step_other(all_nominal_predictors() ,threshold = .001) %>%
  step_dummy(all_nominal_predictors())

#Prep & Bake Recipe
prep <- prep(my_recipe)
baked <- bake(prep ,new_data = train_data)

#Set Up Logistic Regression Model
logRegModel <- logistic_reg() %>%
  set_engine("glm")

#Set Workflow
wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel)

#Train Logistic Regression Model
train <- fit(wf, train_data)

#Get Predictions
predictions <-predict(train,
                      new_data = test_data
                      ,type = "prob")

#Remove p(0) column from df
predictions <- predictions %>% select(-pr0)
#waiting to see what the p(0) column name is
