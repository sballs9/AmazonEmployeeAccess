# Amazon Principal Component Analysis

library(tidymodels)
library(vroom)
library(themis)
library(embed)
library(doParallel)

cl <- makePSOCKcluster(8)

registerDoParallel(cl)

training_data <- vroom("amazon_train.csv")
testing_data <- vroom("amazon_test.csv")

training_data$ACTION <- as.factor(training_data$ACTION)

my_recipe <- recipe(ACTION ~ ., data=training_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) 

tree_grid <- grid_regular(
  mtry(range = c(1, 9)), min_n(),            
  levels = 3                             
)

cv_folds <- vfold_cv(training_data, v = 10, repeats = 1)

tuned_results <- rf_wf |>
  tune_grid(resamples = cv_folds, 
            grid = tree_grid, 
            metrics = metric_set(roc_auc))

best_params <- tuned_results |> 
  select_best(metric = "roc_auc")

best_params

rf_final_wf <- rf_wf %>% 
  finalize_workflow(best_params) %>%
  fit(training_data)

amazon_predictions <- predict(rf_final_wf,
                              new_data = testing_data,
                              type = "prob") 


amazon_predictions <- amazon_predictions$.pred_1

submission <- testing_data %>%
  dplyr::select(id) %>%  # Ensure 'id' is the correct column name
  mutate(ACTION = amazon_predictions) # Add predicted probabilities

# Save the submission file
vroom_write(x=submission, file="./randforest_preds.csv", delim=",")

stopCluster(cl)

