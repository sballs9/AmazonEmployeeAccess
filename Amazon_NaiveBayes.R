# Amazon Naive Bayes

install.packages("discrim")
install.packages("naivebayes")

library(naivebayes)
library(discrim)
library(tidymodels)
library(vroom)

training_data <- vroom("amazon_train.csv")
testing_data <- vroom("amazon_test.csv")

training_data$ACTION <- as.factor(training_data$ACTION)

my_recipe <- recipe(ACTION ~ ., data=training_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) 

bayes_mod <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bayes_mod) 

parm_grid <- grid_regular(Laplace(), smoothness(),             
  levels = 5)

cv_folds <- vfold_cv(training_data, v = 5, repeats = 1)

tuned_results <- nb_wf |>
  tune_grid(resamples = cv_folds, 
            grid = parm_grid, 
            metrics = metric_set(roc_auc))

best_params <- tuned_results |> 
  select_best(metric = "roc_auc")

best_params

nb_final_wf <- nb_wf %>% 
  finalize_workflow(best_params) %>%
  fit(training_data)

amazon_predictions <- predict(nb_final_wf,
                              new_data = testing_data,
                              type = "prob") 


amazon_predictions <- amazon_predictions$.pred_1

submission <- testing_data %>%
  dplyr::select(id) %>%  # Ensure 'id' is the correct column name
  mutate(ACTION = amazon_predictions) # Add predicted probabilities

# Save the submission file
vroom_write(x=submission, file="./naivebayes_preds.csv", delim=",")