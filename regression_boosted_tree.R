# Load libraries and set seed
library(tidyverse)
library(tidymodels)
library(lubridate)
set.seed(42)

# process training set
loans <- read_csv("data/train.csv") %>%
  select(-purpose, -grade, -id, -emp_title) %>%
  mutate(
    addr_state = factor(addr_state),
    application_type = factor(application_type),
    earliest_cr_line = as.numeric(parse_date(earliest_cr_line, format = "%b-%Y")),
    emp_length = factor(
      emp_length,
      ordered = TRUE,
      levels = c(
        "< 1 year",
        "1 year",
        "2 years",
        "3 years",
        "4 years",
        "5 years",
        "6 years",
        "7 years",
        "8 years",
        "9 years",
        "10+ years",
        "n/a"
      )
    ),
    home_ownership = factor(home_ownership),
    initial_list_status = factor(initial_list_status),
    last_credit_pull_d = as.numeric(parse_date(last_credit_pull_d, format = "%b-%Y")),
    sub_grade = factor(sub_grade, ordered = TRUE),
    term = factor(term),
    verification_status = factor(verification_status)
  )

# process testing set
testing_data <- read_csv("data/test.csv") %>%
  select(-purpose, -grade, -id, -emp_title) %>%
  mutate(
    addr_state = factor(addr_state),
    application_type = factor(application_type),
    earliest_cr_line = as.numeric(parse_date(earliest_cr_line, format = "%b-%Y")),
    emp_length = factor(
      emp_length,
      ordered = TRUE,
      levels = c(
        "< 1 year",
        "1 year",
        "2 years",
        "3 years",
        "4 years",
        "5 years",
        "6 years",
        "7 years",
        "8 years",
        "9 years",
        "10+ years",
        "n/a"
      )
    ),
    home_ownership = factor(home_ownership),
    initial_list_status = factor(initial_list_status),
    last_credit_pull_d = as.numeric(parse_date(last_credit_pull_d, format = "%b-%Y")),
    sub_grade = factor(sub_grade, ordered = TRUE),
    term = factor(term),
    verification_status = factor(verification_status)
  )

# no split needed since test set is already seperate
loan_folds <- vfold_cv(data = loans, v = 10, repeats = 3, strata =  money_made_inv)

# Create recipe, remove id column
loan_recipe1 <- recipe(money_made_inv ~ ., data = loans) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_novel(all_nominal(), -all_outcomes()) %>%
  step_normalize(all_numeric(), -all_outcomes())

# Make model
bt_model <- boost_tree(mode = "regression",
                       mtry = tune(),
                       min_n = tune(),
                       trees = tune(),
                       learn_rate = tune()) %>%
  set_engine("xgboost")

# Make workflow
bt_workflow <- workflow() %>%
  add_model(bt_model) %>%
  add_recipe(loan_recipe1)

# Update tuning parameters
bt_params <- parameters(bt_workflow) %>%
  update(mtry = mtry(range = c(1,35)))
bt_grid <- grid_regular(bt_params,levels = 3)

# Save output
save(bt_workflow, loan_folds, bt_grid, file = "bt_tuned_inputs.rda")

# --- Run bt_tune.R ---
load("bt_tuned_inputs.rda")
control <- control_resamples(verbose = TRUE)
bt_tuned <- bt_workflow %>%
  tune_grid(loan_folds, grid = bt_grid, control = control)
saveRDS(bt_tuned, "bt_tuned.rds")

# # Load tuned boosted tree
bt_tuned <- readRDS("bt_tuned.rds")

# Pick optimal tuning params
show_best(bt_tuned, metric = "rmse")
bt_results <- bt_workflow %>%
  finalize_workflow(select_best(bt_tuned, metric = "rmse")) %>%
  fit(loans)

# Predict test set
bt_predictions <- predict(bt_results, new_data = testing_data) %>%
  bind_cols(testing_data %>% select(id)) %>%
  rename(
    Predicted = .pred,
    Id = id
  )

# Write out predictions
write_csv(bt_predictions, "boosted_tree_predictions.csv")
