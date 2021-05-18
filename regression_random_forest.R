# Load libraries and set seed
library(tidyverse)
library(tidymodels)
library(lubridate)
set.seed(42)

# process training set
loans <- read_csv("data/train.csv") %>%
  select(-purpose, -grade, -emp_title) %>%
  mutate(
    addr_state = factor(addr_state),
    application_type = factor(application_type),
    earliest_cr_line = parse_date(earliest_cr_line, format = "%b-%Y"),
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
    last_credit_pull_d = parse_date(last_credit_pull_d, format = "%b-%Y"),
    sub_grade = factor(sub_grade, ordered = TRUE),
    term = factor(term),
    verification_status = factor(verification_status)
  )

# process testing set
testing_data <- read_csv("data/test.csv") %>%
  select(-purpose, -grade, -emp_title) %>%
  mutate(
    addr_state = factor(addr_state),
    application_type = factor(application_type),
    earliest_cr_line = parse_date(earliest_cr_line, format = "%b-%Y"),
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
    last_credit_pull_d = parse_date(last_credit_pull_d, format = "%b-%Y"),
    sub_grade = factor(sub_grade, ordered = TRUE),
    term = factor(term),
    verification_status = factor(verification_status)
  )

# no split needed since test set is already seperate
loan_folds <- vfold_cv(data = loans, v = 10, repeats = 3, strata =  money_made_inv)
ggplot(loans) +
  geom_histogram(mapping = aes(money_made_inv))

# Create recipe, remove id column
loan_recipe1 <- recipe(money_made_inv ~ ., data = loans) %>%
  step_rm(contains("id")) %>%
  step_date(earliest_cr_line, last_credit_pull_d, features = c("year","month")) %>%
  step_rm(earliest_cr_line,last_credit_pull_d) %>%
  step_other(all_nominal(), threshold = 0.005) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_novel(all_nominal(), -all_outcomes()) %>%
  step_normalize(all_numeric(), -all_outcomes())


# Make model
rf_model <- rand_forest(mode = "regression",
                        mtry = tune(),
                        trees = 1000,
                        min_n = tune())%>%
  set_engine("ranger")

# Make workflow
rf_workflow <- workflow() %>%
  add_recipe(loan_recipe1) %>%
  add_model(rf_model)

# Update tuning parameters
### Random forest
rf_params <- parameters(rf_workflow) %>%
  update(mtry = mtry(range = c(1,36)))
rf_grid <- grid_regular(rf_params, levels = 5)

# Save output
save(rf_workflow, loan_folds, rf_grid, file = "rf_tuned_inputs.rda")

# --- Run rf_tune.R ---
control <- control_resamples(verbose = TRUE)
rf_tuned <- rf_workflow %>%
  tune_grid(loan_folds, grid = rf_grid, control = control)
saveRDS(rf_tuned, "rf_tuned.rds")

# # Load tuned random forest
rf_tuned <- readRDS("rf_tuned.rds")

# Pick optimal tuning params
show_best(rf_tuned, metric = "rmse")
rf_results <- rf_workflow %>%
  finalize_workflow(select_best(rf_tuned, metric = "rmse")) %>%
  fit(loans)

# Predict test set
rf_predictions <- predict(rf_results, new_data = testing_data) %>%
  bind_cols(testing_data %>% select(id)) %>%
  rename(
    Predicted = .pred,
    Id = id
  )

# Write out predictions
write_csv(rf_predictions, "random_forest_predictions.csv")
