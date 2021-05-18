library(tidyverse)
library(tidymodels)
set.seed(42)

load("bt_tuned_inputs.rda")
control <- control_resamples(verbose = TRUE)
bt_tuned <- bt_workflow %>%
  tune_grid(loan_folds, grid = bt_grid, control = control)
saveRDS(bt_tuned, "bt_tuned.rds")
