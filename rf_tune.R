library(tidyverse)
library(tidymodels)
set.seed(42)

load("rf_tuned_inputs.rda")
control <- control_resamples(verbose = TRUE)
rf_tuned <- rf_workflow %>%
  tune_grid(loan_folds, grid = rf_grid, control = control)
saveRDS(rf_tuned, "rf_tuned.rds")
