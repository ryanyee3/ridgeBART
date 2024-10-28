# Author: Ryan Yee
# Purpose: friedman simulation study script
# Details: 
# Dependencies: ridgeBART, BART, SoftBart, MLmetrics

dir = "friedman_function/"
study = "friedman_function"

source(paste0(dir, "ridgeBART_wrapper.R"))
source(paste0(dir, "wbart_wrapper.R"))
source(paste0(dir, "softBART_wrapper.R"))
source(paste0(dir, "gpbart_wrapper.R"))

# hyperparameters
# args = commandArgs(TRUE)
# job_id = as.numeric(args[1]) + 1
job_id = 1 # update to change the simulation settings
source(paste0(dir, "settings.R"))
hypers = list(
  model = as.character(settings$model[job_id]),
  n = settings$n[job_id],
  p = settings$p[job_id],
  tau = settings$tau[job_id],
  n_folds = settings$n_folds[job_id],
  test_fold = settings$test_fold[job_id],
  n_trees = settings$n_trees[job_id],
  n_bases = settings$n_bases[job_id],
  int_opt = settings$int_opt[job_id],
  act_opt = settings$act_opt[job_id],
  n_chains = settings$n_chains[job_id],
  nd = settings$nd[job_id],
  burn = settings$burn[job_id]
)

# generate data
set.seed(10101)
source(paste0(dir, "gen_data.R"))

# fit model
if (hypers$model == "ridgeBART"){
  fit = ridgeBART_wrapper(
    Y_train = Y_train,
    X_cont_train = std_X_cont_train,
    Z_mat_train = std_X_cont_train,
    X_cont_test = std_X_cont_test,
    Z_mat_test = std_X_cont_test,
    M = hypers$n_trees, 
    n_bases = hypers$n_bases,
    activation = hypers$act_opt,
    n_chains = hypers$n_chains,
    nd = hypers$nd, burn = hypers$nd
  )
} else if (hypers$model == "wbart"){
  fit = wbart_wrapper(
    Y_train = Y_train,
    X_train_df = std_X_cont_train,
    X_test_df = std_X_cont_test,
    n_chains = hypers$n_chains,
    nd = hypers$nd, burn = hypers$nd
  )
} else if (hypers$model == "softbart"){
  fit = softBART_wrapper(
    Y_train = Y_train,
    X_train = std_X_cont_train,
    X_test = std_X_cont_test,
    n_chains = hypers$n_chains,
    nd = hypers$nd, burn = hypers$nd
  )
} else if (hypers$model == "flexBART"){
  fit = flexBART_wrapper(
    Y_train = Y_train,
    X_cont_train = std_X_cont_train,
    X_cont_test = std_X_cont_test,
    n_chains = hypers$n_chains,
    nd = hypers$nd, burn = hypers$nd
  )
} else if (hypers$model == "gpbart"){
  fit = gpbart_wrapper(
    Y_train = Y_train,
    X_train_df = as.data.frame(std_X_cont_train),
    X_test_df = as.data.frame(std_X_cont_test),
    n_chains = hypers$n_chains,
    nd = hypers$nd, burn = hypers$nd
  )
}

# results
mu_rmse_train = MLmetrics::RMSE(mu_train, fit$train$fit[,"MEAN"])
mu_rmse_test= MLmetrics::RMSE(mu_test, fit$test$fit[,"MEAN"])
tot_train_time = sum(fit$timing)

results = list(
  job_id = job_id,
  model = hypers$model,
  n = hypers$n,
  p = hypers$p,
  tau = hypers$tau,
  n_folds = hypers$n_folds,
  test_fold = hypers$test_fold,
  n_trees = hypers$n_trees,
  n_bases = hypers$n_bases,
  int_opt = hypers$int_opt,
  act_opt = hypers$act_opt,
  n_chains = hypers$n_chains,
  mu_rmse_train = mu_rmse_train,
  mu_rmse_test = mu_rmse_test,
  tot_train_time = tot_train_time
)

name = paste0(study, "_", job_id, "_results")
save(results, file = paste0(name, ".RData"))

