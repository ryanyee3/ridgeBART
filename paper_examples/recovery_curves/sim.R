# Author: Ryan Yee
# Purpose: recovery curve simulation study script
# Details: 
# Dependencies: ridgeBART, BART, SoftBart

dir = "recovery_curves/"
study = "recovery_curves"

source(paste0(dir, "ridgeBART_wrapper.R"))
source(paste0(dir, "wbart_wrapper.R"))
source(paste0(dir, "softBART_wrapper.R"))


# hyperparameters
# args = commandArgs(TRUE)
# job_id = as.numeric(args[1]) + 1
job_id = 1 # update to change simulation settings
source(paste0(dir, "settings.R"))
hypers = list(
  model = as.character(settings$model[job_id]),
  test = as.character(settings$test[job_id]),
  replicate = settings$replicate[job_id],
  n_ind = settings$n_ind[job_id],
  p_cont = settings$p_cont[job_id],
  lambda = settings$lambda[job_id],
  sigma = settings$sigma[job_id],
  n_test_old = settings$n_test_old[job_id],
  n_test_new = settings$n_test_new[job_id],
  n_trees = settings$n_trees[job_id],
  n_bases = settings$n_bases[job_id],
  int_opt = settings$int_opt[job_id],
  act_opt = settings$act_opt[job_id],
  n_chains = settings$n_chains[job_id]
)


# generate data
set.seed(101 * hypers$replicate)
source(paste0(dir, "gen_data.R"))


# fit model
if (hypers$model == "ridgeBART"){
  fit = ridgeBART_wrapper(
    Y_train = Y_train,
    X_cont_train = std_X_cont_train,
    Z_mat_train = std_Z_mat_train,
    X_cont_test = std_X_cont_test,
    Z_mat_test = std_Z_mat_test,
    M = hypers$n_trees, 
    n_bases = hypers$n_bases,
    activation = hypers$act_opt,
    n_chains = hypers$n_chains
  )
} else if (hypers$model == "wbart"){
  X_train_df = cbind(std_X_cont_train, std_Z_mat_train)
  X_test_df = cbind(std_X_cont_test, std_Z_mat_test)
  fit = wbart_wrapper(
    Y_train = Y_train,
    X_train_df = X_train_df,
    X_test_df = X_test_df,
    n_chains = hypers$n_chains
  )
} else if (hypers$model == "softbart"){
  X_train_df = cbind(std_X_cont_train, std_Z_mat_train)
  X_test_df = cbind(std_X_cont_test, std_Z_mat_test)
  fit = softBART_wrapper(
    Y_train = Y_train,
    X_train = X_train_df,
    X_test = X_test_df,
    n_chains = hypers$n_chains
  )
}


# results
mu_rmse_train = MLmetrics::RMSE(mu_train, fit$train$fit[,"MEAN"])
mu_rmse_test= MLmetrics::RMSE(mu_test, fit$test$fit[,"MEAN"])
tot_train_time = sum(fit$timing)

results = list(
  job_id = job_id,
  model = hypers$model,
  test = hypers$test,
  replicate = hypers$replicate,
  n_train = length(Y_train),
  n_ind = hypers$n_ind,
  p_cont = hypers$p_cont,
  lambda = hypers$lambda,
  sigma = hypers$sigma,
  n_test_old = hypers$n_test_old,
  n_test_new = hypers$n_test_new,
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
assign(name, results)
save(results, file = paste0(name, ".RData"))

if (hypers$replicate == 1 & hypers$test == "old"){
  save(
    fit,
    std_X_cont_train,
    std_X_cat_train,
    std_Z_mat_train,
    std_X_cont_test,
    std_X_cat_test,
    std_Z_mat_test,
    mu_train,
    mu_test,
    Y_train,
    Y_test,
    file = paste0(hypers$model, hypers$act_opt, "_n", hypers$n_ind, "_", hypers$test, ".RData")
  )
}

