# Author: Ryan Yee
# Date: October 23, 2024
# Purpose: train benchmark models for basketball data
# Details: 
# Dependencies: rffBART, flexBART, BART, MLmetrics

study = "bart_cv"
data_dir = "../"
script_dir = ""

source(paste0(script_dir, "probit_ridgeBART_wrapper.R"))
source(paste0(script_dir, "pbart_wrapper.R"))

# load data
load(paste0(data_dir, "model_data_2023.RData"))

# settings
args = commandArgs(TRUE)
job_id = as.numeric(args[1]) + 1
source(paste0(script_dir, "settings.R"))
model = as.character(settings$model[job_id])
split = as.integer(settings$split[job_id])
test_index = test_split_list[[split]]
n_chains = 10

# training data
Y_train = as.integer(Y[-test_index])
X_cont_train = as.matrix(std_x_cont[-test_index, c("height", "weight")])
X_cat_train = as.matrix(std_x_cat[-test_index, ])
Z_mat_train = as.matrix(std_z_mat[-test_index, ])

# testing data
Y_test = as.integer(Y[test_index])
X_cont_test = as.matrix(std_x_cont[test_index, c("height", "weight")])
X_cat_test = as.matrix(std_x_cat[test_index, ])
Z_mat_test = as.matrix(std_z_mat[test_index, ])

if (model == "pbart"){
  X_train_df = as.data.frame(cbind(X_cont_train, X_cat_train, Z_mat_train))
  X_train_df$std_player_id = as.factor(X_train_df$std_player_id)
  X_train_df$std_position = as.factor(X_train_df$std_position)
  
  X_test_df = as.data.frame(cbind(X_cont_test, X_cat_test, Z_mat_test))
  colnames(X_test_df) = colnames(X_train_df)
  X_test_df$std_player_id= factor(X_test_df$std_player_id, levels = player_index$std_player_id)
  X_test_df$std_position = factor(X_test_df$std_position, levels = c(0, 1, 2))
}

# fit model
if (model == "cos"){
  fit = probit_ridgeBART_wrapper(
    Y_train = Y_train,
    X_cont_train = X_cont_train,
    X_cat_train = X_cat_train,
    Z_mat_train = Z_mat_train,
    X_cont_test = X_cont_test,
    X_cat_test = X_cat_test,
    Z_mat_test = Z_mat_test,
    cat_levels_list = cat_levels_list,
    activation = "cos",
    M = 50, n_bases = 1,
    rho_nu = 3,
    rho_lambda = qchisq(.5, df = 3) / 3,
    n_chains = 10
  )
} else if (model == "tanh"){
  fit = probit_ridgeBART_wrapper(
    Y_train = Y_train,
    X_cont_train = X_cont_train,
    X_cat_train = X_cat_train,
    Z_mat_train = Z_mat_train,
    X_cont_test = X_cont_test,
    X_cat_test = X_cat_test,
    Z_mat_test = Z_mat_test,
    cat_levels_list = cat_levels_list,
    activation = "tanh",
    M = 50, n_bases = 1,
    rho_nu = 3,
    rho_lambda = qchisq(.5, df = 3) / 3,
    n_chains = 10
  )
} else if (model == "ReLU"){
  fit = probit_ridgeBART_wrapper(
    Y_train = Y_train,
    X_cont_train = X_cont_train,
    X_cat_train = X_cat_train,
    Z_mat_train = Z_mat_train,
    X_cont_test = X_cont_test,
    X_cat_test = X_cat_test,
    Z_mat_test = Z_mat_test,
    cat_levels_list = cat_levels_list,
    activation = "ReLU",
    M = 50, n_bases = 1,
    rho_nu = 3,
    rho_lambda = qchisq(.5, df = 3) / 3,
    n_chains = 10
  )
} else if (model == "pbart"){
  fit = pbart_wrapper(
    Y_train = Y_train,
    X_train_df = X_train_df,
    X_test_df = X_test_df,
    n_chains = 10
  )
}

#results
log_loss_train = MLmetrics::LogLoss(fit$train[,"MEAN"], Y_train)
log_loss_test = MLmetrics::LogLoss(fit$test[,"MEAN"], Y_test)

brier_train = MLmetrics::MSE(fit$train[,"MEAN"], Y_train)
brier_test = MLmetrics::MSE(fit$test[,"MEAN"], Y_test)

mcr_train = 1 - MLmetrics::Accuracy(ifelse(fit$train[,"MEAN"] > .5, 1, 0), Y_train)
mcr_test = 1 - MLmetrics::Accuracy(ifelse(fit$test[,"MEAN"] > .5, 1, 0), Y_test)

train_time = fit$timing

results = list(
  model = model,
  split = split,
  log_loss_train = log_loss_train,
  log_loss_test = log_loss_test,
  brier_train = brier_train,
  brier_test = brier_test,
  mcr_train = mcr_train,
  mcr_test = mcr_test,
  train_time = train_time
)

name = paste0(study, "_", model, "_", split, "_", job_id, "_results")
assign(name, results)
save(list = name, file = paste0(name, ".RData"))

