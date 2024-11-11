# Author: Ryan Yee
# Purpose: get predictions for shot chart visualizations
# Details: 
# Dependencies: ridgeBART, BART

data_dir = "../"
script_dir = ""

source(paste0(script_dir, "probit_ridgeBART_wrapper.R"))
source(paste0(script_dir, "pbart_wrapper.R"))

### settings ###
models = c("cos", "tanh", "ReLU", "pbart")
players = c("king", "freak", "chef", "bs")
settings = expand.grid(model = models, player = players)

### arguments ###
args = commandArgs(TRUE)
job_id = as.numeric(args[1]) + 1
model = settings$model[job_id]
player = settings$player[job_id]

### load data ###
load(paste0(data_dir, "model_data_2023.RData"))

# training data
Y_train = as.integer(Y)
X_cont_train = as.matrix(std_x_cont[, c("height", "weight")])
X_cat_train = as.matrix(std_x_cat)
Z_mat_train = as.matrix(std_z_mat)

# testing data
if (player == "king"){
  std_id = player_index$std_player_id[which(player_index$player_name == "LeBron James")]
  std_pos = std_x_cat$std_position[min(which(std_x_cat$std_player_id == std_id))]
  height = std_x_cont$height[min(which(std_x_cat$std_player_id == std_id))]
  weight = std_x_cont$weight[min(which(std_x_cat$std_player_id == std_id))]
} else if (player == "freak"){
  std_id = player_index$std_player_id[which(player_index$player_name == "Giannis Antetokounmpo")]
  std_pos = std_x_cat$std_position[min(which(std_x_cat$std_player_id == std_id))]
  height = std_x_cont$height[min(which(std_x_cat$std_player_id == std_id))]
  weight = std_x_cont$weight[min(which(std_x_cat$std_player_id == std_id))]
} else if (player == "chef"){
  std_id = player_index$std_player_id[which(player_index$player_name == "Stephen Curry")]
  std_pos = std_x_cat$std_position[min(which(std_x_cat$std_player_id == std_id))]
  height = std_x_cont$height[min(which(std_x_cat$std_player_id == std_id))]
  weight = std_x_cont$weight[min(which(std_x_cat$std_player_id == std_id))]
} else if (player == "bs"){
  std_id = player_index$std_player_id[which(player_index$player_name == "Ben Simmons")]
  std_pos = std_x_cat$std_position[min(which(std_x_cat$std_player_id == std_id))]
  height = std_x_cont$height[min(which(std_x_cat$std_player_id == std_id))]
  weight = std_x_cont$weight[min(which(std_x_cat$std_player_id == std_id))]
}

# shot chart grid points
# data precision is to the nearest foot, predicting every half foot
baseline = scales::rescale(seq(from = baseline_range[1], to = baseline_range[2], by = 0.5), to = c(-.99, .99))
sideline = scales::rescale(seq(from = sideline_range[1], to = sideline_range[2], by = 0.5), to = c(-.99, .99))
Z_mat_test = as.matrix(expand.grid(baseline = baseline, sideline = sideline))
X_cont_test = matrix(rep(c(height, weight), each = nrow(Z_mat_test)), nrow = nrow(Z_mat_test))
X_cat_test = matrix(rep(c(std_id, std_pos), each = nrow(Z_mat_test)), nrow = nrow(Z_mat_test))

if (model == "pbart"){
  X_train_df = as.data.frame(cbind(X_cont_train, X_cat_train, std_z_mat))
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

# save predictions
name = paste0(model, "_", player, "_predictions")
assign(name, fit$test)
save(list = name, file = paste0(name, ".RData"))

