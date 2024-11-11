# Author: Ryan Yee
# Purpose: fit a single GAM to all players
# Details: 
# Dependencies: gam, MLmetrics

data_dir = "../"

# load data
load(paste0(data_dir, "model_data_2023.RData"))

# settings
args = commandArgs(TRUE)
test_fold = as.numeric(args[1])
test_index = test_split_list[[test_fold]]

# training data
Y_train = as.integer(Y[-test_index])
X_cat_train = as.integer(std_x_cat$std_player_id[-test_index])
Z_mat_train = as.matrix(std_z_mat[-test_index,])
train_df = data.frame(Z_mat_train, player = X_cat_train, make = Y_train)

# testing data
Y_test = as.integer(Y[test_index])
X_cat_test = as.integer(std_x_cat$std_player_id[test_index])
Z_mat_test = as.matrix(std_z_mat[test_index,])
test_df = data.frame(Z_mat_test, player = X_cat_test, make = Y_test)

# fit model
fit = mgcv::gam(make ~ s(baseline, sideline, by = player), family = binomial(), data = train_df)

# held-out predictions
p_test = predict(fit, newdata = test_df, type = "response")

# prediction error
rmse_train = MLmetrics::RMSE(fit$fitted.values, Y_train)
rmse_test = MLmetrics::RMSE(p_test, Y_test)

log_loss_train = MLmetrics::LogLoss(fit$fitted.values, Y_train)
log_loss_test = MLmetrics::LogLoss(p_test, Y_test)

mcr_train = 1 - MLmetrics::Accuracy(ifelse(fit$fitted.values>0.5,1,0), Y_train)
mcr_test = 1 - MLmetrics::Accuracy(ifelse(p_test>0.5,1,0), Y_test)

results = list(
  rmse_train = rmse_train, 
  rmse_test = rmse_test, 
  log_loss_train = log_loss_train, 
  log_loss_test = log_loss_test, 
  mcr_train = mcr_train,
  mcr_test = mcr_test
)

# save
name = paste0("player_gam_", test_fold)
assign(name, results)
save(list = name, file = paste0(name, ".RData"))

