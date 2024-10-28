# Author: Ryan Yee
# Purpose: hyperparameter setting for recovery curve simulation study
# Details: 
# Dependencies: 

prev_state = ls()

# data factors
n_ind = seq(200, 1000, by = 200)
lambda = 3 
p_cont = 6
sigma = 0.05
n_test_old = 41
n_test_new = 1000

# ridge hyperparameters
n_trees = 50
n_bases = 1
act_opt = c("ReLU", "cos", "tanh")

# misc
replicate = 1:10
n_chains = 10

ridge_old = expand.grid(
  model = "ridgeBART",
  test = "old",
  replicate = replicate,
  n_ind = n_ind,
  lambda = lambda,
  p_cont = p_cont,
  sigma = sigma,
  n_test_old = n_test_old,
  n_test_new = 0,
  n_trees = n_trees,
  n_bases = n_bases,
  act_opt = act_opt,
  n_chains = n_chains
)

ridge_new = expand.grid(
  model = "ridgeBART",
  test = "new",
  replicate = replicate,
  n_ind = n_ind,
  lambda = lambda,
  p_cont = p_cont,
  sigma = sigma,
  n_test_old = 0,
  n_test_new = n_test_new,
  n_trees = n_trees,
  n_bases = n_bases,
  act_opt = act_opt,
  n_chains = n_chains
)

comps_old = expand.grid(
  model = c("wbart", "softbart"),
  test = "old",
  replicate = replicate,
  n_ind = n_ind,
  lambda = lambda,
  p_cont = p_cont,
  sigma = sigma,
  n_test_old = n_test_old,
  n_test_new = 0,
  n_trees = 0,
  n_bases = 0,
  act_opt = NA,
  n_chains = n_chains
)

comps_new = expand.grid(
  model = c("wbart", "softbart"),
  test = "new",
  replicate = replicate,
  n_ind = n_ind,
  lambda = lambda,
  p_cont = p_cont,
  sigma = sigma,
  n_test_old = 0,
  n_test_new = n_test_new,
  n_trees = 0,
  n_bases = 0,
  act_opt = NA,
  n_chains = n_chains
)

settings = rbind(ridge_old, ridge_new, comps_old, comps_new)
rm(list = setdiff(ls(), c(prev_state, "settings")))
