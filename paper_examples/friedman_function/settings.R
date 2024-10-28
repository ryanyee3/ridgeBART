# Author: Ryan Yee
# Date: October 9, 2024
# Purpose: settings for Friedman study
# Details: 
# Dependencies: scales

# data factors
n = c(500, 1000, 2000, 4000)
p = 5
tau = 1
n_folds = 25
test_fold = 1:25

# ridgeBART factors
n_trees = 50
n_bases = 1
act_opt = c("ReLU", "cos", "tanh")

# sampling settings
n_chains = 10
nd = 1000
burn = 1000

ridge = expand.grid(
  model = "ridgeBART",
  n = n,
  p = p,
  tau = tau,
  n_folds = n_folds,
  test_fold = test_fold,
  n_trees = n_trees,
  n_bases = n_bases,
  act_opt = act_opt,
  n_chains = n_chains,
  nd = nd,
  burn = burn
)

comps = expand.grid(
  model = c("wbart", "softbart", "gpbart", "flexBART"),
  n = n,
  p = p,
  tau = tau,
  n_folds = n_folds,
  test_fold = test_fold,
  n_trees = 0,
  n_bases = 0,
  act_opt = NA,
  n_chains = n_chains,
  nd = nd,
  burn = burn
)

settings = rbind(ridge, comps)
