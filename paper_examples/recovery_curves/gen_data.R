# Author: Ryan Yee
# Purpose: generate data for recovery curve simulation study
# Details: 
# Dependencies: scales

prev_state = ls()

# hyperparameters
n_ind = hypers$n_ind # number of individuals
lambda = hypers$lambda # number of obs. for each individual drawn from 1 + Poisson(lambda)
p_cont = hypers$p_cont # number of continuous predictors
sigma = hypers$sigma # measurement error
n_test_old = hypers$n_test_old # number of grid points to test old X values
n_test_new = hypers$n_test_new # number of observations to test new X values


# data-generating functions

# asymptotic drop
# values for this function are in [-.05, .15] for x in [-1, 1]
A = function(x){return(0.5 * abs((1 / (1 + exp(-2 * x))) - 0.5) - 0.05)}

# initial drop
# values for this function are in [.85, 1] for x in [-1, 1]
B = function(x){return(pmin(1, 1 + cos(5 * x) * 0.15))}

# recovery rate
# values for this function are in [3.6, 27.2] for x in [-1, 1]
C = function(x){return(5 * exp(x))}

# recovery curve function (https://arxiv.org/pdf/1504.06964)
rc = function(t, A, B, C){return((1 - A) * (1 - B * exp(-t / C)))}

# plots of functions
# png(filename = paste0("figures/", "rc_examples.png"), width = 5.5, height = 4, units = "in", res = 500)
# par(mar = c(3, 3, 1, 1), mgp = c(1.8, 0.5, 0), mfrow = c(1, 1))
# plt_grid = seq(-1, 1, by = 0.01)
# plot(plt_grid, A(plt_grid), type = "l", xlab = "x", ylab = "A(x)", main = "Function A")
# plot(plt_grid, B(plt_grid), type = "l", xlab = "x", ylab = "B(x)", main = "Function B")
# 
# # plot 20 recovery curves
# start = 0
# stop = 25
# x = matrix(runif(20 * p_cont, min = -1, max = 1), ncol = p_cont)
# time_grid = seq(start, stop, by = 0.05)
# plot(0, type = "n", xlim = c(start, stop), ylim = c(0, 1), xlab = "t", ylab = "f(t)")
# for (i in 1:nrow(x)){
# 
#   # individual parameters
#   tmp = x[i,]
#   a = A(tmp[1])
#   b = B(tmp[2])
#   c = C(tmp[3])
# 
#   curve = rep(NA, times = length(time_grid))
#   for (j in 1:length(time_grid))curve[j] = rc(time_grid[j], a, b, c)
#   lines(time_grid, curve)
# }
# dev.off()


# generate data

# number of observations for each individual
n_obs = pmin(9, rpois(n_ind, lambda = lambda) + 1)

# determine time points of observations
time_point_list = list()
for (i in 1:n_ind){
  # let most of the observations be at the start
  months = sample(x = c(1, 2, 4, 6, 8, 12, 16, 20, 24), size = n_obs[i], replace = FALSE)
  days = rep(NA, times = n_obs[i])
  for (j in 1:n_obs[i]){
    # jitter day by +/- 15
    days[j] = 30 * months[j] + ceiling((rbeta(1, 2, 2) - .5) * 30 -.5)
  }
  time_point_list[[i]] = days / 30
}

# training features
X_cont_ind = matrix(runif(n_ind * p_cont, min = -1, max = 1), ncol = p_cont)
std_X_cont_train = matrix(ncol = p_cont, nrow = sum(n_obs))
std_X_cat_train = matrix(rep(NA, times = sum(n_obs)))
Z_mat_train = matrix(rep(NA, times = sum(n_obs)))
mu_train = rep(NA, times = sum(n_obs))
index = 1
for (i in 1:n_ind){
  tmp = X_cont_ind[i, ]
  a = A(tmp[1])
  b = B(tmp[2])
  c = C(tmp[3])
  for (j in 1:n_obs[i]){
    t = time_point_list[[i]][j]
    std_X_cont_train[index, ] = tmp
    std_X_cat_train[index, ] = i
    Z_mat_train[index, ] = t
    mu_train[index] = rc(t, a, b, c)
    index = index + 1
  }
}
Y_train = mu_train + rnorm(sum(n_obs), sd = sigma)

# # training data set
# png(filename = paste0("figures/", "rc_data.png"), width = 5.5, height = 4, units = "in", res = 500)
# par(mar = c(3, 3, 1, 1), mgp = c(1.8, 0.5, 0), mfrow = c(1, 1))
# plot(Z_mat_train, Y_train, xlab = "z", ylab = "y", pch = 20, cex = 0.7, col = scales::alpha("black", 0.5))
# dev.off()

# # plot a few examples
# start = 0
# stop = 25
# x = matrix(runif(20 * p_cont, min = -1, max = 1), ncol = p_cont)
# time_grid = seq(start, stop, by = 0.05)
# n_tmp = 9 # must be a perfect square
# ids = sample(1:n_ind, size = n_tmp, replace = FALSE)
# par(mar = c(3, 3, 2, 1), mgp = c(1.8, 0.5, 0), mfrow = c(sqrt(n_tmp), sqrt(n_tmp)))
# for (i in 1:n_tmp){
# 
#   # individual parameters
#   tmp = X_cont_ind[ids[i],]
#   a = A(tmp[1])
#   b = B(tmp[2])
#   c = C(tmp[3])
# 
#   # plot curve
#   curve = rep(NA, times = length(time_grid))
#   for (j in 1:length(time_grid))curve[j] = rc(time_grid[j], a, b, c)
#   plot(time_grid, curve, type = "l")
# 
#   # plot observations
#   tmp_rows = which(std_X_cat_train == ids[i])
#   points(Z_mat_train[tmp_rows], Y_train[tmp_rows], pch = 20, cex = 0.7, col = scales::alpha("black", 0.5))
# }

# testing features (same X)
std_X_cont_test_old = matrix(rep(X_cont_ind, each = n_test_old), ncol = p_cont)
std_X_cat_test_old = matrix(rep(1:n_ind, each = n_test_old))
Z_mat_test_old = matrix(rep(seq(0, 25, length.out = n_test_old), times = n_ind))
mu_test_old = rep(NA, times = (n_ind * n_test_old))
if (hypers$test == "old"){
  for (i in 1:(n_ind * n_test_old)){
    a = A(std_X_cont_test_old[i, 1])
    b = B(std_X_cont_test_old[i, 2])
    c = C(std_X_cont_test_old[i, 3])
    mu_test_old[i] = rc(Z_mat_test_old[i, ], a, b, c)
  }
}
Y_test_old = mu_test_old + rnorm((n_ind * n_test_old), sd = sigma)

# testing features (new X)
std_X_cont_test_new = matrix(runif(n_test_new * p_cont, min = -1, max = 1), ncol = p_cont)
std_X_cat_test_new = matrix(rep(0, times = n_test_new))
Z_mat_test_new = matrix(runif(n_test_new, min = 0, max = 25))
mu_test_new = rep(NA, times = n_test_new)
if (hypers$test == "new"){
  for (i in 1:n_test_new){
    a = A(std_X_cont_test_new[i, 1])
    b = B(std_X_cont_test_new[i, 2])
    c = C(std_X_cont_test_new[i, 3])
    mu_test_new[i] = rc(Z_mat_test_new[i, ], a, b, c)
  }
}
Y_test_new = mu_test_new + rnorm(n_test_new, sd = sigma)

# standardize Z values
std_Z_mat_train = scales::rescale(Z_mat_train, from = c(0, 25), to = c(-1, 1))
std_Z_mat_test_old = scales::rescale(Z_mat_test_old, from = c(0, 25), to = c(-1, 1))
std_Z_mat_test_new = scales::rescale(Z_mat_test_new, from = c(0, 25), to = c(-1, 1))

if (hypers$test == "old"){
  std_X_cont_test = std_X_cont_test_old
  std_X_cat_test = std_X_cat_test_old
  std_Z_mat_test = std_Z_mat_test_old
  mu_test = mu_test_old
  Y_test = Y_test_old
} else if (hypers$test == "new"){
  std_X_cont_test = std_X_cont_test_new
  std_X_cat_test = std_X_cat_test_new
  std_Z_mat_test = std_Z_mat_test_new
  mu_test = mu_test_new
  Y_test = Y_test_new
}

# clean up environment
keep = c(
  "std_X_cont_train", "std_X_cont_test", "std_X_cat_train", "std_X_cat_test",
  "std_Z_mat_train", "std_Z_mat_test", "mu_train", "mu_test", "Y_train", "Y_test"
  )
rm(list = setdiff(ls(), c(prev_state, keep)))
