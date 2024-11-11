# Author: Ryan Yee
# Purpose: generate data for Friedman study
# Details: 
# Dependencies: scales

# hyperparameters
p = hypers$p
n = hypers$n
tau = hypers$tau
n_folds = hypers$n_folds
test_fold = hypers$test_fold

# friedman function
friedman = function(X){
  return(10 * sin(pi * X[,1] * X[,2]) + 20 * (X[,3] - 0.5)^2 + 10 * X[,4] + 5 * X[,5])
}

# generate covariates
X = matrix(runif(n = p * n), ncol = p)
mu = friedman(X)
Y = mu + rnorm(n, mean = 0, sd = (1 / tau))

# rescale covariates
std_X = scales::rescale(X, to = c(-1, 1), from = c(0, 1))

# generate folds
folds = sample(rep(1:n_folds, times = n / n_folds), size = n, replace = FALSE)

# train / test split
std_X_cont_train = std_X[which(folds != test_fold), ]
mu_train = mu[which(folds != test_fold)]
Y_train = Y[which(folds != test_fold)]
std_X_cont_test = std_X[which(folds == test_fold), ]
mu_test = mu[which(folds == test_fold)]
Y_test = Y[which(folds == test_fold)]

