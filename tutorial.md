# Friedman function example

To demonstrate the use of **ridgeBART**, we will fit a model to the [Friedman Function](https://www.sfu.ca/~ssurjano/fried.html) and compare the speed and fit to a model fit with the standard **BART** package.

```r
f = function(x_cont){
  if(ncol(x_cont) < 5) stop("x_cont needs to have at least five columns")
  if(!all(abs(x_cont) <= 1)){
    stop("all entries in x_cont must be between -1 and 1")
  } else{
    x <- (x_cont+1)/2 # convert to [0,1]
    return(10 * sin(pi*x[,1]*x[,2]) + 20 * (x[,3] - 0.5)^2 + 10*x[,4] + 5 * x[,5])
  }
}
```

Note: the Friedman function is defined over the unit square but **ridgeBART** assumes that continuous predictors lie in \[-1,1\]. 
So in this example, we had to transform our predictors to the unit interval.

Now, let's generate some data:

```r
n = 2000
p = 5
sigma = 1

# generate data
set.seed(2024)
X = matrix(unif(n = n * p), nrow = n, ncol = p)
mu = f(X)
Y = mu + rnorm(n, mean = 0, sd = sigma)

# train / test split
folds = 10
test_fold = sample(1:n, size = n / folds, replace = FALSE)
X_train = X[!test_fold,]
mu_train = mu[!test_fold]
Y_train = mu[!test_fold]
X_test = X[test_fold,]
mu_test = mu[test_fold]
Y_test = mu[test_fold]

# output containters
rmse_train = c("ridgeBART" = NA, "BART" = NA)
rmse_test = c("ridgeBART" = NA, "BART" = NA)
timing = c("ridgeBART" = NA, "BART" = NA)
```

Now we are ready to run both `ridgeBART::ridgeBART()` and `BART::wbart()`.
Note that we have hidden the printed output.

```r
bart_time = system_time(
  bart_fit <- BART::wbart(
    x.train = X_train,
    y.train = Y_train,
    x.test = X_test
  )
)
rmse_train["BART"] = sqrt(mean( (mu_train - bart_fit$yhat.train.mean)^2 ))
rmse_test["BART"] = sqrt(mean( (mu_test - bart_fit$yhat.test.mean)^2 ))
timing["BART"] = bart_time["elapsed"]
```

Recall that the main feature of **ridgeBART** is that it can smooth over a t"target" set of covariates.
Since we do not have a "target" set of covariates over which to smooth, we will smooth and split over all covariates.
We do this by setting both `X_cont_train` (i.e. the splitting covariates) and `Z_mat_train` (i.e. the smoothing covariates) equal to `X_train` and respectively for the test data.
This will results in a function recovery which is piecewise smooth.
We will use a ReLU activation, which worked the best in our experiments.

```r
ridge_time = system_time(
  ridge_fit <- ridgeBART::ridgeBART( 
    Y_train = Y_train,
    X_cont_train = X_train,
    Z_mat_train = X_train,
    X_test = X_test,
    Z_mat_test = X_test,
    activation = "ReLU"
  )
)
rmse_train["BART"] = sqrt(mean( (mu_train - ridge_fit$yhat.train.mean)^2 ))
rmse_test["BART"] = sqrt(mean( (mu_test - ridge_fit$yhat.test.mean)^2 ))
timing["BART"] = ridge_time["elapsed"]
```

```r
print("Training RMSE")
print(round(rmse_train, digits = 3))

print("Testing RMSE")
print(round(rmse_test, digits = 3))

print("Timing (seconds):")
print(round(timing, digits = 3))
```

