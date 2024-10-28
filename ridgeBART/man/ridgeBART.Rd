\name{ridgeBART}
\alias{ridgeBART}
\title{
  BART with ridge function outputs.
}
\description{
  Implements Chipman et al. (2010)'s Bayesian additive regression trees (BART) method for nonparametric regression with continuous outcomes. 
  The regression function is represented as a sum of binary regression trees.
  Ridge functions with a user-specified activation are used as the output function for leaf nodes.
}
\usage{
  ridgeBART(
    Y_train,
    X_cont_train = matrix(0, nrow = 1, ncol = 1),
    X_cat_train = matrix(0, nrow = 1, ncol = 1),
    Z_mat_train = matrix(0, nrow = 1, ncol = 1),
    X_cont_test = matrix(0, nrow = 1, ncol = 1),
    X_cat_test = matrix(0L, nrow = 1, ncol = 1),
    Z_mat_test = matrix(0, nrow = 1, ncol = 1),
    unif_cuts = rep(TRUE, times = ncol(X_cont_train)), 
    cutpoints_list = NULL,
    cat_levels_list = NULL,
    sparse = FALSE, p_change = 0.2, sigma0 = 1.0,
    M = 50, n_bases = 1, activation = "ReLU",
    rho_alpha = 2 * M, rho_nu = 3, rho_lambda = stats::qchisq(.5, df = rho_nu) / rho_nu,
    nd = 1000, burn = 1000, thin = 1,
    save_samples = TRUE, save_trees = TRUE, 
    verbose = TRUE, print_every = floor( (nd*thin + burn))/10
  )
}
\arguments{
  \item{Y_train}{Vector of continuous responses for training data}
  \item{X_cont_train}{Matrix of continuous predictors for training data. Note, predictors must be re-scaled to lie in the interval [-1,1]. Default is a 1x1 matrix, which signals that there are no continuous predictors in the training data.}
  \item{X_cat_train}{Integer matrix of categorical predictors for training data. Note categorical levels should be 0-indexed. That is, if a categorical predictor has 10 levels, the values should run from 0 to 9. Default is a 1x1 matrix, which signals that there are no categorical predictors in the training data.}
  \item{Z_mat_train}{Matrix of smoothing predictors for training data. Note, predictors must be re-scaled to lie in the interval [-1,1]. Default is a 1x1 matrix, which signals that there are no continuous predictors in the training data.}
  \item{X_cont_test}{Matrix of continuous predictors for testing data. Default is a 1x1 matrix, which signals that testing data is not provided.}
  \item{X_cat_test}{Integer matrix of categorical predictors for testing data. Default is a 1x1 matrix, which signals that testing data is not provided.}
  \item{Z_mat_test}{Matrix of smoothing predictors for testing data. Note, predictors must be re-scaled to lie in the interval [-1,1]. Default is a 1x1 matrix, which signals that there are no continuous predictors in the training data.}
  \item{unif_cuts}{Vector of logical values indicating whether cutpoints for each continuous predictor should be drawn from a continuous uniform distribution (\code{TRUE}) or a discrete set (\code{FALSE}) specified in \code{cutpoints_list}. Default is \code{TRUE} for each variable in \code{X_cont_train}}
  \item{cutpoints_list}{List of length \code{ncol(X_cont_train)} containing a vector of cutpoints for each continuous predictor. By default, this is set to \code{NULL} so that cutpoints are drawn uniformly from a continuous distribution.}
  \item{cat_levels_list}{List of length \code{ncol(X_cat_train)} containing a vector of levels for each categorical predictor. If the j-th categorical predictor contains L levels, \code{cat_levels_list[[j]]} should be the vector \code{0:(L-1)}. Default is \code{NULL}, which corresponds to the case that no categorical predictors are available.}
  \item{sparse}{Logical, indicating whether or not to perform variable selection based on a sparse Dirichlet prior rather than uniform prior; see Linero 2018. Default is \code{FALSE}.}
  \item{p_change}{Probability of a change proposal in the MCMC. Defaults to 0.2.}
  \item{sigma0}{Prior variance. Defaults to 1.0.}
  \item{M}{Number of trees in the ensemble. Default is 50.}
  \item{n_bases}{Number of smoothing bases used in output function of leaf node. Default is 1.}
  \item{activation}{Specifies the activation function used to form the random basis functions. Options are \code{ReLU}, \code{cos}, or \code{tanh}. See details for further description. Default is \code{ReLU}.}
  \item{rho_alpha}{Concentration parameter of DP prior for the length scale. Default is \code{2*M}.}
  \item{rho_nu}{Shape parameter of base measure distribution choosen by \code{rho_option}. Default is 3.}
  \item{rho_lambda}{Scale parameter of base measure distribution choosen by \code{rho_option}. Default is \code{qchisq(0.5, df = nu)} / nu.}
  \item{nd}{Number of posterior draws to return. Default is 1000.}
  \item{burn}{Number of MCMC iterations to be treated as "warmup" or "burn-in". Default is 1000.}
  \item{thin}{Number of post-warmup MCMC iteration by which to thin. Default is 1.}
  \item{save_samples}{Logical, indicating whether to return all posterior samples. Default is \code{TRUE}. If \code{FALSE}, only posterior mean is returned.}
  \item{save_trees}{Logical, indicating whether or not to save a text-based representation of the tree samples. This representation can be passed to \code{predict_ridgeBART} to make predictions at a later time. Default is \code{FALSE}.}
  \item{verbose}{Logical, inciating whether to print progress to R console. Default is \code{TRUE}.}
  \item{print_every}{As the MCMC runs, a message is printed every \code{print_every} iterations. Default is \code{floor( (nd*thin + burn)/10)} so that only 10 messages are printed.}
}
\details{
  Activation options:
  \code{ReLU}: (h(x) = max(x, 0))
  \code{cos}: (h(x) = sqrt(2) * cos(x))
  \code{tanh}: (h(x) = tanh(x))
}
\value{
  A list containing
  \item{y_mean}{Mean of the training observations (needed by \code{predict_ridgeBART})}
  \item{y_sd}{Standard deviation of the training observations (needed by \code{predict_ridgeBART})}
  \item{yhat.train.mean}{Vector containing posterior mean of evaluations of regression function on training data.}
  \item{yhat.train}{Matrix with \code{nd} rows and \code{length(Y_train)} columns. Each row corresponds to a posterior sample of the regression function and each column corresponds to a training observation. Only returned if \code{save_samples == TRUE}.}
  \item{yhat.test.mean}{Vector containing posterior mean of evaluations of regression function on testing data, if testing data is provided.}
  \item{yhat.test}{If testing data was supplied, matrix containing posterior samples of the regression function evaluated on the testing data. Structure is similar to that of \code{yhat_train}. Only returned if testing data is passed and \code{save_samples == TRUE}.}
  \item{sigma}{Vector containing ALL samples of the residual standard deviation, including burn-in.}
  \item{varcounts}{Matrix that counts the number of times a variable was used in a decision rule in each MCMC iteration. Structure is similar to that of \code{yhat_train}, with rows corresponding to MCMC iteration and columns corresponding to predictors, with continuous predictors listed first followed by categorical predictors}
  \item{trees}{A list (of length \code{M}) of change logs and indexed decison rules and leaf parameters containing textual representations of the regression trees. These strings are parsed by \code{predict_ridgeBART} to reconstruct the C++ representations of the sampled trees.}
}
\seealso{
  \code{\link{probit_ridgeBART}} for binary outcomes.
}
%\references{
%  Put arXiv reference here.
%}
\author{
  Ryan Yee
}
\examples{
  target = function(x){
    if (x <= -0.5) return(-2 * x)
    else if (x <= 0) return(sin(5 * x))
    else if (x <= 0.5) return((x+1)^2)
    else return(log(x))
  }

  sigma <- 0.25
  n_train <- 2000
  n_test <- 100

  set.seed(99)
  X_train = runif(n_train, min = -1, max = 1)
  mu_train = rep(NA, times = n_train)
  for (i in 1:n_train) mu_train[i] = target(X_train[i])
  Y_train = mu_train + rnorm(n_train, mean = 0, sd = sigma)

  X_test = seq(-1, 1, length.out = n_test)
  mu_test = rep(NA, times = n_test)
  for (i in 1:n_test) mu_test[i] = target(X_test[i])
  Y_test = mu_test + rnorm(n_test, mean = 0, sd = sigma)

  \dontrun{
    fit = ridgeBART(
      Y_train = Y_train,
      X_cont_train = X_train,
      Z_mat_train = X_train,
      X_cont_test = X_test,
      Z_mat_test = X_test,
      save_samples = FALSE, save_trees = FALSE,
      nd = 100, burn = 100
    )

    par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(1,2))
    plot(mu_train, fit$yhat.train.mean, pch = 16, cex = 0.5, 
        xlab = "Truth", ylab = "Estimate", main = "Training")
    abline(a = 0, b = 1, col = 'blue')
    plot(mu_test, fit$yhat.test.mean, pch = 16, cex = 0.5, 
        xlab = "Truth", ylab = "Estimate", main = "Testing")
    abline(a = 0, b = 1, col = 'blue')
  }
}
