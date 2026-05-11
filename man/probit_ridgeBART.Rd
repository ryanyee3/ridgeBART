\name{probit_ridgeBART}
\alias{probit_ridgeBART}
\title{
  Probit ridgeBART for binary outcomes.
}
\description{
  Fit a ridgeBART model of a binary responses using the the Albert & Chib (1993) data augmentation for probit models.
}
\usage{
  probit_ridgeBART(
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
    sparse = FALSE, p_change = 0.2,
    M = 50, n_bases = 1, activation = "ReLU",
    rho_alpha = 2 * M, rho_nu = 3, rho_lambda = stats::qchisq(.5, df = rho_nu) / rho_nu,
    nd = 1000, burn = 1000, thin = 1,
    save_samples = TRUE, save_trees = TRUE, 
    verbose = TRUE, print_every = floor( (nd*thin + burn))/10
  )
}
\arguments{
  \item{Y_train}{Integer vector of binary (i.e. 0/1) responses for training data}
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
  \item{M}{Number of trees in the ensemble. Default is 50.}
  \item{n_bases}{Number of smoothing bases used in output function of leaf node. Default is 1.}
  \item{activation}{Specifies the activation function used to form the random basis functions. Options are \code{ReLU}, \code{cos}, or \code{tanh}. See \code{\link{ridgeBART}} for options. Default is \code{ReLU}.}
  \item{rho_alpha}{Confidence parameter of DP prior for the length scale. Default is \code{M}, the number of trees in the emsemble.}
  \item{rho_nu}{Shape parameter of base measure distribution choosen by \code{rho_option}. Default is 3.}
  \item{rho_lambda}{Scale parameter of base measure distribution choosen by \code{rho_option}. Default is qchisq(.5, df = rho_nu) / rho_nu.}
  \item{nd}{Number of posterior draws to return. Default is 1000.}
  \item{burn}{Number of MCMC iterations to be treated as "warmup" or "burn-in". Default is 1000.}
  \item{thin}{Number of post-warmup MCMC iteration by which to thin. Default is 1.}
  \item{save_samples}{Logical, indicating whether to return all posterior samples. Default is \code{TRUE}. If \code{FALSE}, only posterior mean is returned.}
  \item{save_trees}{Logical, indicating whether or not to save a text-based representation of the tree samples. This representation can be passed to \code{predict_ridgeBART} to make predictions at a later time. Default is \code{FALSE}.}
  \item{verbose}{Logical, inciating whether to print progress to R console. Default is \code{TRUE}.}
  \item{print_every}{As the MCMC runs, a message is printed every \code{print_every} iterations. Default is \code{floor( (nd*thin + burn)/10)} so that only 10 messages are printed.}
}
\details{
  See \code{\link{ridgeBART}} for activation options.
  Implements the Albert & Chib (1993) data augmentation strategy for probit regression and models the regression function with a sum-of-trees.
  The marginal prior of any evaluation of the regression function f(x) is a normal distribution centered at \code{mu0} with standard deviation \code{tau * sqrt(M)}. 
As such, for each x, the induced prior for P(Y = 1 | x) places 95\% probability on the interval \code{pnorm(mu0 -2 * tau * sqrt(M)), pnorm(mu0 + 2 * tau * sqrt(M))}. 
  By default, we set \code{tau = 1/sqrt(M * n_bases)} and \code{mu0 = qnorm(mean(Y_train))} to shrink towards the observed mean.
}
\value{
  A list containing
  \item{prob.train.mean}{Vector containing posterior mean of evaluations of regression function on training data.}
  \item{prob.train}{Matrix with \code{nd} rows and \code{length(Y_train)} columns. Each row corresponds to a posterior sample of the regression function and each column corresponds to a training observation. Only returned if \code{save_samples == TRUE}.}
  \item{prob.test.mean}{Vector containing posterior mean of evaluations of regression function on testing data, if testing data is provided.}
  \item{prob.test}{If testing data was supplied, matrix containing posterior samples of the regression function evaluated on the testing data. Structure is similar to that of \code{yhat_train}. Only returned if testing data is passed and \code{save_samples == TRUE}.}
  \item{varcounts}{Matrix that counts the number of times a variable was used in a decision rule in each MCMC iteration. Structure is similar to that of \code{yhat_train}, with rows corresponding to MCMC iteration and columns corresponding to predictors, with continuous predictors listed first followed by categorical predictors}
  \item{trees}{A list (of length \code{M}) of change logs and indexed decison rules and leaf parameters containing textual representations of the regression trees. These strings are parsed by \code{predict_ridgeBART} to reconstruct the C++ representations of the sampled trees.}
}
\seealso{
  \code{\link{ridgeBART}} for continuous outcomes.
}
%\references{
%  Put arXiv reference here.
%}
\author{
  Ryan Yee
}
%\examples{
%##---- Should be DIRECTLY executable !! ----
%##-- ==>  Define data, use random,
%##--	or do  help(data=index)  for the standard data sets.
%
%## The function is currently defined as
%function (x) 
%{
%  }
%}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
