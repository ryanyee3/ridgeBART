\name{predict_ridgeBART}
\alias{predict_ridgeBART}
\title{
  Predicting new observations with previously fit ridgeBART models.
}
\description{
  \code{predict_ridgeBART} can take the output of \code{ridgeBART} or \code{probit_ridgeBART} and use it to make predictions at new inputs.
}
\usage{
  predict_ridgeBART(
    fit, 
    X_cont = matrix(0, nrow = 1, ncol = 1), 
    X_cat = matrix(0, nrow = 1, ncol = 1), 
    Z_mat = matrix(0, nrow = 1, ncol = 1), 
    verbose = FALSE, print_every = 100
  )
}
\arguments{
  \item{fit}{Object returned by \code{ridgeBART} or \code{probit_ridgeBART}}
  \item{X_cont}{Matrix of continuous predictors. Note all predictors must be re-scaled to lie in the interval [-1,1]. Default is a 1x1 matrix, which signals that no continuos predictors are available.}
  \item{X_cat}{Integer matrix of categorical predictors for training data. Note categorical levels should be 0-indexed. That is, if a categorical predictor has 10 levels, the values should run from 0 to 9. Default is a 1x1 matrix, which signals that no categorical predictors.}
  \item{Z_mat}{Matrix of smoothing predictors. Note, predictors must be re-scaled to lie in the interval [-1,1]. Default is a 1x1 matrix, which signals that there are no continuous predictors in the training data.}
  \item{verbose}{Logical, indicating whether or not to print message predictions are being made. Default is \code{FALSE}.}
  \item{print_every}{As the function loops over the MCMC samples, a message is printed to the console every \code{print_every} iterations. Default is 100.}
}
\value{
  A matrix containing posterior samples of the regression function evaluated at the supplied inputs.
  The rows of the matrix correspond to MCMC iterations and the columns correspond to the observations in the supplied data (i.e. rows of \code{X_cont}, \code{X_cat} and/or \code{Z_mat}).
}
\author{
  Ryan Yee
}
