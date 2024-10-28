predict_ridgeBART <- function(
    fit, 
    X_cont = matrix(0, nrow = 1, ncol = 1), 
    X_cat = matrix(0, nrow = 1, ncol = 1), 
    Z_mat = matrix(0, nrow = 1, ncol = 1), 
    verbose = FALSE, print_every = 100
    )
{
  # rffBART does not include an is.probit by default so we'll add it
  if(is.null(fit[["is.probit"]])) fit[["is.probit"]] <- FALSE

  tmp <- .predict_ridgeBART(fit_list = fit$trees,
                           tX_cont = t(X_cont),
                           tX_cat = t(X_cat),
                           tZ_mat = t(Z_mat),
                           probit = fit[["is.probit"]],
                           verbose = verbose,
                           print_every = print_every)
  if(!fit[["is.probit"]]) output <- fit$y_mean + fit$y_sd * tmp
  else output <- tmp
  return(output)
}