# Author: Ryan Yee
# Details: 
# Dependencies: ridgeBART

probit_ridgeBART_wrapper = function(
    Y_train,
    Z_mat_train,
    X_cont_train = matrix(0, nrow = 1, ncol = 1),
    X_cat_train = matrix(0, nrow = 1, ncol = 1),
    Z_mat_test = matrix(0, nrow = 1, ncol = 1),
    X_cont_test = matrix(0, nrow = 1, ncol = 1),
    X_cat_test = matrix(0, nrow = 1, ncol = 1),
    unif_cuts = rep(TRUE, times = ncol(X_cont_train)),
    cutpoints_list = NULL,
    cat_levels_list = NULL,
    activation = "ReLU",
    M = 50, n_bases = 1,
    p_change = 0.20,
    rho_nu = 3, rho_lambda = qchisq(.5, df = 3) / 3,
    nd = 1000, burn = 1000, thin = 1,
    save_samples = TRUE, save_trees = FALSE,
    verbose = TRUE, print_every = floor((nd * thin + burn)/10),
    n_chains = 4
)
{
  N_train = nrow(Z_mat_train)
  N_test = nrow(Z_mat_test)
  
  # Create output containers 
  phat_train_samples = array(NA, dim = c(n_chains * nd, N_train))
  phat_test_samples = array(NA, dim = c(n_chains * nd, N_test))
  timing = rep(NA, times = n_chains)
  
  for (chain in 1:n_chains){
    cat("Starting chain ", chain, " at ", format(Sys.time(), "%b %d %Y %X"), "\n")
    train_time = system.time(
      fit <- ridgeBART::probit_ridgeBART(
        Y_train = Y_train,
        X_cont_train = X_cont_train,
        X_cat_train = X_cat_train,
        Z_mat_train = Z_mat_train,
        unif_cuts = unif_cuts,
        cutpoints_list = cutpoints_list,
        cat_levels_list = cat_levels_list,
        X_cont_test = X_cont_test,
        X_cat_test = X_cat_test,
        Z_mat_test = Z_mat_test,
        activation = activation,
        M = M, n_bases = n_bases,
        p_change = p_change,
        rho_alpha = 2 * M, 
        rho_nu = rho_nu, rho_lambda = rho_lambda,
        nd = nd, burn = burn, thin = thin,
        save_samples = save_samples, save_trees = save_trees,
        verbose = verbose, print_every = print_every
      )
    )
    start_index = (chain - 1) * nd + 1
    end_index = chain * nd
    
    phat_train_samples[start_index:end_index, ] = fit$prob.train
    phat_test_samples[start_index:end_index, ] = fit$prob.test
    timing[chain] = train_time["elapsed"]
  }
  
  # Containers to summarize all the posterior samples of the regression function
  fit_summary_train = array(dim = c(N_train, 3), dimnames = list(c(), c("MEAN", "L95", "U95")))
  fit_summary_test = array(dim = c(N_test, 3), dimnames = list(c(), c("MEAN", "L95", "U95")))
  
  fit_summary_train[,"MEAN"] = apply(phat_train_samples, MARGIN = 2, FUN = mean)
  fit_summary_train[,"L95"] = apply(phat_train_samples, MARGIN = 2, FUN = quantile, probs = 0.025)
  fit_summary_train[,"U95"] = apply(phat_train_samples, MARGIN = 2, FUN = quantile, probs = 0.975)
  
  fit_summary_test[,"MEAN"] = apply(phat_test_samples, MARGIN = 2, FUN = mean)
  fit_summary_test[,"L95"] = apply(phat_test_samples, MARGIN = 2, FUN = quantile, probs = 0.025)
  fit_summary_test[,"U95"] = apply(phat_test_samples, MARGIN = 2, FUN = quantile, probs = 0.975)
  
  return(
    list(timing = timing,
         train = fit_summary_train,
         test = fit_summary_test
    ))
}
