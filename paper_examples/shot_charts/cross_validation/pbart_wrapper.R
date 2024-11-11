# Author: Ryan Yee
# Date: June 7, 2024
# Purpose: wrapper function for fitting wbart in simulation study
# Details: 
# Dependencies: BART

pbart_wrapper = function(
    Y_train, 
    X_train_df, 
    X_test_df, 
    nd = 1000, burn = 1000, thin = 1,
    print_every = floor((nd * thin + burn)/10),
    n_chains = 1
)
{
  N_train = nrow(X_train_df)
  N_test = nrow(X_test_df)
  
  # Create output containers 
  phat_train_samples = array(NA, dim = c(n_chains * nd, N_train))
  phat_test_samples = array(NA, dim = c(n_chains * nd, N_test))
  timing = rep(NA, times = n_chains)
  
  for (chain in 1:n_chains){
    cat("Starting chain ", chain, " at ", format(Sys.time(), "%b %d %Y %X"), "\n")
    train_time = system.time(
      fit <- BART::pbart(
        y.train = Y_train,
        x.train = X_train_df, 
        x.test = X_test_df,
        ndpost = nd, nskip = burn, keepevery = thin,
        printevery = print_every)
    )
    start_index = (chain - 1) * nd + 1
    end_index = chain * nd
    
    phat_train_samples[start_index:end_index, ] = fit$yhat.train
    phat_test_samples[start_index:end_index, ] = fit$yhat.test
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
