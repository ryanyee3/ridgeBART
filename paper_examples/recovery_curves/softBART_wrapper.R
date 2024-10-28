# Author: Ryan Yee
# Purpose: wrapper function for fitting SoftBART in simulation study
# Details: 
# Dependencies: SoftBart

softBART_wrapper = function(
    Y_train,
    X_train = matrix(0, nrow = 1, ncol = 1),
    X_test = matrix(0, nrow = 1, ncol = 1),
    nd = 1000, burn = 1000, thin = 1,
    save_samples = TRUE, save_trees = FALSE,
    verbose = TRUE, print_every = floor((nd * thin + burn)/10),
    n_chains = 4
)
{
  N_train = nrow(X_train)
  N_test = nrow(X_test)
  
  # Create output containers 
  sigma_samples = rep(NA, times = n_chains * nd)
  yhat_train_samples = array(NA, dim = c(n_chains * nd, N_train))
  yhat_test_samples = array(NA, dim = c(n_chains * nd, N_test))
  timing = rep(NA, times = n_chains)
  
  for (chain in 1:n_chains){
    cat("Starting chain ", chain, " at ", format(Sys.time(), "%b %d %Y %X"), "\n")
    train_time = system.time(
      fit <- SoftBart::softbart(
        Y = Y_train,
        X = X_train,
        X_test = X_test,
        opts = SoftBart::Opts(num_burn = burn, 
                              num_thin = thin, 
                              num_save = nd, 
                              num_print = print_every)
      )
    )
    start_index = (chain - 1) * nd + 1
    end_index = chain * nd
    
    sigma_samples[start_index:end_index] = fit$sigma
    yhat_train_samples[start_index:end_index, ] = fit$y_hat_train
    yhat_test_samples[start_index:end_index, ] = fit$y_hat_test
    timing[chain] = train_time["elapsed"]
  }
  
  # Containers to summarize all the posterior samples of the regression function
  fit_summary_train = array(dim = c(N_train, 3), dimnames = list(c(), c("MEAN", "L95", "U95")))
  fit_summary_test = array(dim = c(N_test, 3), dimnames = list(c(), c("MEAN", "L95", "U95")))
  
  fit_summary_train[,"MEAN"] = apply(yhat_train_samples, MARGIN = 2, FUN = mean)
  fit_summary_train[,"L95"] = apply(yhat_train_samples, MARGIN = 2, FUN = quantile, probs = 0.025)
  fit_summary_train[,"U95"] = apply(yhat_train_samples, MARGIN = 2, FUN = quantile, probs = 0.975)
  
  fit_summary_test[,"MEAN"] = apply(yhat_test_samples, MARGIN = 2, FUN = mean)
  fit_summary_test[,"L95"] = apply(yhat_test_samples, MARGIN = 2, FUN = quantile, probs = 0.025)
  fit_summary_test[,"U95"] = apply(yhat_test_samples, MARGIN = 2, FUN = quantile, probs = 0.975)
  
  # Container for samples from posterior predictive
  ystar_summary_train = array(dim = c(N_train, 3), dimnames = list(c(), c("MEAN", "L95", "U95")))
  ystar_summary_test = array(dim = c(N_test, 3), dimnames = list(c(), c("MEAN", "L95", "U95")))
  
  for(i in 1:N_train){
    tmp_ystar = yhat_train_samples[,i] + sigma_samples * rnorm(n = length(sigma_samples), mean = 0, sd = 1)
    ystar_summary_train[i,"MEAN"] = mean(tmp_ystar)
    ystar_summary_train[i,"L95"] = quantile(tmp_ystar, probs = 0.025)
    ystar_summary_train[i,"U95"] = quantile(tmp_ystar, probs = 0.975)
  }
  for(i in 1:N_test){
    tmp_ystar = yhat_test_samples[,i] + sigma_samples * rnorm(n = length(sigma_samples), mean = 0, sd = 1)
    ystar_summary_test[i,"MEAN"] = mean(tmp_ystar)
    ystar_summary_test[i,"L95"] = quantile(tmp_ystar, probs = 0.025)
    ystar_summary_test[i,"U95"] = quantile(tmp_ystar, probs = 0.975)
  }
  
  return(
    list(timing = timing,
         train = list(fit = fit_summary_train, ystar = ystar_summary_train),
         test = list(fit = fit_summary_test, ystar = ystar_summary_test)
    ))
}
