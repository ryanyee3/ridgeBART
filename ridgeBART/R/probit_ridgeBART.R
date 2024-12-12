probit_ridgeBART <- function(
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
    M = 50, n_bases = 1, activation = "ReLU", rotation_option = 0, dp_option = 1,
    rho_alpha = 2 * M, rho_nu = 3, rho_lambda = stats::qchisq(.5, df = rho_nu) / rho_nu,
    nd = 1000, burn = 1000, thin = 1,
    save_samples = TRUE, save_trees = TRUE, 
    verbose = TRUE, print_every = floor( (nd*thin + burn))/10
    )
{
  if(!is.integer(Y_train)) stop("Y_train must be an integer vector")
  if(!all(Y_train %in% c(0,1))) stop("All elements of Y_train must be 0 or 1")
  if(length(X_cat_train) > 1 & is.null(cat_levels_list)) stop("cat_levels_list is null but categorical predictors were provided")

  if(p_change > 1) stop("p_change must be between 0 and 1!")
  
  p_cont <- 0
  p_cat <- 0
  cont_names <- c()
  cat_names <- c()

  intercept_option = 0
  if (intercept_option == 1){
    beta0 = rep(0, times = n_bases + intercept_option)
    beta0[1] = stats::qnorm(mean(Y_train)) / M
  } else {
    beta0 = rep(stats::qnorm(mean(Y_train)) / (M * n_bases), times = n_bases)
  }
  tau = 1 / sqrt(M * (n_bases + intercept_option))
  
  if ((activation) == "cos"){
    activation_option = 1
  } else if (tolower(activation) == "tanh"){
    activation_option = 2
  } else if (tolower(activation) == "relu"){
    activation_option = 5
  }

  if(length(X_cont_train) > 1){
    p_cont <- ncol(X_cont_train)
    if(is.null(colnames(X_cont_train))){
      cont_names <- paste0("X", 1:p_cont)
    } else{
      cont_names <- colnames(X_cont_train)
    }
  } else{
    cont_names <- c()
  }
  
  if(length(X_cat_train) > 1){
    p_cat <- ncol(X_cat_train)
    if(is.null(colnames(X_cat_train))){
      cat_names <- paste0("X", (p_cont + 1):(p_cont + p_cat))
    } else{
      cat_names <- colnames(X_cat_train)
    }
  } else{
    cat_names <- c()
  }
  pred_names <- c(cont_names, cat_names)
  
  fit <- .probit_ridgeBART_fit(
    Y_train = Y_train,
    tX_cont_train = t(X_cont_train),
    tX_cat_train = t(X_cat_train),
    tZ_mat_train = t(Z_mat_train),
    tX_cont_test = t(X_cont_test),
    tX_cat_test = t(X_cat_test),
    tZ_mat_test = t(Z_mat_test),
    unif_cuts = unif_cuts, cutpoints_list = cutpoints_list, 
    cat_levels_list = cat_levels_list, edge_mat_list = NULL,
    graph_split = rep(FALSE, times = ncol(X_cat_train)), graph_cut_type = 0,
    oblique_option = FALSE, prob_aa = 1, x0_option = 2,
    sparse = sparse, a_u = 0.5, b_u = 1, p_change = p_change,
    beta0 = beta0, tau = tau,
    branch_alpha = 0.95, branch_beta = 2.0,
    intercept_option = intercept_option,
    sparse_smooth_option = 0,
    rotation_option = rotation_option,
    dp_option = dp_option,
    activation_option = activation_option, n_bases = n_bases,
    rho_option = 1, rho_prior = rep(1, times = ncol(Z_mat_train)),
    rho_alpha = rho_alpha, rho_nu = rho_nu, rho_lambda = rho_lambda,
    M = M, nd = nd, burn = burn, thin = thin,
    save_samples = save_samples, save_trees = save_trees, 
    verbose = verbose, print_every = print_every
    )
  
  results <- list()
  results[["prob.train.mean"]] <- fit$fit_train_mean
  if(save_samples) results[["prob.train"]] <- fit$fit_train
  if(!is.null(fit$fit_test_mean)){
    results[["prob.test.mean"]] <- fit$fit_test_mean
    if(save_samples) results[["prob.test"]] <- fit$fit_test
  }

  results[["activation_option"]] <- activation_option
  results[["intercept_option"]] <- intercept_option

  varcounts <- fit$var_count
  if(length(pred_names) != ncol(varcounts)){
    warning("There was an issue tracking variable names. Not naming columns of varcounts object")
  } else{
    colnames(varcounts) <- pred_names
  }
  results[["varcounts"]] <- varcounts
  results[["total_accept"]] <- fit$total_accept
  if(save_trees) results[["trees"]] <- fit$trees
  results[["is.probit"]] <- TRUE
  return(results)
}