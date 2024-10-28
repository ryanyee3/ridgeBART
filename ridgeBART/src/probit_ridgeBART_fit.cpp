#include "update_tree.h"
#include "save_trees.h"

// [[Rcpp::export(".probit_ridgeBART_fit")]]
Rcpp::List probit_ridgeBART_fit(
  Rcpp::NumericVector Y_train,
  Rcpp::NumericMatrix tX_cont_train,
  Rcpp::IntegerMatrix tX_cat_train,
  Rcpp::NumericMatrix tZ_mat_train,
  Rcpp::NumericMatrix tX_cont_test,
  Rcpp::IntegerMatrix tX_cat_test,
  Rcpp::NumericMatrix tZ_mat_test,
  Rcpp::LogicalVector unif_cuts,
  Rcpp::Nullable<Rcpp::List> cutpoints_list,
  Rcpp::Nullable<Rcpp::List> cat_levels_list,
  Rcpp::Nullable<Rcpp::List> edge_mat_list,
  Rcpp::LogicalVector graph_split, int graph_cut_type,
  bool oblique_option, double prob_aa, int x0_option,
  bool sparse, double a_u, double b_u, double p_change,
  Rcpp::NumericVector beta0, double tau,
  double branch_alpha, double branch_beta,
  int activation_option, int intercept_option,
  int sparse_smooth_option,
  int n_bases, int rho_option, 
  Rcpp::NumericVector rho_prior,
  double rho_alpha, double rho_nu, double rho_lambda,
  int M, int nd, int burn, int thin,
  bool save_samples, bool save_trees,
  bool verbose, int print_every
  )
{
  Rcpp::RNGScope scope;
  RNG gen;
  
  set_str_conversion set_str; // for converting sets of integers into strings
  
  int n_train = 0;
  int n_test = 0;
  int p_cont = 0;
  int p_cat = 0;
  int p_smooth = 0;
  
  parse_training_data(n_train, p_cont, p_cat, p_smooth, tX_cont_train, tX_cat_train, tZ_mat_train);
  if(Y_train.size() != n_train) Rcpp::stop("Number of observations in Y_train does not match number of rows in training design matrices");
  parse_testing_data(n_test, tX_cont_test, tX_cat_test, tZ_mat_test, p_cat, p_cont, p_smooth);
  
  int p_split = p_cont + p_cat;
  
  if(verbose){
    Rcpp::Rcout << "n_train = " << n_train << " n_test = " << n_test;
    Rcpp::Rcout << " p_cont = " << p_cont << "  p_cat = " << p_cat << "  p_smooth = " << p_smooth << std::endl;
  }
  
  std::vector<std::set<double>> cutpoints;
  if(p_cont > 0){
    if(cutpoints_list.isNotNull()){
      Rcpp::List tmp_cutpoints  = Rcpp::List(cutpoints_list);
      parse_cutpoints(cutpoints, p_cont, tmp_cutpoints, unif_cuts);
    }
  }
  
  std::vector<std::set<int>> cat_levels;
  std::vector<int> K; // number of levels for the different categorical variables
  std::vector<std::vector<edge>> edges;
  
  if(p_cat > 0){
    if(cat_levels_list.isNotNull()){
      Rcpp::List tmp_cat_levels = Rcpp::List(cat_levels_list);
      parse_cat_levels(cat_levels, K, p_cat, tmp_cat_levels);
    } else{
      Rcpp::stop("Must provide list of categorical levels!");
    }
    if(edge_mat_list.isNotNull()){
      Rcpp::List tmp_edge_mat = Rcpp::List(edge_mat_list);
      parse_graphs(edges, p_cat, K, tmp_edge_mat, graph_split);
    }
  }
  
  double* allfit_train = new double[n_train];
  double* residual = new double[n_train];
  double* latents = new double[n_train]; // holds the latent variables
  
  std::vector<double> allfit_test;
  if(n_test > 0) allfit_test.resize(n_test);
  
  // set up the data info object for training data
  data_info di_train;
  di_train.n = n_train;
  di_train.p_cont = p_cont;
  di_train.p_cat = p_cat;
  di_train.p_smooth = p_smooth;
  di_train.p_split = p_split;
  if(p_cont > 0) di_train.x_cont = tX_cont_train.begin();
  if(p_cat > 0) di_train.x_cat = tX_cat_train.begin();
  if(p_smooth > 0) di_train.z_mat = tZ_mat_train.begin();
  di_train.rp = residual;
  
  // set up the data info object for testing data
  data_info di_test;
  if(n_test > 0){
    di_test.n = n_test;
    di_test.p_cont = p_cont;
    di_test.p_cat = p_cat;
    di_test.p_smooth = p_smooth;
    di_test.p_split = p_split;
    if(p_cont > 0) di_test.x_cont = tX_cont_test.begin();
    if(p_cat > 0)  di_test.x_cat = tX_cat_test.begin();
    if(p_smooth > 0) di_test.z_mat = tZ_mat_test.begin();
  }
  
  // declare stuff for variable selection
  std::vector<double> theta(p_split, 1.0/ (double) p_split);
  double u = 1.0/(1.0 + (double) p_split);
  std::vector<int> var_count(p_split, 0); // count how many times a variable has been used in a splitting rule
  int rule_count = 0; // how many total decision rules are there in the ensemble
  
  tree_prior_info tree_pi;
  tree_pi.alpha = branch_alpha;
  tree_pi.beta = branch_beta;
  tree_pi.prob_bd = 1 - p_change;
  tree_pi.theta = &theta;
  tree_pi.var_count = &var_count;
  tree_pi.rule_count = &rule_count;
  tree_pi.x0_option = &x0_option;
  tree_pi.rho_option = &rho_option;
  tree_pi.activation_option = &activation_option;
  tree_pi.intercept_option = intercept_option;
  tree_pi.sparse_smooth = &sparse_smooth_option;
  
  if(p_cont > 0){
    tree_pi.unif_cuts = unif_cuts.begin(); // do we use uniform cutpoints?
    tree_pi.cutpoints = &cutpoints;
  }
  
  if(p_cat > 0){
    tree_pi.cat_levels = &cat_levels;
    tree_pi.edges = &edges;
    tree_pi.K = &K;
    tree_pi.graph_split = graph_split.begin();
    tree_pi.graph_cut_type = graph_cut_type;
  }

  // rff prior info
  tree_pi.D = n_bases;
  arma::vec tmp_beta0(n_bases + intercept_option, arma::fill::zeros);
  for (int i = 0; i < beta0.size(); ++i) tmp_beta0[i] = beta0[i];
  tree_pi.beta0 = tmp_beta0;
  tree_pi.V = pow(tau, 2) * arma::mat(n_bases + intercept_option, n_bases + intercept_option, arma::fill::eye);
  tree_pi.inv_V = arma::inv(tree_pi.V);
  tree_pi.det_V = arma::det(tree_pi.V);
  tree_pi.bVb = arma::as_scalar(tree_pi.beta0.t() * tree_pi.inv_V * tree_pi.beta0);

  // declare stuff for length scale prior
  int leaf_count = 0;
  tree_pi.rho_alpha = &rho_alpha;
  tree_pi.rho_nu = &rho_nu;
  tree_pi.rho_lambda = &rho_lambda;
  // one map per smoothing variable
  // key = sampled value, value = number of leaves using value
  std::vector<std::map<double, int>> sampled_rhos;
  tree_pi.rho_samples = &sampled_rhos;
  if (rho_option != 0) sampled_rhos.resize(p_smooth);

  // length scale stuff if we don't use DP
  std::vector<double> rho_vec(p_smooth);
  for (int i = 0; i < p_smooth; ++i) rho_vec[i] = rho_prior[i];
  tree_pi.rho_prior = &rho_vec;

  double prob_cat = p_cat / (double) p_split;
  tree_pi.oblique_option = oblique_option;
  tree_pi.prob_aa = &prob_aa;
  tree_pi.prob_cat = &prob_cat;
  
  double sigma = 1.0; // remember for probit, sigma is always 1!
  
  // stuff for MCMC loop
  int total_draws = 1 + burn + (nd-1)*thin;
  int sample_index = 0;
  int accept = 0;
  int total_accept = 0; // counts how many trees we change in each iteration
  rule_diag_t rule_diag;

  rho_diag_t rho_diag;

  tree::npv bnv; // for checking that our ss map and our trees are not totally and utterly out of sync
  double tmp_mu; // for holding the value of mu when we're doing the backfitting
  int nid; // for holding the node id of where a changed was made used for writing to change log
  int change_type; // for encoding the type of update we've proposed
  int save_iter; // for keeping track of the save iteration we are on
  
  // initialize the trees
  std::vector<tree> t_vec(M);
  std::vector<suff_stat> ss_train_vec(M);
  std::vector<suff_stat> ss_test_vec(M);
  
  // set the dimension of all the leaves
  // we need to do this for the linear algebra to work during the initial tree traversal
  // we won't necessarily have to do this if we start from an arbitrary tree ensemble
  for(int m = 0; m < M; ++m){
    tree::npv bn_vec;
    t_vec[m].get_bots(bn_vec);
    for(tree::npv_it it = bn_vec.begin(); it != bn_vec.end(); ++it) {
      std::set<int> ancestors;
      if (*tree_pi.sparse_smooth == 1) (*it)->get_ancestors(ancestors);
      (*it)->set_int_opt(tree_pi.intercept_option);
      (*it)->set_leaf_dim(di_train.p_smooth, tree_pi.D, tree_pi.intercept_option);
      (*it)->set_b(draw_b(tree_pi, gen));
      (*it)->set_rho(draw_rho(sampled_rhos, leaf_count, di_train, tree_pi, gen));
      arma::vec tmp_rho = (*it)->get_rho();
      (*it)->set_w(draw_omega(tmp_rho, ancestors, di_train, tree_pi, gen));
      (*it)->set_act_opt(*tree_pi.activation_option);
    }
  }

  for(int i = 0; i < n_train; ++i) allfit_train[i] = 0.0;
  
  for(int m = 0; m < M; ++m){
    // do an initial tree traversal
    // this is kind of silly when t is just a stump
    // but it may help if we were to allow start from an arbitrary ensemble
    tree_traversal(ss_train_vec[m], t_vec[m], di_train);
    
    // get the fit of each tree
    for(suff_stat_it ss_it = ss_train_vec[m].begin(); ss_it != ss_train_vec[m].end(); ++ss_it){
      if(ss_it->second.size() > 0){
        leaf_ft l = t_vec[m].get_ptr(ss_it->first)->get_leaf();
        // arma::vec tmp_mu = evaluate_entire_leaf(ss_it, di_train, l, tree_pi);
        arma::vec tmp_mu = l.eval_leaf(get_z_mat(ss_it, di_train));
        for (int i = 0; i < tmp_mu.size(); ++i){
          allfit_train[ss_it->second[i]] += tmp_mu(i);
        }
      }
    }
    if(n_test > 0) tree_traversal(ss_test_vec[m], t_vec[m], di_test);
  }

  // draw the initial set of latents and compute the residual
  for(int i = 0; i < n_train; i++){
    if(Y_train[i] == 1) latents[i] = gen.lo_trunc_norm(allfit_train[i], 0.0);
    else if(Y_train[i] == 0) latents[i] = gen.hi_trunc_norm(allfit_train[i], 0.0);
    else{
      Rcpp::Rcout << " Outcome for observation i = " << i+1 << " is " << Y_train[i] << std::endl;
      Rcpp::stop("For probit regression, all outcomes must be 1 or 0.");
    }
    residual[i] = latents[i] - allfit_train[i];
  }

  // output containers
  arma::vec fit_train_mean = arma::zeros<arma::vec>(n_train); // posterior mean for training data
  arma::vec fit_test_mean = arma::zeros<arma::vec>(1); // posterior mean for testing data (if any)
  if(n_test > 0) fit_test_mean.zeros(n_test); // arma::set.size can initialize with garbage values
  
  arma::mat fit_train = arma::zeros<arma::mat>(1,1); // posterior samples for training data
  arma::mat fit_test = arma::zeros<arma::mat>(1,1); // posterior samples for testing data (if any)
  if(save_samples){
    // if we are saving all samples, then we resize the containers accordingly
    fit_train.zeros(nd, n_train);
    if(n_test > 0) fit_test.zeros(nd, n_test);
  }

  arma::vec sigma_samples(total_draws);
  arma::vec total_accept_samples(total_draws);
  arma::mat tree_depths = arma::zeros<arma::mat>(total_draws, M);

  // this is all for obliqueBART
  arma::vec aa_proposed_samples(1); // how many axis-aligned rules proposed in this iteration
  arma::vec aa_rejected_samples(1); // how many axis-aligned rules rejected in this iteration
  arma::vec cat_proposed_samples(1);
  arma::vec cat_rejected_samples(1);
  arma::vec obl_proposed_samples(1); // how many oblique rules proposed in this iteration
  arma::vec obl_rejected_samples(1); // how many oblique rules rejected in this iteration
  if (oblique_option){
    arma::vec aa_proposed_samples(total_draws); // how many axis-aligned rules proposed in this iteration
    arma::vec aa_rejected_samples(total_draws); // how many axis-aligned rules rejected in this iteration
    arma::vec cat_proposed_samples(total_draws);
    arma::vec cat_rejected_samples(total_draws);
    arma::vec obl_proposed_samples(total_draws); // how many oblique rules proposed in this iteration
    arma::vec obl_rejected_samples(total_draws); // how many oblique rules rejected in this iteration
  }
  
  arma::mat theta_samples(1,1); // unless we're doing DART, no need to waste space
  if(sparse) theta_samples.set_size(total_draws, p_split);
  arma::mat var_count_samples(total_draws, p_split); // always useful to see how often we're splitting on variables in the ensemble
  
  // Rcpp::List tree_draws(nd);
  std::vector<std::vector<std::string>> change_log_vec(M);
  std::vector<std::vector<std::map<int,double>>> phi_log_vec(M);
  std::vector<std::vector<arma::mat>> w_log_vec(M);
  std::vector<std::vector<arma::vec>> b_log_vec(M);
  std::vector<std::vector<std::map<int,arma::vec>>> beta_log_vec(M);

  Rcpp::Rcout << "Starting MCMC Loop" << std::endl;
  
  // main MCMC loop starts here
  for(int iter = 0; iter < total_draws; iter++){
    if(verbose){
    // remember that R is 1-indexed
      if( (iter < burn) && (iter % print_every == 0)){
        Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << total_draws << "; Warmup" << std::endl;
        Rcpp::checkUserInterrupt();
      } else if(((iter> burn) && (iter%print_every == 0)) || (iter == burn) ){
        Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << total_draws << "; Sampling" << std::endl;
        Rcpp::checkUserInterrupt();
      } else if( iter == total_draws-1){
        Rcpp::Rcout << "  MCMC Iteration: " << iter+1 << " of " << total_draws << "; Sampling" << std::endl;
        Rcpp::checkUserInterrupt();
      }
    }

    // at the start of the loop, we need to update the latents
    for(int i = 0; i < n_train; i++){
      // residual is latent - allfit
      if(Y_train[i] == 1) latents[i] = gen.lo_trunc_norm(allfit_train[i], 0.0);
      else if(Y_train[i] == 0) latents[i] = gen.hi_trunc_norm(allfit_train[i], 0.0);
      else{
        Rcpp::Rcout << " Outcome for observation i = " << i+1 << " is " << Y_train[i] << std::endl;
        Rcpp::stop("For probit regression, all outcomes must be 1 or 0.");
      }
      residual[i] = latents[i] - allfit_train[i];
    }

    // loop over trees
    total_accept = 0;
    rule_diag.reset(); // reset running counts of proposed and rejected rules
    rho_diag.reset();
    for(int m = 0; m < M; m++){
      for(suff_stat_it ss_it = ss_train_vec[m].begin(); ss_it != ss_train_vec[m].end(); ++ss_it){
        // loop over the bottom nodes in m-th tree)
        if(ss_it->second.size() > 0){
          leaf_ft l = t_vec[m].get_ptr(ss_it->first)->get_leaf();
          // arma::vec tmp_mu = evaluate_entire_leaf(ss_it, di_train, l, tree_pi);
          arma::vec tmp_mu = l.eval_leaf(get_z_mat(ss_it, di_train));
          for (int i = 0; i < tmp_mu.size(); ++i){
            // remove fit of m-th tree from allfit: allfit[i] -= tmp_mu
            // for partial residual: we could compute Y - allfit (now that allfit has fit of m-th tree removed)
            // numerically this is exactly equal to adding tmp_mu to the value of residual
            allfit_train[ss_it->second[i]] -= tmp_mu(i); // adjust the value of allfit
            residual[ss_it->second[i]] += tmp_mu(i);
          }
        }
      } // this whole loop is O(n)
      
      update_tree(t_vec[m], ss_train_vec[m], ss_test_vec[m], rho_diag, sampled_rhos, iter, leaf_count, nid, accept, change_type, rule_diag, sigma, di_train, di_test, tree_pi, gen); // update the tree
      if ( (iter >= burn) && ( (iter - burn)%thin == 0) && (save_trees)){
        save_iter = iter - burn;
        write_fit_logs(change_log_vec[m], phi_log_vec[m], w_log_vec[m], b_log_vec[m], beta_log_vec[m], tree_pi, t_vec[m], nid, save_iter, change_type, accept, set_str);
      }
      total_accept += accept;
    
      // now we need to update the value of allfit
      for(suff_stat_it ss_it = ss_train_vec[m].begin(); ss_it != ss_train_vec[m].end(); ++ss_it){
        if(ss_it->second.size() > 0){
          leaf_ft l = t_vec[m].get_ptr(ss_it->first)->get_leaf();
          // arma::vec tmp_mu = evaluate_entire_leaf(ss_it, di_train, l, tree_pi);
          arma::vec tmp_mu = l.eval_leaf(get_z_mat(ss_it, di_train));
          for (int i = 0; i < tmp_mu.size(); ++i){
            // add fit of m-th tree back to allfit and subtract it from the value of the residual
            allfit_train[ss_it->second[i]] += tmp_mu(i);
            residual[ss_it->second[i]] -= tmp_mu(i);
          }
        }
      } // this loop is also O(n)
    } // closes loop over all of the trees

    // ready to update rho
    for (int i = 0; i < rho_diag.accepted.size(); i++) add_rho(sampled_rhos, rho_diag.accepted[i], leaf_count, di_train);
    for (int i = 0; i < rho_diag.pruned.size(); i++) remove_rho(sampled_rhos, rho_diag.pruned[i], leaf_count, di_train);
    
    if(sparse){
      // update_theta_u(theta, u, var_count, p_split, a_u, b_u, gen);
      for(int j = 0; j < p_split; j++){
        theta_samples(iter, j) = theta[j];
        var_count_samples(iter,j) = var_count[j];
      }
    } else{
      for(int j = 0; j < p_split; j++) var_count_samples(iter, j) = var_count[j];
    }

    total_accept_samples(iter) = total_accept; // how many trees changed in this iteration
    if (oblique_option){
      aa_proposed_samples(iter) = rule_diag.aa_prop;
      aa_rejected_samples(iter) = rule_diag.aa_rej;
      cat_proposed_samples(iter) = rule_diag.cat_prop;
      cat_rejected_samples(iter) = rule_diag.cat_rej;
      obl_proposed_samples(iter) = rule_diag.obl_prop;
      obl_rejected_samples(iter) = rule_diag.obl_rej;
    }

    // save depth of tree at each iteration
    for(int m = 0; m < M; ++m) tree_depths(iter,m) = t_vec[m].get_tree_depth();
    
    if( (iter >= burn) && ( (iter - burn)%thin == 0)){
      sample_index = (int) ( (iter-burn)/thin);

      // total_accept_samples(sample_index) = total_accept; // how many trees changed in this iteration
      // // time to write each tree as a string
      // if(save_trees){
      //   Option 1 in coatless' answer:
      //   // https://stackoverflow.com/questions/37502121/assigning-rcpp-objects-into-an-rcpp-list-yields-duplicates-of-the-last-element
      //   Rcpp::CharacterVector tree_string_vec(M);
      //   for(int m = 0; m < M; m++){
      //     tree_string_vec[m] = write_tree(t_vec[m], tree_pi, set_str);
      //   }
      //   tree_draws[sample_index] = tree_string_vec; // dump a character vector holding each tree's draws into an element of an Rcpp::List
      // }
      
      if(save_samples){
        for(int i = 0; i < n_train; i++){
          fit_train(sample_index,i) = R::pnorm(allfit_train[i], 0.0, 1.0, true, false);
          fit_train_mean(i) += R::pnorm(allfit_train[i], 0.0, 1.0, true, false);
          // fit_train_raw(sample_index,i) = allfit_train[i];
        }
      } else{
        for(int i = 0; i < n_train; i++) fit_train_mean(i) += R::pnorm(allfit_train[i], 0.0, 1.0, true, false);
      }
      
      if(n_test > 0){
        for(int i = 0; i < n_test; i++) allfit_test[i] = 0.0; // reset the value of allfit_test
        for(int m = 0; m < M; m++){
          for(suff_stat_it ss_it = ss_test_vec[m].begin(); ss_it != ss_test_vec[m].end(); ++ss_it){
            if(ss_it->second.size() > 0){
              leaf_ft l = t_vec[m].get_ptr(ss_it->first)->get_leaf();
              // arma::vec tmp_mu = evaluate_entire_leaf(ss_it, di_test, l, tree_pi);
              arma::vec tmp_mu = l.eval_leaf(get_z_mat(ss_it, di_test));
              for (int i = 0; i < tmp_mu.size(); ++i) allfit_test[ss_it->second[i]] += tmp_mu(i);
            }
          } // loop over the keys in the m-th sufficient stat map
        } // closes loop over trees
        
        if(save_samples){
          for(int i = 0; i < n_test; i++){
            fit_test(sample_index,i) = R::pnorm(allfit_test[i], 0.0, 1.0, true, false);
            fit_test_mean(i) += R::pnorm(allfit_test[i], 0.0, 1.0, true, false);
          }
        } else{
          for(int i = 0; i < n_test; i++) fit_test_mean(i) += R::pnorm(allfit_test[i], 0.0, 1.0, true, false);
        }
      } // closes if checking whether we have any test set observations
    } // closes if that checks whether we should save anything in this iteration
  } // closes the main MCMC for loop

  Rcpp::Rcout << "The MCMC loop has ended" << std::endl;

  fit_train_mean /= ( (double) nd);
  if(n_test > 0) fit_test_mean /= ( (double) nd);

  Rcpp::List rho_out;
  if (rho_option != 0){
    for (int i = 0; i < sampled_rhos.size(); ++i){
      Rcpp::NumericMatrix tmp_mat(sampled_rhos[i].size(), 2);
      int j = 0;
      for (std::map<double,int>::iterator it = sampled_rhos[i].begin(); it != sampled_rhos[i].end(); ++it){
        tmp_mat(j, 0) = it->first;
        tmp_mat(j, 1) = it->second;
        ++j;
      }
      rho_out.push_back(tmp_mat);
    }
  }
  
  Rcpp::List results;
  
  results["fit_train_mean"] = fit_train_mean;
  if(save_samples){
    results["fit_train"] = fit_train;
  }
  if(n_test > 0){
    results["fit_test_mean"] = fit_test_mean;
    if(save_samples){
      results["fit_test"] = fit_test;
    }
  }
  results["sigma"] = sigma_samples;
  results["total_accept"] = total_accept_samples;
  results["tree_depths"] = tree_depths;
  results["var_count"] = var_count_samples;

  if(save_trees){
    // Rcpp::Rcout << "Saving tree outputs (this may take a while)" << std::endl;
    // clock_t t = clock();
    Rcpp::List tree_logs(M);
    for (int m = 0; m < M; m++) tree_logs[m] = parse_fit_logs(change_log_vec[m], phi_log_vec[m], w_log_vec[m], b_log_vec[m], beta_log_vec[m], tree_pi, di_train);
    results["trees"] = tree_logs;
    // t = clock() - t;
    // Rcpp::Rcout << "Finished saving trees in " << (float)t/CLOCKS_PER_SEC << " seconds." << std::endl;
  }
  if(sparse){
    results["theta"] = theta_samples;
  }
  if(oblique_option){
    results["aa_proposed"] = aa_proposed_samples;
    results["aa_rejected"] = aa_rejected_samples;
    results["cat_proposed"] = cat_proposed_samples;
    results["cat_rejected"] = cat_rejected_samples;
    results["obl_proposed"] = obl_proposed_samples;
    results["obl_rejected"] = obl_rejected_samples;
  }
  if(rho_option != 0) results["rhos"] = rho_out;
  return results;
}
