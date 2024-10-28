#include "draw_tree.h"
#include "data_parsing_funs.h"
#include "update_tree.h"
#include "save_trees.h"

Rcpp::List drawTree(Rcpp::NumericMatrix tX_cont,
                    Rcpp::IntegerMatrix tX_cat,
                    Rcpp::NumericMatrix tZ_mat,
                    Rcpp::LogicalVector unif_cuts,
                    Rcpp::Nullable<Rcpp::List> cutpoints_list,
                    Rcpp::Nullable<Rcpp::List> cat_levels_list,
                    Rcpp::Nullable<Rcpp::List> edge_mat_list,
                    Rcpp::LogicalVector graph_split, int graph_cut_type,
                    bool oblique_option, double prob_aa, int x0_option,
                    bool sparse, double a_u, double b_u,
                    Rcpp::NumericVector beta0, double tau,
                    double lambda, double nu,
                    int activation_option, int intercept_option,
                    int n_bases, int rho_option, 
                    Rcpp::NumericVector rho_prior,
                    double rho_alpha, double rho_nu, double rho_lambda,
                    int M, bool verbose, int print_every)
{
  Rcpp::RNGScope scope;
  RNG gen;
  set_str_conversion set_str; // for converting sets of integers into string
  
  int n = 0;
  int p_cont = 0;
  int p_cat = 0;
  int p_smooth = 0;
  
  parse_training_data(n, p_cont, p_cat, p_smooth, tX_cont, tX_cat, tZ_mat);
  int p_split = p_cont + p_cat;
  
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

  data_info di;
  di.n = n;
  di.p_cont = p_cont;
  di.p_cat = p_cat;
  di.p_smooth = p_smooth;
  di.p_split = p_split;
  if(p_cont > 0) di.x_cont = tX_cont.begin();
  if(p_cat > 0) di.x_cat = tX_cat.begin();
  if(p_smooth > 0) di.z_mat = tZ_mat.begin();

  std::vector<double> theta(p_split, 1.0/ (double) p_split);
  // double u = 1.0/(1.0 + (double) p); unused. consider adding back in if we turn on sparse
  std::vector<int> var_count(p_split, 0); // count how many times a variable has been used in a splitting rule
  int rule_count = 0; // how many total decision rules are there in the ensemble
  
  tree_prior_info tree_pi;
  tree_pi.theta = &theta;
  tree_pi.var_count = &var_count;
  tree_pi.rule_count = &rule_count;
  tree_pi.x0_option = &x0_option;
  tree_pi.rho_option = &rho_option;
  tree_pi.activation_option = &activation_option;
  tree_pi.intercept_option = intercept_option;
  
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
  tree_pi.V = tau * arma::mat(n_bases + intercept_option, n_bases + intercept_option, arma::fill::eye);
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
  if (rho_option == 1) sampled_rhos.resize(p_smooth);

  // length scale stuff if we don't use DP
  std::vector<double> rho_vec(p_smooth);
  for (int i = 0; i < p_smooth; ++i) rho_vec[i] = rho_prior[i];
  tree_pi.rho_prior = &rho_vec;

  double prob_cat = p_cat / (double) p_split;
  tree_pi.oblique_option = oblique_option;
  tree_pi.prob_aa = &prob_aa;
  tree_pi.prob_cat = &prob_cat;
  
  tree t;
  suff_stat ss;
  arma::vec tmp_mu;
  Rcpp::NumericVector fit(n);

  draw_tree(t, sampled_rhos, leaf_count, di, tree_pi, gen);
  
  t.print();

  tree_traversal(ss, t, di);

  for(suff_stat_it ss_it = ss.begin(); ss_it != ss.end(); ++ss_it){
    leaf_ft l = t.get_ptr(ss_it->first)->get_leaf(); // get the leaf
    arma::vec tmp_mu = l.eval_leaf(get_z_mat(ss_it, di));
    for (int i = 0; i < tmp_mu.size(); ++i){
      fit[ss_it->second[i]] += tmp_mu(i);
    }
  }
  
  // a little overkill but not a big deal
  std::vector<std::vector<std::string>> change_log_vec(1);
  std::vector<std::vector<std::map<int,double>>> phi_log_vec(1);
  std::vector<std::vector<arma::mat>> w_log_vec(1);
  std::vector<std::vector<arma::vec>> b_log_vec(1);
  std::vector<std::vector<std::map<int,arma::vec>>> beta_log_vec(1);
  // these variables don't matter
    // but we have to pass by reference
  int nid = 0;
  int iter = 0;
  int change_type = 0;
  int accepted = 0;
  write_fit_logs(change_log_vec[1], phi_log_vec[1], w_log_vec[1], b_log_vec[1], beta_log_vec[1], tree_pi, t, nid, iter, change_type, accepted, set_str);
  
  Rcpp::List tree_logs(1);
  parse_fit_logs(change_log_vec[1], phi_log_vec[1], w_log_vec[1], b_log_vec[1], beta_log_vec[1], tree_pi, di);

  Rcpp::List results;
  results["fit"] = fit;
  results["trees"] = tree_logs;
  return results;
  
}


// [[Rcpp::export(".drawEnsemble")]]
Rcpp::List drawEnsemble(Rcpp::NumericMatrix tX_cont,
                        Rcpp::IntegerMatrix tX_cat,
                        Rcpp::NumericMatrix tZ_mat,
                        Rcpp::LogicalVector unif_cuts,
                        Rcpp::Nullable<Rcpp::List> cutpoints_list,
                        Rcpp::Nullable<Rcpp::List> cat_levels_list,
                        Rcpp::Nullable<Rcpp::List> edge_mat_list,
                        Rcpp::LogicalVector graph_split, int graph_cut_type,
                        bool oblique_option, double prob_aa, int x0_option,
                        bool sparse, double a_u, double b_u,
                        Rcpp::NumericVector beta0, double tau,
                        double lambda, double nu,
                        int activation_option, int intercept_option,
                        int n_bases, int rho_option, 
                        Rcpp::NumericVector rho_prior,
                        double rho_alpha, double rho_nu, double rho_lambda,
                        int M, bool verbose, int print_every)
{
  Rcpp::RNGScope scope;
  RNG gen;
  set_str_conversion set_str; // for converting sets of integers into string

  int n = 0;
  int p_cont = 0;
  int p_cat = 0;
  int p_smooth = 0;
  
  parse_training_data(n, p_cont, p_cat, p_smooth, tX_cont, tX_cat, tZ_mat);
  int p_split = p_cont + p_cat;
  
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

  data_info di;
  di.n = n;
  di.p_cont = p_cont;
  di.p_cat = p_cat;
  di.p_smooth = p_smooth;
  di.p_split = p_split;
  if(p_cont > 0) di.x_cont = tX_cont.begin();
  if(p_cat > 0) di.x_cat = tX_cat.begin();
  if(p_smooth > 0) di.z_mat = tZ_mat.begin();

  std::vector<double> theta(p_split, 1.0/ (double) p_split);
  //double u = 1.0/(1.0 + (double) p); // unused; consider adding it back in if we turn on sparse
  std::vector<int> var_count(p_split, 0); // count how many times a variable has been used in a splitting rule
  int rule_count = 0; // how many total decision rules are there in the ensemble
  
  tree_prior_info tree_pi;
  tree_pi.theta = &theta;
  tree_pi.var_count = &var_count;
  tree_pi.rule_count = &rule_count;
  tree_pi.x0_option = &x0_option;
  tree_pi.rho_option = &rho_option;
  tree_pi.activation_option = &activation_option;
  tree_pi.intercept_option = intercept_option;
  
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
  tree_pi.V = tau * arma::mat(n_bases + intercept_option, n_bases + intercept_option, arma::fill::eye);
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
  if (rho_option == 1) sampled_rhos.resize(p_smooth);

  // length scale stuff if we don't use DP
  std::vector<double> rho_vec(p_smooth);
  for (int i = 0; i < p_smooth; ++i) rho_vec[i] = rho_prior[i];
  tree_pi.rho_prior = &rho_vec;

  double prob_cat = p_cat / (double) p_split;
  tree_pi.oblique_option = oblique_option;
  tree_pi.prob_aa = &prob_aa;
  tree_pi.prob_cat = &prob_cat;
  
  Rcpp::IntegerVector num_clusters(M);
  Rcpp::IntegerVector num_singletons(M);
  Rcpp::IntegerVector num_empty(M);
  Rcpp::IntegerVector max_cluster_size(M);
  Rcpp::IntegerVector min_cluster_size(M);
  Rcpp::NumericMatrix tree_fits(n,M);
  Rcpp::IntegerMatrix leaf_id(n,M);
  Rcpp::NumericVector fit(n);
  arma::mat kernel = arma::zeros<arma::mat>(n,n); // kernel(i,ii) counts #times obs i & j in same leaf
  
  for(int i = 0; i < n; ++i) fit[i] = 0.0;

  // a little overkill but not a big deal
  std::vector<std::vector<std::string>> change_log_vec(M);
  std::vector<std::vector<std::map<int,double>>> phi_log_vec(M);
  std::vector<std::vector<arma::mat>> w_log_vec(M);
  std::vector<std::vector<arma::vec>> b_log_vec(M);
  std::vector<std::vector<std::map<int,arma::vec>>> beta_log_vec(M);
  
  for(int m = 0; m < M; ++m){
    num_clusters[m] = 0;
    num_singletons[m] = 0;
    num_empty[m] = 0;
    max_cluster_size[m] = 0;
    min_cluster_size[m] = 0;
  }
  
  for(int i = 0; i < n; ++i){
    for(int m = 0; m < M; ++m){
      leaf_id(i,m) = 0;
    }
  }
  
  tree t;
  suff_stat ss;
  arma::vec tmp_mu;
  
  for(int m = 0; m < M; m++){
    if(verbose && m % print_every == 0) Rcpp::Rcout << "Drawing tree " << m+1 << " of " << M << std::endl; 
    t.to_null();
    draw_tree(t, sampled_rhos, leaf_count, di, tree_pi, gen);
    ss.clear();
    tree_traversal(ss,t,di);
    num_clusters[m] = ss.size();
    int singleton = 0;
    int empty = 0;
    int max_size = 0;
    int min_size = n;
    
    for(suff_stat_it ss_it = ss.begin(); ss_it != ss.end(); ++ss_it){
      
      int cluster_size = ss_it->second.size();
      if(cluster_size == 1) ++singleton;
      if(cluster_size == 0) ++empty;
      if(cluster_size > max_size) max_size = cluster_size;
      if(cluster_size < min_size) min_size = cluster_size;
      leaf_ft l = t.get_ptr(ss_it->first)->get_leaf(); // get the leaf
      arma::vec tmp_mu = l.eval_leaf(get_z_mat(ss_it, di)); // get the jump in this leaf
      for (int i = 0; i < tmp_mu.size(); ++i){
        tree_fits(ss_it->second[i],m) = tmp_mu(i);
        fit[ss_it->second[i]] += tmp_mu(i);
      }
      
      if(cluster_size > 1){
        for(int_it it = ss_it->second.begin(); it != ss_it->second.end(); ++it){
          leaf_id(*it, m) = ss_it->first; // save the id of the leaf
          for(int_it iit = it; iit != ss_it->second.end(); ++iit){
            if(*it != *iit){
              kernel(*it, *iit) += 1.0;
              kernel(*iit, *it) += 1.0;
            } else{
              kernel(*it, *iit) += 1.0;
            }
          } // closes second loop over obs in leaf
        } // closes loop over obs in leaf
      }
      num_singletons[m] = singleton;
      num_empty[m] = empty;
      max_cluster_size[m] = max_size;
      min_cluster_size[m] = min_size;
    } // closes loop over leafs
    // these variables don't matter
    // but we have to pass by reference
    int nid = 0;
    int iter = 0;
    int change_type = 0;
    int accepted = 0;
    write_fit_logs(change_log_vec[m], phi_log_vec[m], w_log_vec[m], b_log_vec[m], beta_log_vec[m], tree_pi, t, nid, iter, change_type, accepted, set_str);
  }

  Rcpp::List tree_logs(M);
  for (int m = 0; m < M; m++) tree_logs[m] = parse_fit_logs(change_log_vec[m], phi_log_vec[m], w_log_vec[m], b_log_vec[m], beta_log_vec[m], tree_pi, di);

  kernel /= (double) M;
  Rcpp::List results;
  results["fit"] = fit;
  results["trees"] = tree_logs;
  results["tree_fits"] = tree_fits;
  results["leaf"] = leaf_id;
  results["num_leafs"] = num_clusters;
  results["num_singletons"] = num_singletons;
  results["num_empty"] = num_empty;
  results["max_leaf_size"] = max_cluster_size;
  results["min_leaf_size"] = min_cluster_size;
  results["kernel"] = kernel;
  return results;
}
