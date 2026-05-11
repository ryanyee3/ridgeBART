#include "update_tree.h"
#include "save_trees.h"

// [[Rcpp::export(".predict_ridgeBART")]]
Rcpp::NumericMatrix predict_ridgeBART(Rcpp::List fit_list,
                                     Rcpp::NumericMatrix tX_cont,
                                     Rcpp::IntegerMatrix tX_cat,
                                     Rcpp::NumericMatrix tZ_mat,
                                     bool probit = false,
                                     bool verbose = true, int print_every = 50)
{
  set_str_conversion set_str; // for converting sets of integers into strings

  int n = 0;
  int p_cont = 0;
  int p_cat = 0;
  int p_smooth = 0;
  
  parse_training_data(n, p_cont, p_cat, p_smooth, tX_cont, tX_cat, tZ_mat);
  int p_split = p_cont + p_cat;
  data_info di;
  di.n = n;
  di.p_cont = p_cont;
  di.p_cat = p_cat;
  di.p_smooth = p_smooth;
  di.p_split = p_split;
  if(p_cont > 0) di.x_cont = tX_cont.begin();
  if(p_cat > 0) di.x_cat = tX_cat.begin();
  if(p_smooth > 0) di.z_mat = tZ_mat.begin();
  
  // int nd = tree_draws.size();
  // Rcpp::CharacterVector first_tree_vec = tree_draws[0];
  // int M = first_tree_vec.size();
  int M = fit_list.size();
  std::vector<std::vector<std::string>> change_log_vec(M);
  std::vector<std::vector<std::map<int,double>>> phi_log_vec(M);
  std::vector<std::vector<arma::mat>> w_log_vec(M);
  std::vector<std::vector<arma::vec>> b_log_vec(M);
  std::vector<std::vector<std::map<int,arma::vec>>> beta_log_vec(M);
  int activation_option;
  int intercept_option;
  for (int m = 0; m < M; m++){
    Rcpp::List tmp_list = fit_list[m];
    parse_fit_list(change_log_vec[m], phi_log_vec[m], w_log_vec[m], b_log_vec[m], beta_log_vec[m], activation_option, intercept_option, tmp_list);
  }

  // we draw new betas in every sample iteration
  // so the number of beta logs is the number of draws
  int nd = beta_log_vec[0].size();

  if(verbose){  
    Rcpp::Rcout << "nd = " << nd << " M = " << M;
    Rcpp::Rcout << " n = " << n << " p_cont = " << p_cont << " p_cat = " << p_cat << " p_smooth = " << p_smooth << std::endl;
  }

  std::vector<double> allfit(n);
  Rcpp::NumericMatrix pred_out(nd,n);

  std::vector<tree> t_vec(M);
  std::vector<int> last_log_index(M, 0); // initialize a vector of zeros to record the last index where a change was made for each tree
  for(int iter = 0; iter < nd; iter++){
    if(verbose && (iter%print_every == 0)){
      Rcpp::Rcout << "  Iteration: " << iter << " of " << nd <<std::endl;
      Rcpp::checkUserInterrupt();
    }
    for (int m = 0; m < M; m++){
      read_fit_logs(t_vec[m], last_log_index[m], iter, change_log_vec[m], phi_log_vec[m], w_log_vec[m], b_log_vec[m], beta_log_vec[m], activation_option, intercept_option, set_str);
    }
    fit_ensemble(allfit, t_vec, di);
    if(probit){
      for(int i = 0; i < n; i++) pred_out(iter,i) = R::pnorm(allfit[i], 0.0, 1.0, true, false);
    } else{
      for(int i = 0; i < n; i++) pred_out(iter,i) = allfit[i];
    } 
  } // closes loop over all draws of the ensemble
  return pred_out;
}
