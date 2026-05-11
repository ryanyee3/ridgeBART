#include "structs.h"

// [[Rcpp::export(".eval_bases")]]
Rcpp::NumericMatrix eval_bases(Rcpp::List bases_list,
                               Rcpp::NumericMatrix Z_mat,
                               int activation_option,
                               int intercept_option,
                               bool probit = false,
                               bool verbose = true, int print_every = 50)
{
  Rcpp::List tmp_list = bases_list[0];
  int M = tmp_list.size();
  int n = Z_mat.rows();
  int nd = bases_list.size();
  Rcpp::NumericMatrix pred_out(nd, n);

  // get arma::mat using advanced constructor
  // https://stackoverflow.com/questions/57516726/converting-r-matrix-to-armamat-with-list
  arma::mat Z_arma(Z_mat.begin(), Z_mat.rows(), Z_mat.cols(), false);

  if(verbose) Rcpp::Rcout << "Starting bases evaluations" << std::endl;
  for(int iter = 0; iter < nd; iter++){
    if(verbose && (iter%print_every == 0)){
      Rcpp::Rcout << "  Iteration: " << iter << " of " << nd <<std::endl;
      Rcpp::checkUserInterrupt();
    }
    Rcpp::List forest_list = bases_list[iter];
    std::vector<double> allfit(n, 0.0); // initialize length n vector of zeros
    for (int m = 0; m < M; m++){
      Rcpp::List tree_list = forest_list[m];
      leaf_ft leaf;
      leaf.act_opt = activation_option;
      leaf.intercept = intercept_option;
      arma::vec tmp_beta = tree_list["beta"];
      leaf.beta = tmp_beta;
      arma::mat tmp_omega = tree_list["omega"];
      leaf.w = tmp_omega;
      arma::vec tmp_b = tree_list["b"];
      leaf.b = tmp_b;
      arma::vec tmp_vec = leaf.eval_leaf(Z_arma);
      for (int i = 0; i < n; i++) allfit[i] += tmp_vec[i];
    }
    if(probit){
      for(int i = 0; i < n; i++) pred_out(iter,i) = R::pnorm(allfit[i], 0.0, 1.0, true, false);
    } else{
      for(int i = 0; i < n; i++) pred_out(iter,i) = allfit[i];
    } 
  } // closes loop over all draws of the ensemble
  return pred_out;
}
