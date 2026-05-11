#ifndef GUARD_data_parsing_funs_h
#define GUARD_data_parsing_funs_h

#include "structs.h"

void parse_cutpoints(std::vector<std::set<double>> &cutpoints, int p_cont, Rcpp::List &tmp_cutpoints, Rcpp::LogicalVector &unif_cuts);
void parse_cat_levels(std::vector<std::set<int>> &cat_levels, std::vector<int> &K, int &p_cat, Rcpp::List &tmp_cat_levels);
void parse_training_data(int &n_train, int &p_cont, int &p_cat, int &p_smooth, Rcpp::NumericMatrix &tX_cont_train, Rcpp::IntegerMatrix &tX_cat_train, Rcpp::NumericMatrix &tZ_mat_train);
void parse_testing_data(int &n_test, Rcpp::NumericMatrix &tX_cont_test, Rcpp::IntegerMatrix &tX_cat_test, Rcpp::NumericMatrix &tZ_mat_test, const int &p_cat, const int &p_cont, const int &p_smooth);
void parse_z_mat(arma::mat &tZ_arma, Rcpp::NumericMatrix &tZ_input);

#endif
