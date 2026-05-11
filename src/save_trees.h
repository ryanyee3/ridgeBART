#ifndef GUARD_save_trees_h
#define GUARD_save_trees_h

#include "rand_basis_funs.h"

// writing functions
void write_fit_logs(std::vector<std::string> &change_log, std::vector<std::map<int,double>> &phi_log, std::vector<arma::mat> &w_log, std::vector<arma::vec> &b_log, std::vector<std::map<int,arma::vec>> &beta_log, tree_prior_info &tree_pi, tree &t, int &nid, int &iter, int &change_type, int &accepted, set_str_conversion &set_str);
void write_rule(std::ostringstream &os, std::vector<std::map<int,double>> &phi_log, tree_prior_info &tree_pi, rule_t &rule, set_str_conversion &set_str);
void write_leaf(std::ostringstream &os, std::vector<arma::mat> &w_log, std::vector<arma::vec> &b_log, leaf_ft &leaf);

// parsing functions
Rcpp::List parse_fit_logs(std::vector<std::string> &change_log, std::vector<std::map<int,double>> &phi_log, std::vector<arma::mat> &w_log, std::vector<arma::vec> &b_log, std::vector<std::map<int,arma::vec>> &beta_log, tree_prior_info &tree_pi, data_info &di);
void parse_fit_list(std::vector<std::string> &change_log, std::vector<std::map<int,double>> &phi_log, std::vector<arma::mat> &w_log, std::vector<arma::vec> &b_log, std::vector<std::map<int,arma::vec>> &beta_log, int &activation_option, int &intercept_option, Rcpp::List &fit_list);

// reading functions
void read_fit_logs(tree &t, int &last_log_index, int &sample_iter, std::vector<std::string> &change_log, std::vector<std::map<int,double>> &phi_log, std::vector<arma::mat> &w_log, std::vector<arma::vec> &b_log, std::vector<std::map<int,arma::vec>> &beta_log, int &activation_option, int &intercept_option, set_str_conversion &set_str);
void read_rule(std::istringstream &node_ss, rule_t &rule, std::vector<std::map<int,double>> &phi_log, char &type, set_str_conversion &set_str);
void read_leaf(std::istringstream &ss, leaf_ft &leaf, std::vector<arma::mat> &w_log, std::vector<arma::vec> &b_log, int &activation_option, int &intercept_option);

#endif
