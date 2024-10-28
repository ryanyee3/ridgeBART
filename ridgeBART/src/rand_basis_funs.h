#ifndef GUARD_funs_h
#define GUARD_funs_h

#include "polytope_funs.h"

void tree_traversal(suff_stat &ss, tree &t, data_info &di);
arma::mat get_z_mat(suff_stat_it &ss_it, data_info &di);
// arma::vec evaluate_entire_leaf(suff_stat_it &ss_it, data_info &di, leaf_ft &l, tree_prior_info &tree_pi);
// arma::mat evaluate_entire_activation(suff_stat_it &ss_it, data_info &di, leaf_ft &l, tree_prior_info &tree_pi);
arma::vec get_res_vec(suff_stat_it &ss_it, data_info &di);
arma::mat calc_precision(arma::mat phi_x, double &sigma, tree_prior_info &tree_pi);
arma::mat calc_theta(arma::mat phi_x, arma::vec res_vec, double &sigma, tree_prior_info &tree_pi);
void fit_ensemble(std::vector<double> &fit, std::vector<tree> &t_vec, data_info &di);
arma::mat draw_omega(arma::vec &rho, std::set<int> &ancestors, data_info &di, tree_prior_info &tree_pi, RNG &gen);
arma::vec draw_b(tree_prior_info &tree_pi, RNG &gen);
void add_rho(std::vector<std::map<double,int>> &sampled_rhos, arma::vec &accepted_rho, int &leaf_count, data_info &di);
void remove_rho(std::vector<std::map<double,int>> &sampled_rhos, arma::vec &pruned_rho, int &leaf_count, data_info &di);
arma::vec draw_rho(std::vector<std::map<double,int>> &sampled_rhos, int &leaf_count, data_info &di, tree_prior_info &tree_pi, RNG &gen);

#endif
