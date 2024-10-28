#ifndef GUARD_update_tree_h
#define GUARD_update_tree_h

#include "rule_funs.h"
#include "rand_basis_funs.h"

void compute_suff_stat_grow(suff_stat &orig_suff_stat, suff_stat &new_suff_stat, int &nx_nid, rule_t &rule, tree &t, data_info &di);
void compute_suff_stat_prune(suff_stat &orig_suff_stat, suff_stat &new_suff_stat, int &nl_nid, int &nr_nid, int &np_nid, tree &t, data_info &di);

double compute_lil(suff_stat &ss, int &nid, double &sigma, data_info &di, tree_prior_info &tree_pi, leaf_ft &l);
void draw_betas(tree &t, suff_stat &ss, double &sigma, data_info &di, tree_prior_info &tree_pi, RNG &gen);

void grow_tree(tree &t, suff_stat &ss_train, suff_stat &ss_test, rho_diag_t &rho_diag, std::vector<std::map<double,int>> &sampled_rhos, int &iter, int &leaf_count, int &nx_nid, int &accept, rule_diag_t &rule_diag, double &sigma, data_info &di_train, data_info &di_test, tree_prior_info &tree_pi, RNG &gen);
void prune_tree(tree &t, suff_stat &ss_train, suff_stat &ss_test, rho_diag_t &rho_diag, std::vector<std::map<double,int>> &sampled_rhos, int &leaf_count, int &nx_nid, int &accept, double &sigma, data_info &di_train, data_info &di_test, tree_prior_info &tree_pi, RNG &gen);
void change_tree(tree &t, suff_stat &ss_train, rho_diag_t &rho_diag, std::vector<std::map<double,int>> &sampled_rhos, int &leaf_count, int &accept, double &sigma, data_info &di_train, tree_prior_info &tree_pi, RNG &gen);
void update_tree(tree &t, suff_stat &ss_train, suff_stat &ss_test, rho_diag_t &rho_diag, std::vector<std::map<double,int>> &sampled_rhos, int &iter, int &leaf_count, int &nid, int &accept, int &change_type, rule_diag_t &rule_diag, double &sigma, data_info &di_train, data_info &di_test, tree_prior_info &tree_pi, RNG &gen);

#endif
