#ifndef GUARD_draw_tree_h
#define GUARD_draw_tree_h

#include "rule_funs.h"
#include "rand_basis_funs.h"

void draw_tree(tree &t, std::vector<std::map<double,int>> &sampled_rhos, int &leaf_count, data_info &di, tree_prior_info &tree_pi, RNG &gen);
#endif
