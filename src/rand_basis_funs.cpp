#include "rand_basis_funs.h"

void tree_traversal(suff_stat &ss, tree &t, data_info &di)
{
  double* xx_cont = 0;
  int* xx_cat = 0;
  tree::tree_cp bn;
  int nid;
  ss.clear(); // clear out the sufficient statistic map
  
  tree::npv bnv;
  t.get_bots(bnv);
  suff_stat_it ss_it; // used to look up to which element of ss a particular bottom node corresponds
  
  // add an element to suff stat map for each bottom node
  for(tree::npv_it it = bnv.begin(); it != bnv.end(); ++it){
    nid = (*it)->get_nid(); // bnv is a vector of pointers. it points to elements of bnv so we need (*it) to access the members of these elements
    // add element to sufficient stats map with key = nid, value = empty vector to hold index of observations assigned to this node
    ss.insert(std::pair<int,std::vector<int>>(nid,std::vector<int>()));
  }
  
  /*
  // for debugging purposes
  Rcpp::Rcout << "[tree_traversal]: added elements to suff. stat. map for each bottom node!" << std::endl;
  for(suff_stat_it ss_it = ss.begin(); ss_it != ss.end(); ++ss_it){
    Rcpp::Rcout << "  nid = " << ss_it->first << "  n = " << ss_it->second.size() << std::endl;
  }
  Rcpp::Rcout << "[tree_traversal]: ready to start traversal!" << std::endl;
  */
  
  // ready to loop over all observations
  for(int i = 0; i < di.n; i++){
    if(di.x_cont != 0) xx_cont = di.x_cont + i * di.p_cont;
    if(di.x_cat != 0) xx_cat = di.x_cat + i * di.p_cat;
    bn = t.get_bn(xx_cont, xx_cat);
    if(bn == 0){
      Rcpp::Rcout << "i = " << i << std::endl;
      t.print();
      Rcpp::stop("[tree_traversal]: could not find bottom node!");
    }
    else{
      nid = bn->get_nid();
      // Rcpp::Rcout << "The bottom node for obs " << i << " is " << nid << std::endl;
      if(ss.count(nid) != 1) Rcpp::stop("[tree_traversal]: bottom node not included in sufficient statistic map!"); // should never be encountered
      else{
        ss_it = ss.find(nid); // iterator now set to element of ss corresponding to the bottom node holding observation i
        ss_it->second.push_back(i);
      } // closes if/else checking that i-th observation's bottom node was in our map
    } // closes if/else checking that i-th observation maps to valid bottom node
  } // closes loop over all observation
}

arma::mat get_z_mat(suff_stat_it &ss_it, data_info &di)
{
  int n_obs = ss_it->second.size();
  arma::mat Z_mat(n_obs, di.p_smooth); // store the smooth covariates for obs in the leaf
  // loop through observations in leaf
  for (int i = 0; i < n_obs; ++i){
    for (int j = 0; j < di.p_smooth; ++j){
      Z_mat(i, j) = di.z_mat[j + ss_it->second[i] * di.p_smooth];
      }
  }
  return Z_mat;
}

// arma::vec evaluate_entire_leaf(suff_stat_it &ss_it, data_info &di, leaf_ft &l, tree_prior_info &tree_pi)
// {
//   arma::mat Z_mat = get_z_mat(ss_it, di);
//   if (*tree_pi.activation_option == 1){
//     return l.evaluate_cos(Z_mat);
//   } else if (*tree_pi.activation_option == 2){
//     return l.evaluate_tanh(Z_mat);
//   } else if (*tree_pi.activation_option == 3){
//     return l.evaluate_constant(Z_mat);
//   } else {
//     Rcpp::Rcout << "activation option = " << tree_pi.activation_option << std::endl;
//     Rcpp::stop("[evaluate_entire_leaf]: invalid activation option!");
//   }
// }

// arma::mat evaluate_entire_activation(suff_stat_it &ss_it, data_info &di, leaf_ft &l, tree_prior_info &tree_pi)
// {
//   arma::mat Z_mat = get_z_mat(ss_it, di);
//   if (*tree_pi.activation_option == 1){
//     return l.evaluate_cos_activation(Z_mat);
//   } else {
//     Rcpp::Rcout << "activation option = " << tree_pi.activation_option << std::endl;
//     Rcpp::stop("[evaluate_entire_leaf]: invalid activation option!");
//   }
// }

arma::vec get_res_vec(suff_stat_it &ss_it, data_info &di)
{
  int n_obs = ss_it->second.size();
  arma::vec res_vec(n_obs); // store the smooth covariates for obs in the leaf
  // loop through observations in leaf
  for (int i = 0; i < n_obs; ++i) res_vec[i] = di.rp[ss_it->second[i]];
  return res_vec;
}

arma::mat calc_precision(arma::mat phi_x, double &sigma, tree_prior_info &tree_pi)
{
  if (phi_x.size() > 0){
    arma::mat P = 1.0/pow(sigma, 2.0) * (phi_x.t() * phi_x) + tree_pi.inv_V;
    return P;
  } else {
    return tree_pi.inv_V; // if not observations fall into the leaf, we just get V^-1
  }
}

arma::mat calc_theta(arma::mat phi_x, arma::vec res_vec, double &sigma, tree_prior_info &tree_pi)
{
  if (phi_x.size() > 0){
    arma::mat Theta = 1.0/pow(sigma, 2.0) * phi_x.t() * res_vec + tree_pi.inv_V * tree_pi.beta0;
    return Theta;
  } else {
    return tree_pi.inv_V * tree_pi.beta0;
  }
}

void fit_ensemble(std::vector<double> &fit, std::vector<tree> &t_vec, data_info &di){
  if(fit.size() != di.n) Rcpp::stop("[fit_ensemble]: size of fit must be equal to di.n!"); // honestly should never get triggered
  double* xx_cont = 0;
  int* xx_cat = 0;
  arma::vec tmp_vec(di.p_smooth);
  for(int i = 0; i < di.n; i++){
    if(di.x_cont != 0) xx_cont = di.x_cont + i * di.p_cont;
    if(di.x_cat != 0) xx_cat = di.x_cat + i * di.p_cat;
    if(di.z_mat != 0) {
        for (int j = 0; j < di.p_smooth; ++j) tmp_vec[j] = di.z_mat[j + i * di.p_smooth];
    }
    fit[i] = 0.0;
    for(int m = 0; m < t_vec.size(); m++) {
      // Rcpp::Rcout << "Fitting tree " << m << std::endl;
      // double tmp_fit = t_vec[m].evaluate(tmp_vec, xx_cont, xx_cat); // involves a tree traversal
      // Rcpp::Rcout << "The t.evaluate fit of tree " << m << " is " << tmp_fit << std::endl;
      fit[i] += t_vec[m].evaluate(tmp_vec, xx_cont, xx_cat); // involves a tree traversal
    }
  }
}

arma::mat draw_omega(arma::vec &rho, std::set<int> &ancestors, data_info &di, tree_prior_info &tree_pi, RNG &gen)
{
  arma::mat omega(di.p_smooth, tree_pi.D, arma::fill::zeros);
  if (*tree_pi.sparse_smooth == 1 && ancestors.size() > 0){
    // sparse smooth option has been chosen and we have split on at least one continuous variable
    // omega gets nonzero entries for variables that have been split on
    // if a variable has not been split on, the column corresponding to that varaibles gets zeros
    for (int i = 0; i < di.p_smooth; ++i){
      if (ancestors.count(i) > 0) omega.row(i) = gen.std_norm_vec(tree_pi.D).t() / rho[i];
    }
  } else for (int i = 0; i < di.p_smooth; ++i) omega.row(i) = gen.std_norm_vec(tree_pi.D).t() / rho[i];
  return omega;
}

arma::vec draw_b(tree_prior_info &tree_pi, RNG &gen)
{
  arma::vec b(tree_pi.D, arma::fill::zeros);
  if (*tree_pi.activation_option == 1){
    // rff prior
    b = gen.unif_vec(tree_pi.D, 0, 2 * M_PI);
  } else {
    b = gen.std_norm_vec(tree_pi.D);
  }
  return b;
}

void add_rho(std::vector<std::map<double,int>> &sampled_rhos, arma::vec &accepted_rho, int &leaf_count, data_info &di)
{
  // loop over smoothing parameters
  for (int i = 0; i < di.p_smooth; ++i){
    // increment a previously drawn length-scale
    if (sampled_rhos[i].count(accepted_rho[i]) == 1) ++ sampled_rhos[i].at(accepted_rho[i]);
    // otherwise, add a new element
    else sampled_rhos[i].insert(std::pair<double,int>(accepted_rho[i], 1));
  }
  ++leaf_count;
}

void remove_rho(std::vector<std::map<double,int>> &sampled_rhos, arma::vec &pruned_rho, int &leaf_count, data_info &di)
{
  // loop over smoothing parameters
  for (int i = 0; i < di.p_smooth; ++i){
    // decrement a previously drawn length-scale
    if (sampled_rhos[i].count(pruned_rho[i]) == 1 && sampled_rhos[i].at(pruned_rho[i]) > 0){
      -- sampled_rhos[i].at(pruned_rho[i]);
      if (sampled_rhos[i].at(pruned_rho[i]) == 0) sampled_rhos[i].erase(pruned_rho[i]);
    } else {
      // otherwise, somthing has gone wrong
      Rcpp::Rcout << "trying to remove " << pruned_rho[i] << " from sampled_rhos" << std::endl;
      Rcpp::Rcout << "the values in sampled_rhos are " << std::endl;
      for (int i = 0; i < sampled_rhos.size(); ++i){
        Rcpp::Rcout << i << "th parameter length scale:" << std::endl;
        for (std::map<double,int>::iterator it = sampled_rhos[i].begin(); it != sampled_rhos[i].end(); ++it){
            Rcpp::Rcout << "length scale: " << it->first << " selected " << it->second << " times." << std::endl;
        }
    }
      Rcpp::stop("[remove_rho]: trying to remove a length scale that was never sampled!");
    }
  }
  --leaf_count;
}

arma::vec draw_rho(std::vector<std::map<double,int>> &sampled_rhos, int &leaf_count, data_info &di, tree_prior_info &tree_pi, RNG &gen)
{
  if (*tree_pi.rho_option == 0){
    // set to fixed length scale
    return *tree_pi.rho_prior;
  } else {
    // draw from DP
    double output;
    arma::vec rho(di.p_smooth);
    for (int i = 0; i < di.p_smooth; ++i){
      // draw from base measure with probability alpha / (alpha + leaf_count)
      // otherwise, draw from previously sampled point
      if ((*tree_pi.rho_alpha / (*tree_pi.rho_alpha + leaf_count)) > gen.uniform()){
        if (*tree_pi.rho_option == 1){
          // rho option 1: F0 = Gamma(nu/2, nu*lambda/2)
          rho[i] = gen.gamma((*tree_pi.rho_nu) / 2, (*tree_pi.rho_nu) * (*tree_pi.rho_lambda) / 2);
        } else if (*tree_pi.rho_option == 2){
          // rho option 2: F0 = InvGamma(nu/2, nu*lambda/2)
          rho[i] = ((*tree_pi.rho_nu) * (*tree_pi.rho_lambda)) / gen.chi_square(*tree_pi.rho_nu);
        } else{
          Rcpp::Rcout << "[draw_omega]: " << '"' << *tree_pi.rho_option << '"' << " is an invalid rho_option." << std::endl;
          Rcpp::stop("Invalid rho_option!");
        }
    } else {
        int index = ceil(gen.uniform() * leaf_count);
        int counter = 0;
        for (std::map<double,int>::iterator it = sampled_rhos[i].begin(); it != sampled_rhos[i].end(); ++it){
          counter += it->second;
          if (counter >= index){
            output = it->first;
            break;
          }
        }
        rho[i] = output;
      }
    }
    return rho;
  }
}
