#include "update_tree.h"

void compute_suff_stat_grow(suff_stat &orig_suff_stat, suff_stat &new_suff_stat, int &nx_nid, rule_t &rule, tree &t, data_info &di)
{
  double* xx_cont;
  int* xx_cat;
  int i;
  int l_count;
  int r_count;
  
  // we are growing tree from node nx, which has id of nx_nid
  
  int nxl_nid = 2*nx_nid; // id of proposed left child of nx
  int nxr_nid = 2*nx_nid+1; // id of proposed right child of nx
  
  suff_stat_it nx_it = orig_suff_stat.find(nx_nid); // iterator at element for nx in original sufficient statistic map
  new_suff_stat.clear();
  
  // copy orig_suff_stat into new_suff_stat
  for(suff_stat_it it = orig_suff_stat.begin(); it != orig_suff_stat.end(); ++it){
    new_suff_stat.insert(std::pair<int,std::vector<int>>(it->first, it->second));
  }
  
  // now we manipulate new_suff_stat to drop nx and add nxl and nxr
  new_suff_stat.insert(std::pair<int,std::vector<int>>(nxl_nid, std::vector<int>())); // create map element for left child of nx
  new_suff_stat.insert(std::pair<int,std::vector<int>>(nxr_nid, std::vector<int>())); // create map element for right child of nx
  new_suff_stat.erase(nx_nid); // remove map element for nx as it is not a bottom leaf node in new tree
  
  suff_stat_it nxl_it = new_suff_stat.find(nxl_nid); // iterator at element for nxl in new sufficient stat map
  suff_stat_it nxr_it = new_suff_stat.find(nxr_nid); // iterator at element for nxr in new sufficient stat map
  
  // loop over all observation that were assigned to nx in original tree
  // note:
  //   nx_it->first is just the node id for nx (nx_nid)
  //   nx_it->second is a vector of integers containing the indicies of observations that land in nx
  // in helper.h we defined int_it as std::vector<int>::iterator
  
  if(nx_it->second.size() > 0){
    for(int_it it = nx_it->second.begin(); it != nx_it->second.end(); ++it){
      i = *it;
      if(di.x_cont != 0) xx_cont = di.x_cont + i * di.p_cont;
      if(di.x_cat != 0) xx_cat = di.x_cat + i * di.p_cat;
      
      if(rule.is_cat){
        // categorical rule
        // we need to see whether i-th observation's value of the categorical pred goes to left or right
        // std::set.count returns 1 if the value is in the set and 0 otherwise
        l_count = rule.l_vals.count(xx_cat[rule.v_cat]);
        r_count = rule.r_vals.count(xx_cat[rule.v_cat]);
        if(l_count == 1 && r_count == 0) nxl_it->second.push_back(i);
        else if(l_count == 0 && r_count == 1) nxr_it->second.push_back(i);
        else if(l_count == 1 && r_count == 1) Rcpp::stop("[compute_ss_grow]: observation goes to both left & right child...");
        else{
          Rcpp::Rcout << "i = " << i << "v = " << rule.v_cat+1 << "  value = " << xx_cat[rule.v_cat] << std::endl;
          Rcpp::Rcout << "left values:";
          for(set_it levels_it = rule.l_vals.begin(); levels_it != rule.l_vals.end(); ++levels_it) Rcpp::Rcout << " " << *levels_it;
          Rcpp::Rcout << std::endl;
          
          Rcpp::Rcout << "right values:";
          for(set_it levels_it = rule.r_vals.begin(); levels_it != rule.r_vals.end(); ++levels_it) Rcpp::Rcout << " " << *levels_it;
          Rcpp::Rcout << std::endl;
          
          Rcpp::stop("[compute_ss_grow]: could not assign observation to left or right child in categorical split!");
        } // closes if/else determining whether obs is in l_vals or r_vals
      } else{
        double tmp_x = 0.0;
        for(rc_it phi_it = rule.phi.begin(); phi_it != rule.phi.end(); ++phi_it) tmp_x += (phi_it->second) * xx_cont[phi_it->first];
        if(tmp_x < rule.c) nxl_it->second.push_back(i);
        else if(tmp_x >= rule.c) nxr_it->second.push_back(i);
        else{
          Rcpp::Rcout << "i = " << i << " phi'x = " << tmp_x << " c = " << rule.c << std::endl;
          Rcpp::stop("[compute_ss_grow]: could not assign observation to left or right child in oblique split!");
        }
      } // closes if/else determing whether it is continuous or categorical rule
    } // closes loop over all entries in nx
  }  // closes if checking if nx has empty size
}

void compute_suff_stat_prune(suff_stat &orig_suff_stat, suff_stat &new_suff_stat, int &nl_nid, int &nr_nid, int &np_nid, tree &t, data_info &di)
{
  //int i;
  if(orig_suff_stat.count(nl_nid) != 1) Rcpp::stop("[compute_ss_prune]: did not find left node in suff stat map");
  if(orig_suff_stat.count(nr_nid) != 1) Rcpp::stop("[compute_ss_prune]: did not find right node in suff stat map");
  
  suff_stat_it nl_it = orig_suff_stat.find(nl_nid); // iterator at element for nl in original suff stat map
  suff_stat_it nr_it = orig_suff_stat.find(nr_nid); // iterator at element for nr in original suff stat map
  
  new_suff_stat.clear();
  // this makes a completely new copy of orig_suff_stat
  for(suff_stat_it ss_it = orig_suff_stat.begin(); ss_it != orig_suff_stat.end(); ++ss_it){
    new_suff_stat.insert(std::pair<int,std::vector<int>>(ss_it->first, ss_it->second));
  }
  new_suff_stat.insert(std::pair<int,std::vector<int>>(np_nid, std::vector<int>())); // add element for np in new suff stat map
  new_suff_stat.erase(nl_nid); // delete element for nl in new suff stat map since nl has been pruned
  new_suff_stat.erase(nr_nid); // delete element for nr in new suff stat map since nr has been pruned
  
  if(new_suff_stat.count(np_nid) != 1) Rcpp::stop("[compute_ss_prune]: didn't create element in new suff stat map for np correctly");
  suff_stat_it np_it = new_suff_stat.find(np_nid); // iterator at element for np in new suff stat map
  
  // time to populate np_it
  // first let's add the elements from nl_it
  
  if(nl_it->second.size() > 0){
    for(int_it it = nl_it->second.begin(); it != nl_it->second.end(); ++it) np_it->second.push_back( *it );
  }
  if(nr_it->second.size() > 0){
    for(int_it it = nr_it->second.begin(); it != nr_it->second.end(); ++it) np_it->second.push_back( *it );
  }
}

double compute_lil(suff_stat &ss, int &nid, double &sigma, data_info &di, tree_prior_info &tree_pi, leaf_ft &l)
{
  // reminder posterior of jump mu is N(P^-1 Theta, P^-1)
  if(ss.count(nid) != 1) Rcpp::stop("[compute_lil]: did not find node in suff stat map!");
  suff_stat_it ss_it = ss.find(nid);

  // arma::mat phi = evaluate_entire_activation(ss_it, di, l, tree_pi);
  arma::mat phi = l.get_phi(get_z_mat(ss_it, di));
  arma::vec res_vec = get_res_vec(ss_it, di);
  arma::vec Theta = calc_theta(phi, res_vec, sigma, tree_pi);
  arma::mat P = calc_precision(phi, sigma, tree_pi);

  arma::mat L = arma::chol(P, "lower"); // P = L * L.t()
  arma::vec nu = arma::solve(arma::trimatl(L), Theta); // nu = L^-1 Theta

  // leaf lil contribution is 1/2 * [Theta^T P^-1 Theta - log(|P|)] = 1/2 * (nu^t nu) - log|L|
  return 0.5 * arma::as_scalar(nu.t() * nu) - log(arma::det(L));
}

void draw_betas(tree &t, suff_stat &ss, double &sigma, data_info &di, tree_prior_info &tree_pi, RNG &gen)
{
  arma::mat phi;
  arma::vec res_vec;
  arma::mat P;
  arma::mat Theta;
  arma::vec mu;
  arma::mat cov;
  tree::tree_p bn;

  for (suff_stat_it ss_it = ss.begin(); ss_it != ss.end(); ++ss_it){
    bn = t.get_ptr(ss_it->first);
    if(bn == 0) Rcpp::stop("[draw_beta]: could not find node that is in suff stat map in the tree");
    else {
      leaf_ft l = bn->get_leaf();
      // phi = evaluate_entire_activation(ss_it, di, l, tree_pi);
      phi = l.get_phi(get_z_mat(ss_it, di));
      res_vec = get_res_vec(ss_it, di);
      P = calc_precision(phi, sigma, tree_pi);
      Theta = calc_theta(phi, res_vec, sigma, tree_pi);

      bn->set_beta(gen.mvnormal(Theta, P));
    }
  }
}

void grow_tree(tree &t, suff_stat &ss_train, suff_stat &ss_test, rho_diag_t &rho_diag, std::vector<std::map<double,int>> &sampled_rhos, int &iter, int &leaf_count, int &nx_nid, int &accept, rule_diag_t &rule_diag, double &sigma, data_info &di_train, data_info &di_test, tree_prior_info &tree_pi, RNG &gen)
{
  
  std::vector<int> bn_nid_vec; // vector to hold the id's of all of the bottom nodes in the tree
  for(suff_stat_it ss_it = ss_train.begin(); ss_it != ss_train.end(); ++ss_it) bn_nid_vec.push_back(ss_it->first);
  
  int ni = floor(gen.uniform() * bn_nid_vec.size()); // randomly pick the index of the node from which we will grow
  nx_nid = bn_nid_vec[ni]; // id of the node from which we are growing.
  tree::tree_p nx = t.get_ptr(nx_nid); // pointer to the node from which we are growing. refer to this node as nx
  tree::tree_cp nxp = nx->get_p(); // pointer to parent of nx in tree
  
  // we are ready to compute the log transition ratio:
  double q_grow_old = tree_pi.prob_b; // transition prob. of growing old tree into new tree
  double q_prune_new = 1.0 - tree_pi.prob_b; // transition prob. of pruning new true into old tree
  
  // int nleaf_old = t.get_nbots(); // number of leaves in old tree
  int nnog_old = t.get_nnogs(); // number of nodes in old tree with no grandchildren (nog node)
  int nnog_new = nnog_old; // number of nodes in new tree with no grandchildren
  
  if(nxp == 0){
    // nx is the root node so transition always propose growing it
    q_grow_old = 1.0;
    nnog_new = 1; // nx has no grandchildren in new tree
  } else if(nxp->is_nog()){
    // parent of nx has no grandchildren in old tree
    // in new tree nxp has grandchildren but nx does not
    // hence nnog_new = nnod_old
    nnog_new = nnog_old;
  } else{
    // parent of nx has grandchildren in old tree and will continue to do so in new tree
    // nx has no grandchildren in the new tree
    nnog_new = 1 + nnog_old;
  }
  
  // numerator of transition ratio: P(uniformly pick a nog node in new tree) * P(decide to prune new tree)
  // denominator of transition rate: P(uniformly pick a leaf node in old tree) * P(decide to grow old tree)
  
  double log_trans_ratio = (log(q_prune_new) - log( (double) nnog_new)) - (log(q_grow_old) - log( (double) q_grow_old));

  // Rcpp::Rcout << "The log_trans_ratio is " << log_trans_ratio;

  // for prior ratio:
  // numerator: p(grow at nx) * (1 - p(grow at nxl)) * (1 - p(grow at nxr))
  // denominator: (1 - p(grow at nx))
  // we need 1 - P(grow at nx in old tree) = 1 - alpha(1 + depth(nx))^(-beta) in denominator
  // we need P(grow at nx in new) (1 - P(grow at nxl in Tnew))(1 - P(grow at nxr in Tnew)) in numerator
  
  double p_grow_nx = tree_pi.alpha/pow(1.0 + (double) nx->get_depth(), tree_pi.beta); // prior prob of growing tree at nx
  double p_grow_nxl = tree_pi.alpha/pow(2.0 + (double) nx->get_depth(), tree_pi.beta); // prior prob of growing tree at nxl. remember depth of nxl is 1 + depth of nx
  double p_grow_nxr = tree_pi.alpha/pow(2.0 + (double) nx->get_depth(), tree_pi.beta); // prior prob of growing tree at nxr. remember depth of nxr is 1 + depth of nx
  double log_prior_ratio = log(p_grow_nx) + log(1.0 - p_grow_nxl) + log(1.0 - p_grow_nxr) - log(1.0 - p_grow_nx);
  
  // we now are ready to draw a decision rule
  rule_t rule;
  // if(analytic_centers.find(nx_nid) == analytic_centers.end()) Rcpp::stop("Couldn't find analytic center of node being split!");
  // std::vector<double> tmp_x0 = analytic_centers.find(nx_nid)->second; // analytic center of nx's polytope
  draw_rule(rule, t, nx_nid, di_train, tree_pi, gen); // draw the rule.
  // at this point we have the proposed rule and are ready to update our sufficient statistic map
  suff_stat prop_ss_train;
  compute_suff_stat_grow(ss_train, prop_ss_train, nx_nid, rule, t, di_train); // figure out which training observations from nx move to nxl and nxr
  suff_stat prop_ss_test;
  if(di_test.n > 0){
    compute_suff_stat_grow(ss_test, prop_ss_test, nx_nid, rule, t, di_test); // figure out which testing observation from nx more to nxl and nxr
  }
  
  /*
     25 Dec: temporarily, we can reject proposals that do not have one observation in each leaf
     this vastly simplifies the analytic center computation
   
   */

  int nxl_nid = 2*nx_nid; // id for the left child of nx
  int nxr_nid = 2*nx_nid+1; // id for right child of nx

  // get leaf info
  // parent
  leaf_ft lp = nx->get_leaf();

  // get ancestors, add proposed split variable(s) to ancestors if it is a continuous split
  std::set<int> ancestors;
  if (*tree_pi.sparse_smooth == 1){
    nx->get_ancestors(ancestors);
    if (!rule.is_cat) for (rc_it it = rule.phi.begin(); it != rule.phi.end(); ++it) ancestors.insert(it->first);
  }

  // child leaves need to be drawn
  // left leaf
  leaf_ft ll;
  ll.b = draw_b(tree_pi, gen);
  ll.rho = draw_rho(sampled_rhos, leaf_count, di_train, tree_pi, gen);
  ll.w = draw_omega(ll.rho, ancestors, di_train, tree_pi, gen);
  ll.act_opt = *tree_pi.activation_option;
  ll.intercept = tree_pi.intercept_option;
  // right leaf
  leaf_ft lr;
  lr.b = draw_b(tree_pi, gen);
  lr.rho = draw_rho(sampled_rhos, leaf_count, di_train, tree_pi, gen);
  lr.w = draw_omega(lr.rho, ancestors, di_train, tree_pi, gen);
  lr.act_opt = *tree_pi.activation_option;
  lr.intercept = tree_pi.intercept_option;

  double nxl_lil = compute_lil(prop_ss_train, nxl_nid, sigma, di_train, tree_pi, ll); // nxl's contribution to log marginal likelihood of new tree
  double nxr_lil = compute_lil(prop_ss_train, nxr_nid, sigma, di_train, tree_pi, lr); // nxr's contribution to log marginal likelihood of new tree
  double nx_lil = compute_lil(ss_train, nx_nid, sigma, di_train, tree_pi, lp); // nx's contribution to log marginal likelihood of old tree 

  // likelihood ratio also needs to include some constants from prior on jumps condition on tree
  // in GROW move, the new tree has one extra leaf so there's an additional factor of |V|^(-0.5) * exp(-0.5 beta0^T V^(-1) beta0) from leaf prior in the numerator
  // double prior_contribution = arma::as_scalar(tree_pi.beta0.t() * tree_pi.inv_V * tree_pi.beta0);
  double log_like_ratio = nxl_lil + nxr_lil - nx_lil - 0.5 * (log(tree_pi.det_V) + tree_pi.bVb);

  double log_alpha = log_like_ratio + log_prior_ratio + log_trans_ratio; // MH ratio
  if(log_alpha > 0){
    // Rcpp::Rcout << "the log_lik is " << log_like_ratio << std::endl;
    // Rcpp::Rcout << "nxl_lil is " << nxl_lil << std::endl;
    // Rcpp::Rcout << "nxr_lil is " << nxr_lil << std::endl;
    // Rcpp::Rcout << "nx_lil is " << nx_lil << std::endl;
    // Rcpp::Rcout << "0.5 * log(det(V)) is " << 0.5 * log(tree_pi.det_V) << std::endl;
    // Rcpp::Rcout << "[grow_tree]: the MH ratio is " << exp(log_alpha) << "." << std::endl;
    // Rcpp::Rcout << "[grow_tree]: warning, MH ratio greater than one!" << std::endl;
    log_alpha = 0.0; // if MH ratio greater than 1, we set it equal to 1. this is almost never needed
  } 
  if(gen.log_uniform() <= log_alpha){
    // accept the transition!
    ++(*tree_pi.rule_count); // increment running count of total number of splitting rules

    if(rule.is_cat){
      // we accepted a categorical rule
      ++(rule_diag.cat_prop); // increment count of proposed categorical rules
      int v_raw = rule.v_cat + di_train.p_cont;
      ++(tree_pi.var_count->at(v_raw));
    } else{
      // continuous rule
      if(rule.phi.size() == 1){
        // turned out to be an axis-aligned rule
        ++(rule_diag.aa_prop);
        // need to increment count for the variable involed
        // rule.phi.begin() points to first (and only) element in phi
        // variable index is the key.
        ++(tree_pi.var_count->at(rule.phi.begin()->first));
      } else{
        ++(rule_diag.obl_prop); // increment count of propose oblique rules
      }
    }
    // we need to update ss, the sufficient statistic object
    // this accounting is checked in test_grow_tree();

    suff_stat_it nxl_it = prop_ss_train.find(nxl_nid); // iterator at element for nxl in the proposed suff_stat map
    suff_stat_it nxr_it = prop_ss_train.find(nxr_nid); // iterator at element for nxr in the proposed suff_stat map
    
    if(nxl_it == prop_ss_train.end() || nxr_it == prop_ss_train.end()){
      // couldn't find a key in prop_ss_train equal to nxl_nid or nxr_nid
      Rcpp::Rcout << "[grow_tree]: sufficient stat map for training data not updated correctly in grow move!" << std::endl;
      Rcpp::Rcout << "  left child id = " << nxl_nid << "  right child = " << nxr_nid << std::endl;
      Rcpp::Rcout << "  available ids in map:";
      for(suff_stat_it it = prop_ss_train.begin(); it != prop_ss_train.end(); ++it) Rcpp::Rcout << " " << it->first;
      Rcpp::Rcout << std::endl;
      Rcpp::stop("missing id for either left or right child in proposed suff_stat_map!");
    }
    
    ss_train.insert(std::pair<int,std::vector<int>>(nxl_nid, nxl_it->second)); // add element for nxl in sufficient stat map
    ss_train.insert(std::pair<int,std::vector<int>>(nxr_nid, nxr_it->second)); // add element for nxr in sufficient stat map
    ss_train.erase(nx_nid); // remove element for nx in sufficient stat map
    if(di_test.n > 0){
      nxl_it = prop_ss_test.find(nxl_nid);
      nxr_it = prop_ss_test.find(nxr_nid);
      if(nxl_it == prop_ss_test.end() || nxr_it == prop_ss_test.end()){
        // couldn't find a key in prop_ss_train equal to nxl_nid or nxr_nid
        Rcpp::Rcout << "[grow_tree]: sufficient stat map for testing data not updated correctly in grow move!" << std::endl;
        Rcpp::Rcout << "  left child id = " << nxl_nid << "  right child = " << nxr_nid << std::endl;
        Rcpp::Rcout << "  available ids in map:";
        for(suff_stat_it it = prop_ss_test.begin(); it != prop_ss_test.end(); ++it) Rcpp::Rcout << " " << it->first;
        Rcpp::Rcout << std::endl;
        Rcpp::stop("missing id for either left or right child in proposed suff_stat_map!");
      }
      ss_test.insert(std::pair<int,std::vector<int>>(nxl_nid, nxl_it->second));
      ss_test.insert(std::pair<int,std::vector<int>>(nxr_nid, nxr_it->second));
      ss_test.erase(nx_nid);
    }

    // update length scale prior
    if (*tree_pi.rho_option != 0){
      if (iter == 0){ // always need to update on the first iteration
        // we accepted the proposal and it is the first iteration
        // so we just need to add the length scale of the child nodes
        // add_rho(sampled_rhos, ll.rho, leaf_count, di_train);
        // add_rho(sampled_rhos, lr.rho, leaf_count, di_train);

        rho_diag.accepted.push_back(ll.rho);
        rho_diag.accepted.push_back(lr.rho);
        
      } else { // after first iteration, only update when a proposal is accepted
        // // add children
        // add_rho(sampled_rhos, ll.rho, leaf_count, di_train);
        // add_rho(sampled_rhos, lr.rho, leaf_count, di_train);
        // // remove parent
        // remove_rho(sampled_rhos, lp.rho, leaf_count, di_train);

        rho_diag.accepted.push_back(ll.rho);
        rho_diag.accepted.push_back(lr.rho);
        rho_diag.pruned.push_back(lp.rho);

      }
    } // closes loop checking if we are using a DP prior for length scale

    t.birth(nx_nid, rule); // actually do the birth
    t.get_ptr(nxl_nid)->set_leaf(ll); // set left-child leaf
    t.get_ptr(nxr_nid)->set_leaf(lr); // set right-child leaf
    accept = 1;
    
    // get_child_analytic_center(analytic_centers, ss_train, nx_nid, nxl_nid, t, di_train, gen);
    // get_child_analytic_center(analytic_centers, ss_train, nx_nid, nxr_nid, t, di_train, gen);
    
    // now we need to compute the analytic centers of the two new leaf nodes
    //get_child_analytic_centers(analytic_centers, nx_nid, t, di_train.p_cont, gen);
  } else{
    accept = 0;

    // update length scale prior if it is the first iteration
    if (iter == 0){ // only need to update on the first iteration
      if (*tree_pi.rho_option != 0){
        // we rejected and it is the first iteration
        // so we just need to add the length scale of the root node, which is the parent leaf
        // add_rho(sampled_rhos, lp.rho, leaf_count, di_train);
        rho_diag.accepted.push_back(lp.rho);
      }
      
    }
    
    if(rule.is_cat){
      ++rule_diag.cat_prop;
      ++rule_diag.cat_rej;
    } else{
      if(rule.phi.size() == 1){
        ++rule_diag.aa_prop;
        ++rule_diag.aa_rej;
      } else{
        ++rule_diag.obl_prop;
        ++rule_diag.obl_rej;
      }
    }
    // don't do anything with rule counters or variable splitting counters etc.
  }
}

void prune_tree(tree &t, suff_stat &ss_train, suff_stat &ss_test, rho_diag_t &rho_diag, std::vector<std::map<double,int>> &sampled_rhos, int &leaf_count, int &nx_nid, int &accept, double &sigma, data_info &di_train, data_info &di_test, tree_prior_info &tree_pi, RNG &gen)
{
  // first we randomly select a nog node
  tree::npv nogs_vec; // vector of pointers to nodes w/ no grandchildren
  t.get_nogs(nogs_vec);

  int ni = floor(gen.uniform() * nogs_vec.size());
  tree::tree_p nx = nogs_vec[ni]; // pointer to node whose children we will prune
  tree::tree_p nxl = nx->get_l(); // left child that will be pruned
  tree::tree_p nxr = nx->get_r(); // right child that will be pruned
  
  // transition ratio stuff
  double q_prune_old = 1.0 - tree_pi.prob_b; // transition prob that we prune old tree
  double q_grow_new = tree_pi.prob_b; // transition prob that we grow new tree
  tree::tree_p nxp = nx->get_p(); // pointer to parent node of nx in old tree
  if(nxp == 0) q_grow_new = 1.0; // nx is top node so new tree is just root and we'd always propose a GROW when encountering new tree
  else q_grow_new = tree_pi.prob_b; // nx is not top node so  given T_new, we propose grow with prob tree_pi.pb

  int nleaf_new = t.get_nbots() - 1; // new tree has one less leaf node than old tree
  int nnog_old = t.get_nnogs(); // number of nodes with no grandchildren in old tree
  
  // numerator of transition ratio: P(uniformly pick a leaf node in new tree) * P(decide to grow newtree)
  // denominator of transition ratio: P(uniformly pick a nog node in old tree) * P(decide to prune old tree)

  double log_trans_ratio = (log(q_grow_new) - log(nleaf_new)) - (log(q_prune_old) - log(nnog_old));

  // prior ratio
  // numerator: we need [1 - P(grow at nx in Tnew)] = 1 - tree_pi.alpha/pow(1 + nx->get_depth(), tree_pi.beta)
  // denom: we need [P(grow at nx in Told)] x [1 - P(grow at nxl in Told)] x [1 - P(grow at nxr in Told)]
  double p_grow_nx = tree_pi.alpha/pow(1.0 + (double) nx->get_depth(), tree_pi.beta); // prior prob of growing tree at nx
  double p_grow_nxl = tree_pi.alpha/pow(2.0 + (double) nx->get_depth(), tree_pi.beta); // prior prob of growing tree at nxl, left child of nx
  double p_grow_nxr = tree_pi.alpha/pow(2.0 + (double) nx->get_depth(), tree_pi.beta); // prior prob of growing tree nxr, right child of nx
  double log_prior_ratio = log(1.0 - p_grow_nx) - (log(1.0 - p_grow_nxl) + log(1.0 - p_grow_nxr) + log(p_grow_nx));
  
  // likelihood ratio
  suff_stat prop_ss_train;
  suff_stat prop_ss_test;
  nx_nid = nx->get_nid(); // id for nx
  int nxl_nid = nxl->get_nid(); // id for nxl
  int nxr_nid = nxr->get_nid(); // id for nxr
  
  compute_suff_stat_prune(ss_train, prop_ss_train, nxl_nid, nxr_nid, nx_nid, t, di_train); // create a sufficient statistic map for the new tree
  if(di_test.n > 0) compute_suff_stat_prune(ss_test, prop_ss_test, nxl_nid, nxr_nid, nx_nid, t, di_test);

  // get leaf info
  std::set<int> ancestors;
  if (*tree_pi.sparse_smooth == 1) nx->get_ancestors(ancestors);
  // draw new leaf for parent node
  leaf_ft lp;
  lp.b = draw_b(tree_pi, gen);
  lp.rho = draw_rho(sampled_rhos, leaf_count, di_train, tree_pi, gen);
  lp.w = draw_omega(lp.rho, ancestors, di_train, tree_pi, gen);
  lp.act_opt = *tree_pi.activation_option;
  lp.intercept = tree_pi.intercept_option;
  // child leaves need to be drawn
  // left leaf
  leaf_ft ll = nxl->get_leaf();
  // right leaf
  leaf_ft lr = nxr->get_leaf();
  
  double nxl_lil = compute_lil(ss_train, nxl_nid, sigma, di_train, tree_pi, ll);
  double nxr_lil = compute_lil(ss_train, nxr_nid, sigma, di_train, tree_pi, lr);
  double nx_lil = compute_lil(prop_ss_train, nx_nid, sigma, di_train, tree_pi, lp);

  // old tree has one more leaf node than new tree so there is additional factor of
  // |V|^(-0.5) * exp(-0.5 beta0^T V^(-1) beta0) in denominator of likelihood ratio that comes from the prior on leafs
  // double prior_contribution = arma::as_scalar(tree_pi.beta0.t() * tree_pi.inv_V * tree_pi.beta0);
  double log_like_ratio = nx_lil - nxl_lil - nxr_lil + 0.5 * (log(tree_pi.det_V) + tree_pi.bVb);

  double log_alpha = log_like_ratio + log_prior_ratio + log_trans_ratio;
  if(log_alpha > 0) log_alpha = 0; // if MH greater than we, set it equal to 1
  if(gen.log_uniform() <= log_alpha){
    // accept the proposal!
    accept = 1;
    // need to decrement several counters
    --(*tree_pi.rule_count);
    
    if(nx->get_is_cat()){
      // we pruned away a categorical rule
      --(tree_pi.var_count->at((nx->get_v_cat()) + di_train.p_cont));
    } else{
      rule_t tmp_rule = nx->get_rule();
      if(tmp_rule.phi.size() == 1){
        // we pruned away an axis-aligned rule
        --(tree_pi.var_count->at(tmp_rule.phi.begin()->first));
      } else{
        // for now do nothing.
      }
    }
    
    // // need to remove the analytic centers of the pruned nodes
    // analytic_centers.erase(nxl_nid);
    // analytic_centers.erase(nxr_nid);
    
    // need to adjust ss
    // this accounting is checked in test_prune_tree();
    suff_stat_it nx_it = prop_ss_train.find(nx_nid); // iterator at element for nx in suff stat map for new tree
    if(nx_it == prop_ss_train.end()){
      // did not find nx_nid in the keys of prop_ss_train
      Rcpp::Rcout << "[prune_tree]: did not find id of new leaf node in the keys of training sufficient stat map" << std::endl;
      Rcpp::Rcout << "  id of new leaf: " << nx_nid << std::endl;
      Rcpp::Rcout << "  ids in map:";
      for(suff_stat_it it = prop_ss_train.begin(); it != prop_ss_train.end(); ++it) Rcpp::Rcout << " " << it->first;
      Rcpp::Rcout << std::endl;
      Rcpp::stop("missing id for new leaf node in prune move in training sufficient stat map");
    } else{
      ss_train.erase(nxl_nid); // delete entry for nxl in suff stat map
      ss_train.erase(nxr_nid); // delete entry for nxr in suff stat map
      ss_train.insert(std::pair<int, std::vector<int>>(nx_nid, nx_it->second)); // add an entry for nx in suff stat map
    }
    
    if(di_test.n > 0){
      nx_it = prop_ss_test.find(nx_nid);
      if(nx_it == prop_ss_test.end()){
        // did not find nx_nid in the keys of prop_ss_test
        Rcpp::Rcout << "[prune_tree]: did not find id of new leaf node in the keys of testing sufficient stat map" << std::endl;
        Rcpp::Rcout << "  id of new leaf: " << nx_nid << std::endl;
        Rcpp::Rcout << "  ids in map:";
        for(suff_stat_it it = prop_ss_test.begin(); it != prop_ss_test.end(); ++it) Rcpp::Rcout << " " << it->first;
        Rcpp::Rcout << std::endl;
        Rcpp::stop("missing id for new leaf node in prune move in training sufficient stat map");
      } else{
        ss_test.erase(nxl_nid); // delete entry for nxl in suff stat map
        ss_test.erase(nxr_nid); // delete entry for nxr in suff stat map
        ss_test.insert(std::pair<int, std::vector<int>>(nx_nid, nx_it->second)); // add an entry for nx in suff stat map
      }
    }

    if (*tree_pi.rho_option != 0){
      // add parent
      // add_rho(sampled_rhos, lp.rho, leaf_count, di_train);
      rho_diag.accepted.push_back(lp.rho);
      // remove children
      // remove_rho(sampled_rhos, ll.rho, leaf_count, di_train);
      // remove_rho(sampled_rhos, lr.rho, leaf_count, di_train);
      rho_diag.pruned.push_back(ll.rho);
      rho_diag.pruned.push_back(lr.rho);
    } // closes loop checking if we are using a DP prior for length scale

    t.death(nx_nid); // actually perform the death
    nx->set_leaf(lp); // set leaf value
  } else{
    accept = 0;
  }
}

void change_tree(tree &t, suff_stat &ss_train, rho_diag_t &rho_diag, std::vector<std::map<double,int>> &sampled_rhos, int &leaf_count, int &accept, double &sigma, data_info &di_train, tree_prior_info &tree_pi, RNG &gen)
{

  // t.print();

  std::vector<int> bn_nid_vec; // vector to hold the id's of all of the bottom nodes in the tree
  for(suff_stat_it ss_it = ss_train.begin(); ss_it != ss_train.end(); ++ss_it){
    bn_nid_vec.push_back(ss_it->first);
    // Rcpp::Rcout << ss_it->first;
  }
  // Rcpp::Rcout << std::endl;
  
  // compute lil of old tree
  double lil_t = 0.0;
  for (int i = 0; i < bn_nid_vec.size(); i++){
    leaf_ft l = t.get_ptr(bn_nid_vec[i])->get_leaf();
    lil_t += compute_lil(ss_train, bn_nid_vec[i], sigma, di_train, tree_pi, l);
  }

  // draw leaves for new tree
  leaf_ft l;
  std::vector<leaf_ft> leaf_vec(bn_nid_vec.size(), l); // vector to store leaves
  double lil_tstar = 0.0;
  for (int i = 0; i < bn_nid_vec.size(); i++){
    std::set<int> ancestors;
    if (*tree_pi.sparse_smooth == 1) t.get_ptr(bn_nid_vec[i])->get_ancestors(ancestors);
    // draw new leaf
    leaf_vec[i].b = draw_b(tree_pi, gen);
    leaf_vec[i].rho = draw_rho(sampled_rhos, leaf_count, di_train, tree_pi, gen);
    leaf_vec[i].w = draw_omega(leaf_vec[i].rho, ancestors, di_train, tree_pi, gen);
    leaf_vec[i].act_opt = *tree_pi.activation_option;
    leaf_vec[i].intercept = tree_pi.intercept_option;

    // compute lil contribution
    lil_tstar += compute_lil(ss_train, bn_nid_vec[i], sigma, di_train, tree_pi, leaf_vec[i]);
  }

  // compute MH acceptance probability
  // remember, the tree structure doesn't change so P(T) = P(T^*) cancels out
  // the transition probabilities also cancel out since the reversal of a CHANGE is also a CHANGE
  double log_alpha = lil_tstar - lil_t;
  if (log_alpha > 0) log_alpha = 0.0;
  if (gen.log_uniform() <= log_alpha){
    // accept the transition!
    // we do not need to update the sufficient statistic object because none of the rules changed

    // update length scale prior
    if (*tree_pi.rho_option != 0){
      for (int i = 0; i < bn_nid_vec.size(); i++){
        leaf_ft tmp_leaf = t.get_ptr(bn_nid_vec[i])->get_leaf();
        // add_rho(sampled_rhos, leaf_vec[i].rho, leaf_count, di_train);
        // remove_rho(sampled_rhos, tmp_leaf.rho, leaf_count, di_train);
        rho_diag.accepted.push_back(leaf_vec[i].rho);
        rho_diag.pruned.push_back(tmp_leaf.rho);
      }
    }

    // update leaves
    for (int i = 0; i < bn_nid_vec.size(); i++) t.get_ptr(bn_nid_vec[i])->set_leaf(leaf_vec[i]);

    // leaf_ft tmp_leaf;
    // for (int i = 0; i < bn_nid_vec.size(); i++){
    //   tree::tree_p nx = t.get_ptr(bn_nid_vec[i]);

    //   if (*tree_pi.rho_option != 0){
    //     // update length scale prior
    //     tmp_leaf.clear();
    //     tmp_leaf = nx->get_leaf();
    //     add_rho(sampled_rhos, leaf_vec[i].rho, leaf_count, di_train);
    //     remove_rho(sampled_rhos, tmp_leaf.rho, leaf_count, di_train);
    //   }

    //   // update the leaf
    //   nx->set_leaf(leaf_vec[i]);
    // }

    accept = 1;
  } else {
    accept = 0;
  }

}

void update_tree(tree &t, suff_stat &ss_train, suff_stat &ss_test, rho_diag_t &rho_diag, std::vector<std::map<double,int>> &sampled_rhos, int &iter, int &leaf_count, int &nid, int &accept, int &change_type, rule_diag_t &rule_diag, double &sigma, data_info &di_train, data_info &di_test, tree_prior_info &tree_pi, RNG &gen)
{
  accept = 0; // initialize indicator of MH acceptance to 0 (reject)
  double PBDx = tree_pi.prob_bd; // prob of proposing a birth or death move (typically 1)
  double PBx = tree_pi.prob_b; // prob of proposing a birth move (typically 0.5)
  
  // if tree is just the root, we must always GROW
  if(t.get_treesize() == 1) {
    PBx = 1.0;
    PBDx = 1.0;
  }
  
  if (gen.uniform() < PBDx){
    // grow or prune
    if(gen.uniform() < PBx){
      grow_tree(t, ss_train, ss_test, rho_diag, sampled_rhos, iter, leaf_count, nid, accept, rule_diag, sigma, di_train, di_test, tree_pi, gen);
      change_type = 1;
    } else{
      prune_tree(t, ss_train, ss_test, rho_diag, sampled_rhos, leaf_count, nid, accept, sigma, di_train, di_test, tree_pi, gen);
      change_type = 2;
    } 
  } else {
    change_tree(t, ss_train, rho_diag, sampled_rhos, leaf_count, accept, sigma, di_train, tree_pi, gen);
    change_type = 3;
  }

  // by this point, the decision tree has been updated so we can draw new jumps.
  draw_betas(t, ss_train, sigma, di_train, tree_pi, gen);
}
