
#include "rule_funs.h"

void draw_rule(rule_t &rule, tree &t, int &nid, data_info &di, tree_prior_info &tree_pi, RNG &gen){
  rule.clear();
  // draw_cont_rule(rule, t, nid, x0, di, tree_pi, gen);
  if(gen.uniform() < *(tree_pi.prob_cat)) draw_cat_rule(rule, t, nid, di, tree_pi, gen);
  else draw_cont_rule(rule, t, nid, di, tree_pi, gen);
}

void draw_cat_rule(rule_t &rule, tree &t, int &nid, data_info &di, tree_prior_info &tree_pi, RNG &gen){
  rule.is_cat = true;
  rule.v_cat = floor(di.p_cat * gen.uniform());
  tree::tree_p nx = t.get_ptr(nid); // at what node are we proposing this rule.
  int rule_counter = 0;
  std::set<int> avail_levels = tree_pi.cat_levels->at(rule.v_cat); // get the full set of levels for this variable
  nx->get_rg_cat(rule.v_cat, avail_levels); // determine the set of levels available at nx.
  // if there is only one level left for this variable at nx, we will just propose a trivial split
  // and will reset the value of avail_levels to be the full set of all levels for the variable
  if(avail_levels.size() <= 1) avail_levels = tree_pi.cat_levels->at(rule.v_cat);
  
  rule.l_vals.clear();
  rule.r_vals.clear();
  
  if(tree_pi.graph_split[rule.v_cat] == 1 && tree_pi.edges->at(rule.v_cat).size() > 0){
    // if we explicitly say to use the graph to split the variables
    //graph_partition(avail_levels, rule.l_vals, rule.r_vals, tree_pi.edges->at(rule.v_cat), tree_pi.K->at(rule.v_cat), tree_pi.graph_cut_type, gen);
    graph_partition(rule.l_vals, rule.r_vals, tree_pi.edges->at(rule.v_cat), avail_levels, tree_pi.graph_cut_type, gen);
  } else{
    // otherwise we default to splitting the available levels uniformly at random: prob 0.5 to go to each child
    rule_counter = 0;
    //double tmp_prob = 0.5;
    //if(tree_pi.a_cat > 0  && tree_pi.b_cat > 0) tmp_prob = gen.beta(tree_pi.a_cat, tree_pi.b_cat);
    //else tmp_prob = 0.5;
    
    while( ((rule.l_vals.size() == 0) || (rule.r_vals.size() == 0)) && rule_counter < 1000 ){
      rule.l_vals.clear();
      rule.r_vals.clear();
      for(set_it it = avail_levels.begin(); it != avail_levels.end(); ++it){
        //if(gen.uniform() <= tmp_prob) rule.l_vals.insert(*it);
        if(gen.uniform() <= 0.5) rule.l_vals.insert(*it);
        else rule.r_vals.insert(*it);
      }
      ++(rule_counter);
    }
    if(rule_counter == 1000){
      Rcpp::stop("[draw_cat_rule]: failed to generate valid categorical split in 1000 attempts"); // this should almost surely not get triggered.
    }
  }
  if( (rule.l_vals.size() == 0) || (rule.r_vals.size() == 0) ){
    Rcpp::stop("[draw_cat_rule]: proposed an invalid categorical rule!");
  }
}

// by construction x0 is a point inside the polytope already
void draw_cont_rule(rule_t &rule, tree &t, int &nid, data_info &di, tree_prior_info &tree_pi, RNG &gen)
{
  rule.is_cat = false;
  rule.phi.clear();
  
  // we will have to recurse up the tree to get the previous contraints
  // no matter what type of rule we choose
  // std::vector<std::map<int, double>> phi_vec;
  // std::vector<double> c_vec;
  // std::vector<double> aa_lower(di.p_cont, -1.0);
  // std::vector<double> aa_upper(di.p_cont, 1.0);
  
  // t.get_ptr(nid)->get_cont_constraints(phi_vec, c_vec, aa_lower, aa_upper, di.p_cont);

  // axis-aligned
  double c_upper = 1.0; // upper bound for range of cutpoints in axis aligned split
  double c_lower = -1.0; // lower bound for range of cutpoints in axis aligned split
  tree::tree_p nx = t.get_ptr(nid); // at what node are we proposing this rule.

  // for now pick direction uniformly
  int v_aa = floor(di.p_cont * gen.uniform());
  rule.phi.insert(std::pair<int,double>(v_aa, 1.0));

  // decided how to pick cutpoint
  if(tree_pi.unif_cuts[v_aa] == 0){
    // draw the cutpoint from cutpoint_list
    c_lower = *(tree_pi.cutpoints->at(v_aa).begin()); // returns smallest element in set
    c_upper = *(tree_pi.cutpoints->at(v_aa).rbegin()); // reverse iterator, returns largest value in set
    nx->get_rg_aa(v_aa, c_lower, c_upper);
    if(c_lower >= c_upper){
      // this is a weird tree and we'll just propose a trivial split
      c_lower = *(tree_pi.cutpoints->at(v_aa).begin());
      c_upper = *(tree_pi.cutpoints->at(v_aa).rbegin());
    }
    std::vector<double> valid_cutpoints;
    if(tree_pi.cutpoints->at(v_aa).count(c_lower) != 1 || tree_pi.cutpoints->at(v_aa).count(c_upper) != 1){
      // c_lower and c_upper were not found in the set of available cutpoints
      Rcpp::Rcout << "[draw_rule]: attempting to select a cutpoint from given set" << std::endl;
      Rcpp::Rcout << "  lower bound is: " << c_lower << " count in set is " << tree_pi.cutpoints->at(v_aa).count(c_lower) << std::endl;
      Rcpp::Rcout << "  upper bound is: " << c_upper << " count in set is " << tree_pi.cutpoints->at(v_aa).count(c_upper) << std::endl;
      //Rcpp::Rcout << "  cutpoints are:";
      //for(std::set<double>::iterator it = tree_pi.cutpoints->at(rule.v_aa).begin(); it != tree_pi.cutpoints->at(rule.v_aa).end(); ++it) Rcpp::Rcout << " " << *it;
      //Rcpp::Rcout << std::endl;
      Rcpp::stop("we should never have a c that is outside the pre-defined set of cutpoints!");
    }
    // we want to draw from the cutpoints exclusive of c_lower & c_upper;
    // i.e. we want to start with the one just after c_lower and just before c_upper
    // std::set::lower_bound: iterator at first element that is not considered to come before
    // std::set::upper_bound: iterator at first element considered to come after
    // if value is not in set, lower_bound and upper_bound give same result
    // if value is in set: lower bound returns the value, upper bound returns the next value
    for(std::set<double>::iterator it = tree_pi.cutpoints->at(v_aa).upper_bound(c_lower); it != tree_pi.cutpoints->at(v_aa).lower_bound(c_upper); ++it){
      valid_cutpoints.push_back(*it);
    }
    int num_cutpoints = valid_cutpoints.size();
    if(num_cutpoints < 1){
      // no valid splits are available; we will just pick something, all of the observations will go to one child anyway...
      valid_cutpoints.clear();
      for(std::set<double>::iterator it = tree_pi.cutpoints->at(v_aa).begin(); it != tree_pi.cutpoints->at(v_aa).end(); ++it){
        valid_cutpoints.push_back(*it);
      }
      num_cutpoints = valid_cutpoints.size();
    }
    // at this point, valid cutpoints is a vector containing the available cutpoints at this node. we pick one uniformly.
    rule.c = valid_cutpoints[floor(gen.uniform() * num_cutpoints)];

  } else {
    // pick cutpoint uniformally
    double c_upper = 1.0;
    double c_lower = -1.0;
    t.get_ptr(nid)->get_rg_aa(v_aa, c_lower, c_upper);
    rule.c = gen.uniform(c_lower, c_upper);
  }
  
  // if(gen.uniform() < *tree_pi.prob_aa){
  //   // axis-aligned
  //   // for now pick direction uniformly
  //   int v_aa = floor(di.p_cont * gen.uniform());
  //   rule.phi.insert(std::pair<int,double>(v_aa, 1.0));
  //   //rule.c = gen.uniform(aa_lower[v_aa], aa_upper[v_aa]);
  // } else{
  //   // actually oblique
  //   if( (di.p_cont == 2 && nid == 1) || tree_pi.phi_option == 1){
  //     // always pick a 2-sparse phi 
  //     int v1 = floor(di.p_cont * gen.uniform());
  //     int v2 = floor(di.p_cont * gen.uniform());
  //     int counter = 0;
  //     while( (v1 == v2) && counter < 500){
  //       v2 = floor(di.p_cont * gen.uniform());
  //       ++counter;
  //     }
  //     if(v1 == v2) Rcpp::stop("[draw_cont_rule]: could not find two different non-zero indices for phi!");
  //     else{
  //       rule.phi.insert(std::pair<int,double>(v1, gen.normal()));
  //       rule.phi.insert(std::pair<int,double>(v2, gen.normal()));
  //       rule.normalize_phi(); // normalize phi
  //     }
  //   } else if(tree_pi.phi_option == 2 || tree_pi.phi_option == 3){
  //     // pick a random hyperplane
  //     // find the null space of the normal vector
  //     // use either a random element of the null space for the new rule (option == 2)
  //     // or a random linear combination of elements of the null space (option == 3)
  //     int rand_index = floor(phi_vec.size() * gen.uniform());
  //     arma::rowvec rand_arma_vec(di.p_cont, arma::fill::zeros);
  //     for(rc_it it = phi_vec[rand_index].begin(); it != phi_vec[rand_index].end(); ++it) rand_arma_vec(it->first) = it->second;
  //     arma::mat null_space = arma::null(arma::mat(rand_arma_vec));
  //     arma::vec new_phi = null_space.col(0);
      
  //     if(tree_pi.phi_option == 2){
  //       int rand_index2 = 0;
  //       if(null_space.n_cols > 1) rand_index2 = floor(null_space.n_cols * gen.uniform());
  //       new_phi = null_space.col(rand_index2);
  //     } else{
  //       if(null_space.n_cols > 1){
  //         arma::vec rand_weights = arma::randn(null_space.n_cols); // vector of standard normals
  //         new_phi = null_space * rand_weights;
  //       }
  //     }
  //     // now we have new_phi
      
  //     // if it only has one non-zero element, we need to follow convention
  //     // that axis-aligned splits are of the form x[j] < c; and not have -1*x[j] < c
  //     arma::uvec new_phi_support = find(arma::abs(new_phi) > 1e-9); // find the non-zero elements
  //     if(new_phi_support.n_elem == 1) new_phi(new_phi_support(0)) = 1.0;
  //     for(int j = 0; j < di.p_cont; ++j){
  //       if(abs(new_phi(j)) > 1e-9) rule.phi.insert(std::pair<int,double>(j, new_phi(j))); // tolerance is 1e-9 for 0. this may be too permissive, but 1e-16 might be too strict
  //     }
  //     rule.normalize_phi(); // normalize phi
  //   } else if(tree_pi.phi_option == 4){
  //     // generate a uniform direction
  //     rule.phi.clear();
  //     gen.unif_direction(rule.phi, di.p_cont);
  //   } else{
  //     Rcpp::stop("invalid phi_option");
  //   }
  // } // closes if/else determining if it is axis-aligned or oblique
  // rule.c = 0.0;
  // std::vector<double> new_x0;
  // for(std::vector<double>::iterator it = x0.begin(); it != x0.end(); ++it) new_x0.push_back(*it);

  // if(tree_pi.x0_option == 1){
  //   // use the analytic center so do nothing
  // } else if(tree_pi.x0_option == 2){
  //   // draw the point randomly
  //   //Rcpp::Rcout << "About to try lin_ess" << std::endl;
  //   lin_ess(new_x0, phi_vec, c_vec, x0, di.p_cont, gen);
  // } else Rcpp::stop("x0_option should be 0 or 1");
  // // now set  to be phi'x0
  // for(rc_it it = rule.phi.begin(); it != rule.phi.end(); ++it) rule.c += it->second * new_x0[it->first];
}
