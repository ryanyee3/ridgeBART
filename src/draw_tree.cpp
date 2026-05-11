#include "draw_tree.h"

void draw_tree(tree &t, std::vector<std::map<double,int>> &sampled_rhos, int &leaf_count, data_info &di, tree_prior_info &tree_pi, RNG &gen)
{
  //Rcpp::Rcout << "[draw tree]: started!" << std::endl;

  // initialize the tree
  t.to_null(); // prune tree back to root
  // give the root node a leaf
  t.set_int_opt(tree_pi.intercept_option);
  t.set_leaf_dim(di.p_smooth, tree_pi.D, tree_pi.intercept_option);
  t.set_b(gen.unif_vec(tree_pi.D, 0, 2 * M_PI));
  
  t.set_rho(draw_rho(sampled_rhos, leaf_count, di, tree_pi, gen));
  arma::vec tmp_rho = t.get_rho();
  if (*tree_pi.rho_option == 1) add_rho(sampled_rhos, tmp_rho, leaf_count, di);
  std::set<int> ancestors;
  if (*tree_pi.sparse_smooth == 1) t.get_ancestors(ancestors);
  t.set_w(draw_omega(tmp_rho, ancestors, di, tree_pi, gen));
  t.set_act_opt(*tree_pi.activation_option);

  tree::npv bnv;
  int dnx; // depth of node nx
  int nx_nid;
  int max_depth = 0; // depth of deepest leaf
  int prev_max_depth = 0; // max depth from previous iteration
  
  double PGnx = 0.0; // probability of growing at node nx
  bool grow = true;
  int counter = 0;
  
  // stuff for decision rules
  rule_t rule;
  while(grow && counter < 100){
    prev_max_depth = max_depth;
    bnv.clear();
    t.get_bots(bnv); // get the bottom nodes of the tree (could be done more efficiently but not super important)
    
    //Rcpp::Rcout << "Starting round " << counter;
    //Rcpp::Rcout << "  tree size = " << t.get_treesize();
    
    for(tree::npv_it l_it = bnv.begin(); l_it != bnv.end(); ++l_it){
      dnx = (*l_it)->get_depth(); // remember l_it is a pointer to an element in bnv, which is itself a pointer, hence the need for (*)->
      if(dnx > max_depth) max_depth = dnx; // the node we're at is deeper than the maximum depth of the tree in the previous iteration
    }
    
    //Rcpp::Rcout << "  max depth = " << max_depth << std::endl;
    
    if( (max_depth < prev_max_depth) || (max_depth > 1 + prev_max_depth) ){
      // each time through the loop we can only grow the deepest leaf nodes
      // we should *never* encounter this condition but it's here to be safe
      Rcpp::Rcout << "max_depth = " << max_depth << " prev_max_depth = " << prev_max_depth << std::endl;
      Rcpp::stop("[draw_tree]: max depth should be prev_max_depth or prev_max_depth + 1!");
    } else if(max_depth == prev_max_depth && max_depth != 0){
      // tree didn't grow in the last iteration so we should break out of the loop
      break;
    } else {
      grow = false;
      //Rcpp::Rcout << "max_depth = " << max_depth << " prev_max_depth = " << prev_max_depth << std::endl;
      for(tree::npv_it l_it = bnv.begin(); l_it != bnv.end(); ++l_it){
        dnx = (*l_it)->get_depth();
        nx_nid = (*l_it)->get_nid();
        //Rcpp::Rcout << "trying node " << (*l_it)->get_nid() << "at depth " << dnx << std::endl;
        if(dnx == max_depth){
          // current node nx is at the maximum depth, we will try to grow the tree from nx
          PGnx = tree_pi.alpha/pow(1.0 + (double) dnx, tree_pi.beta);
          double tmp_unif = gen.uniform();
          //Rcpp::Rcout << "  node " << (*l_it)->get_nid() << " PGnx = " << PGnx << " tmp_unif = " << tmp_unif;

          if(tmp_unif < PGnx){
            //Rcpp::Rcout << " can grow...";
            grow = true;
            // we're actually going to grow the tree!
            draw_rule(rule, t, nx_nid, di, tree_pi, gen);
            leaf_ft lp = (*l_it)->get_leaf();

            std::set<int> ancestors;
            if (*tree_pi.sparse_smooth == 1){
              (*l_it)->get_ancestors(ancestors);
              if (!rule.is_cat) for (rc_it it = rule.phi.begin(); it != rule.phi.end(); ++it) ancestors.insert(it->first);
            }

            // draw left leaf
            leaf_ft ll;
            ll.b = gen.unif_vec(tree_pi.D, 0, 2 * M_PI);
            ll.rho = draw_rho(sampled_rhos, leaf_count, di, tree_pi, gen);
            ll.w = draw_omega(ll.rho, ancestors, di, tree_pi, gen);
            ll.act_opt = *tree_pi.activation_option;
            ll.intercept = tree_pi.intercept_option;
            // right leaf
            leaf_ft lr;
            lr.b = gen.unif_vec(tree_pi.D, 0, 2 * M_PI);
            lr.rho = draw_rho(sampled_rhos, leaf_count, di, tree_pi, gen);
            lr.w = draw_omega(lr.rho, ancestors, di, tree_pi, gen);
            lr.act_opt = *tree_pi.activation_option;
            lr.intercept = tree_pi.intercept_option;
            t.birth(nx_nid, rule); // actually do the birth
            t.get_ptr(2*nx_nid)->set_leaf(ll); // set left-child leaf
            t.get_ptr(2*nx_nid+1)->set_leaf(lr); // set right-child leaf
            // update rho tracking
            if (*tree_pi.rho_option == 1){
              add_rho(sampled_rhos, ll.rho, leaf_count, di);
              add_rho(sampled_rhos, lr.rho, leaf_count, di);
              remove_rho(sampled_rhos, lp.rho, leaf_count, di);
            }
          } // closes if checking that we're actually trying to grow the tree
        } else{
          //Rcpp::Rcout << "  node " << (*l_it)->get_nid() << " not at max depth. moving on";
        }// closes if/else checking that we're at node at the deepest level of the tree
        //Rcpp::Rcout << std::endl;
      } // closes for loop over all bottom nodes in the tree
    } // closes if/else checking that max depth is valid
    ++(counter);
  } // closes main while loop
  
  // now that we have drawn the decision tree, let's draw the jumps
  bnv.clear();
  t.get_bots(bnv);
  for(tree::npv_it l_it = bnv.begin(); l_it != bnv.end(); ++l_it){
    (*l_it)->set_beta(gen.mvnormal(tree_pi.beta0, tree_pi.inv_V));
  }
}
