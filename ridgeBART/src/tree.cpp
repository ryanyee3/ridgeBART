#include <string>
#include <vector>
#include <map>
#include "tree.h"

//--------------------------------------------------
// constructors

tree::tree(){
  leaf.b = arma::vec(1, arma::fill::zeros);
  leaf.w = arma::mat(1, 1, arma::fill::zeros);
  leaf.beta = arma::vec(1, arma::fill::zeros);

  rule.is_cat = false;
  rule.phi = std::map<int, double>();
  rule.c = 0.0;
  
  rule.v_cat = 0;
  rule.l_vals = std::set<int>();
  rule.r_vals = std::set<int>();
  
  p = 0;
  l = 0;
  r = 0;
}

tree::tree(int p, int D){
  leaf.b = arma::vec(D, arma::fill::zeros);
  leaf.w = arma::mat(p, D, arma::fill::zeros);
  leaf.beta = arma::vec(D, arma::fill::zeros);

  rule.is_cat = false;
  rule.phi = std::map<int, double>();
  rule.c = 0.0;
  
  rule.v_cat = 0;
  rule.l_vals = std::set<int>();
  rule.r_vals = std::set<int>();
  
  p = 0;
  l = 0;
  r = 0;
}

// for the destructor: cut back to one node
void tree::to_null()
{
  size_t ts = get_treesize();
  while(ts > 1){
    npv nv;
    get_nogs(nv); // get nodes with no grandchildren
    for(size_t i = 0; i < nv.size(); i++){
      delete nv[i]->l;
      delete nv[i]->r;
      nv[i]->l=0;
      nv[i]->r=0;
    }
    ts = get_treesize();
  }
  leaf.clear();
  rule.clear();
  p = 0;
  l = 0;
  r = 0;
}

// print
void tree::print() const // pc is flag to print children
{
  size_t id = get_nid();
  
  if(get_ntype() == 't') Rcpp::Rcout << "tree size:" << get_treesize() << std::endl;
  Rcpp::Rcout << "id." << id;
  if(get_ntype() == 'b'){
    int D = leaf.b.size();
    Rcpp::Rcout << "  b: ";
    for(arma_vec_it it = leaf.b.begin(); it != leaf.b.end(); ++it) Rcpp::Rcout << " " << *it;
    Rcpp::Rcout << std::endl;
    Rcpp::Rcout << "  w: ";
    int col = 0;
    for(arma_mat_it it = leaf.w.begin(); it != leaf.w.end(); ++it){
      Rcpp::Rcout << " " << *it;
      col += 1;
      if (col == D){
        col = 0;
        Rcpp::Rcout << std::endl;
        if (it != leaf.w.end() -1) Rcpp::Rcout << "   ";
      }
    }
    Rcpp::Rcout << "  beta: ";
    for(arma_vec_it it = leaf.beta.begin(); it != leaf.beta.end(); ++it) Rcpp::Rcout << " " << *it;
    Rcpp::Rcout << std::endl;
  }
  else if(get_ntype() == 't' && get_treesize() == 1){
    // tree is just the top node
    int D = leaf.b.size();
    Rcpp::Rcout << "  b: ";
    for(arma_vec_it it = leaf.b.begin(); it != leaf.b.end(); ++it) Rcpp::Rcout << " " << *it;
    Rcpp::Rcout << std::endl;
    Rcpp::Rcout << "  w: ";
    int col = 0;
    for(arma_mat_it it = leaf.w.begin(); it != leaf.w.end(); ++it){
      Rcpp::Rcout << " " << *it;
      col += 1;
      if (col == D){
        col = 0;
        Rcpp::Rcout << std::endl << "   ";
      }
    }
    Rcpp::Rcout << "  beta: ";
    for(arma_vec_it it = leaf.beta.begin(); it != leaf.beta.end(); ++it) Rcpp::Rcout << " " << *it;
    Rcpp::Rcout << std::endl;
  } else{ // internal node or nnog or top node
    if(rule.is_cat){
      Rcpp::Rcout << " categorical split on X_cat[" << rule.v_cat+1 << "]" << std::endl;
      Rcpp::Rcout << "   left levels: ";
      for(set_it it = rule.l_vals.begin(); it != rule.l_vals.end(); ++it) Rcpp::Rcout << " " << *it;
      Rcpp::Rcout << std::endl;
      Rcpp::Rcout << "    right levels: ";
      for(set_it it = rule.r_vals.begin(); it != rule.r_vals.end(); ++it) Rcpp::Rcout << " " << *it;
      Rcpp::Rcout << std::endl;
    } else{
      // continuous split, check whether it is axis-aligned or not.
      if(rule.phi.size() == 1){
        // axis-aligned rule
        rc_it it = rule.phi.begin();
        Rcpp::Rcout << " axis-aligned split: X_cont[" << it->first+1 << "] < " << rule.c << std::endl;
      } else{
        Rcpp::Rcout << " oblique split: ";
        for(rc_it it = rule.phi.begin(); it != rule.phi.end(); ++it){
          if(it == rule.phi.begin()) Rcpp::Rcout << it->second << "* X_cont[" << it->first+1 << "]";
          else Rcpp::Rcout << " + " << it->second << "*X_cont[" << it->first+1 << "]";
        }
        Rcpp::Rcout << " < " << rule.c << std::endl;
      } // closes if/else determining whether continuous rule is axis-aligned or oblique
    } // closes if/else determining whether it is a categorial or continuous rule
  } // closes if/else determining the type of node we're at.
  
  if(l){
    l->print();
    r->print();
  }
}


//--------------------------------------------------
//operators

// this overloads = if we say tree1 = tree2

tree& tree::operator=(const tree& rhs)
{
   if(&rhs != this) {
      to_null(); //kill left hand side (this)
      cp(this,&rhs); //copy right hand side to left hand side
   }
   return *this;
}

// tree-level gets

// get a vector of pointers to the bottom nodes
void tree::get_bots(npv& bv)
{
  if(l) { //have children
    l->get_bots(bv);
    r->get_bots(bv);
  } else bv.push_back(this);
}

//get a vector of pointers to the no grandchildren nodes
void tree::get_nogs(npv& nv)
{
  if(l) { //have children
    if((l->l) || (r->l)) {  //have grandchildren
      if(l->l) l->get_nogs(nv);
      if(r->l) r->get_nogs(nv);
    } else nv.push_back(this);
  }
}

bool tree::is_nog() const
{
  bool isnog=true;
  if(l) {
    if(l->l || r->l) isnog=false; //one of the children has children.
  } else isnog=false; //no children
  return isnog;
}

//get a vector of pointers to *ALL* nodes
void tree::get_nodes(npv& v)
{
  v.push_back(this);
  if(l) {
    l->get_nodes(v);
    r->get_nodes(v);
  }
}

void tree::get_nodes(cnpv& v)  const
{
  v.push_back(this);
  if(l) {
    l->get_nodes(v);
    r->get_nodes(v);
  }
}


// get the size of the tree (i.e. number of nodes)
int tree::get_treesize() const
{
   if(!l) return 1;  //if bottom node, tree size is 1
   else return (1+l->get_treesize()+r->get_treesize());
}
//number of nodes with no
int tree::get_nnogs() const
{
  if(!l) return 0; // this is a bottom node
  if(l->l || r->l) return(l->get_nnogs() + r->get_nnogs()); // this has at least one grandchild
  else return 1; // this is a nog node
}
// count number of bottom nodes
int tree::get_nbots() const
{
  if(!l) return 1; // this is a bottom node
  else return(l->get_nbots() + r->get_nbots());
}

// get depth of current node
int tree::get_depth() const
{
   if(!p) return 0; //no parents
   else return (1+p->get_depth());
}

// gets depth of the entire tree
int tree::get_tree_depth() 
{
  tree::npv bn_vec; // vector to store pointers to bottom nodes of tree
  this->get_bots(bn_vec); // get pointers to all bottom nodes
  int max_node_id = 0;
	int current_node_id = 0;
  // loop over bottom nodes
  for(tree::npv::iterator npv_it = bn_vec.begin(); npv_it != bn_vec.end(); ++npv_it){
    // push to map where first entry is nid and second entry is mu value
	  current_node_id = (*npv_it)->get_nid();
    if (current_node_id >= max_node_id) max_node_id = current_node_id;
  }
  return floor(log(max_node_id) / log(2));
}

// get the node id
int tree::get_nid() const
//recursion up the tree
{
   if(!p) return 1; //if you don't have a parent, you are the top
   if(this==p->l) return 2*(p->get_nid()); //if you are a left child
   else return 2*(p->get_nid())+1; //else you are a right child
}

// get the node type
char tree::get_ntype() const
{
   //t:top, b:bottom, n:no grandchildren, i:internal
  if(p == 0) return 't';
  if(l == 0) return 'b';
  if( (l->l != 0) && (r->l != 0)) return 'n'; // no grandchildren
  return 'i';
  /*
   if(!p) return 't';
   if(!l) return 'b';
   if(!(l->l) && !(r->l)) return 'n';
   return 'i';
  */
}

tree::tree_p tree::get_ptr(int nid)
{
  if(this->get_nid() == nid) return this; //found it
  if(!l) return 0; //no children, did not find it
  tree_p lp = l->get_ptr(nid);
  if(lp) return lp; //found on left
  tree_p rp = r->get_ptr(nid);
  if(rp) return rp; //found on right
  return 0; //never found it
}

tree::tree_cp tree::get_bn(double* x_cont, int* x_cat)
{
  if(!l) return this; // node has no left child, so it must be a leaf
  double tmp_x = 0.0;
  int l_count = 0;
  int r_count = 0;
  
  if(rule.is_cat){
    if(x_cat != 0){
      l_count = rule.l_vals.count(x_cat[rule.v_cat]);
      r_count = rule.r_vals.count(x_cat[rule.v_cat]);
      if(l_count == 1 && r_count == 0) return l->get_bn(x_cont, x_cat);
      else if(l_count == 0 && r_count == 1) return r->get_bn(x_cont, x_cat);
      else if(l_count == 0 && r_count == 0){
        Rcpp::Rcout << "[get_bn]: could not find value of categorical predictor in either left or right cutset!" << std::endl;
        return 0;
      } else{ // l_count == 1 & r_count == 1
        Rcpp::Rcout << "[get_bn]: value of categorical predictor in both left & right cutset!" << std::endl;
        return 0;
      }
    } else{
      Rcpp::Rcout << "[get_bn]: encountered categorical decision rule but no categorical predictors were supplied!" << std::endl;
      return 0;
    }
  } else{
    // continuous decision rule
    tmp_x = 0.0;
    for(rc_it it = rule.phi.begin(); it != rule.phi.end(); ++it){
      tmp_x += x_cont[it->first] * (it->second);
    }
    if(tmp_x < rule.c) return l->get_bn(x_cont, x_cat);
    else if(tmp_x >= rule.c) return r->get_bn(x_cont, x_cat);
    else{
      Rcpp::Rcout << "[get_bn]: Could not resolve continuous decision rule." << std::endl;
      Rcpp::Rcout << "  phi'x = " << tmp_x << " c = " << rule.c << std::endl;
      return 0;
    }
  } // closes if/else determining the type of rule
}

// overload evaluate
// takes a vector z to evaluate a single observation
double tree::evaluate(arma::vec &z, double* x_cont, int* x_cat)
{
  tree::tree_cp bn = get_bn(x_cont, x_cat);
  if(bn == 0) Rcpp::stop("[tree::evaluate]: unable to find valid bottom node.]");// arma::vec(1, fill::zeros); // when we don't have a valid bottom node, return 0
  else {
    // leaf_ft l = bn->get_leaf();
    // Rcpp::Rcout << "at tree.evaluate()." << std::endl;
    // Rcpp::Rcout << "z = " << z << std::endl;
    // Rcpp::Rcout << "l.beta = " << l.beta << std::endl;
    // Rcpp::Rcout << "l.w = " << l.w << std::endl;
    // Rcpp::Rcout << "l.b = " << l.b << std::endl;
    return bn->get_leaf().eval_leaf(z);
  }
}
// takes a matrix Z to evaluate multiple observations in the same leaf
// arma::vec tree::evaluate(arma::mat &Z, double* x_cont, int* x_cat)
// {
//   tree::tree_cp bn = get_bn(x_cont, x_cat);
//   if(bn == 0) return std::nan(""); // when we don't have a valid bottom node, return nan
//   else return bn->leaf.eval_leaf(Z);
// }

// birth
void tree::birth(int nid, rule_t rule)
{
  tree_p np = get_ptr(nid); // get the pointer to the node being split
  if(!np){
    Rcpp::Rcout << "Trying birth at node w/ id " << nid << " but pointer is invalid!" << std::endl;
    Rcpp::stop("[birth]: invalid pointer to node!");
  }
  if(np->l) Rcpp::stop("[birth]: trying to split a node that already has children!");
  
  tree_p l = new tree; // initialize the new tree that will be the left child of nid
  l->rule.clear();
  l->leaf.clear();

  tree_p r = new tree; // initialize the new tree that will be the left child of nid
  r->rule.clear();
  r->leaf.clear();
  
  np->l = l;
  np->r = r;
  
  // np->leaf.clear(); // leaf of parent gets cleared during grow
  np->rule.is_cat = rule.is_cat;
  np->rule.phi = rule.phi;
  np->rule.c = rule.c;
  np->rule.v_cat = rule.v_cat;
  np->rule.l_vals = rule.l_vals;
  np->rule.r_vals = rule.r_vals;
  l->p = np;
  r->p = np;
  
}

// perform a death
void tree::death(int nid)
{
  tree_p nb = get_ptr(nid);
  if(!nb) Rcpp::stop("[death]: missing pointer for nid!");
  
  if(nb->is_nog()){
    delete nb->l;
    delete nb->r;
    
    nb->l = 0; // nb now has no children so set corresponding pointers to 0
    nb->r = 0; // nb now has no children so set corresponding pointers to 0

    // reset the rule values
    nb->rule.clear();
  } else Rcpp::stop("[death]: cannot perform death move on a node with grandchildren.");
}

// we initially track the axis-aligned constraints differently than the actually oblique ones
// once we get to the root, we add in the axis-aligned constraints
void tree::get_cont_constraints(std::vector<std::map<int, double>> &phi_vec, std::vector<double> &b_vec,
                                std::vector<double> &aa_lower, std::vector<double> &aa_upper, int &p_cont)
{
  std::map<int, double> tmp_a; // will always be phi (if this is left child) or -phi (if this is right child)
  double tmp_sign = 1.0; // 1.0 when this is the left child, -1.0 when this is the right child
  double tmp_b = 0.0;
  if(p){
    if(!p->rule.is_cat){
      // this is the child of a node with continuous rule
      // if this is the left child, constraint is phi'x < c
      // if this is the right child, constraint is phi'x > c, equivalent to (-phi)'x < (-c)
      if(p->rule.phi.size() == 1){
        // this is an axis-aligned rule.
        // we will update the entries in aa_lower and aa_upper.
        int v = p->rule.phi.begin()->first;
        if(this == p->l){
          // this is the left child of the parent with rule x[v]< c
          // the new *upper* bound for x[v] at this is min(c, previous upper bound)
          if(p->rule.c < aa_upper[v]) aa_upper[v] = p->rule.c;
        }
        if(this == p->r){
          // this is the right child of the parent with rule x[v] > c
          // the new *lower* bound for x[v] at this is max(c, previous lower bound)
          if(p->rule.c > aa_lower[v]) aa_lower[v] = p->rule.c;
        }
        
      } else{
        if(this == p->l) tmp_sign = 1.0;
        else tmp_sign = -1.0;
        tmp_a.clear();
        tmp_b = tmp_sign * p->rule.c;
        for(rc_it it = p->rule.phi.begin(); it != p->rule.phi.end(); ++it){
          tmp_a.insert(std::pair<int,double>(it->first, tmp_sign * it->second));
        }
        phi_vec.push_back(tmp_a);
        b_vec.push_back(tmp_b);
      }
    } else{
      // categorical rule does not affect continuous constraints
    }
    p->get_cont_constraints(phi_vec, b_vec, aa_lower, aa_upper, p_cont);
  } else{
    // this is the root. need to append the bounds from axis-aligned rules
    
    for(int j = 0; j < p_cont; ++j){
      tmp_a.clear();
      tmp_a.insert(std::pair<int,double>(j, -1.0));
      phi_vec.push_back(tmp_a);
      b_vec.push_back(-1.0 * aa_lower[j]);
      
      tmp_a.clear();
      tmp_a.insert(std::pair<int,double>(j, 1.0));
      phi_vec.push_back(tmp_a);
      b_vec.push_back(aa_upper[j]);
    }
  }
  
}

void tree::get_recent_phi(std::map<int,double> &last_phi, double &last_c, int &last_nid, bool &is_left){
  if(last_nid == 0){
    // still looking
    if(p){
      if(!p->rule.is_cat){
        // found the most recent ancestor with continuous rule
        last_phi.clear();
        last_nid = p->get_nid();
        for(rc_it it = p->rule.phi.begin(); it != p->rule.phi.end(); ++it) last_phi.insert(std::pair<int,double>(it->first, it->second));
        last_c = p->rule.c;
        if(this==p->l) is_left = true;
        else is_left = false;
      }
    }
  }
}

void tree::get_rg_cat(int &v, std::set<int> &levels){
// recruse up the tree:
//    1. we will initialize levels to be the set of all levels for x_cat[v]
//    2. if this has a parent that splits on v, check if this is p->l or p->r.
//        * if this is the left child of a node that splits on v, then replace levels with p->rule.l_vals & set recurse = false
//
  bool recurse = true;
  if(p){
    // this has a parent.
    if(p->rule.is_cat && p->rule.v_cat == v){
      // parent of this splits on v
      if(this == p->l){
        levels.clear();
        for(set_it it = p->rule.l_vals.begin(); it != p->rule.l_vals.end(); ++it) levels.insert(*it);
        recurse = false; // no need to continue recursing up the tree
      } else if(this == p->r){
        levels.clear();
        for(set_it it = p->rule.r_vals.begin(); it != p->rule.r_vals.end(); ++it) levels.insert(*it);
        recurse = false; // no need to continue recursing up the tree
      } else Rcpp::stop("[get_rg_cat]: this was not equal to either left or right child of its parent!");
    }
    if(recurse) p->get_rg_cat(v, levels);
  }
}

void tree::get_rg_aa(int &v, double &c_lower, double &c_upper){
  if(p){
    
    if( (!p->rule.is_cat) && p->rule.phi.size() == 1){
      if(v == p->rule.phi.begin()->first){
        if(this == p->l){
          // this is the left child of the parent with rule x[v]< c
          // the new *upper* bound for x[v] at this is min(c, previous upper bound)
          if(p->rule.c < c_upper) c_upper = p->rule.c;
        }
        if(this == p->r){
          // this is the right child of the parent with rule x[v] > c
          // the new *lower* bound for x[v] at this is max(c, previous lower bound)
          if(p->rule.c > c_lower) c_lower = p->rule.c;
        }
      }
    }
    p->get_rg_aa(v, c_lower, c_upper);
  }
}

void tree::get_ancestors(std::set<int> &ancestors){
  if(p){
    if (!p->rule.is_cat){
      // the parent has a continuous splitting rule
      for (rc_it it = p->rule.phi.begin(); it != p->rule.phi.end(); ++it) ancestors.insert(it->first);
    }
    p->get_ancestors(ancestors);
  }
}

//private functions

//copy tree o to tree n
void tree::cp(tree_p n, tree_cp o)
//assume n has no children (so we don't have to kill them)
//recursion down
{
  if(n->l) Rcpp::stop("[cp]:tree n has children.");
  // if we haven't stopped by now, it's valid to continue to copying

  n->leaf.b = o->leaf.b;
  n->leaf.w = o->leaf.w;
  n->leaf.beta = o->leaf.beta;
  // not 100% sure if it's valid to do n->rule = o->rule
  // but better to be safe and deliberately copy all members
  n->rule.is_cat = o->rule.is_cat;
  n->rule.phi = o->rule.phi;
  n->rule.c = o->rule.c;
  n->rule.v_cat = o->rule.v_cat;
  n->rule.l_vals = o->rule.l_vals;
  n->rule.r_vals = o->rule.r_vals;
  
  if(o->l){
    // if o has children
    n->l = new tree; // create new tree for n's left child
    (n->l)->p = n; // assign parent of n's left child as n
    cp(n->l, o->l); // recurse for left child
    n->r = new tree;
    (n->r)->p = n;
    cp(n->r, o->r);
  }
}

