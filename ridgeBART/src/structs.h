#ifndef GUARD_structs_h
#define GUARD_structs_h

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstddef>
#include <vector>

typedef std::vector<int>::iterator int_it; // iterator type for vectors of ints
typedef std::vector<double>::iterator dbl_it; // iterator type vectors of doubles
typedef std::set<int>::iterator set_it; // iterator type for sets of integers
typedef std::map<int, double>::const_iterator rc_it; // iterator type for random combination
typedef std::map<int, std::vector<int>> suff_stat; // sufficient statistic map: key is node id, value is vector of observations that land in that node
typedef suff_stat::iterator suff_stat_it; // iterator for sufficient statistc map

struct edge{
  int source;
  int sink;
  double weight;
  edge(){source = 0; sink = 0; weight = 1.0;}
  edge(int source_in, int sink_in){source = source_in; sink = sink_in; weight = 1.0;}
  edge(int source_in, int sink_in, double weight_in){source = source_in; sink = sink_in; weight = weight_in;}
  void copy_edge(edge edge_in){source = edge_in.source; sink = edge_in.sink; weight = edge_in.weight;} // unlikely to ever use this
};

typedef std::map<int, std::vector<edge> > edge_map;
typedef std::vector<edge>::iterator edge_vec_it;
typedef std::map<int, std::vector<edge>>::iterator edge_map_it;

class leaf_ft{
  public:
    arma::vec b;
    arma::mat w;
    arma::vec beta;
    arma::vec rho;
    int act_opt;
    int intercept; // 0 for no itercept, 1 for intercept
    leaf_ft(){
      b = arma::vec(1, arma::fill::zeros); 
      w = arma::mat(1, 1, arma::fill::zeros); 
      beta = arma::vec(1, arma::fill::zeros);
      rho = arma::vec(1, arma::fill::zeros);
      act_opt = 0;
      intercept = 0;
    }
    void set_dim(int p, int D){
      // p is number of smoothing parameters
      // D is number of random bases
      b.zeros(D);
      w.zeros(p, D);
      beta.zeros(D + intercept);
      rho.zeros(p);
    }
    void clear(){
      b.zeros(1); 
      w.zeros(1, 1); 
      beta.zeros(1);
      rho.zeros(1);
      act_opt = 0;
      intercept = 0;
    }
    // calculates Zw + b
    arma::mat calc_basis(arma::vec z){return z.t() * w + b.t();};
    arma::mat calc_basis(arma::mat Z){
      arma::mat Zw = Z * w;
      return Zw.each_row() + b.t();
    }
    // returns h(Zw+b), where h is activation
    arma::mat get_phi(arma::vec z){
      arma::mat tmp_mat(1, w.n_cols, arma::fill::zeros);
      if (act_opt == 1) tmp_mat = sqrt(2) * arma::cos(calc_basis(z)); // rff / cos activation
      else if (act_opt == 2) tmp_mat = arma::tanh(calc_basis(z)); // tanh activation
      else if (act_opt == 3) tmp_mat = 1 / (1 - arma::exp(-calc_basis(z))); // sigmoid
      else if (act_opt == 4) tmp_mat = z; // linear activation (NOTE: n_bases must be set to p_smooth to work)
      else if (act_opt == 5) tmp_mat = arma::max(calc_basis(z), arma::vec(z.n_elem, arma::fill::zeros)); // reLU activation
      else if (act_opt == 6) tmp_mat = calc_basis(z) / (1 - arma::exp(-calc_basis(z))); // "swish" activation
      else if (act_opt == 7) tmp_mat = arma::sin(calc_basis(z)) / calc_basis(z); // sinc activation
      else if (act_opt == 8) tmp_mat = arma::mat(1, 1, arma::fill::ones); // constant activation (NOTE: n_bases must be set to 1 to work)
      else {
        Rcpp::Rcout << "'" << act_opt << "' passed as activation option." << std::endl;
        Rcpp::stop("[get_phi]: invalid activation option.");
      }
      if (intercept == 0) return tmp_mat;
      else if (intercept == 1) return arma::join_horiz(arma::mat(1, 1, arma::fill::ones), tmp_mat);
      else Rcpp::stop("[get_phi]: intercept option must be set to 0 or 1!");
      // if (act_opt == 1) return sqrt(2) * arma::cos(calc_basis(z)); // rff / cos activation
      // else if (act_opt == 2) return arma::tanh(calc_basis(z)); // tanh activation
      // else if (act_opt == 3) return 1 / (1 - arma::exp(-calc_basis(z))); // sigmoid
      // else if (act_opt == 4) return z; // linear activation (NOTE: n_bases must be set to p_smooth to work)
      // else if (act_opt == 5) return arma::max(calc_basis(z), arma::vec(z.n_elem, arma::fill::zeros)); // reLU activation
      // else if (act_opt == 6) return calc_basis(z) / (1 - arma::exp(-calc_basis(z))); // "swish" activation
      // else if (act_opt == 7) return arma::sin(calc_basis(z)) / calc_basis(z); // just tryin stuff, wavelet?
      // else if (act_opt == 8) return arma::vec(1, arma::fill::ones); // constant activation (NOTE: n_bases must be set to 1 to work)
      // else Rcpp::stop("[get_phi]: invalid activation option.");
    }
    arma::mat get_phi(arma::mat Z){
      arma::mat tmp_mat(Z.n_rows, w.n_cols);
      if (act_opt == 1) tmp_mat = sqrt(2) * arma::cos(calc_basis(Z)); // rff / cos activation
      else if (act_opt == 2) tmp_mat = arma::tanh(calc_basis(Z)); // tanh activation
      else if (act_opt == 3) tmp_mat = 1 / (1 - arma::exp(-calc_basis(Z))); // sigmoid
      else if (act_opt == 4) tmp_mat = Z; // linear activation (NOTE: n_bases must be set to p_smooth to work)
      else if (act_opt == 5) tmp_mat = arma::max(calc_basis(Z), arma::mat(Z.n_rows, w.n_cols, arma::fill::zeros)); // reLU activation
      else if (act_opt == 6) tmp_mat = calc_basis(Z) / (1 - arma::exp(-calc_basis(Z))); // "swish" activation
      else if (act_opt == 7) tmp_mat = arma::sin(calc_basis(Z)) / calc_basis(Z); // sinc activation
      else if (act_opt == 8) tmp_mat = arma::mat(Z.n_rows, 1, arma::fill::ones); // constant activation (NOTE: n_bases must be set to 1 to work)
      else {
        Rcpp::Rcout << "'" << act_opt << "' passed as activation option." << std::endl;
        Rcpp::stop("[get_phi]: invalid activation option.");
      }
      if (intercept == 0) return tmp_mat;
      else if (intercept == 1) return arma::join_horiz(arma::vec(Z.n_rows, arma::fill::ones), tmp_mat);
      else Rcpp::stop("[get_phi]: intercept option must be set to 0 or 1!");
      // if (act_opt == 1) return sqrt(2) * arma::cos(calc_basis(Z)); // rff / cos activation
      // else if (act_opt == 2) return arma::tanh(calc_basis(Z)); // tanh activation
      // else if (act_opt == 3) return 1 / (1 - arma::exp(-calc_basis(Z))); // sigmoid
      // else if (act_opt == 4) return Z; // linear activation (NOTE: n_bases must be set to p_smooth to work)
      // else if (act_opt == 5) return arma::max(calc_basis(Z), arma::mat(Z.n_rows, w.n_cols, arma::fill::zeros)); // reLU activation
      // else if (act_opt == 6) return calc_basis(Z) / (1 - arma::exp(-calc_basis(Z))); // "swish" activation
      // else if (act_opt == 7) return arma::sin(calc_basis(Z)) / calc_basis(Z); // just tryin stuff, wavelet?
      // else if (act_opt == 8) return arma::mat(Z.n_rows, 1, arma::fill::ones); // constant activation (NOTE: n_bases must be set to 1 to work)
      // else Rcpp::stop("[get_phi]: invalid activation option.");
    }
    // full evaluation of h(Zw + b)* beta
    double eval_leaf(arma::vec z){return arma::as_scalar(get_phi(z) * beta);};
    arma::vec eval_leaf(arma::mat Z){return get_phi(Z) * beta;};
};

typedef arma::vec::const_iterator arma_vec_it;
typedef arma::mat::const_iterator arma_mat_it;

class rule_t{
public:
  bool is_cat; // is it a categorical split
  std::map<int, double> phi; // weights of random combination
  double c; // cutpoint
  
  int v_cat; // index of variable of the categorical variable on which we split (always between 0 and p_cat)
  std::set<int> l_vals; // holds unique values of levels of v associated w/ left child
  std::set<int> r_vals; // holds unique values of levels of v associated w/ right child
  
  rule_t(){
    is_cat = false;
    phi = std::map<int,double>();
    c = 0.0;
    v_cat = 0;
    l_vals = std::set<int>();
    r_vals = std::set<int>();
  }
  
  void clear(){
    is_cat = false;
    phi.clear();
    c = 0.0;
    v_cat = 0;
    l_vals.clear();
    r_vals.clear();
  }

  void normalize_phi(){
    if(phi.size() > 0){
      double sq_sum = 0.0;
      for(std::map<int, double>::iterator it = phi.begin(); it != phi.end(); ++it) sq_sum += pow(it->second, 2.0);
      double norm = sqrt(sq_sum);
      for(std::map<int,double>::iterator it = phi.begin(); it != phi.end(); ++it) it->second /= norm;
    }
  }

};

// structure for diagnostics tracking how often we propose/reject certain types of rules
struct rule_diag_t{
  int aa_prop;
  int aa_rej;
  int cat_prop;
  int cat_rej;
  int obl_prop;
  int obl_rej;
  rule_diag_t(){aa_prop = 0; aa_rej = 0; cat_prop = 0; cat_rej = 0; obl_prop = 0; obl_rej = 0;}
  void reset(){aa_prop = 0; aa_rej = 0; cat_prop = 0; cat_rej = 0; obl_prop = 0; obl_rej = 0;}
};

// structure for tracking rho parameter
struct rho_diag_t{
  std::vector<arma::vec> accepted;
  std::vector<arma::vec> pruned;
  rho_diag_t(){accepted = std::vector<arma::vec>(); pruned = std::vector<arma::vec>();}
  void reset(){accepted.clear(); pruned.clear();}
};

//structures for graph partitioning


// class holding data dimensions and pointers to the covariate data
class data_info{
public:
  int n; // number of observations
  int p_cont; // number of continuous predictors
  int p_cat; // number of categorical predictors
  int p_smooth; // number of smoothing predictors
  int p_split; // total number of splitting predictors (likely will never every use this)
  double* x_cont; // pointer to the matrix of continuous predictors
  int* x_cat; // pointer to matrix of categorical predictors (levels coded as integers, beginning with 0)
  double* z_mat; // pointer to matrix of smoothing predictors
  double* rp; // partial residual;
  data_info(){n = 0; p_cont = 0; p_cat = 0; p_smooth = 0; p_split = 0; x_cont = 0; x_cat = 0; z_mat = 0; rp = 0;}
};



// holds hyperparameters for regression tree prior
class tree_prior_info{
public:
  double alpha; // 1st parameter of the branching process prior
  double beta; // 2nd parameter in branching process prior
  
  double prob_bd; // prob of proposing a grow (birth) or prune (death) move. almost always set to 1
  double prob_b; // prob of proposing a grow (birth) move. almost always set to 0.5
  
  double* prob_cat; // probability of a categorical rule.
  double* prob_aa; // conditional on splitting on continuous variables,

  std::vector<double>* theta; // prob. that we pick one variable out of p_cont + p_cat
  std::vector<int> *var_count; // counts how many times we split on a single variable
  int* rule_count; // how many total rules are there in the ensemble

  // unif_cuts passed an Rcpp::LogicalVector
  int* unif_cuts; // unif_cuts = 1 to draw cuts uniformly, = 0 to draw from pre-specified cutpoints, = minimum integer for NA
  std::vector<std::set<double> >* cutpoints;
  
  std::vector<std::set<int>> *cat_levels; // holds the levels of the categorical variables
  std::vector<std::vector<edge>> *edges; // vector of edges for the graph-structured categorical levels
  std::vector<int> *K; // number of levels per categorical variable
  int* graph_split; // do we split categorical variables using the supplied graphs?
  int graph_cut_type; // determines how we generate the partition
  
  bool oblique_option;
  bool dense_phi;
  int phi_option;
  int* x0_option;

  // length scale parameters
  int* dp_option;
  std::vector<std::map<double, int>> *rho_samples;
  int* rho_option; // indicates how we will choose the length scale (0 = fixed, 1 = Dirchilet Process prior)
  std::vector<double> *rho_prior;
  double *rho_alpha;
  double *rho_nu;
  double *rho_lambda;

  // hyperparameters will go here eventually
  int* activation_option;
  int intercept_option; // 0 for no intercept term, 1 for intercept term
  int* sparse_smooth; // 0 for regular smoothing, 1 for sparse smoothing
  int* rand_rot; // 0 for not rotation, 1 for random rotation of omega
  int D; // number of bases in a single leaf
  arma::vec beta0; // prior beta mean
  arma::mat V; // prior beta covariance
  arma::mat inv_V; // inverse of V, saving reduces number calculations
  double det_V; // determinant of V, saving reduces number calculations
  double bVb; // double to store beta0^T V^-1 beta0 which is a constant used in the lil_ratio calculation
  
  // constructor
  tree_prior_info(){
    alpha = 0.95;
    beta = 2.0;
    prob_bd = 1.0;
    prob_b = 0.5;
    prob_cat = 0; // 0 pointer
    prob_aa = 0; // 0 pointer
    theta = 0; // 0 pointer
    var_count = 0; // 0 pointer
    rule_count = 0; // 0 pointer
    
    unif_cuts = 0; // 0 pointer
    cutpoints = 0; // 0 pointer
    
    cat_levels = 0; // 0 pointer
    edges = 0; // 0 pointer
    K = 0; // 0 pointer
  
    graph_split = 0; // 0 pointer
    graph_cut_type = 0;
    
    oblique_option = false;
    dense_phi = false;
    phi_option = 0;
    x0_option = 0; // 0 pointer

    dp_option = 0; // 0 pointer
    rho_samples = 0; // 0 pointer
    rho_option = 0; // 0 pointer
    rho_prior = 0; // 0 pointer
    rho_nu = 0; // 0 pointer
    rho_lambda = 0; // 0 pointer

    activation_option = 0; // 0 pointer
    intercept_option = 0;
    sparse_smooth = 0; // 0 pointer
    rand_rot = 0; // 0 pointer
    D = 0;
    beta0 = arma::vec(D + intercept_option, arma::fill::zeros); // init prior mean with zeros
    V = arma::mat(D + intercept_option, D + intercept_option, arma::fill::eye); // init prior covariance with identity matrix
    inv_V = arma::inv(V);
    det_V = arma::det(V);
  }

  void set_D(int x){
    D = x + intercept_option;
    beta0 = arma::vec(D, arma::fill::zeros); // init prior mean with zeros
    set_V(arma::mat(D, D, arma::fill::eye)); // init prior covariance with identity matrix
  }

  void set_V(arma::mat Sigma){
    V = Sigma;
    inv_V = arma::inv(V);
    det_V = arma::det(V);
    bVb = arma::as_scalar(beta0.t() * inv_V * beta0);
  }
};



// silly class to convert sets of integers into character strings
class set_str_conversion{
public:
  std::map<std::string,char> str_to_hex_lookup;
  std::map<char, std::string> hex_to_str_lookup;
  
  set_str_conversion(){
    str_to_hex_lookup["0000"] = '0';
    str_to_hex_lookup["0001"] = '1';
    str_to_hex_lookup["0010"] = '2';
    str_to_hex_lookup["0011"] = '3';
    str_to_hex_lookup["0100"] = '4';
    str_to_hex_lookup["0101"] = '5';
    str_to_hex_lookup["0110"] = '6';
    str_to_hex_lookup["0111"] = '7';
    str_to_hex_lookup["1000"] = '8';
    str_to_hex_lookup["1001"] = '9';
    str_to_hex_lookup["1010"] = 'a';
    str_to_hex_lookup["1011"] = 'b';
    str_to_hex_lookup["1100"] = 'c';
    str_to_hex_lookup["1101"] = 'd';
    str_to_hex_lookup["1110"] = 'e';
    str_to_hex_lookup["1111"] = 'f';
    
    hex_to_str_lookup['0'] = "0000";
    hex_to_str_lookup['1'] = "0001";
    hex_to_str_lookup['2'] = "0010";
    hex_to_str_lookup['3'] = "0011";
    hex_to_str_lookup['4'] = "0100";
    hex_to_str_lookup['5'] = "0101";
    hex_to_str_lookup['6'] = "0110";
    hex_to_str_lookup['7'] = "0111";
    hex_to_str_lookup['8'] = "1000";
    hex_to_str_lookup['9'] = "1001";
    hex_to_str_lookup['a'] = "1010";
    hex_to_str_lookup['b'] = "1011";
    hex_to_str_lookup['c'] = "1100";
    hex_to_str_lookup['d'] = "1101";
    hex_to_str_lookup['e'] = "1110";
    hex_to_str_lookup['f'] = "1111";
  }
  
  std::string set_to_hex(int &K, std::set<int> &vals){
    // we divide the full set {0, 1, ... , K-1} into blocks of 4
    // block 0 {0,1,2,3}, block 1 {4,5,6,7}, etc.
    // we sweep over each block and see whether or not each element is in the set vals
    // this creates a binary string of length 4, which we then convert into a single character w/ our lookup table
    
    int num_blocks = K/4;
    std::string tmp_str(4,'0'); // temporary string of length 4, overwritten with each block
    std::string hex_str(num_blocks+1,'0'); // our outputted string, initialized for the empty set
    std::map<std::string, char>::iterator str_ch_it; // iterator for looking up in str_to_hex_lookup
    
    for(int blk_id = 0; blk_id <= num_blocks; blk_id++){
      tmp_str.assign(4,'0'); // reset the temporary string to all 0's
      for(int j = 0; j < 4; j++){
        if(vals.count(4*blk_id + j) == 1){
          // if the integer 4*blk_id + j is in the set vals, we make the j-th element of tmp_str = 1
          tmp_str[j] = '1';
        }
      } // closes loop over elements of each block
      str_ch_it = str_to_hex_lookup.find(tmp_str);
      if(str_ch_it == str_to_hex_lookup.end()){
        Rcpp::Rcout << "[set_to_hex]: temporary string " << tmp_str << std::endl;
        Rcpp::stop("string not found in str_to_hex_lookup!");
      } else{
        hex_str[blk_id] = str_ch_it->second;
      }
    } // closes loop over the blocks
    return hex_str;
  }
  
  std::set<int> hex_to_set(int &K, std::string &hex_str){
    
    int num_blocks = K/4;
    if(hex_str.size() != num_blocks+1){
      Rcpp::Rcout << "[hex_to_set]: hex_str = " << hex_str << " is wrong size" << std::endl;
      Rcpp::Rcout << "[hex_to_set]: for K = " << K << " values, hex_str must be of length " << num_blocks+1 << std::endl;
      Rcpp::stop("hex_str is of wrong size!");
    }
    std::map<char, std::string>::iterator ch_str_it; // iterator for looking up in hex_to_str_lookup
    std::string tmp_str;
    std::set<int> vals;
    
    for(int blk_id = 0; blk_id <= num_blocks; blk_id++){
      // std::string's [] lets us look up on a character-by-character basis
      ch_str_it = hex_to_str_lookup.find(hex_str[blk_id]);
      if(ch_str_it == hex_to_str_lookup.end()){
        Rcpp::Rcout << "[hex_to_set]: character " << hex_str[blk_id] << std::endl;
        Rcpp::stop("character not found in hex_to_str_lookup!");
      } else{
        tmp_str = ch_str_it->second;
        for(int j = 0; j < 4; j++){
          if(tmp_str[j] == '1') vals.insert(4*blk_id+j);
        }
      } // closes if/else checking that element of hex_str is a key in hex_to_set_lookup
    } // closes loop over the elements of hex_str
    
    return vals;
  }
  
}
;

#endif
