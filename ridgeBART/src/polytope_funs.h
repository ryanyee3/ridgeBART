//
//  polytope_funs.hpp
//  
//
//  Created by Sameer Deshpande on 11/19/23.
//

#ifndef polytope_funs_hpp
#define polytope_funs_hpp

#include "tree.h"

// functions to help draw oblique rules
// given a phi, compute it's norm
double calc_phi_norm(std::map<int, double> &phi);

// given a phi and an x, return phi'x
double calc_phi_x(std::map<int, double> &phi, std::vector<double> &x);

// given a point x0 and hyperplane phi'x = c
// computes the projection of the point to the hyperplane.
void calc_proj(std::vector<double> &x0, std::map<int, double> &phi, double &c);

void calc_norms(std::vector<std::map<int,double>> &phi_vec, std::vector<double> &phi_norms);

// helper function that checks whether a constraint is met
bool check_constraints(std::vector<std::map<int,double>> &phi_vec,
                       std::vector<double> &b_vec,
                       std::vector<double> &x);




// helper function for computing gradient of log-barrier function
void calc_gradient(std::vector<std::map<int,double>> &phi_vec, std::vector<double> &phi_norms,
                   std::vector<double> &c_vec, int &p, std::vector<double> &x, arma::vec &g);

void calc_hessian(std::vector<std::map<int,double>> &phi_vec, std::vector<double> &phi_norms,
                  std::vector<double> &c_vec, int &p, std::vector<double> &x, arma::mat &H);


// this function finds the analytic center of a polytope defined by linear constraints in the form phi^T x < c
// the inputs for this function are:
//  (i) the linear constraints in the form phi^t x < c
//      where phi is a sparse matrix represented as an Rcpp::List
//      and c is an Rcpp::NumericVector of constants
// (ii) the dimension of the constrained space, p, represented as and integer


/*
double find_shift(double prob,
    std::vector<std::vector<double>> X,
    Rcpp::List phi_mat_list,
    Rcpp::NumericVector c_vec_input,
    int p);

std::vector<double> nested_constraint(Rcpp::List phi_mat_list,
    Rcpp::NumericVector c_vec_input,
    Rcpp::NumericVector x0_vec_input,
    Rcpp::List new_phi_mat_list,
    Rcpp::NumericVector new_c_vec_input,
     int p);

*/

void lin_ess(std::vector<double> &sampled_point,
             std::vector<std::map<int,double>> &phi_vec,
             std::vector<double> &b_vec,
             std::vector<double> &init_point,
             int &p,
             RNG &gen);


double find_shift(std::vector<std::vector<double>> &init_samples,
                  std::vector<std::map<int,double>> &phi_vec,
                  std::vector<double> &b_vec,
                  int &p,
                  double prob = 0.5);

void print_constraints(std::vector<std::map<int, double>> &phi_vec,
                       std::vector<double> &b_vec);

void get_init_point(std::vector<double> &x0,
                    std::vector<std::map<int, double>> &parent_phi_vec,
                    std::vector<double> &parent_c_vec,
                    std::vector<std::map<int,double>> &child_phi_vec,
                    std::vector<double> &child_c_vec,
                    int &p,RNG &gen);

//void get_init_point(std::vector<double> &x0, std::map<int,double> &last_phi,
//                    double &last_c, std::vector<std::map<int,double>> &phi_vec,
//                    std::vector<double> &b_vec,bool &is_left);

void get_child_analytic_center(std::map<int, std::vector<double>> &analytic_centers, suff_stat &ss, int &nx_nid, int &child_nid, tree &t, data_info &di, RNG &gen);

void calc_analytic_center(std::vector<double> &x0_vec, std::vector<std::map<int,double>> &phi_vec, std::vector<double> &c_vec, int &p, int max_iter = 50);

arma::vec calc_res_vec(arma::mat &A, arma::vec &b, arma::vec &x, arma::vec &y, arma::vec &v, arma::vec &g);
void inf_analytic_center(arma::vec &x, arma::mat &A, arma::vec &b);


#endif /* polytope_funs_h */
