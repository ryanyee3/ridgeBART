#ifndef RNG_H
#define RNG_H
#include "data_parsing_funs.h"

using std::vector;

class RNG
{
public:
  // continuous distributions
  double uniform(double x = 0.0, double y = 1.0);
  double exponential(double lambda);
  double log_uniform();
  double gumbel();
  double normal(double mu = 0.0, double sd = 1.0);
  double gamma(double shape, double scale);
  double chi_square(double df);
  double laplace(double mu = 0.0, double b = 1.0);
  double cauchy(double loc = 0.0, double scale = 1.0);
  
  double beta(double a1, double a2);
  int categorical(std::vector<double> &probs);
  
  void dirichlet(std::vector<double> &theta, std::vector<double> &concentration);
  
  
  // discrete
  int multinomial(const int &R, const std::vector<double> &probs);
  int multinomial(const int &R, std::vector<double>* probs);
  
  // vectors
  arma::vec unif_vec(int d, double x, double y);
  arma::vec std_norm_vec(int d); // vector of standard normals
  arma::vec std_laplace_vec(int d); // vector of standard laplaces
  arma::vec std_cauchy_vec(int d); // vector of standard cauchys
  arma::mat std_norm_mat(int nrow, int ncol); // matrix of standard normals
  arma::mat std_laplace_mat(int nrow, int ncol); // matrix of standard laplaces
  arma::mat std_cauchy_mat(int nrow, int ncol); // matrix of standard cauchys
  
  // sample from multivariate normal N(P^-1m, P^-1)
  arma::vec mvnormal(arma::vec m, arma::mat P);
  
  // sample from N(0,1) truncated to be > lo
  double lo_trunc_std_norm(double lo);
  
  // sample from N(mean,1) truncated to be > lo
  double lo_trunc_norm(double mean, double lo);
  
  // sample from N(mean, 1) truncated to be < lo
  double hi_trunc_norm(double mean, double lo);

  void unif_direction(std::map<int,double> &phi, int dim);
  
};
#endif // RNG_H
