//
//  polytope_funs.cpp
//  
//
//  Created by Sameer Deshpande on 11/19/23.
//

#include "polytope_funs.h"

double calc_phi_norm(std::map<int, double> &phi){
  double sq_norm = 0.0;
  for(rc_it it = phi.begin(); it != phi.end(); ++it) sq_norm += pow(it->second, 2.0);
  return sqrt(sq_norm);
}

// given a phi and an x, return phi'x
double calc_phi_x(std::map<int, double> &phi, std::vector<double> &x){
  double phi_x = 0.0;
  for(rc_it it = phi.begin(); it != phi.end(); ++it) phi_x += (it->second) * x[it->first];
  return phi_x;
}


// given a point x0 and hyperplane phi'x = c
// computes the projection of the point to the hyperplane.
void calc_proj(std::vector<double> &x0, std::map<int, double> &phi, double &c){
  double phi_norm = calc_phi_norm(phi); // norm of phi
  double phi_x = calc_phi_x(phi,x0); // phi'(init_x0)
  
  double k = (c - phi_x)/pow(phi_norm, 2.0);
  for(rc_it it = phi.begin(); it != phi.end(); ++it){
    x0[it->first] += k * (it->second);
  }
  phi_x = calc_phi_x(phi,x0);
  if( abs(c - phi_x ) > 1e-12 ){
    Rcpp::Rcout << " x0 is now :";
    for(std::vector<double>::iterator it = x0.begin(); it != x0.end(); ++it) Rcpp::Rcout << " " << *it;
    Rcpp::Rcout << std::endl;
    Rcpp::Rcout << " phi'x0 = " << phi_x << " c = " << c << std::endl;
    Rcpp::Rcout << abs(c - phi_x) << std::endl;
    Rcpp::stop("[calc_proj]: abs( c - phi'x0 ) > 1e-16. Mistake in calculating the projection");
  }
}

void calc_norms(std::vector<std::map<int,double>> &phi_vec, std::vector<double> &phi_norms){
  phi_norms.clear(); // we are changing the value of phi_norms!!!
  for(std::vector<std::map<int,double>>::iterator it = phi_vec.begin(); it != phi_vec.end(); ++it){
    phi_norms.push_back(calc_phi_norm(*it));
  }
}


// helper function that checks whether a constraint is met
bool check_constraints(std::vector<std::map<int,double>> &phi_vec,
                       std::vector<double> &b_vec,
                       std::vector<double> &x)
{
  bool results = true;
  double phi_x = 0.0;
  int num_constraints = phi_vec.size();
  for(int i = 0; i < num_constraints; ++i){
    phi_x = calc_phi_x(phi_vec[i], x);
    if(phi_x >= b_vec[i]){
      results = false;
      break;
    }
  }
  return results;
}
void lin_ess(std::vector<double> &sampled_point,
             std::vector<std::map<int,double>> &phi_vec,
             std::vector<double> &b_vec,
             std::vector<double> &init_point,
             int &p,
             RNG &gen)
{
  sampled_point.clear(); // clear out the sampled point
  int num_constraints = phi_vec.size(); // how many constraints do we have
  // draw auxillary vector from N(0,I)
  std::vector<double> v_vec;
  for(int j = 0;  j < p; ++j) v_vec.push_back(gen.normal());
  std::vector<double> atv; // A^Tv
  std::vector<double> atx; // A^Tx

  for(std::vector<std::map<int,double>>::iterator it = phi_vec.begin(); it != phi_vec.end(); ++it){
    atv.push_back(calc_phi_x(*it, v_vec));
    atx.push_back(calc_phi_x(*it, init_point));
  }
  std::vector<double> r;
  for(int i = 0; i < num_constraints; ++i) r.push_back(sqrt(pow(atx[i], 2.0) + pow(atv[i], 2.0)));
  // now we are ready to find the angles of intersection
  // there could be zero, one, or two points that intersect with the ellipse for each constraint
  // if there are zero, we don't have to worry because we have already checked that x0 satisfies the constraints
  // if there are one or two, we will add them to a std::vector<vector<double>>
  std::vector<double> theta_vec;
  // the paper and package don't agree on the formula for this step
  // ryan tested both methods and found the package was correct: https://github.com/alpiges/LinConGauss/blob/eb86d5424cf2761f0c4f7e274aa0821150dc933f/src/LinConGauss/sampling/active_intersections.py#L23
  // define some variable to help with the intermediate steps
  double acos_arg;
  double pos_angle;
  double neg_angle;
  for(int i = 0; i < num_constraints; ++i){
    acos_arg = b_vec[i] / r[i]; // this will eventually be the argument for  std::acos
    // only continue if acos_input is in the function domain [-1, 1]
    // if it isn't, there is no intersection
    if (acos_arg <= 1 && acos_arg >= -1){
      // calculate the two angles where the i-th constraint intersects the ellipse
      pos_angle = std::acos(acos_arg) + 2 * std::atan(atv[i] / (r[i] + atx[i]));
      neg_angle = (-1) * std::acos(acos_arg) + 2 * std::atan(atv[i] / (r[i] + atx[i]));
      
      // put these angles in theta_vec
      theta_vec.push_back(pos_angle);
      theta_vec.push_back(neg_angle);
    }
  }
  // make sure all thetas are in the interval [0,2pi]
  for(std::vector<double>::iterator it = theta_vec.begin(); it != theta_vec.end(); ++it){
    if(*it < 0) (*it) += 2*M_PI;
    if(*it > 2*M_PI) (*it) -= 2 * M_PI;
  }
  // sort theta_vec
  std::sort(theta_vec.begin(), theta_vec.end());
  // pad theta_vec with 0 and 2 pi
  theta_vec.insert(theta_vec.begin(), 0.0);
  theta_vec.insert(theta_vec.end(), 2.0 * M_PI);
  
  // define some constants to help with the intermediate calculations
  double mid_pt;
  std::vector<double> x_new_vec;
  std::vector<std::vector<double>> active_intervals;
  double interval_length = 0;
  for(std::vector<double>::iterator it = theta_vec.begin(); it < theta_vec.end()-1; ++it){
    // calculate current midpoint for current interval
    mid_pt = 0.5 * ( *(it) + *(it+1));
    x_new_vec.clear();
    for(int j = 0; j < p; ++j) x_new_vec.push_back(init_point[j] * std::cos(mid_pt) + v_vec[j] * std::sin(mid_pt));
    if(check_constraints(phi_vec, b_vec, x_new_vec)){
      std::vector<double> tmp_int;
      tmp_int.push_back(*(it));
      tmp_int.push_back(*(it+1));
      active_intervals.push_back(tmp_int);
      interval_length += *(it+1) - *(it); // increment total length of active interval
    }
  }
  double rand_point = gen.uniform() * interval_length;
  // find the corresponding theta;
  double rand_theta;
  double len_to_go = rand_point;
  double tmp_int_len;
  int c_xi = 0; // iterator inside while loop
  
  while(len_to_go > 0 && c_xi < active_intervals.size()){
    tmp_int_len = active_intervals[c_xi][1]-active_intervals[c_xi][0]; // length of current interval
    if(len_to_go <= tmp_int_len) rand_theta = active_intervals[c_xi][0] + len_to_go;
    len_to_go -= tmp_int_len;
    ++c_xi;
  }
  // return point on ellipse corresponding to our randomly chosen theta
  for(int j = 0; j < p; ++j) sampled_point.push_back(init_point[j] * std::cos(rand_theta) + v_vec[j] * std::sin(rand_theta));
}



// helper function for computing gradient of log-barrier function
void calc_gradient(std::vector<std::map<int,double>> &phi_vec, std::vector<double> &phi_norms,
                   std::vector<double> &c_vec, int &p, std::vector<double> &x, arma::vec &g){
    
  g.zeros(p); // we are changing the value of g here!!!
  std::map<int, double> phi_i; // map to store current phi
  std::map<int,double>::iterator it; // create iterator for std::map
  double g_j; // variable to store progress on calculating the jth entry of gradient
  double dot_prod; // variable to store phi_i^T x
  double num; // numerator
  double denom; // denominator
    
    // loop over gradient indices
  for(int j = 0; j < p; ++j){
    g_j = 0.0;
    // loop over constraints
    for(int i = 0; i < phi_vec.size(); ++i){
      phi_i = phi_vec[i]; // store current constraint vector
      // if the jth entry is nonzero
      if(phi_i.count(j) == 1){
        //dot_prod = 0.0;
        //calc_dot_prod(phi_i, x, dot_prod);
        dot_prod = calc_phi_x(phi_i,x);
        denom = c_vec[i] - dot_prod;
                
        // check denom is not equal to zero
        if(denom == 0){
          Rcpp::Rcout << "Distance to constraint is zero." << std::endl;
          Rcpp::stop("We must start with a feasible x!!");
        }
                
        num = phi_norms[i] * (phi_i.find(j)->second);
        g_j += num / denom;
      }
    } // closes loop over constraints
    g(j) = g_j;
  } // closes loop over gradient indices
  //Rcpp::Rcout << "Gradient calculated. g = " << g << std::endl;
}

void calc_hessian(std::vector<std::map<int,double>> &phi_vec, std::vector<double> &phi_norms,
                  std::vector<double> &c_vec, int &p, std::vector<double> &x, arma::mat &H){
    
  H.zeros(p, p); // we are changing the value of H here!!
  std::map<int, double> phi_i; // map to store current phi
  std::map<int,double>::iterator it; // create iterator for std::map
  double h_jk; // var storing progress of Hessian entry
  double dot_prod;
  double num;
  double denom;
    
    // loop over Hessian rows
  for(int j = 0; j < p; ++j){
  // loop over Hessian cols
    for(int k = 0; k < p; ++k){
      h_jk = 0.0;
      // loop over constraints
      for(int i = 0; i < phi_vec.size(); ++i){
        phi_i = phi_vec[i];
        // check if the jth and kth entry are nonzero
        if(phi_i.count(j) + phi_i.count(k) == 2){
          dot_prod = calc_phi_x(phi_i, x);
          denom = std::pow((c_vec[i] - dot_prod), 2);
                    
          // check denom is not equal to zero
          if(denom == 0){
            Rcpp::Rcout << "Distance to constraint is zero." << std::endl;
            Rcpp::stop("We must start with a feasible x!!");
          }
          num = phi_norms[i] * (phi_i.find(j)->second) * (phi_i.find(k)->second);
          h_jk += num / denom;
        }
      } // closes loop over constraints
      H(j, k) = h_jk;
    } // closes loop over Hessian cols
  } // closes loop over Hessian rows
}


// this function finds the analytic center of a polytope defined by linear constraints in the form phi^T x < c
// the inputs for this function are:
//  (i) the linear constraints in the form phi^t x < c
//      where phi is a sparse matrix represented as an Rcpp::List
//      and c is an Rcpp::NumericVector of constants
// (ii) the dimension of the constrained space, p, represented as and integer

void print_constraints(std::vector<std::map<int, double>> &phi_vec,
                       std::vector<double> &b_vec)
{
  int num_constraints = phi_vec.size();
  for(int cix = 0; cix < num_constraints; ++cix){
    for(rc_it it = phi_vec[cix].begin(); it != phi_vec[cix].end(); ++it){
      if(it == phi_vec[cix].begin()) Rcpp::Rcout << it->second << "*X[" << it->first+1 << "]";
      else Rcpp::Rcout << " + " << it->second << "*X[" << it->first+1 << "]";
    }
    Rcpp::Rcout << " < " << b_vec[cix] << std::endl;
  }
}

double find_shift(std::vector<std::vector<double>> &init_samples,
                  std::vector<std::map<int,double>> &phi_vec,
                  std::vector<double> &b_vec,
                  int &p,
                  double prob)
{
  int n_samples = init_samples.size(); // how many initial samples do we have?
  int num_constraints = phi_vec.size();
  std::vector<double> shift_vec; //
  
  // for each sample, find the maximum value of phi'x-b for all constraints
  for(std::vector<std::vector<double>>::iterator x_it = init_samples.begin(); x_it != init_samples.end(); ++x_it){
    double tmp_dist = -1.0 * (calc_phi_x(phi_vec[0], *x_it) - b_vec[0]);
    double min = tmp_dist;

    for(int phi_ix = 0; phi_ix < num_constraints; ++phi_ix){
      tmp_dist = -1.0 * (calc_phi_x(phi_vec[phi_ix], *x_it) - b_vec[phi_ix]);
      if(tmp_dist < min) min = tmp_dist;
    }
    shift_vec.push_back(min);
  }
  // sort the distances to constraints in ascending order
  std::sort(shift_vec.begin(), shift_vec.end());

  int index = (int) floor(prob * n_samples);
  double shift = 0.5 * (shift_vec[index] + shift_vec[index-1]);
  return shift;
}

void get_init_point(std::vector<double> &x0,
                    std::vector<std::map<int, double>> &parent_phi_vec,
                    std::vector<double> &parent_c_vec,
                    std::vector<std::map<int,double>> &child_phi_vec,
                    std::vector<double> &child_c_vec,
                    int &p,
                    RNG &gen)
{
  // check if x0 is in the child
  if(!check_constraints(child_phi_vec, child_c_vec, x0)){
    // x0 is not in the child, so we will sample a bunch of points from the parent
    std::vector<double> tmp_sample; // temporarily holds our sampled point
    std::vector<std::vector<double>> init_samples;
    for(int rep = 0; rep < 500; ++rep){
      lin_ess(tmp_sample, parent_phi_vec, parent_c_vec, x0, p, gen);
      if(check_constraints(child_phi_vec, child_c_vec, tmp_sample)){
        x0.clear();
        // found a point in the child polytope, no need to run nested scheme
        for(std::vector<double>::iterator it = tmp_sample.begin(); it != tmp_sample.end(); ++it) x0.push_back(*it);
      } else init_samples.push_back(tmp_sample);
    } // close first sampling loop
    
    if(init_samples.size() > 0){
      // none of our samples reached the child.
      // we can shift the constraints and try again
      //Rcpp::Rcout << " Found " << init_samples.size() << " points in the parent polytope!" << std::endl;
      std::vector<double> tmp_x0;
      for(std::vector<double>::iterator x0_it = x0.begin(); x0_it != x0.end(); ++x0_it) tmp_x0.push_back(*x0_it);
      
      double tmp_prob = 1.0 - 1.0/( (double) init_samples.size() );
      //double tmp_prob = 0.5;
      double shift = find_shift(init_samples, child_phi_vec, child_c_vec, p, tmp_prob);
      
      // actually shift the hyperplanes
      std::vector<double> tmp_c;
      for(std::vector<double>::iterator c_it = child_c_vec.begin(); c_it != child_c_vec.end(); ++c_it) tmp_c.push_back(*c_it - shift);
      tmp_x0.clear();
      for(std::vector<std::vector<double>>::iterator it = init_samples.begin(); it != init_samples.end(); ++it){
        if(check_constraints(child_phi_vec, tmp_c, *it)){
          tmp_x0.clear();
          tmp_x0 = *(it);
          break;
        }
      }
      if(tmp_x0.size() != 0){
        bool in_child = check_constraints(child_phi_vec, child_c_vec, tmp_x0);
        int counter = 0;
        while(!in_child && counter < 20){
          // draw samples from lin-ess inside nested domain
          //Rcpp::Rcout << "  nested sampler counter = " << counter;
          init_samples.clear();
          for(int rep = 0; rep < 50; ++rep){
            lin_ess(tmp_sample, child_phi_vec, tmp_c, tmp_x0, p ,gen);
            if(!check_constraints(child_phi_vec, child_c_vec, tmp_sample)) init_samples.push_back(tmp_sample);
            else{
              // we found a point after sampling within the nested constraint
              x0.clear();
              for(std::vector<double>::iterator x0_it = tmp_sample.begin(); x0_it != tmp_sample.end(); ++x0_it) x0.push_back(*x0_it);
              in_child = true;
              init_samples.clear();
              break;
            }
          } // closes sampling loop
          if(init_samples.size() > 0){
            //Rcpp::Rcout << " found " << init_samples.size() << " samples in the shifted polytope";
            //tmp_prob = 1.0 - 1.0/( (double) init_samples.size());
            tmp_prob = 0.5;
            shift = find_shift(init_samples, child_phi_vec, child_c_vec, p, tmp_prob);
            tmp_c.clear();
            for(std::vector<double>::iterator c_it = child_c_vec.begin(); c_it != child_c_vec.end(); ++c_it) tmp_c.push_back(*c_it - shift);
            tmp_x0.clear();
            for(std::vector<std::vector<double>>::iterator it = init_samples.begin(); it != init_samples.end(); ++it){
              if(check_constraints(child_phi_vec, tmp_c, *it)){
                tmp_x0.clear();
                tmp_x0 = *(it);
                // Rcpp::Rcout << " new point is ";
                // for(std::vector<double>::iterator x0_it = tmp_x0.begin(); x0_it != tmp_x0.end(); ++x0_it) Rcpp::Rcout << " " << *x0_it;
                // Rcpp::Rcout << std::endl;
                break;
              }
            }
            if(tmp_x0.size() == 0){
              Rcpp::Rcout << " something is up with the shift!" << std::endl;
              
              Rcpp::Rcout << " Child constraints are " << std::endl;
              print_constraints(child_phi_vec, child_c_vec);
              
              Rcpp::Rcout << " Shifts is: " << shift << std::endl;
              
              Rcpp::Rcout << " Shifted constraints are: " << std::endl;
              print_constraints(child_phi_vec, tmp_c);
              
              Rcpp::stop("COULD NOT FIND POINT IN SHIFTED POLYTOPE");
              
            }
          }
          ++counter;
        } // closes while loop
        if(!in_child && counter == 20) Rcpp::Rcout << "we goofed somewhere" << std::endl;
      } else Rcpp::stop("could not find point in initial shifted polytope");
    }
  } // closes if checking whether initial point x0 was in child polytope
}

void get_child_analytic_center(std::map<int, std::vector<double>> &analytic_centers,
                               suff_stat &ss,
                               int &nx_nid,
                               int &child_nid,
                               tree &t,
                               data_info &di,
                               RNG &gen)
{
  tree::tree_p nx = t.get_ptr(nx_nid);
  std::vector<std::map<int,double>> parent_phi_vec;
  std::vector<double> parent_c_vec;
  std::vector<double> parent_aa_lower(di.p_cont, -1.0);
  std::vector<double> parent_aa_upper(di.p_cont, 1.0);
  nx->get_cont_constraints(parent_phi_vec, parent_c_vec, parent_aa_lower, parent_aa_upper, di.p_cont);

  if(analytic_centers.find(nx_nid) == analytic_centers.end()){
    Rcpp::Rcout << "[get_child_analytic_center]: Trying to find analytic center for node " << nx_nid;
    Rcpp::Rcout << "Only have analytic centers for nodes:";
    for(std::map<int,std::vector<double>>::iterator it = analytic_centers.begin(); it != analytic_centers.end(); ++it) Rcpp::Rcout << " " << it->first;
    Rcpp::Rcout << std::endl;
    Rcpp::stop("Could not find analytic center for requested parent!");
  } else{
    // we have the parent analytic center.
    // now get the analytic center of the parent
    std::map<int,std::vector<double>>::iterator parent_center_it = analytic_centers.find(nx_nid);
    std::vector<double> child_center = parent_center_it->second;
    
    // get pointer to the new child node in tree
    tree::tree_p child_nx = t.get_ptr(child_nid);
    
    // get all the constraints for the new child node
    std::vector<std::map<int, double>> child_phi_vec;
    std::vector<double> child_c_vec;
    std::vector<double> child_aa_lower(di.p_cont, -1.0);
    std::vector<double> child_aa_upper(di.p_cont, 1.0);
    child_nx->get_cont_constraints(child_phi_vec, child_c_vec, child_aa_lower, child_aa_upper, di.p_cont);

    int n = child_c_vec.size();
    int d = child_center.size();

    arma::mat A(n, d, arma::fill::zeros);
    for (int i = 0; i < n; ++i){
      std::map<int, double> tmp_map = child_phi_vec[i];
      for (std::map<int, double>::iterator it = tmp_map.begin(); it != tmp_map.end(); ++it){
        A(i, it->first) = it->second;
      }
    }
    arma::vec b(n, arma::fill::zeros);
    for (int i = 0; i < n; ++i) b[i] = child_c_vec[i];
    arma::vec x(d, arma::fill::zeros);
    for (int i = 0; i < d; ++i) x[i] = child_center[i];
    arma::vec x0 = x;

    inf_analytic_center(x, A, b);
    for (int i = 0; i < d; ++i) child_center[i] = x[i];
    analytic_centers.insert(std::pair<int, std::vector<double>>(child_nid, child_center));

    if(!check_constraints(child_phi_vec, child_c_vec, child_center)){
      Rcpp::Rcout << "somthing went wrong computing the analytic center." << std::endl;
      Rcpp::Rcout << "A: " << A << std::endl;
      Rcpp::Rcout << "b: ";
      for (int i = 0; i < n; ++i) Rcpp::Rcout << b[i] << ", ";
      Rcpp::Rcout << std::endl;
      Rcpp::Rcout << "x0: ";
      for (int i = 0; i < d; ++i) Rcpp::Rcout << x0[i] << ", ";
      Rcpp::Rcout << std::endl;
      Rcpp::Rcout << "x: ";
      for (int i = 0; i < d; ++i) Rcpp::Rcout << x[i] << ", ";
      Rcpp::Rcout << std::endl;
      Rcpp::stop("point does not satisfy constraints.");
    }

    // // look up the suff_stat_map to see what observations reach the child node
    // suff_stat_it ss_it = ss.find(child_nid);
    // if (ss_it != ss.end()){
    //   // check if the parent analytic center satisfies the child's contraints
    //   // if it does, use that as the initial point
    //   if (check_constraints(child_phi_vec, child_c_vec, child_center)); // do nothing
    //   else {
    //     // check to see if there are observations in the leaf that satisfy the contraints and are not on the boundary
    //     if(ss_it->second.size() > 0){
    //       // we actually have a point in the child polytope
    //       // loop through all the points in the polytope until we find one that is not on the boundary
    //       for (int i = 0; i < ss_it->second.size(); ++i){
    //         child_center.clear();
    //         int j = ss_it->second[i];
    //         for (int k = 0; k < di.p_cont; ++k) child_center.push_back(di.x_cont[k + j * di.p_cont]);
    //         if (check_constraints(child_phi_vec, child_c_vec, child_center)) break; // we found a point!
    //         else if (i == ss_it->second.size() - 1){
    //           Rcpp::Rcout << "[get_child_analytic_center]: no points in node " << child_nid << " satisty the constraints constraints." << std::endl;
    //           // Rcpp::stop("All points in this node fall on the boundary.");
    //         }
    //       } // closes for loop over points in leaf
    //     } else{
    //       // no observation reaches leaf. we will use LinEss+Nested sampling to draw a point inside the child polytope
    //       //Rcpp::Rcout << "[get_child_analytic_center]: No sample at node " << child_nid << "! Trying liness + nested sampling scheme" << std::endl;
    //       get_init_point(child_center, parent_phi_vec, parent_c_vec, child_phi_vec, child_c_vec, di.p_cont, gen);
    //     } // closes if/else checking that we have observations in the leaf
    //   } // closes if/else finding an initialization point
    //   // at this point, child center should hold coordinates for a single point in the child polytope
    //   // just as a sanity check, however:
    //   if(!check_constraints(child_phi_vec, child_c_vec, child_center)){
    //     Rcpp::Rcout << "[get_child_analytic_center]: Trying to find analytic center of node " << child_nid << std::endl;
    //     t.print();
    //     Rcpp::Rcout << "  The constraints of child polytope are:" << std::endl;
    //     print_constraints(child_phi_vec, child_c_vec);
    //     Rcpp::Rcout << " The putative point in the polytope is :";
    //     for(std::vector<double>::iterator x0_it = child_center.begin(); x0_it != child_center.end(); ++x0_it) Rcpp::Rcout << " " << *x0_it;
    //     Rcpp::Rcout << std::endl;
    //     Rcpp::stop("Putative point is not in polytope!");
    //   } else{
    //     calc_analytic_center(child_center, child_phi_vec, child_c_vec, di.p_cont);
    //     analytic_centers.insert(std::pair<int, std::vector<double>>(child_nid, child_center));
    //   } // closes if/else checking that initializer for analytic center computation actually lies in the child polytope
    // } else {
    //   Rcpp::Rcout << "[get_child_analytic_center]: Trying to find analytic center for new leaf " << child_nid << std::endl;
    //   Rcpp::Rcout << "  We have entries for following leafs in suff_stat:";
    //   for(suff_stat_it tmp_it = ss.begin(); tmp_it != ss.end(); ++tmp_it) Rcpp::Rcout << " " << tmp_it->first;
    //   Rcpp::Rcout << std::endl;
    //   Rcpp::stop("Could not find entry for new leaf in sufficient statistic map!");
    // } // closes if/else checking that we have ss_map information for the child node
  } // closes if/else checking that we have analytic center for parent
}

arma::vec calc_res_vec(arma::mat &A, arma::vec &b, arma::vec &x, arma::vec &y, arma::vec &v, arma::vec &g)
{
    arma::vec rda = A.t() * v;
    arma::mat rdb = g + v;
    arma::mat rp = y + A * x - b;
    arma::vec r = arma::join_vert(rda, rdb, rp);
    return -r;
}

void inf_analytic_center(arma::vec &x, arma::mat &A, arma::vec &b)
{   
    arma::vec x0 = x;
    int n = b.size();
    int d = x.size();

    if (A.n_rows != n) Rcpp::Rcout << "A has " << A.n_rows << " rows but b has " << n << " elements!" << std::endl;
    if (A.n_cols != d) Rcpp::Rcout << "A has " << A.n_cols << " cols but x has " << d << " elements!" << std::endl;

    // Code for reading in arguments from R
    // int n = b_vec.size(); // number of constraints
    // int d = x0.size(); // dimension of space

    // if (A_mat.nrow() != n) Rcpp::Rcout << "A_mat has " << A_mat.nrow() << " but b has " << n << " elements!" << std::endl;
    // if (A_mat.ncol() != d) Rcpp::Rcout << "A_mat has " << A_mat.ncol() << " but x has " << d << " elements!" << std::endl;

    // arma::mat A(n, d, arma::fill::zeros);
    // for (int i = 0; i < n; ++i){
    //     for (int j = 0; j < d; ++j) A(i, j) = A_mat(i, j);
    // }

    // arma::vec b(n, arma::fill::zeros);
    // for (int i = 0; i < n; ++i) b[i] = b_vec[i];

    // arma::vec x(d, arma::fill::zeros);
    // for (int i = 0; i < d; ++i) x[i] = x0[i];

    // hyperparameters
    // using default values recommended by: https://web.stanford.edu/class/ee364b/lectures/accpm_notes.pdf
    double alpha = 0.01;
    double beta = 0.5;
    int max_it = 100;
    double epsilon = 1 / pow(10, 12);
    double t0 = 1;

    // initialization
    arma::vec v(n, arma::fill::zeros);
    arma::vec y(n, arma::fill::zeros);
    for (int i = 0; i < n; ++i){
        arma::vec a(d, arma::fill::zeros);
        for (int j = 0; j < d; ++j) a[j] = A(i, j);
        if (arma::as_scalar(b[i] - a.t() * x) > 0) y[i] = arma::as_scalar(b[i] - a.t() * x);
        else y[i] = 1;
    }
    
    arma::vec g(n, arma::fill::zeros);
    arma::mat H(n, n, arma::fill::zeros);

    arma::vec rp(n);
    arma::vec r(2 * n + d, arma::fill::zeros);
    
    arma::vec dx(d, arma::fill::zeros);
    arma::vec dy(n, arma::fill::zeros);
    arma::vec dv(n, arma::fill::zeros);

    // Rcpp::NumericVector x_out(d);

    // optimization loop
    int iter = 0;
    while (iter < max_it){

        g = -1/y;
        H = arma::diagmat(1 / arma::pow(y, 2.0));

        // residuals
        rp = y + A * x - b;
        r = calc_res_vec(A, b, x, y, v, g);

        // Rcpp::Rcout << iter << ": " << arma::norm(r, 2) << " " << x << std::endl;
        // Rcpp::Rcout << iter << ": " << r << std::endl;

        // stopping conditions
        if (arma::norm(y + A * x - b, 2) < epsilon && arma::norm(r, 2) < epsilon) break;

        dx = arma::inv(A.t() * H * A) * (A.t() * g - A.t() * H * rp);
        dy = -A * dx - rp;
        dv = -H * dy - g - v;

        // step size via backtracking line search
        arma::vec x_new(d, arma::fill::zeros);
        arma::vec y_new(n, arma::fill::zeros);
        arma::vec v_new(n, arma::fill::zeros);
        double t = t0;
        int it = 0;
        int done = 0;

        // feasibility check (all entries of y are strictly greater than zero)
        while (it < max_it){
            y_new = y + t * dy;
            for (int i = 0; i < n; ++i){
                if (y_new[i] <= 0){
                    t *= beta;
                    ++it;
                    break;
                } else done = 1;
            }
            if (done == 1) break;
        }
        if (it == max_it){
            // if we hit this, step size is below computer precision, so it is likely that the contraint set is empty
            Rcpp::Rcout << "[inf_analytic_center]: maximum iterations reached in feasibility contraint check." << std::endl;
            Rcpp::stop("Unable to find a feasible step size in 50 iterations.");
        }
        else it = 0;

        // check we haven't stepped to far (if residuals increase significantly, we have gone too far)
        while(it < max_it){
            x_new = x + t * dx;
            y_new = y + t * dy;
            v_new = v + t * dv;
            if (arma::norm(calc_res_vec(A, b, x_new, y_new, v_new, g), 2) > (1 - alpha * t) * arma::norm(r, 2)){
                t *= beta;
                ++it;
            } else break;
        }
        if (it == max_it){
            // if we hit this, step size is below computer precision, so it is likely that the contraint set is empty
            Rcpp::Rcout << "[inf_analytic_center]: maximum iterations reached in residual tracking check." << std::endl;
            Rcpp::stop("Unable to find an acceptable step size in 50 iterations.");
        }
        else it = 0;

        // update variables
        x += t * dx;
        y += t * dy;
        v += t * dv;

        ++iter;
    }

    // if (iter == max_it){
    //     Rcpp::Rcout << "WARNING: stopping conditions for infeasible start analytic center not met after " << max_it << " iterations." << std::endl;
    //     Rcpp::Rcout << "[inf_analytic_center]: returning point may not be feasbile or the analytic center." << std::endl;
    //     Rcpp::Rcout << "A: " << A << std::endl;
    //     Rcpp::Rcout << "b: ";
    //     for (int i = 0; i < n; ++i) Rcpp::Rcout << b[i] << ", ";
    //     Rcpp::Rcout << std::endl;
    //     Rcpp::Rcout << "x0: ";
    //     for (int i = 0; i < d; ++i) Rcpp::Rcout << x0[i] << ", ";
    //     Rcpp::Rcout << std::endl;
    //     Rcpp::Rcout << "x: ";
    //     for (int i = 0; i < d; ++i) Rcpp::Rcout << x[i] << ", ";
    //     Rcpp::Rcout << std::endl;
    //     Rcpp::stop("[inf_analytic_center]: hit maximum iterations.");
    // }
}

/*
void get_child_analytic_centers(std::map<int, std::vector<double>> &analytic_centers, int &nid, tree &t,int &p_cont)
{
  
  int nxl_nid = 2*nid;
  int nxr_nid = 2*nid+1;
  
  int last_nid = 0; // id of the most recent ancestor with a continuous rule
  std::map<int,double> last_phi; // normal vec of hyperplane from most recent ancestor w/ continuous rule
  double last_c = 0.0; // intercept for most recent continuous rule
  bool is_left; // is current node on the left or right branch from most recent ancestor w/ continuous rule
  
  std::vector<double> new_center(p_cont, 0.0); // eventually the analytic center of polytope

  tree::tree_p nxl = t.get_ptr(nxl_nid);
  tree::tree_p nxr = t.get_ptr(nxr_nid);
  
  std::vector<std::map<int,double>> phi_vec; // store the normal directions of hyperplanes
  std::vector<double> b_vec; // store the upper bound
  std::vector<double> aa_lower(p_cont, -1.0); // lower bounds of each input dimension
  std::vector<double> aa_upper(p_cont, 1.0); // upper bounds of each input dimension
  
  
  if(nxl){
    //Rcpp::Rcout << "  trying to compute analytic center for " << nxl_nid << std::endl;
    phi_vec.clear();
    b_vec.clear();
    for(int j = 0; j < p_cont; ++j){
      aa_lower[j] = -1.0;
      aa_upper[j] = 1.0;
    }
    last_nid = 0;
    last_c = 0.0;
    is_left = true;
    last_phi.clear();
    
    nxl->get_recent_phi(last_phi, last_c, last_nid, is_left);
    nxl->get_cont_constraints(phi_vec, b_vec, aa_lower, aa_upper, p_cont);
    
    if(last_nid == 0){
      // no ancestor of nxl involved in a continuous rule.
      // analytic center must be the origin
      Rcpp::Rcout << "last nid = 0!" << std::endl;
      for(int j = 0; j < p_cont; ++j) new_center[j] = 0.0;
    } else{
      new_center.clear();
      // let's look up the analytic center of the polytope for nid's most recent ancestor w/ a continuous rule
      std::map<int, std::vector<double>>::iterator old_center_it = analytic_centers.find(last_nid);
      // and let's take that as a putative center for the polytope at nid
      for(std::vector<double>::iterator it = old_center_it->second.begin(); it != old_center_it->second.end(); ++it) new_center.push_back(*it);
      // based on that point, we will compute an initial point for
     
      get_init_point(new_center, last_phi, last_c, phi_vec, b_vec, is_left);
      // at this point, new_center should be a point inside the polytope for nid
      
      calc_analytic_center(new_center, phi_vec, b_vec, p_cont);
    }
    // at this point, new_center should be the origin or the analytic center of nid's polytope
    // as a final sanity check, let's ensure it's in the polytope
    if(!check_constraints(phi_vec, b_vec, new_center)){
      Rcpp::stop("[get_analytic_center]: Putative analytic center is not in the polytope!!");
    } else{
      analytic_centers.insert(std::pair<int, std::vector<double>>(nxl_nid, new_center));
    }
  } else{
    Rcpp::Rcout << "[get_analytic_center]: Could not find nid = " << nxl_nid << " in tree!" << std::endl;
    t.print();
    Rcpp::stop("Tried finding center of region of a node that does not exist!");
  }
  
  
  if(nxr){
    //Rcpp::Rcout << "  trying to compute analytic center for " << nxr_nid << std::endl;

    phi_vec.clear();
    b_vec.clear();
    for(int j = 0; j < p_cont; ++j){
      aa_lower[j] = -1.0;
      aa_upper[j] = 1.0;
    }
    last_nid = 0;
    last_c = 0.0;
    is_left = true;
    last_phi.clear();
    
    nxr->get_recent_phi(last_phi, last_c, last_nid, is_left);
    nxr->get_cont_constraints(phi_vec, b_vec, aa_lower, aa_upper, p_cont);
    
    if(last_nid == 0){
      // no ancestor of nxr involved in a continuous rule.
      // analytic center must be the origin
      for(int j = 0; j < p_cont; ++j) new_center[j] = 0.0;
    } else{
      new_center.clear();
      // let's look up the analytic center of the polytope for nid's most recent ancestor w/ a continuous rule
      std::map<int, std::vector<double>>::iterator old_center_it = analytic_centers.find(last_nid);
      // and let's take that as a putative center for the polytope at nid
      for(std::vector<double>::iterator it = old_center_it->second.begin(); it != old_center_it->second.end(); ++it) new_center.push_back(*it);
      // based on that point, we will compute an initial point for
      get_init_point(new_center, last_phi, last_c, phi_vec, b_vec, is_left);
      // at this point, new_center should be a point inside the polytope for nid
      calc_analytic_center(new_center, phi_vec, b_vec, p_cont);
    }
    // at this point, new_center should be the origin or the analytic center of nid's polytope
    // as a final sanity check, let's ensure it's in the polytope
    if(!check_constraints(phi_vec, b_vec, new_center)){
      Rcpp::stop("[get_analytic_center]: Putative analytic center is not in the polytope!!");
    } else{
      analytic_centers.insert(std::pair<int, std::vector<double>>(nxr_nid, new_center));
    }
  } else{
    Rcpp::Rcout << "[get_analytic_center]: Could not find nid = " << nxr_nid << " in tree!" << std::endl;
    t.print();
    Rcpp::stop("Tried finding center of region of a node that does not exist!");
  }
}
*/

void calc_analytic_center(std::vector<double> &x0_vec, std::vector<std::map<int,double>> &phi_vec, std::vector<double> &c_vec, int &p, int max_iter)
{
  // calculate norms of each phi_i
  // we need to use this in each iteration so it will be better
  // to calculate them all once at the start
  std::vector<double> phi_norms;
  calc_norms(phi_vec, phi_norms);
  // optimizer loop
  arma::vec g(p); // gradient
  arma::mat H(p, p); // Hessian
  arma::vec step(p); // update
  // hyperparameters
  double t = 1.0; // step size
  double epsilon = std::pow(2, -16);
  double dx = 1.0;
  // track progress
  std::vector<std::vector<double>> x_steps;
  std::vector<arma::vec> g_steps;
  std::vector<arma::mat> h_steps;
  // loop until we hit max iterations or we stop moving
  int iter = 0;
  while (iter < max_iter && dx > epsilon){
    //Rcpp::Rcout << "Starting iteration " << iter << std::endl;
    // calculate gradient
    calc_gradient(phi_vec, phi_norms, c_vec, p, x0_vec, g);
    g_steps.push_back(g); // track progress of g
    // calculate hessian
    calc_hessian(phi_vec, phi_norms, c_vec, p, x0_vec, H);
    h_steps.push_back(H); // track progress of H
    // calculate step
    step = arma::solve(H, g, arma::solve_opts::allow_ugly);
    // update point and distance moved
    dx = 0.0;
    for (int i = 0; i < p; ++i){
      x0_vec[i] -= t * step(i);
      dx += std::pow(step(i), 2);
    }
    dx = std::sqrt(dx);
    x_steps.push_back(x0_vec); // track progress of x
    ++iter;
  }
  
  for(std::vector<double>::iterator it = x0_vec.begin(); it != x0_vec.end(); ++it){
    if(abs(*it) < 1e-9) *it = 0.0;
  }
  
}
