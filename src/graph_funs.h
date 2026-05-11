

#ifndef GUARD_graph_funs_h
#define GUARD_graph_funs_h

#include "structs.h"
#include "rng.h"
#include "tree.h"


// we will pass a list of Rcpp::NumericMatrices, which are computed using igraph::get_data_frame
// this function reads those matrices and builds a vector of edges
inline void parse_edge_mat(std::vector<edge> &edges, Rcpp::NumericMatrix &edge_mat, int &n_vertex)
{
  int n_edges = edge_mat.rows();
  edges.clear();
  if(edge_mat.cols() == 3){
    for(int i = 0; i < n_edges; i++){
      edges.push_back(edge( (int) edge_mat(i,0), (int) edge_mat(i,1), edge_mat(i,2)));
    }
  } else if(edge_mat.cols() == 2){
    for(int i = 0; i < n_edges; i++){
      edges.push_back(edge( (int) edge_mat(i,0), (int) edge_mat(i,1), 1.0));
    }
  } else{
    Rcpp::stop("[parse_edge_mat]: The matrix edge_mat must have 2 columns (unweighted graph) or 3 columns (weighted graph)");
  }
}

// takes in the List of edge_mat's
inline void parse_graphs(std::vector<std::vector<edge>> &edges, int &p_cat, std::vector<int> &K, Rcpp::List &tmp_edge_mats, Rcpp::LogicalVector &graph_split)
{
  edges.clear();
  edges.resize(p_cat, std::vector<edge>());
  if(tmp_edge_mats.size() == p_cat){
    for(int j = 0; j < p_cat; j++){
      if(graph_split(j) == 1){
        Rcpp::NumericMatrix edge_mat = Rcpp::as<Rcpp::NumericMatrix>(tmp_edge_mats[j]);
        parse_edge_mat(edges[j], edge_mat, K[j]);
      } else{
        // do nothing
      }
    }
  } else{
    Rcpp::Rcout << "[parse_graphs]: detected " << p_cat << " categorical variables";
    Rcpp::Rcout << " edge_mat_list has length " << tmp_edge_mats.size() << std::endl;
    Rcpp::stop("edge_mat_list must have length equal to p_cat!");
  }
}

void build_symmetric_edge_map(edge_map &emap, std::vector<edge> &edges, std::set<int> &vertices);
std::vector<edge> get_induced_edges(std::vector<edge> &edges, std::set<int> &vertex_subset);
void dfs(int v, std::map<int, bool> &visited, std::vector<int> &comp, edge_map &emap);
void find_components(std::vector<std::vector<int> > &components, std::vector<edge> &edges, std::set<int> &vertices);
void get_unique_edges(std::vector<edge> &edges);

arma::mat get_adjacency_matrix(edge_map emap);

arma::mat floydwarshall(std::vector<edge> &edges, std::set<int> &vertices);

void boruvka(std::vector<edge> &mst_edges, std::vector<edge> &edges, std::set<int> &vertices);
void wilson(std::vector<edge> &mst_edges, std::vector<edge> &edges, std::set<int> &vertices, RNG &gen);
void hotspot(std::set<int> &l_vals, std::set<int> &r_vals, std::vector<edge> &edges, std::set<int> &vertices, RNG &gen, bool debug = false);

void delete_unif_edge(std::set<int> &l_vals, std::set<int> &r_vals, std::vector<edge> &edges, std::set<int> &vertices, RNG &gen);
void signcheck_split(std::set<int> &l_vals, std::set<int> &r_vals, std::vector<edge> &edges, std::set<int> &vertices);

void graph_partition(std::set<int> &l_vals, std::set<int> &r_vals, std::vector<edge> &orig_edges, std::set<int> &avail_levels, int &cut_type, RNG &gen);


#endif /* funs_h */
