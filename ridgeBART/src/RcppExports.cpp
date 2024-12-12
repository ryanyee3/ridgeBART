// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// eval_bases
Rcpp::NumericMatrix eval_bases(Rcpp::List bases_list, Rcpp::NumericMatrix Z_mat, int activation_option, int intercept_option, bool probit, bool verbose, int print_every);
RcppExport SEXP _ridgeBART_eval_bases(SEXP bases_listSEXP, SEXP Z_matSEXP, SEXP activation_optionSEXP, SEXP intercept_optionSEXP, SEXP probitSEXP, SEXP verboseSEXP, SEXP print_everySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type bases_list(bases_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type Z_mat(Z_matSEXP);
    Rcpp::traits::input_parameter< int >::type activation_option(activation_optionSEXP);
    Rcpp::traits::input_parameter< int >::type intercept_option(intercept_optionSEXP);
    Rcpp::traits::input_parameter< bool >::type probit(probitSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type print_every(print_everySEXP);
    rcpp_result_gen = Rcpp::wrap(eval_bases(bases_list, Z_mat, activation_option, intercept_option, probit, verbose, print_every));
    return rcpp_result_gen;
END_RCPP
}
// get_bases
Rcpp::List get_bases(Rcpp::List fit_list, Rcpp::NumericMatrix tX_cont, Rcpp::IntegerMatrix tX_cat, Rcpp::NumericMatrix tZ_mat, bool probit, bool verbose, int print_every);
RcppExport SEXP _ridgeBART_get_bases(SEXP fit_listSEXP, SEXP tX_contSEXP, SEXP tX_catSEXP, SEXP tZ_matSEXP, SEXP probitSEXP, SEXP verboseSEXP, SEXP print_everySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type fit_list(fit_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type tX_cont(tX_contSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type tX_cat(tX_catSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type tZ_mat(tZ_matSEXP);
    Rcpp::traits::input_parameter< bool >::type probit(probitSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type print_every(print_everySEXP);
    rcpp_result_gen = Rcpp::wrap(get_bases(fit_list, tX_cont, tX_cat, tZ_mat, probit, verbose, print_every));
    return rcpp_result_gen;
END_RCPP
}
// predict_ridgeBART
Rcpp::NumericMatrix predict_ridgeBART(Rcpp::List fit_list, Rcpp::NumericMatrix tX_cont, Rcpp::IntegerMatrix tX_cat, Rcpp::NumericMatrix tZ_mat, bool probit, bool verbose, int print_every);
RcppExport SEXP _ridgeBART_predict_ridgeBART(SEXP fit_listSEXP, SEXP tX_contSEXP, SEXP tX_catSEXP, SEXP tZ_matSEXP, SEXP probitSEXP, SEXP verboseSEXP, SEXP print_everySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type fit_list(fit_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type tX_cont(tX_contSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type tX_cat(tX_catSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type tZ_mat(tZ_matSEXP);
    Rcpp::traits::input_parameter< bool >::type probit(probitSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type print_every(print_everySEXP);
    rcpp_result_gen = Rcpp::wrap(predict_ridgeBART(fit_list, tX_cont, tX_cat, tZ_mat, probit, verbose, print_every));
    return rcpp_result_gen;
END_RCPP
}
// probit_ridgeBART_fit
Rcpp::List probit_ridgeBART_fit(Rcpp::NumericVector Y_train, Rcpp::NumericMatrix tX_cont_train, Rcpp::IntegerMatrix tX_cat_train, Rcpp::NumericMatrix tZ_mat_train, Rcpp::NumericMatrix tX_cont_test, Rcpp::IntegerMatrix tX_cat_test, Rcpp::NumericMatrix tZ_mat_test, Rcpp::LogicalVector unif_cuts, Rcpp::Nullable<Rcpp::List> cutpoints_list, Rcpp::Nullable<Rcpp::List> cat_levels_list, Rcpp::Nullable<Rcpp::List> edge_mat_list, Rcpp::LogicalVector graph_split, int graph_cut_type, bool oblique_option, double prob_aa, int x0_option, bool sparse, double a_u, double b_u, double p_change, Rcpp::NumericVector beta0, double tau, double branch_alpha, double branch_beta, int activation_option, int intercept_option, int sparse_smooth_option, int rotation_option, int dp_option, int n_bases, int rho_option, Rcpp::NumericVector rho_prior, double rho_alpha, double rho_nu, double rho_lambda, int M, int nd, int burn, int thin, bool save_samples, bool save_trees, bool verbose, int print_every);
RcppExport SEXP _ridgeBART_probit_ridgeBART_fit(SEXP Y_trainSEXP, SEXP tX_cont_trainSEXP, SEXP tX_cat_trainSEXP, SEXP tZ_mat_trainSEXP, SEXP tX_cont_testSEXP, SEXP tX_cat_testSEXP, SEXP tZ_mat_testSEXP, SEXP unif_cutsSEXP, SEXP cutpoints_listSEXP, SEXP cat_levels_listSEXP, SEXP edge_mat_listSEXP, SEXP graph_splitSEXP, SEXP graph_cut_typeSEXP, SEXP oblique_optionSEXP, SEXP prob_aaSEXP, SEXP x0_optionSEXP, SEXP sparseSEXP, SEXP a_uSEXP, SEXP b_uSEXP, SEXP p_changeSEXP, SEXP beta0SEXP, SEXP tauSEXP, SEXP branch_alphaSEXP, SEXP branch_betaSEXP, SEXP activation_optionSEXP, SEXP intercept_optionSEXP, SEXP sparse_smooth_optionSEXP, SEXP rotation_optionSEXP, SEXP dp_optionSEXP, SEXP n_basesSEXP, SEXP rho_optionSEXP, SEXP rho_priorSEXP, SEXP rho_alphaSEXP, SEXP rho_nuSEXP, SEXP rho_lambdaSEXP, SEXP MSEXP, SEXP ndSEXP, SEXP burnSEXP, SEXP thinSEXP, SEXP save_samplesSEXP, SEXP save_treesSEXP, SEXP verboseSEXP, SEXP print_everySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type Y_train(Y_trainSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type tX_cont_train(tX_cont_trainSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type tX_cat_train(tX_cat_trainSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type tZ_mat_train(tZ_mat_trainSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type tX_cont_test(tX_cont_testSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type tX_cat_test(tX_cat_testSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type tZ_mat_test(tZ_mat_testSEXP);
    Rcpp::traits::input_parameter< Rcpp::LogicalVector >::type unif_cuts(unif_cutsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type cutpoints_list(cutpoints_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type cat_levels_list(cat_levels_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type edge_mat_list(edge_mat_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::LogicalVector >::type graph_split(graph_splitSEXP);
    Rcpp::traits::input_parameter< int >::type graph_cut_type(graph_cut_typeSEXP);
    Rcpp::traits::input_parameter< bool >::type oblique_option(oblique_optionSEXP);
    Rcpp::traits::input_parameter< double >::type prob_aa(prob_aaSEXP);
    Rcpp::traits::input_parameter< int >::type x0_option(x0_optionSEXP);
    Rcpp::traits::input_parameter< bool >::type sparse(sparseSEXP);
    Rcpp::traits::input_parameter< double >::type a_u(a_uSEXP);
    Rcpp::traits::input_parameter< double >::type b_u(b_uSEXP);
    Rcpp::traits::input_parameter< double >::type p_change(p_changeSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type beta0(beta0SEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< double >::type branch_alpha(branch_alphaSEXP);
    Rcpp::traits::input_parameter< double >::type branch_beta(branch_betaSEXP);
    Rcpp::traits::input_parameter< int >::type activation_option(activation_optionSEXP);
    Rcpp::traits::input_parameter< int >::type intercept_option(intercept_optionSEXP);
    Rcpp::traits::input_parameter< int >::type sparse_smooth_option(sparse_smooth_optionSEXP);
    Rcpp::traits::input_parameter< int >::type rotation_option(rotation_optionSEXP);
    Rcpp::traits::input_parameter< int >::type dp_option(dp_optionSEXP);
    Rcpp::traits::input_parameter< int >::type n_bases(n_basesSEXP);
    Rcpp::traits::input_parameter< int >::type rho_option(rho_optionSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type rho_prior(rho_priorSEXP);
    Rcpp::traits::input_parameter< double >::type rho_alpha(rho_alphaSEXP);
    Rcpp::traits::input_parameter< double >::type rho_nu(rho_nuSEXP);
    Rcpp::traits::input_parameter< double >::type rho_lambda(rho_lambdaSEXP);
    Rcpp::traits::input_parameter< int >::type M(MSEXP);
    Rcpp::traits::input_parameter< int >::type nd(ndSEXP);
    Rcpp::traits::input_parameter< int >::type burn(burnSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< bool >::type save_samples(save_samplesSEXP);
    Rcpp::traits::input_parameter< bool >::type save_trees(save_treesSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type print_every(print_everySEXP);
    rcpp_result_gen = Rcpp::wrap(probit_ridgeBART_fit(Y_train, tX_cont_train, tX_cat_train, tZ_mat_train, tX_cont_test, tX_cat_test, tZ_mat_test, unif_cuts, cutpoints_list, cat_levels_list, edge_mat_list, graph_split, graph_cut_type, oblique_option, prob_aa, x0_option, sparse, a_u, b_u, p_change, beta0, tau, branch_alpha, branch_beta, activation_option, intercept_option, sparse_smooth_option, rotation_option, dp_option, n_bases, rho_option, rho_prior, rho_alpha, rho_nu, rho_lambda, M, nd, burn, thin, save_samples, save_trees, verbose, print_every));
    return rcpp_result_gen;
END_RCPP
}
// ridgeBART_fit
Rcpp::List ridgeBART_fit(Rcpp::NumericVector Y_train, Rcpp::NumericMatrix tX_cont_train, Rcpp::IntegerMatrix tX_cat_train, Rcpp::NumericMatrix tZ_mat_train, Rcpp::NumericMatrix tX_cont_test, Rcpp::IntegerMatrix tX_cat_test, Rcpp::NumericMatrix tZ_mat_test, Rcpp::LogicalVector unif_cuts, Rcpp::Nullable<Rcpp::List> cutpoints_list, Rcpp::Nullable<Rcpp::List> cat_levels_list, Rcpp::Nullable<Rcpp::List> edge_mat_list, Rcpp::LogicalVector graph_split, int graph_cut_type, bool oblique_option, double prob_aa, int x0_option, bool sparse, double a_u, double b_u, double p_change, Rcpp::NumericVector beta0, double tau, double branch_alpha, double branch_beta, double sigma0, double lambda, double nu, int activation_option, int intercept_option, int sparse_smooth_option, int rotation_option, int dp_option, int n_bases, int rho_option, Rcpp::NumericVector rho_prior, double rho_alpha, double rho_nu, double rho_lambda, int M, int nd, int burn, int thin, bool save_samples, bool save_trees, bool verbose, int print_every);
RcppExport SEXP _ridgeBART_ridgeBART_fit(SEXP Y_trainSEXP, SEXP tX_cont_trainSEXP, SEXP tX_cat_trainSEXP, SEXP tZ_mat_trainSEXP, SEXP tX_cont_testSEXP, SEXP tX_cat_testSEXP, SEXP tZ_mat_testSEXP, SEXP unif_cutsSEXP, SEXP cutpoints_listSEXP, SEXP cat_levels_listSEXP, SEXP edge_mat_listSEXP, SEXP graph_splitSEXP, SEXP graph_cut_typeSEXP, SEXP oblique_optionSEXP, SEXP prob_aaSEXP, SEXP x0_optionSEXP, SEXP sparseSEXP, SEXP a_uSEXP, SEXP b_uSEXP, SEXP p_changeSEXP, SEXP beta0SEXP, SEXP tauSEXP, SEXP branch_alphaSEXP, SEXP branch_betaSEXP, SEXP sigma0SEXP, SEXP lambdaSEXP, SEXP nuSEXP, SEXP activation_optionSEXP, SEXP intercept_optionSEXP, SEXP sparse_smooth_optionSEXP, SEXP rotation_optionSEXP, SEXP dp_optionSEXP, SEXP n_basesSEXP, SEXP rho_optionSEXP, SEXP rho_priorSEXP, SEXP rho_alphaSEXP, SEXP rho_nuSEXP, SEXP rho_lambdaSEXP, SEXP MSEXP, SEXP ndSEXP, SEXP burnSEXP, SEXP thinSEXP, SEXP save_samplesSEXP, SEXP save_treesSEXP, SEXP verboseSEXP, SEXP print_everySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type Y_train(Y_trainSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type tX_cont_train(tX_cont_trainSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type tX_cat_train(tX_cat_trainSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type tZ_mat_train(tZ_mat_trainSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type tX_cont_test(tX_cont_testSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type tX_cat_test(tX_cat_testSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type tZ_mat_test(tZ_mat_testSEXP);
    Rcpp::traits::input_parameter< Rcpp::LogicalVector >::type unif_cuts(unif_cutsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type cutpoints_list(cutpoints_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type cat_levels_list(cat_levels_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type edge_mat_list(edge_mat_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::LogicalVector >::type graph_split(graph_splitSEXP);
    Rcpp::traits::input_parameter< int >::type graph_cut_type(graph_cut_typeSEXP);
    Rcpp::traits::input_parameter< bool >::type oblique_option(oblique_optionSEXP);
    Rcpp::traits::input_parameter< double >::type prob_aa(prob_aaSEXP);
    Rcpp::traits::input_parameter< int >::type x0_option(x0_optionSEXP);
    Rcpp::traits::input_parameter< bool >::type sparse(sparseSEXP);
    Rcpp::traits::input_parameter< double >::type a_u(a_uSEXP);
    Rcpp::traits::input_parameter< double >::type b_u(b_uSEXP);
    Rcpp::traits::input_parameter< double >::type p_change(p_changeSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type beta0(beta0SEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< double >::type branch_alpha(branch_alphaSEXP);
    Rcpp::traits::input_parameter< double >::type branch_beta(branch_betaSEXP);
    Rcpp::traits::input_parameter< double >::type sigma0(sigma0SEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< int >::type activation_option(activation_optionSEXP);
    Rcpp::traits::input_parameter< int >::type intercept_option(intercept_optionSEXP);
    Rcpp::traits::input_parameter< int >::type sparse_smooth_option(sparse_smooth_optionSEXP);
    Rcpp::traits::input_parameter< int >::type rotation_option(rotation_optionSEXP);
    Rcpp::traits::input_parameter< int >::type dp_option(dp_optionSEXP);
    Rcpp::traits::input_parameter< int >::type n_bases(n_basesSEXP);
    Rcpp::traits::input_parameter< int >::type rho_option(rho_optionSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type rho_prior(rho_priorSEXP);
    Rcpp::traits::input_parameter< double >::type rho_alpha(rho_alphaSEXP);
    Rcpp::traits::input_parameter< double >::type rho_nu(rho_nuSEXP);
    Rcpp::traits::input_parameter< double >::type rho_lambda(rho_lambdaSEXP);
    Rcpp::traits::input_parameter< int >::type M(MSEXP);
    Rcpp::traits::input_parameter< int >::type nd(ndSEXP);
    Rcpp::traits::input_parameter< int >::type burn(burnSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< bool >::type save_samples(save_samplesSEXP);
    Rcpp::traits::input_parameter< bool >::type save_trees(save_treesSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type print_every(print_everySEXP);
    rcpp_result_gen = Rcpp::wrap(ridgeBART_fit(Y_train, tX_cont_train, tX_cat_train, tZ_mat_train, tX_cont_test, tX_cat_test, tZ_mat_test, unif_cuts, cutpoints_list, cat_levels_list, edge_mat_list, graph_split, graph_cut_type, oblique_option, prob_aa, x0_option, sparse, a_u, b_u, p_change, beta0, tau, branch_alpha, branch_beta, sigma0, lambda, nu, activation_option, intercept_option, sparse_smooth_option, rotation_option, dp_option, n_bases, rho_option, rho_prior, rho_alpha, rho_nu, rho_lambda, M, nd, burn, thin, save_samples, save_trees, verbose, print_every));
    return rcpp_result_gen;
END_RCPP
}
// drawEnsemble
Rcpp::List drawEnsemble(Rcpp::NumericMatrix tX_cont, Rcpp::IntegerMatrix tX_cat, Rcpp::NumericMatrix tZ_mat, Rcpp::LogicalVector unif_cuts, Rcpp::Nullable<Rcpp::List> cutpoints_list, Rcpp::Nullable<Rcpp::List> cat_levels_list, Rcpp::Nullable<Rcpp::List> edge_mat_list, Rcpp::LogicalVector graph_split, int graph_cut_type, bool oblique_option, double prob_aa, int x0_option, bool sparse, double a_u, double b_u, Rcpp::NumericVector beta0, double tau, double lambda, double nu, int activation_option, int intercept_option, int n_bases, int rho_option, Rcpp::NumericVector rho_prior, double rho_alpha, double rho_nu, double rho_lambda, int M, bool verbose, int print_every);
RcppExport SEXP _ridgeBART_drawEnsemble(SEXP tX_contSEXP, SEXP tX_catSEXP, SEXP tZ_matSEXP, SEXP unif_cutsSEXP, SEXP cutpoints_listSEXP, SEXP cat_levels_listSEXP, SEXP edge_mat_listSEXP, SEXP graph_splitSEXP, SEXP graph_cut_typeSEXP, SEXP oblique_optionSEXP, SEXP prob_aaSEXP, SEXP x0_optionSEXP, SEXP sparseSEXP, SEXP a_uSEXP, SEXP b_uSEXP, SEXP beta0SEXP, SEXP tauSEXP, SEXP lambdaSEXP, SEXP nuSEXP, SEXP activation_optionSEXP, SEXP intercept_optionSEXP, SEXP n_basesSEXP, SEXP rho_optionSEXP, SEXP rho_priorSEXP, SEXP rho_alphaSEXP, SEXP rho_nuSEXP, SEXP rho_lambdaSEXP, SEXP MSEXP, SEXP verboseSEXP, SEXP print_everySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type tX_cont(tX_contSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type tX_cat(tX_catSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type tZ_mat(tZ_matSEXP);
    Rcpp::traits::input_parameter< Rcpp::LogicalVector >::type unif_cuts(unif_cutsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type cutpoints_list(cutpoints_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type cat_levels_list(cat_levels_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type edge_mat_list(edge_mat_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::LogicalVector >::type graph_split(graph_splitSEXP);
    Rcpp::traits::input_parameter< int >::type graph_cut_type(graph_cut_typeSEXP);
    Rcpp::traits::input_parameter< bool >::type oblique_option(oblique_optionSEXP);
    Rcpp::traits::input_parameter< double >::type prob_aa(prob_aaSEXP);
    Rcpp::traits::input_parameter< int >::type x0_option(x0_optionSEXP);
    Rcpp::traits::input_parameter< bool >::type sparse(sparseSEXP);
    Rcpp::traits::input_parameter< double >::type a_u(a_uSEXP);
    Rcpp::traits::input_parameter< double >::type b_u(b_uSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type beta0(beta0SEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< int >::type activation_option(activation_optionSEXP);
    Rcpp::traits::input_parameter< int >::type intercept_option(intercept_optionSEXP);
    Rcpp::traits::input_parameter< int >::type n_bases(n_basesSEXP);
    Rcpp::traits::input_parameter< int >::type rho_option(rho_optionSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type rho_prior(rho_priorSEXP);
    Rcpp::traits::input_parameter< double >::type rho_alpha(rho_alphaSEXP);
    Rcpp::traits::input_parameter< double >::type rho_nu(rho_nuSEXP);
    Rcpp::traits::input_parameter< double >::type rho_lambda(rho_lambdaSEXP);
    Rcpp::traits::input_parameter< int >::type M(MSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type print_every(print_everySEXP);
    rcpp_result_gen = Rcpp::wrap(drawEnsemble(tX_cont, tX_cat, tZ_mat, unif_cuts, cutpoints_list, cat_levels_list, edge_mat_list, graph_split, graph_cut_type, oblique_option, prob_aa, x0_option, sparse, a_u, b_u, beta0, tau, lambda, nu, activation_option, intercept_option, n_bases, rho_option, rho_prior, rho_alpha, rho_nu, rho_lambda, M, verbose, print_every));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_ridgeBART_eval_bases", (DL_FUNC) &_ridgeBART_eval_bases, 7},
    {"_ridgeBART_get_bases", (DL_FUNC) &_ridgeBART_get_bases, 7},
    {"_ridgeBART_predict_ridgeBART", (DL_FUNC) &_ridgeBART_predict_ridgeBART, 7},
    {"_ridgeBART_probit_ridgeBART_fit", (DL_FUNC) &_ridgeBART_probit_ridgeBART_fit, 43},
    {"_ridgeBART_ridgeBART_fit", (DL_FUNC) &_ridgeBART_ridgeBART_fit, 46},
    {"_ridgeBART_drawEnsemble", (DL_FUNC) &_ridgeBART_drawEnsemble, 30},
    {NULL, NULL, 0}
};

RcppExport void R_init_ridgeBART(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
