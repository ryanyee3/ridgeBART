#include "save_trees.h"

void write_fit_logs(std::vector<std::string> &change_log, std::vector<std::map<int,double>> &phi_log, std::vector<arma::mat> &w_log, std::vector<arma::vec> &b_log, std::vector<std::map<int,arma::vec>> &beta_log, 
    tree_prior_info &tree_pi, tree &t, int &nid, int &iter, int &change_type, int &accepted, set_str_conversion &set_str)
{
    // save the beta for each tree
    // we need to do this every time regardless of whether or not a proposal was accepted
    tree::npv bn_vec; // vector to store pointers to bottom nodes of tree
    t.get_bots(bn_vec); // get pointers to all bottom nodes
    std::map<int,arma::vec> tmp_beta;

    // loop over bottom nodes
    // push to map where first entry is nid and second entry is beta value
    for(tree::npv::iterator npv_it = bn_vec.begin(); npv_it != bn_vec.end(); ++npv_it) tmp_beta.insert(std::pair<int,arma::vec>((*npv_it)->get_nid(), (*npv_it)->get_beta()));

    // update beta_log
    beta_log.push_back(tmp_beta);

    std::ostringstream os;
    os.precision(32);
    os << iter << " "; // record the sampling iteration

    // if this is the first iteration after burn-in, we need to write down the full tree
    if (iter == 0){

        // get every node id in the tree
        tree::cnpv nds;
        rule_t tmp_rule;
        leaf_ft tmp_leaf;
        t.get_nodes(nds);

        // loop over node ids
        for(tree::cnpv_it nd_it = nds.begin(); nd_it != nds.end(); ++nd_it){

            // record the node id and get node type
            os << (*nd_it)->get_nid() << " ";
            char ntype = (*nd_it)->get_ntype();

            // condition checks that we are not a bottom node or a stump
            if ((ntype != 'b') && !((ntype == 't') && (!(*nd_it)->l))){
                // record rule info for interior nodes
                tmp_rule.clear();
                tmp_rule = (*nd_it)->get_rule();
                write_rule(os, phi_log, tree_pi, tmp_rule, set_str);
            } else { // if node is not an interior node, then it is a leaf node
                // record leaf info for terminal nodes
                os << "l "; // l tells us a leaf is coming next
                tmp_leaf.clear();
                tmp_leaf = (*nd_it)->get_leaf();
                write_leaf(os, w_log, b_log, tmp_leaf);
            }

        } // closes loop over node ids

        // write changes to change_log
        change_log.push_back(os.str());

    } else if (accepted == 1){ // we have accepted a change and it is not the first iteration

        if (change_type == 1){ // grow

            os << nid << " g "; // g tells us we accepted a grow proposal
            leaf_ft tmp_leaf;

            if (t.get_ptr(nid) == 0){ // nid does not exist (this should never hit)
                Rcpp::Rcout << "could not find node id " << nid << " on this tree." << std::endl;
                Rcpp::stop("[write_tree_log]: node id not found.");
            } else {

                // record new rule
                rule_t rule = t.get_ptr(nid)->get_rule();
                write_rule(os, phi_log, tree_pi, rule, set_str);

                // record info from left child
                tmp_leaf.clear();
                tmp_leaf = t.get_ptr(nid)->get_l()->get_leaf();
                write_leaf(os, w_log, b_log, tmp_leaf);

                 // record info from right child
                tmp_leaf.clear();
                tmp_leaf = t.get_ptr(nid)->get_r()->get_leaf();
                write_leaf(os, w_log, b_log, tmp_leaf);

            }            

        } else if (change_type == 2){ // prune

            os << nid << " p " << std::endl; // p tells use we accepted a prune proposal, followed by the id of the node we pruned
            leaf_ft tmp_leaf;

            // record leaf info for newly drawn parent leaf
            tmp_leaf.clear();
            tmp_leaf = t.get_ptr(nid)->get_leaf();
            write_leaf(os, w_log, b_log, tmp_leaf);

        } else if (change_type == 3){

            // instead of an nid, store the number of leaves we changed
            // when we read this back it will be stored in the nid variable
            int faux_nid = bn_vec.size(); 
            os << faux_nid << " c " << std::endl; // c tells us we accepted a change proposal

            // record leaf info
            for(tree::npv::iterator npv_it = bn_vec.begin(); npv_it != bn_vec.end(); ++npv_it){
                os << (*npv_it)->get_nid() << " "; // node of leaf
                leaf_ft tmp_leaf = (*npv_it)->get_leaf();
                write_leaf(os, w_log, b_log, tmp_leaf);
            }

        } else { // this should never hit
            Rcpp::Rcout << "Change type " << change_type << " was passed but does not exist." << std::endl;
            Rcpp::stop("[write_tree_log]: invalid change type.");
        }
        // write changes to change_log
        change_log.push_back(os.str());
    }
}

void write_rule(std::ostringstream &os, std::vector<std::map<int,double>> &phi_log, tree_prior_info &tree_pi, rule_t &rule, set_str_conversion &set_str)
{
    if (rule.is_cat){ // categorical rule
        int K = tree_pi.K->at(rule.v_cat);
        os << "k " << rule.v_cat << " " << K << " "; // k tells us the rule is categorical
        os << set_str.set_to_hex(K, rule.l_vals) << " ";
        os << set_str.set_to_hex(K, rule.r_vals) << " ";
    } else { // continuous rule
        phi_log.push_back(rule.phi); // record the phi in the phi_log
        int pid = phi_log.size() - 1; // record index of accepted phi in phi_list
        os << "c " << rule.c << " " << pid << " "; // c tells us the rule is continuous, followed by the cutpoint and index of the rule and cutpoint in phi_log
    }
    os << std::endl;
}

void write_leaf(std::ostringstream &os, std::vector<arma::mat> &w_log, std::vector<arma::vec> &b_log, leaf_ft &leaf)
{
    w_log.push_back(leaf.w);
    b_log.push_back(leaf.b);
    int wid = w_log.size() - 1;
    int bid = b_log.size() - 1;
    if (wid == bid){
        os << wid << " " << std::endl; // record the index of the leaf info in w_log and b_log
    } else {
        Rcpp::Rcout << "w_log has size " << wid + 1 << " but b_log has size " << bid + 1 << "." << std::endl;
        Rcpp::stop("[write_leaf]: w_log and b_log must be the same size!");
    }
}

Rcpp::List parse_fit_logs(std::vector<std::string> &change_log, std::vector<std::map<int,double>> &phi_log, std::vector<arma::mat> &w_log, std::vector<arma::vec> &b_log, std::vector<std::map<int,arma::vec>> &beta_log, tree_prior_info &tree_pi, data_info &di)
{
    // R data structures
    Rcpp::CharacterVector change_list(change_log.size());
    Rcpp::List phi_list(phi_log.size());
    Rcpp::List w_list(w_log.size());
    Rcpp::List b_list(b_log.size());
    Rcpp::List beta_list(beta_log.size());

    // write change_log to change_list
    for (int i = 0; i < change_log.size(); ++i) change_list[i] = change_log[i];
    
    // write phi_log to phi_list
    for (int i = 0; i < phi_log.size(); ++i){
        Rcpp::NumericMatrix tmp_phi(phi_log[i].size(), 2);
        int k = 0;
        for (std::map<int,double>::iterator it = phi_log[i].begin(); it != phi_log[i].end(); ++it){
            tmp_phi(k, 0) = it->first;
            tmp_phi(k, 1) = it->second;
            ++k;
        }
        phi_list[i] = tmp_phi;
    }

    // write w_log to w_list
    for (int i = 0; i < w_log.size(); ++i) w_list[i] = w_log[i];

    // write b_log to b_list
    for (int i = 0; i < b_log.size(); ++i) b_list[i] = b_log[i];

    // write beta_log to beta_list
    for (int i = 0; i < beta_log.size(); ++i){
        // it is faster to specify the size of tmp_beta before populating
        // it is also faster to assign the names of the elements at the end
        // details: https://stackoverflow.com/questions/34181135/converting-an-stdmap-to-an-rcpplist
        Rcpp::List tmp_beta(beta_log[i].size());
        Rcpp::CharacterVector tmp_names(beta_log[i].size());
        int j = 0;
        for (std::map<int,arma::vec>::iterator it = beta_log[i].begin(); it != beta_log[i].end(); ++it) {
            tmp_beta[j] = it->second;
            tmp_names[j] = std::to_string(it->first);
            ++j;
        }
        tmp_beta.attr("names") = tmp_names;
        beta_list[i] = tmp_beta;
    }
    
    // dump all of our outputs into an Rcpp::List
    Rcpp::List fit_list;
    fit_list["change_log"] = change_list;
    fit_list["phi_log"] = phi_list;
    fit_list["w_log"] = w_list;
    fit_list["b_log"] = b_list;
    fit_list["beta_log"] = beta_list;
    fit_list["activation_option"] = *tree_pi.activation_option;
    fit_list["intercept_option"] = tree_pi.intercept_option;
    
    return fit_list;
}

void parse_fit_list(std::vector<std::string> &change_log, std::vector<std::map<int,double>> &phi_log, std::vector<arma::mat> &w_log, std::vector<arma::vec> &b_log, std::vector<std::map<int,arma::vec>> &beta_log, int &activation_option, int &intercept_option, Rcpp::List &fit_list)
{
    Rcpp::CharacterVector change_list = fit_list["change_log"];
    Rcpp::List phi_list = fit_list["phi_log"];
    Rcpp::List w_list = fit_list["w_log"];
    Rcpp::List b_list = fit_list["b_log"];
    Rcpp::List beta_list = fit_list["beta_log"];
    activation_option = fit_list["activation_option"];
    intercept_option = fit_list["intercept_option"];

    // parse change_list to change_log
    for (int i = 0; i < change_list.size(); ++i) change_log.push_back(Rcpp::as<std::string>(change_list[i]));

    // parse phi_list to phi_log
    for (int i = 0; i < phi_list.size(); ++i){
        Rcpp::NumericMatrix tmp_phi_mat = phi_list[i];
        std::map<int,double> tmp_phi;
        for (int j = 0; j < tmp_phi_mat.nrow(); ++j) tmp_phi.insert(std::pair<int,double>(tmp_phi_mat(j, 0), tmp_phi_mat(j, 1)));
        phi_log.push_back(tmp_phi);
    }

    // parse w_list to w_log
    for (int i = 0; i < w_list.size(); ++i) w_log.push_back(w_list[i]);

    // parse b_list to b_log
    for (int i = 0; i < b_list.size(); ++i) b_log.push_back(b_list[i]);

    // parse beta_list
    for (int i = 0; i < beta_list.size(); ++i){
        Rcpp::List tmp_list = beta_list[i];
        Rcpp::CharacterVector names = tmp_list.names();
        std::map<int,arma::vec> tmp_map;
        for (int j = 0; j < names.size(); ++j){
            std::string str_key = Rcpp::as<std::string>(names[j]);
            tmp_map.insert(std::pair<int,arma::vec>(std::stoi(str_key), tmp_list[str_key]));
        }
        beta_log.push_back(tmp_map);
    }
}

void read_fit_logs(tree &t, int &last_log_index, int &sample_iter, std::vector<std::string> &change_log, std::vector<std::map<int,double>> &phi_log, std::vector<arma::mat> &w_log, std::vector<arma::vec> &b_log, std::vector<std::map<int,arma::vec>> &beta_log, int &activation_option, int &intercept_option, set_str_conversion &set_str)
{
    std::string log_string;
    std::istringstream log_ss;
    int change_iter;

    if (last_log_index == change_log.size()){
        // there are no more changes left
        // set change_iter to something silly so we don't try to make any changes to the tree
        change_iter = -1;
    } else {
        // we haven't reached the last change yet
        // check what iteration the change was made on
        log_string = change_log[last_log_index];
        log_ss.str(log_string);
        log_ss >> change_iter;
    }

    // if it is the first iteration, there will be a full tree saved
    if (change_iter == 0){

        t.to_null();
        std::string node_string; // string for an individual node of the tree
        rule_t tmp_rule;
        leaf_ft tmp_leaf;
        int nid;
        char log_type;

        while (log_ss){
            
            std::getline(log_ss, node_string, '\n');

            if (node_string.size() > 0){

                std::istringstream node_ss(node_string);
                node_ss >> nid;
                node_ss >> log_type;

                if (log_type == 'k' || log_type == 'c'){ // rule
                    read_rule(node_ss, tmp_rule, phi_log, log_type, set_str);
                    t.birth(nid, tmp_rule);
                } else if (log_type == 'l'){ // terminal node (leaf)
                    read_leaf(node_ss, tmp_leaf, w_log, b_log, activation_option, intercept_option);
                    t.get_ptr(nid)->set_leaf(tmp_leaf);
                } else {
                    Rcpp::Rcout << "Error: log type " << log_type << " unknown." << std::endl;
                    Rcpp::stop("[read_fit_log]: unknown log type!");
                }
            }

        } // closes while loop over log_ss

        ++last_log_index; // increment index of last change

    } else if (sample_iter == change_iter){ // we made a change on this iteration
        
        int nid;
        log_ss >> nid;
        char change_type;
        log_ss >> change_type;

        if (change_type == 'g'){
            
            // update rule
            rule_t tmp_rule;
            char rule_type;
            log_ss >> rule_type;
            read_rule(log_ss, tmp_rule, phi_log, rule_type, set_str);
            t.birth(nid, tmp_rule);

            // update leaves
            leaf_ft tmp_leaf;

            // leaf child
            read_leaf(log_ss, tmp_leaf, w_log, b_log, activation_option, intercept_option);
            t.get_ptr(2 * nid)->set_leaf(tmp_leaf);

            // right child
            read_leaf(log_ss, tmp_leaf, w_log, b_log, activation_option, intercept_option);
            t.get_ptr(2 * nid + 1)->set_leaf(tmp_leaf);


        } else if (change_type == 'p'){
            
            // kill children
            t.death(nid);

            // assign leaf to parent
            leaf_ft tmp_leaf;
            read_leaf(log_ss, tmp_leaf, w_log, b_log, activation_option, intercept_option);
            t.get_ptr(nid)->set_leaf(tmp_leaf);

        } else if (change_type == 'c'){
            // update leaves
            // remember, for change rule we stored the number of bottom leaves in nid
            for (int i = 0; i < nid; i++){
                
                // node id of leaf
                int tmp_nid;
                log_ss >> tmp_nid;

                // leaf info
                leaf_ft tmp_leaf;
                read_leaf(log_ss, tmp_leaf, w_log, b_log, activation_option, intercept_option);

                // set leaf
                t.get_ptr(tmp_nid)->set_leaf(tmp_leaf);
            }
        } else {
            Rcpp::Rcout << "unknown change type " << change_type << std::endl;
            Rcpp::stop("[read_tree_log]: unknown change type passed.");
        }

        ++last_log_index; // increment index of last change

    }
    // update the betas (do this every time)
    std::map<int,arma::vec> beta_map = beta_log[sample_iter];
    tree::tree_p bn;
    for (std::map<int,arma::vec>::iterator it = beta_map.begin(); it != beta_map.end(); ++it){
        bn = t.get_ptr(it->first); // first entry in map is nid
        bn->set_beta(it->second); // second entry in map is beta
    }
}

void read_rule(std::istringstream &ss, rule_t &rule, std::vector<std::map<int,double>> &phi_log, char &type, set_str_conversion &set_str)
{
    // we are updating the rule!!!
    rule.clear();

    if (type == 'k'){ // categorical rule

        int K; // number of categorical levels
        std::string l_hex; // string representation of the l_vals in a categorical rule
        std::string r_hex; // string representation of the r_vals in a categorical rule

        rule.is_cat = true;
        ss >> rule.v_cat;
        ss >> K;
        ss >> l_hex;
        ss >> r_hex;
        rule.l_vals = set_str.hex_to_set(K, l_hex);
        rule.r_vals = set_str.hex_to_set(K, r_hex);

    } else if (type == 'c'){ // continuous rule

        int pid; // phi id
        std::map<int,double> tmp_phi; // for reading in phi's

        ss >> rule.c;
        ss >> pid;
        rule.phi = phi_log[pid];

    } else {
        Rcpp::Rcout << "unknown rule type " << type << std::endl;
        Rcpp::stop("[read_rule]: unknown rule type passed.");
    }
}

void read_leaf(std::istringstream &ss, leaf_ft &leaf, std::vector<arma::mat> &w_log, std::vector<arma::vec> &b_log, int &activation_option, int &intercept_option)
{
    // we are updating the leaf!!!
    leaf.clear();
    
    // get log index
    int log_id;
    ss >> log_id;

    // get w and b
    arma::mat tmp_w = w_log[log_id];
    arma::vec tmp_b = b_log[log_id];

    // set w and b in leaf
    leaf.w = tmp_w;
    leaf.b = tmp_b;
    leaf.act_opt = activation_option;
    leaf.intercept = intercept_option;
}
