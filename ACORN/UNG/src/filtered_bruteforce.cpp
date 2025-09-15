#include <omp.h>
#include <queue>
#include <numeric>
#include <iostream>
#include "search_queue.h"
#include "filtered_bruteforce.h"


namespace ANNS {    

    // for computing groundtruth: process all query with the same label set together
    float FilteredBruteforce::search(std::shared_ptr<IStorage> base_storage, std::shared_ptr<IStorage> query_storage,
                                     std::shared_ptr<DistanceHandler> distance_handler, std::string scenario, 
                                     uint32_t num_threads, IdxType K, std::pair<IdxType, float>* results) {
        _base_storage = base_storage;
        _query_storage = query_storage;
        _distance_handler = distance_handler;
        _results = results;
        _K = K;

        // init trie index only for base label sets
        init_trie_index(false);

        // iterate each query
        std::vector<float> num_cmps(_query_storage->get_num_points());
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic, 1)
        for (auto query_vec_id=0; query_vec_id<_query_storage->get_num_points(); ++query_vec_id) {          
            std::vector<IdxType> target_group_ids;
            const auto& query_label_set = _query_storage->get_label_set(query_vec_id);

            // equality or nofilter scenario: locate the base vector ids that are equal
            if (scenario == "equality" || scenario == "nofilter") {
                auto node = base_trie_index.find_exact_match(query_label_set);
                if (node)
                    target_group_ids.emplace_back(node->group_id);

            // overlap or containment scenario: locate the base vector ids that are super sets
            } else {
                compute_base_super_sets(scenario, query_label_set, target_group_ids);
            }

            // find the nearest neighbors for each query vector
            num_cmps[query_vec_id] = answer_one_query(query_vec_id, target_group_ids);
        }

        return std::accumulate(num_cmps.begin(), num_cmps.end(), 0);
    }



    // for computing groundtruth: process all query with the same label set together
    void FilteredBruteforce::run(std::shared_ptr<IStorage> base_storage, std::shared_ptr<IStorage> query_storage,
                                 std::shared_ptr<DistanceHandler> distance_handler, std::string scenario, 
                                 uint32_t num_threads, IdxType K, std::pair<IdxType, float>* results) {
        _base_storage = base_storage;
        _query_storage = query_storage;
        _distance_handler = distance_handler;
        _results = results;
        _K = K;

        // init trie index for base and query label sets
        std::cout << "- Scenario: " << scenario << std::endl;
        init_trie_index();

        // iterate each query group
        for (auto query_group_id=1; query_group_id<query_group_id_to_label_set.size(); ++query_group_id) {           
            std::vector<IdxType> target_group_ids;
            const auto& query_label_set = query_group_id_to_label_set[query_group_id];

            // equality or nofilter scenario: locate the base vector ids that are equal
            if (scenario == "equality" || scenario == "nofilter") {
                auto node = base_trie_index.find_exact_match(query_label_set);
                if (node)
                    target_group_ids.emplace_back(node->group_id);

            // overlap or containment scenario: locate the base vector ids that are super sets
            } else {
                compute_base_super_sets(scenario, query_label_set, target_group_ids);
            }

            // find the nearest neighbors for each query vector
            omp_set_num_threads(num_threads);
            #pragma omp parallel for schedule(dynamic, 1)
            for (auto query_vec_id : query_group_id_to_vec_ids[query_group_id]) 
                answer_one_query(query_vec_id, target_group_ids);
        }
    }



    // initialize trie index for base and query label sets
    void FilteredBruteforce::init_trie_index(bool for_query) {

        // create groups for base label sets
        IdxType new_group_id = 1;
        for (auto vec_id=0; vec_id<_base_storage->get_num_points(); ++vec_id) {
            auto group_id = base_trie_index.insert(_base_storage->get_label_set(vec_id), new_group_id);
            if (group_id+1 > base_group_id_to_vec_ids.size())
                base_group_id_to_vec_ids.resize(group_id+1);
            base_group_id_to_vec_ids[group_id].emplace_back(vec_id);
        }
        std::cout << "- Number of groups in the base data: " << new_group_id-1 << std::endl;
        if (!for_query)
            return;

        // create groups for query label sets
        new_group_id = 1;
        for (auto vec_id=0; vec_id<_query_storage->get_num_points(); ++vec_id) {
            auto query_label_set = _query_storage->get_label_set(vec_id);
            auto group_id = query_trie_index.insert(query_label_set, new_group_id);
            if (group_id+1 > query_group_id_to_vec_ids.size()) {
                query_group_id_to_vec_ids.resize(group_id+1);
                query_group_id_to_label_set.resize(group_id+1);
                query_group_id_to_label_set[group_id] = query_label_set;
            }
            query_group_id_to_vec_ids[group_id].emplace_back(vec_id);
        }
        std::cout << "- Number of groups in the query data: " << new_group_id-1 << std::endl;
    }



    // get all base super sets for a query label set
    void FilteredBruteforce::compute_base_super_sets(std::string scenario, const std::vector<LabelType>& query_label_set,
                                                     std::vector<IdxType>& base_super_set_group_ids) {

        // push the super set entrances to queue
        std::vector<std::shared_ptr<TrieNode>> super_set_entrances;
        base_trie_index.get_super_set_entrances(query_label_set, super_set_entrances, false, scenario=="containment");
        std::queue<std::shared_ptr<TrieNode>> q;
        for (const auto& node : super_set_entrances)
            q.push(node);

        // find all super sets
        base_super_set_group_ids.clear();
        while (!q.empty()) {
            auto cur = q.front();
            q.pop();
            if (cur->group_id > 0)
                base_super_set_group_ids.emplace_back(cur->group_id);
            for (const auto& child : cur->children)
                q.push(child.second);
        }
    }

    

    // execute each query
    float FilteredBruteforce::answer_one_query(IdxType query_vec_id, const std::vector<IdxType>& target_group_ids) {
        auto dim = _base_storage->get_dim();
        SearchQueue search_queue;
        search_queue.reserve(_K);
        float num_cmps = 0;

        // iterate each base vector in each target group
        for (const auto& base_group_id : target_group_ids) {
            for (IdxType i=0; i<base_group_id_to_vec_ids[base_group_id].size(); ++i) {
                if (i+1 < base_group_id_to_vec_ids[base_group_id].size())
                    _base_storage->prefetch_vec_by_id(base_group_id_to_vec_ids[base_group_id][i+1]);
                const auto& base_vec_id = base_group_id_to_vec_ids[base_group_id][i];
                float distance = _distance_handler->compute(_query_storage->get_vector(query_vec_id), 
                                                            _base_storage->get_vector(base_vec_id), dim);
                search_queue.insert(base_vec_id, distance);
            }
            num_cmps += base_group_id_to_vec_ids[base_group_id].size();
        }

        // write to results
        bool enough_answer = true;
        for (auto k=0; k<_K; ++k) {
            if (k < search_queue.size())
                _results[query_vec_id*_K+k] = std::make_pair(search_queue[k].id, search_queue[k].distance);
            else {
                _results[query_vec_id*_K+k] = std::make_pair(-1, -1);
                enough_answer = false;
            }
        }
        if (!enough_answer)
            std::cout << "! Warning: query " << query_vec_id << " has less than " << _K << " answers, the calculated recall will be smaller!" << std::endl;

        return num_cmps;
    }
};