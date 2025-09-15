#include <omp.h>
#include <iostream>
#include <algorithm>
#include <boost/filesystem.hpp>
#include "utils.h"
#include "vamana.h"

namespace fs = boost::filesystem;



namespace ANNS {

    void Vamana::build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler, 
                       std::shared_ptr<Graph> graph, IdxType max_degree, IdxType Lbuild, float alpha, 
                       uint32_t num_threads, IdxType max_candidate_size) {
        
        if (_verbose) {
            std::cout << "Building Vamana index ..." << std::endl;
            std::cout << "- max_degree: " << max_degree << std::endl;
            std::cout << "- Lbuild: " << Lbuild << std::endl;
            std::cout << "- alpha: " << alpha << std::endl;
            std::cout << "- max_candidate_size: " << max_candidate_size << std::endl;
            std::cout << "- num_threads: " << num_threads << std::endl;
        }
        
        _base_storage = base_storage;
        _distance_handler = distance_handler;
        _graph = graph;

        _num_threads = num_threads;
        _max_degree = max_degree;
        _Lbuild = Lbuild;
        _alpha = alpha;
        _max_candidate_size = max_candidate_size;

        if (_verbose)
            std::cout << "Computing entry point ..." << std::endl;
        _entry_point = _base_storage->choose_medoid(num_threads, distance_handler);

        if (_verbose)
            std::cout << "Linking the graph ..." << std::endl;
        link();

        if (_verbose)
            std::cout << "Finish." << std::endl << SEP_LINE;
    }



    void Vamana::link() {
        auto num_points = _base_storage->get_num_points();
        auto dim = _base_storage->get_dim();
        SearchCacheList search_cache_list(_num_threads, num_points, _Lbuild);

        omp_set_num_threads(_num_threads);
        #pragma omp parallel for schedule(dynamic, 1)
        for (auto id = 0; id < num_points; ++id) {
            auto search_cache = search_cache_list.get_free_cache();

            // search for point 
            const char* query = _base_storage->get_vector(id);
            iterate_to_fixed_point(query, search_cache, true, id);

            // prune for candidate neighbors
            std::vector<IdxType> pruned_list;
            prune_neighbors(id, search_cache->expanded_list, pruned_list, search_cache);

            // update neighbors and insert the reversed edge
            {
                std::lock_guard<std::mutex> lock(_graph->neighbor_locks[id]);
                _graph->neighbors[id] = pruned_list;
            }
            inter_insert(id, pruned_list, search_cache);

            // clean and print
            search_cache_list.release_cache(search_cache);
            if (_verbose && id % 10000 == 0)
                std::cout << "\r" << (100.0 * id) / num_points << "%" << std::flush;
        }

        if (_verbose)
            std::cout << "\rStarting final cleanup ..." << std::endl;
        #pragma omp parallel for schedule(dynamic, 1)
        for (auto id = 0; id < num_points; ++id)
            if (_graph->neighbors[id].size() > _max_degree) {

                // prepare candidates
                std::vector<Candidate> candidates;
                for (auto& neighbor : _graph->neighbors[id]) 
                    candidates.emplace_back(neighbor, _distance_handler->compute(_base_storage->get_vector(id), 
                                                                                 _base_storage->get_vector(neighbor), dim));
                
                // prune neighbors
                std::vector<IdxType> new_neighbors;
                auto search_cache = search_cache_list.get_free_cache();
                prune_neighbors(id, candidates, new_neighbors, search_cache);
                _graph->neighbors[id] = new_neighbors;
            }
    }



    IdxType Vamana::iterate_to_fixed_point(const char* query, std::shared_ptr<SearchCache> search_cache, 
                                           bool record_expanded, IdxType target_id) {
        auto dim = _base_storage->get_dim();
        auto& search_queue = search_cache->search_queue;
        auto& visited_set = search_cache->visited_set;
        auto& expanded_list = search_cache->expanded_list;
        search_queue.clear();
        visited_set.clear();
        expanded_list.clear();
        std::vector<IdxType> neighbors;
        
        // entry point
        search_queue.insert(_entry_point, _distance_handler->compute(query, _base_storage->get_vector(_entry_point), dim));
        IdxType num_cmps = 1;

        // greedily expand closest nodes
        while (search_queue.has_unexpanded_node()) {
            const Candidate& cur = search_queue.get_closest_unexpanded();
            if (record_expanded && target_id != cur.id)
                expanded_list.push_back(cur);

            // iterate neighbors
            {
                std::lock_guard<std::mutex> lock(_graph->neighbor_locks[cur.id]);
                neighbors = _graph->neighbors[cur.id];
            }
            for (auto i=0; i<neighbors.size(); ++i) {

                // prefetch
                if (i+1 < neighbors.size()) {
                    visited_set.prefetch(neighbors[i+1]);
                    _base_storage->prefetch_vec_by_id(neighbors[i+1]);
                }

                // skip if visited
                auto& neighbor = neighbors[i];
                if (visited_set.check(neighbor)) 
                    continue;
                visited_set.set(neighbor);

                // push to search queue
                search_queue.insert(neighbor, _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim));
                num_cmps++;
            }
        }
        return num_cmps;
    }



    void Vamana::prune_neighbors(IdxType id, std::vector<Candidate>& candidates, std::vector<IdxType>& pruned_list, 
                                    std::shared_ptr<SearchCache> search_cache) {
        auto dim = _base_storage->get_dim();
        pruned_list.clear();
        pruned_list.reserve(_max_degree);

        // init candidates
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.distance < b.distance;
        });
        auto candidate_size = std::min((IdxType)(candidates.size()), _max_candidate_size);

        // init occlude factor
        auto& occlude_factor = search_cache->occlude_factor;
        occlude_factor.clear();
        occlude_factor.insert(occlude_factor.end(), candidate_size, 0.0f);

        // prune neighbors
        float cur_alpha = 1;
        while (cur_alpha <= _alpha && pruned_list.size() < _max_degree) {
            for (auto i=0; i<candidate_size && pruned_list.size() < _max_degree; ++i) {
                if (occlude_factor[i] > cur_alpha) 
                    continue;

                // set to float::max so that is not considered again
                occlude_factor[i] = std::numeric_limits<float>::max();
                if (candidates[i].id != id)
                    pruned_list.push_back(candidates[i].id);

                // update occlude factor for the following candidates
                for (auto j=i+1; j<candidate_size; ++j) {
                    if (occlude_factor[j] > _alpha)
                        continue;
                    auto distance_ij = _distance_handler->compute(_base_storage->get_vector(candidates[i].id), 
                                                                _base_storage->get_vector(candidates[j].id), dim);
                    occlude_factor[j] = (distance_ij == 0) ? std::numeric_limits<float>::max() 
                                                        : std::max(occlude_factor[j], candidates[j].distance / distance_ij);
                }
            }
            cur_alpha *= 1.2f;
        }
    }



    void Vamana::inter_insert(IdxType src, std::vector<IdxType>& src_neighbors, std::shared_ptr<SearchCache> search_cache) {
        auto dim = _base_storage->get_dim();

        // insert the reversed edge
        for (auto& dst : src_neighbors) {
            bool need_prune = false;
            std::vector<Candidate> candidates;

            // try to add edge dst -> src
            {
                std::lock_guard<std::mutex> lock(_graph->neighbor_locks[dst]);
                auto& dst_neighbors = _graph->neighbors[dst];
                if (std::find(dst_neighbors.begin(), dst_neighbors.end(), src) == dst_neighbors.end()) {
                    if (dst_neighbors.size() < (IdxType)(default_paras::GRAPH_SLACK_FACTOR * _max_degree)) 
                        dst_neighbors.push_back(src);
                    else {
                        candidates.reserve(dst_neighbors.size() + 1);
                        for (auto& neighbor : dst_neighbors)
                            candidates.emplace_back(neighbor, 0);
                        candidates.emplace_back(src, 0);
                        need_prune = true;
                    }
                }
            }

            // prune the neighbors of dst
            if (need_prune) {
                for (auto& candidate : candidates)
                    candidate.distance = _distance_handler->compute(_base_storage->get_vector(dst), 
                                                                    _base_storage->get_vector(candidate.id), dim);
                std::vector<IdxType> new_dst_neighbors;
                prune_neighbors(dst, candidates, new_dst_neighbors, search_cache);
                {
                    std::lock_guard<std::mutex> lock(_graph->neighbor_locks[dst]);
                    _graph->neighbors[dst] = new_dst_neighbors;
                }
            }
        }
    }



    void Vamana::statistics() {
        float num_points = _base_storage->get_num_points();
        std::cout << "Number of points: " << num_points << std::endl;

        float num_edges = 0;
        IdxType min_degree = std::numeric_limits<IdxType>::max(), max_degree = 0;
        for (auto id=0; id<num_points; ++id) {
            num_edges += _graph->neighbors[id].size();
            min_degree = std::min(min_degree, (IdxType)_graph->neighbors[id].size());
            max_degree = std::max(max_degree, (IdxType)_graph->neighbors[id].size());
        }
        std::cout << "Number of edges: " << num_edges << std::endl;
        std::cout << "Min degree: " << min_degree << std::endl;
        std::cout << "Max degree: " << max_degree << std::endl;

        float avg_degree = num_edges / num_points;
        std::cout << "Average degree: " << avg_degree << std::endl;
    }



    void Vamana::save(std::string& index_path_prefix) {
        fs::create_directories(index_path_prefix);
        std::cout << "Saving index to " << index_path_prefix << " ..." << std::endl;

        // save meta data
        std::map<std::string, std::string> meta_data;
        meta_data["max_degree"] = std::to_string(_max_degree);
        meta_data["Lbuild"] = std::to_string(_Lbuild);
        meta_data["alpha"] = std::to_string(_alpha);
        meta_data["max_candidate_size"] = std::to_string(_max_candidate_size);
        meta_data["build_num_threads"] = std::to_string(_num_threads);
        meta_data["entry_point"] = std::to_string(_entry_point);
        std::string meta_filename = index_path_prefix + "meta";
        write_kv_file(meta_filename, meta_data);

        // save graph data
        std::string graph_filename = index_path_prefix + "graph";
        _graph->save(graph_filename);

        // print
        std::cout << "- Index saved." << std::endl;
    }

    

    void Vamana::load(std::string& index_path_prefix, std::shared_ptr<Graph> graph) {
        std::cout << "Loading index from " << index_path_prefix << " ..." << std::endl;
            
        // load meta data
        std::string meta_filename = index_path_prefix + "meta";
        auto meta_data = parse_kv_file(meta_filename);
        _entry_point = std::stoi(meta_data["entry_point"]);

        // load graph data
        std::string graph_filename = index_path_prefix + "graph";
        _graph = graph;
        _graph->load(graph_filename);

        // print
        std::cout << "- Index loaded." << std::endl;
    }



    void Vamana::search(std::shared_ptr<IStorage> base_storage, std::shared_ptr<IStorage> query_storage, 
                        std::shared_ptr<DistanceHandler> distance_handler, IdxType K, IdxType Lsearch, 
                        uint32_t num_threads, std::pair<IdxType, float>* results, std::vector<IdxType>& num_cmps) {
        auto num_points = base_storage->get_num_points();
        auto num_queries = query_storage->get_num_points();
        _base_storage = base_storage;
        _query_storage = query_storage;
        _distance_handler = distance_handler;

        // preparation
        if (K > Lsearch) {
            std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
            exit(-1);
        }
        SearchCacheList search_cache_list(num_threads, num_points, Lsearch);

        // run queries
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic, 1)
        for (auto id = 0; id < num_queries; ++id) {
            auto search_cache = search_cache_list.get_free_cache(); 
            const char* query = _query_storage->get_vector(id);
            num_cmps[id] = iterate_to_fixed_point(query, search_cache);

            // write results then clean
            for (auto k=0; k<K; ++k) {
                if (k < search_cache->search_queue.size()) {
                    results[id*K+k].first = search_cache->search_queue[k].id;
                    results[id*K+k].second = search_cache->search_queue[k].distance;
                } else
                    results[id*K+k].first = -1;
            }
            search_cache_list.release_cache(search_cache);
        }
    }
}