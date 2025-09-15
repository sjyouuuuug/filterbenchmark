#include <omp.h>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <boost/filesystem.hpp>
#include "utils.h"
#include "vamana.h"
#include "uni_nav_graph.h"

namespace fs = boost::filesystem;



namespace ANNS {

    void UniNavGraph::build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler, 
                            std::string scenario, std::string index_name, uint32_t num_threads, IdxType num_cross_edges,
                            IdxType max_degree, IdxType Lbuild, float alpha) {
        auto all_start_time = std::chrono::high_resolution_clock::now();
        _base_storage = base_storage;
        _num_points = base_storage->get_num_points();
        _distance_handler = distance_handler;
        std::cout << "- Scenario: " << scenario << std::endl;

        // index parameters
        _index_name = index_name;
        _num_cross_edges = num_cross_edges;
        _max_degree = max_degree;
        _Lbuild = Lbuild;
        _alpha = alpha;
        _num_threads = num_threads;
        _scenario = scenario;

        // build the trie tree index to divide groups
        std::cout << "Dividing groups and building the trie tree index ..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        build_trie_and_divide_groups();
        _graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
        prepare_group_storages_graphs();
        _label_processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::high_resolution_clock::now() - start_time).count();
        std::cout << "- Finished in " << _label_processing_time << " ms" << std::endl;

        // build graph index for each group
        build_graph_for_all_groups();

        // for label equality scenario, there is no need for label navigating graph and cross-group edges
        if (_scenario == "equality") {
            add_offset_for_uni_nav_graph();
        } else {

            // build the label navigating graph
            build_label_nav_graph();

            // build cross-group edges
            build_cross_group_edges();
        }

        // index time
        _index_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::high_resolution_clock::now() - all_start_time).count();
    }



    void UniNavGraph::build_trie_and_divide_groups() {

        // create groups for base label sets
        IdxType new_group_id = 1;
        for (auto vec_id=0; vec_id<_num_points; ++vec_id) {
            const auto& label_set = _base_storage->get_label_set(vec_id);
            auto group_id = _trie_index.insert(label_set, new_group_id);

            // deal with new label set
            if (group_id+1 > _group_id_to_vec_ids.size()) {
                _group_id_to_vec_ids.resize(group_id+1);
                _group_id_to_label_set.resize(group_id+1);
                _group_id_to_label_set[group_id] = label_set;
            }
            _group_id_to_vec_ids[group_id].emplace_back(vec_id);
        }

        // logs
        _num_groups = new_group_id-1;
        std::cout << "- Number of groups: " << _num_groups << std::endl;    
    }
            


    void UniNavGraph::get_min_super_sets(const std::vector<LabelType>& query_label_set, std::vector<IdxType>& min_super_set_ids, 
                                         bool avoid_self, bool need_containment) {
        min_super_set_ids.clear();

        // obtain the candidates
        std::vector<std::shared_ptr<TrieNode>> candidates;
        _trie_index.get_super_set_entrances(query_label_set, candidates, avoid_self, need_containment);

        // special cases
        if (candidates.empty())
            return;
        if (candidates.size() == 1) {
            min_super_set_ids.emplace_back(candidates[0]->group_id);
            return;
        }

        // obtain the minimum size
        std::sort(candidates.begin(), candidates.end(), 
                  [](const std::shared_ptr<TrieNode>& a, const std::shared_ptr<TrieNode>& b) {
                      return a->label_set_size < b->label_set_size;
                  });
        auto min_size = _group_id_to_label_set[candidates[0]->group_id].size();
        
        // get the minimum super sets
        for (auto candidate : candidates) {
            const auto& cur_group_id = candidate->group_id;
            const auto& cur_label_set = _group_id_to_label_set[cur_group_id];
            bool is_min = true;
            
            // check whether contains existing minimum super sets (label ids are in ascending order)
            if (cur_label_set.size() > min_size) {
                for (auto min_group_id : min_super_set_ids) {
                    const auto& min_label_set = _group_id_to_label_set[min_group_id];
                    if (std::includes(cur_label_set.begin(), cur_label_set.end(), min_label_set.begin(), min_label_set.end())) {
                        is_min = false;
                        break;
                    }
                }
            }

            // add to the minimum super sets
            if (is_min) 
                min_super_set_ids.emplace_back(cur_group_id);
        }
    }



    void UniNavGraph::prepare_group_storages_graphs() {
        _new_vec_id_to_group_id.resize(_num_points);

        // reorder the vectors
        _group_id_to_range.resize(_num_groups+1);
        _new_to_old_vec_ids.resize(_num_points);
        IdxType new_vec_id = 0;
        for (auto group_id=1; group_id<=_num_groups; ++group_id) {
            _group_id_to_range[group_id].first = new_vec_id;
            for (auto old_vec_id : _group_id_to_vec_ids[group_id]) {
                _new_to_old_vec_ids[new_vec_id] = old_vec_id;
                _new_vec_id_to_group_id[new_vec_id] = group_id;
                ++new_vec_id;
            }
            _group_id_to_range[group_id].second = new_vec_id;
        }

        // reorder the underlying storage
        _base_storage->reorder_data(_new_to_old_vec_ids);

        // init storage and graph for each group
        _group_storages.resize(_num_groups + 1);
        _group_graphs.resize(_num_groups + 1);
        for (auto group_id=1; group_id<=_num_groups; ++group_id) {
            auto start = _group_id_to_range[group_id].first;
            auto end = _group_id_to_range[group_id].second;
            _group_storages[group_id] = create_storage(_base_storage, start, end);
            _group_graphs[group_id] = std::make_shared<Graph>(_graph, start, end);
        }
    }



    void UniNavGraph::build_graph_for_all_groups() {
        std::cout << "Building graph for each group ..." << std::endl;
        omp_set_num_threads(_num_threads);
        auto start_time = std::chrono::high_resolution_clock::now();

        // build vamana index
        if (_index_name == "Vamana") {
            _vamana_instances.resize(_num_groups + 1);
            _group_entry_points.resize(_num_groups + 1);

            #pragma omp parallel for schedule(dynamic, 1)
            for (auto group_id=1; group_id<=_num_groups; ++group_id) {
                if (group_id % 100 == 0)
                    std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
                
                // if there are less than _max_degree points in the group, just build a complete graph
                const auto& range = _group_id_to_range[group_id];
                if (range.second - range.first <= _max_degree) {
                    build_complete_graph(_group_graphs[group_id], range.second - range.first);
                    _vamana_instances[group_id] = std::make_shared<Vamana>(_group_storages[group_id], _distance_handler,
                                                                           _group_graphs[group_id], 0);

                // build the vamana graph
                } else {                
                    _vamana_instances[group_id] = std::make_shared<Vamana>(false);
                    _vamana_instances[group_id]->build(_group_storages[group_id], _distance_handler, 
                                                       _group_graphs[group_id], _max_degree, _Lbuild, _alpha, 1);
                }

                // set entry point
                _group_entry_points[group_id] = _vamana_instances[group_id]->get_entry_point() + range.first;
            }
        
        // if none of the above
        } else {
            std::cerr << "Error: invalid index name " << _index_name << std::endl;
            exit(-1);
        }

        _build_graph_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - start_time).count();
        std::cout << "\r- Finished in " << _build_graph_time << " ms" << std::endl;
    }



    void UniNavGraph::build_complete_graph(std::shared_ptr<Graph> graph, IdxType num_points) {
        for (auto i=0; i<num_points; ++i)
            for (auto j=0; j<num_points; ++j)
                if (i != j)
                    graph->neighbors[i].emplace_back(j);
    }



    void UniNavGraph::build_label_nav_graph() {
        std::cout << "Building label navigation graph... " << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups+1);
        omp_set_num_threads(_num_threads);
        
        // obtain out-neighbors
        #pragma omp parallel for schedule(dynamic, 256)
        for (auto group_id=1; group_id<=_num_groups; ++group_id) {
            if (group_id % 100 == 0)
                std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
            std::vector<IdxType> min_super_set_ids;
            get_min_super_sets(_group_id_to_label_set[group_id], min_super_set_ids, true);
            _label_nav_graph->out_neighbors[group_id] = min_super_set_ids;
        }

        // obtain in-neighbors
        for (auto group_id=1; group_id<=_num_groups; ++group_id)
            for (auto each : _label_nav_graph->out_neighbors[group_id])
                _label_nav_graph->in_neighbors[each].emplace_back(group_id);

        _build_LNG_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::high_resolution_clock::now() - start_time).count();
        std::cout << "\r- Finished in " << _build_LNG_time << " ms" << std::endl;
    }



    void UniNavGraph::add_offset_for_uni_nav_graph() {
        omp_set_num_threads(_num_threads);
        #pragma omp parallel for schedule(dynamic, 4096)
        for (auto i=0; i<_num_points; ++i)
            for (auto& neighbor : _graph->neighbors[i])
                neighbor += _group_id_to_range[_new_vec_id_to_group_id[i]].first;
    }



    void UniNavGraph::build_cross_group_edges() {
        std::cout << "Building cross-group edges ..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // allocate memory for storaging cross-group neighbors
        std::vector<SearchQueue> cross_group_neighbors;
        cross_group_neighbors.resize(_num_points);
        for (auto point_id=0; point_id<_num_points; ++point_id)
            cross_group_neighbors[point_id].reserve(_num_cross_edges);

        // allocate memory for search caches
        size_t max_group_size = 0;
        for (auto group_id=1; group_id<=_num_groups; ++group_id)
            max_group_size = std::max(max_group_size, _group_id_to_vec_ids[group_id].size());
        SearchCacheList search_cache_list(_num_threads, max_group_size, _Lbuild);
        omp_set_num_threads(_num_threads);

        // for each group
        for (auto group_id=1; group_id<=_num_groups; ++group_id) {
            if (_label_nav_graph->in_neighbors[group_id].size() > 0) {
                if (group_id % 100 == 0)
                    std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
                IdxType offset = _group_id_to_range[group_id].first;

                // query vamana index
                if (_index_name == "Vamana") {
                    auto index = _vamana_instances[group_id];
                    if (_num_cross_edges > _Lbuild) {
                        std::cerr << "Error: num_cross_edges should be less than or equal to Lbuild" << std::endl;
                        exit(-1);
                    }

                    // for each in-neighbor group
                    for (auto in_group_id : _label_nav_graph->in_neighbors[group_id]) {
                        const auto& range = _group_id_to_range[in_group_id];

                        // take each vector in the group as the query
                        #pragma omp parallel for schedule(dynamic, 1)
                        for (auto vec_id=range.first; vec_id<range.second; ++vec_id) {
                            const char* query = _base_storage->get_vector(vec_id);
                            auto search_cache = search_cache_list.get_free_cache(); 
                            index->iterate_to_fixed_point(query, search_cache);

                            // update the cross-group edges for vec_id
                            for (auto k=0; k<search_cache->search_queue.size(); ++k)
                                cross_group_neighbors[vec_id].insert(search_cache->search_queue[k].id + offset, 
                                                                     search_cache->search_queue[k].distance);
                            search_cache_list.release_cache(search_cache);
                        }
                    }
                
                // if none of the above
                } else {
                    std::cerr << "Error: invalid index name " << _index_name << std::endl;
                    exit(-1);
                }
            }
        }

        // add additional edges
        std::vector<std::vector<std::pair<IdxType, IdxType>>> additional_edges(_num_groups+1);
        #pragma omp parallel for schedule(dynamic, 256)
        for (IdxType group_id=1; group_id <= _num_groups; ++group_id) {
            const auto& cur_range = _group_id_to_range[group_id];
            std::unordered_set<IdxType> connected_groups;

            // obtain connected groups
            for (IdxType i=cur_range.first; i<cur_range.second; ++i)
                for (IdxType j=0; j<cross_group_neighbors[i].size(); ++j)
                    connected_groups.insert(_new_vec_id_to_group_id[cross_group_neighbors[i][j].id]);

            // add additional cross-group edges for unconnected groups
            for (IdxType out_group_id : _label_nav_graph->out_neighbors[group_id])
                if (connected_groups.find(out_group_id) == connected_groups.end()) {
                    IdxType cnt = 0;
                    for (auto vec_id=cur_range.first; vec_id<cur_range.second && cnt < _num_cross_edges; ++vec_id) {
                        auto search_cache = search_cache_list.get_free_cache(); 
                        _vamana_instances[out_group_id]->iterate_to_fixed_point(_base_storage->get_vector(vec_id), search_cache);

                        for (auto k=0; k<search_cache->search_queue.size() && k<_num_cross_edges / 2; ++k) {
                            additional_edges[group_id].emplace_back(vec_id,
                                                                    search_cache->search_queue[k].id + _group_id_to_range[out_group_id].first);
                            cnt += 1;
                        }
                        search_cache_list.release_cache(search_cache);
                    }
                }
        }

        // add offset for uni-nav graph
        add_offset_for_uni_nav_graph();

        // merge cross-group edges
        #pragma omp parallel for schedule(dynamic, 4096)
        for (auto point_id=0; point_id<_num_points; ++point_id)
            for (auto k=0; k<cross_group_neighbors[point_id].size(); ++k)
                _graph->neighbors[point_id].emplace_back(cross_group_neighbors[point_id][k].id);

        // merge additional cross-group edges
        #pragma omp parallel for schedule(dynamic, 256)
        for (IdxType group_id=1; group_id <= _num_groups; ++group_id) {
            for (const auto& [from_id, to_id] : additional_edges[group_id])
                _graph->neighbors[from_id].emplace_back(to_id);
        }

        _build_cross_edges_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::high_resolution_clock::now() - start_time).count();
        std::cout << "\r- Finish in " << _build_cross_edges_time << " ms" << std::endl;
    }



    void UniNavGraph::search(std::shared_ptr<IStorage> query_storage, std::shared_ptr<DistanceHandler> distance_handler, 
                             uint32_t num_threads, IdxType Lsearch, IdxType num_entry_points, std::string scenario,
                             IdxType K, std::pair<IdxType, float>* results, std::vector<float>& num_cmps) {
        auto num_queries = query_storage->get_num_points();
        _query_storage = query_storage;
        _distance_handler = distance_handler;
        _scenario = scenario;

        // preparation
        if (K > Lsearch) {
            std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
            exit(-1);
        }
        SearchCacheList search_cache_list(num_threads, _num_points, Lsearch);

        // run queries
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic, 1)
        for (auto id = 0; id < num_queries; ++id) {
            auto search_cache = search_cache_list.get_free_cache(); 
            const char* query = _query_storage->get_vector(id);
            SearchQueue cur_result;

            // for overlap or nofilter scenario
            if (scenario == "overlap" || scenario == "nofilter") {
                num_cmps[id] = 0;
                search_cache->visited_set.clear();
                cur_result.reserve(K);

                // obtain entry group
                std::vector<IdxType> entry_group_ids;
                if (scenario == "overlap")
                    get_min_super_sets(_query_storage->get_label_set(id), entry_group_ids, false, false);
                else
                    get_min_super_sets({}, entry_group_ids, true, true);

                // for each entry group
                for (const auto& group_id : entry_group_ids) {
                    std::vector<IdxType> entry_points;
                    get_entry_points_given_group_id(num_entry_points, search_cache->visited_set, group_id, entry_points);

                    // graph search and dump to current result
                    num_cmps[id] += iterate_to_fixed_point(query, search_cache, id, entry_points, true, false); 
                    for (auto k=0; k<search_cache->search_queue.size() && k<K; ++k)
                        cur_result.insert(search_cache->search_queue[k].id, search_cache->search_queue[k].distance);
                }

            // for the other scenarios: containment, equality
            } else {
            
                // obtain entry points
                auto entry_points = get_entry_points(_query_storage->get_label_set(id), num_entry_points, search_cache->visited_set);
                if (entry_points.empty()) {
                    num_cmps[id] = 0;
                    for (auto k=0; k<K; ++k)
                        results[id*K+k].first = -1;
                    continue;
                }

                // graph search
                num_cmps[id] = iterate_to_fixed_point(query, search_cache, id, entry_points);  
                cur_result = search_cache->search_queue;
            }

            // write results
            for (auto k=0; k<K; ++k) {
                if (k < cur_result.size()) {
                    results[id*K+k].first = _new_to_old_vec_ids[cur_result[k].id];
                    results[id*K+k].second = cur_result[k].distance;
                } else
                    results[id*K+k].first = -1;
            }

            // clean
            search_cache_list.release_cache(search_cache);
        }
    }



    std::vector<IdxType> UniNavGraph::get_entry_points(const std::vector<LabelType>& query_label_set, 
                                                       IdxType num_entry_points, VisitedSet& visited_set) {
        std::vector<IdxType> entry_points;
        entry_points.reserve(num_entry_points);
        visited_set.clear();
        
        // obtain entry points for label-equality scenario
        if (_scenario == "equality") {
            auto node = _trie_index.find_exact_match(query_label_set);
            if (node == nullptr)
                return entry_points;
            get_entry_points_given_group_id(num_entry_points, visited_set, node->group_id, entry_points);
            
        // obtain entry points for label-containment scenario
        } else if (_scenario == "containment") {
            std::vector<IdxType> min_super_set_ids;
            get_min_super_sets(query_label_set, min_super_set_ids);
            for (auto group_id : min_super_set_ids)
                get_entry_points_given_group_id(num_entry_points, visited_set, group_id, entry_points);

        } else {
            std::cerr << "Error: invalid scenario " << _scenario << std::endl;
            exit(-1);
        }

        return entry_points;
    }



    void UniNavGraph::get_entry_points_given_group_id(IdxType num_entry_points, VisitedSet& visited_set, 
                                                      IdxType group_id, std::vector<IdxType>& entry_points) {
        const auto& group_range = _group_id_to_range[group_id];

        // not enough entry points, use all of them
        if (group_range.second - group_range.first <= num_entry_points) {
            for (auto i=0; i<group_range.second - group_range.first; ++i)
                entry_points.emplace_back(i + group_range.first);
            return;
        }

        // add the entry point of the group
        const auto& group_entry_point = _group_entry_points[group_id];
        visited_set.set(group_entry_point);
        entry_points.emplace_back(group_entry_point);
        
        // randomly sample the other entry points
        for (auto i=1; i<num_entry_points; ++i) {
            auto entry_point = rand() % (group_range.second - group_range.first) + group_range.first;
            if (visited_set.check(entry_point) == false) {
                visited_set.set(entry_point);
                entry_points.emplace_back(i + group_range.first);
            }
        }
    }



    IdxType UniNavGraph::iterate_to_fixed_point(const char* query, std::shared_ptr<SearchCache> search_cache, 
                                                IdxType target_id, const std::vector<IdxType>& entry_points,
                                                bool clear_search_queue, bool clear_visited_set) {
        auto dim = _base_storage->get_dim();
        auto& search_queue = search_cache->search_queue;
        auto& visited_set = search_cache->visited_set;
        std::vector<IdxType> neighbors;
        if (clear_search_queue)
            search_queue.clear();
        if (clear_visited_set)
            visited_set.clear();
        
        // entry point
        for (const auto& entry_point : entry_points)
            search_queue.insert(entry_point, _distance_handler->compute(query, _base_storage->get_vector(entry_point), dim));
        IdxType num_cmps = entry_points.size();

        // greedily expand closest nodes
        while (search_queue.has_unexpanded_node()) {
            const Candidate& cur = search_queue.get_closest_unexpanded();

            // iterate neighbors
            {
                std::lock_guard<std::mutex> lock(_graph->neighbor_locks[cur.id]);
                neighbors = _graph->neighbors[cur.id];
            }
            for (auto i=0; i<neighbors.size(); ++i) {

                // prefetch
                if (i+1 < neighbors.size() && visited_set.check(neighbors[i+1]) == false)
                    _base_storage->prefetch_vec_by_id(neighbors[i+1]);

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



    void UniNavGraph::save(std::string index_path_prefix) {
        fs::create_directories(index_path_prefix);
        std::cout << "Saving index to " << index_path_prefix << " ..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        // save meta data
        std::map<std::string, std::string> meta_data;
        statistics();
        meta_data["num_points"] = std::to_string(_num_points);
        meta_data["num_groups"] = std::to_string(_num_groups);
        meta_data["index_name"] = _index_name;
        meta_data["max_degree"] = std::to_string(_max_degree);
        meta_data["Lbuild"] = std::to_string(_Lbuild);
        meta_data["alpha"] = std::to_string(_alpha);
        meta_data["build_num_threads"] = std::to_string(_num_threads);
        meta_data["scenario"] = _scenario;
        meta_data["num_cross_edges"] = std::to_string(_num_cross_edges);
        meta_data["index_time(ms)"] = std::to_string(_index_time);
        meta_data["label_processing_time(ms)"] = std::to_string(_label_processing_time);
        meta_data["build_graph_time(ms)"] = std::to_string(_build_graph_time);
        meta_data["build_LNG_time(ms)"] = std::to_string(_build_LNG_time);
        meta_data["build_cross_edges_time(ms)"] = std::to_string(_build_cross_edges_time);
        meta_data["graph_num_edges"] = std::to_string(_graph_num_edges);
        meta_data["LNG_num_edges"] = std::to_string(_LNG_num_edges);
        meta_data["index_size(MB)"] = std::to_string(_index_size);
        std::string meta_filename = index_path_prefix + "meta";
        write_kv_file(meta_filename, meta_data);

        // save vectors and label sets
        std::string bin_file = index_path_prefix + "vecs.bin";
        std::string label_file = index_path_prefix + "labels.txt";
        _base_storage->write_to_file(bin_file, label_file);

        // save group id to label set
        std::string group_id_to_label_set_filename = index_path_prefix + "group_id_to_label_set";
        write_2d_vectors(group_id_to_label_set_filename, _group_id_to_label_set);

        // save group id to range
        std::string group_id_to_range_filename = index_path_prefix + "group_id_to_range";
        write_2d_vectors(group_id_to_range_filename, _group_id_to_range);

        // save group id to entry point
        std::string group_entry_points_filename = index_path_prefix + "group_entry_points";
        write_1d_vector(group_entry_points_filename, _group_entry_points);

        // save new to old vec ids
        std::string new_to_old_vec_ids_filename = index_path_prefix + "new_to_old_vec_ids";
        write_1d_vector(new_to_old_vec_ids_filename, _new_to_old_vec_ids);

        // save trie index
        std::string trie_filename = index_path_prefix + "trie";
        _trie_index.save(trie_filename);

        // save graph data
        std::string graph_filename = index_path_prefix + "graph";
        _graph->save(graph_filename);

        // print
        std::cout << "- Index saved in " << std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
    }



    void UniNavGraph::load(std::string index_path_prefix, const std::string& data_type) {
        std::cout << "Loading index from " << index_path_prefix << " ..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
            
        // load meta data
        std::string meta_filename = index_path_prefix + "meta";
        auto meta_data = parse_kv_file(meta_filename);
        _num_points = std::stoi(meta_data["num_points"]);

        // load vectors and label sets
        std::string bin_file = index_path_prefix + "vecs.bin";
        std::string label_file = index_path_prefix + "labels.txt";
        _base_storage = create_storage(data_type, false);
        _base_storage->load_from_file(bin_file, label_file);

        // load group id to label set
        std::string group_id_to_label_set_filename = index_path_prefix + "group_id_to_label_set";
        load_2d_vectors(group_id_to_label_set_filename, _group_id_to_label_set);

        // load group id to range
        std::string group_id_to_range_filename = index_path_prefix + "group_id_to_range";
        load_2d_vectors(group_id_to_range_filename, _group_id_to_range);

        // load group id to entry point
        std::string group_entry_points_filename = index_path_prefix + "group_entry_points";
        load_1d_vector(group_entry_points_filename, _group_entry_points);

        // load new to old vec ids
        std::string new_to_old_vec_ids_filename = index_path_prefix + "new_to_old_vec_ids";
        load_1d_vector(new_to_old_vec_ids_filename, _new_to_old_vec_ids);

        // load trie index
        std::string trie_filename = index_path_prefix + "trie";
        _trie_index.load(trie_filename);

        // load graph data
        std::string graph_filename = index_path_prefix + "graph";
        _graph = std::make_shared<Graph>(_base_storage->get_num_points());
        _graph->load(graph_filename);

        // print
        std::cout << "- Index loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
    }

    void UniNavGraph::statistics() {

        // number of edges in the unified navigating graph
        _graph_num_edges = 0;
        for (IdxType i=0; i<_num_points; ++i)
            _graph_num_edges += _graph->neighbors[i].size();

        // number of edges in the label navigating graph
        _LNG_num_edges = 0;
        if (_label_nav_graph != nullptr)            
            for (IdxType i=1; i<=_num_groups; ++i)
                _LNG_num_edges += _label_nav_graph->out_neighbors[i].size();

        // index size
        _index_size = 0;
        for (IdxType i=1; i<=_num_groups; ++i)
            _index_size += _group_id_to_label_set[i].size() * sizeof(LabelType);
        _index_size += _group_id_to_range.size() * sizeof(IdxType) * 2;
        _index_size += _group_entry_points.size() * sizeof(IdxType);
        _index_size += _new_to_old_vec_ids.size() * sizeof(IdxType);
        _index_size += _trie_index.get_index_size();
        _index_size += _graph->get_index_size();

        // return as MB
        _index_size /= 1024 * 1024;
    }
}