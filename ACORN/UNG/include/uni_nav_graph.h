#ifndef UNG_H
#define UNG_H

#include "trie.h"
#include "graph.h"
#include "storage.h"
#include "distance.h"
#include "search_cache.h"
#include "label_nav_graph.h"
#include "vamana.h"


namespace ANNS {

    class UniNavGraph {
        public:
            UniNavGraph() = default;
            ~UniNavGraph() = default;

            void build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler, 
                       std::string scenario, std::string index_name, uint32_t num_threads, IdxType num_cross_edges,
                       IdxType max_degree, IdxType Lbuild, float alpha);
            
            void search(std::shared_ptr<IStorage> query_storage, std::shared_ptr<DistanceHandler> distance_handler, 
                        uint32_t num_threads, IdxType Lsearch, IdxType num_entry_points, std::string scenario,
                        IdxType K, std::pair<IdxType, float>* results, std::vector<float>& num_cmps);

            // I/O
            void save(std::string index_path_prefix);
            void load(std::string index_path_prefix, const std::string& data_type);

        private:

            // data
            std::shared_ptr<IStorage> _base_storage, _query_storage;
            std::shared_ptr<DistanceHandler> _distance_handler;
            std::shared_ptr<Graph> _graph;
            IdxType _num_points;

            // trie index and vector groups
            IdxType _num_groups;
            TrieIndex _trie_index;
            std::vector<IdxType> _new_vec_id_to_group_id;
            std::vector<std::vector<IdxType>> _group_id_to_vec_ids;
            std::vector<std::vector<LabelType>> _group_id_to_label_set;
            void build_trie_and_divide_groups();

            // label navigating graph
            std::shared_ptr<LabelNavGraph> _label_nav_graph = nullptr;
            void get_min_super_sets(const std::vector<LabelType>& query_label_set, std::vector<IdxType>& min_super_set_ids, 
                                    bool avoid_self=false, bool need_containment=true);
            void build_label_nav_graph();

            // prepare vector storage for each group
            std::vector<IdxType> _new_to_old_vec_ids;
            std::vector<std::pair<IdxType, IdxType>> _group_id_to_range;
            std::vector<std::shared_ptr<IStorage>> _group_storages;
            void prepare_group_storages_graphs();

            // graph indices for each graph
            std::string _index_name;
            std::vector<std::shared_ptr<Graph>> _group_graphs;
            std::vector<IdxType> _group_entry_points;
            void build_graph_for_all_groups();
            void build_complete_graph(std::shared_ptr<Graph> graph, IdxType num_points);
            std::vector<std::shared_ptr<Vamana>> _vamana_instances;

            // index parameters for each graph
            IdxType _max_degree, _Lbuild;
            float _alpha;
            uint32_t _num_threads;
            std::string _scenario;

            // cross-group edges
            IdxType _num_cross_edges;
            std::vector<SearchQueue> _cross_group_neighbors;
            void build_cross_group_edges();

            // obtain the final unified navigating graph
            void add_offset_for_uni_nav_graph();

            // obtain entry_points
            std::vector<IdxType> get_entry_points(const std::vector<LabelType>& query_label_set, 
                                                  IdxType num_entry_points, VisitedSet& visited_set);
            void get_entry_points_given_group_id(IdxType num_entry_points, VisitedSet& visited_set, 
                                                 IdxType group_id, std::vector<IdxType>& entry_points);

            // search in graph
            IdxType iterate_to_fixed_point(const char* query, std::shared_ptr<SearchCache> search_cache, 
                                           IdxType target_id, const std::vector<IdxType>& entry_points,
                                           bool clear_search_queue=true, bool clear_visited_set=true);

            // statistics
            float _index_time, _label_processing_time, _build_graph_time;
            float _build_LNG_time = 0, _build_cross_edges_time = 0, _index_size;
            IdxType _graph_num_edges, _LNG_num_edges;
            void statistics();
    };
}

#endif // UNG_H