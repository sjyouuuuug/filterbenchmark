#ifndef VAMANA_H
#define VAMANA_H

#include "storage.h"
#include "distance.h"
#include "graph.h"
#include "search_cache.h"


namespace ANNS {

    class Vamana {
            
        public:
            Vamana(bool verbose = true) { _verbose = verbose; };
            Vamana(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler,
                   std::shared_ptr<Graph> graph, IdxType entry_point) 
                        : _base_storage(base_storage), _distance_handler(distance_handler),
                        _graph(graph), _entry_point(entry_point), _verbose(false) {}
            ~Vamana() = default;

            void build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler, 
                       std::shared_ptr<Graph> graph, IdxType max_degree, IdxType Lbuild, float alpha, 
                       uint32_t num_threads, IdxType max_candidate_size = default_paras::MAX_CANDIDATE_SIZE);

            void search(std::shared_ptr<IStorage> base_storage, std::shared_ptr<IStorage> query_storage, 
                        std::shared_ptr<DistanceHandler> distance_handler, IdxType K, IdxType Lsearch, 
                        uint32_t num_threads, std::pair<IdxType, float>* results, std::vector<IdxType>& num_cmps);

            // search the graph
            IdxType get_entry_point() { return _entry_point; }
            IdxType iterate_to_fixed_point(const char* query, std::shared_ptr<SearchCache> search_cache, 
                                           bool record_expanded = false, IdxType target_id = -1);

            // stats and I/O
            void statistics();
            void save(std::string& index_path_prefix);
            void load(std::string& index_path_prefix, std::shared_ptr<Graph> graph);

        private:

            // data
            std::shared_ptr<IStorage> _base_storage, _query_storage;
            std::shared_ptr<DistanceHandler> _distance_handler;

            // build parameters
            IdxType _max_degree, _Lbuild, _max_candidate_size;
            float _alpha;
            uint32_t _num_threads;

            // build the graph
            IdxType _entry_point;
            std::shared_ptr<Graph> _graph;
            void link();
            void prune_neighbors(IdxType id, std::vector<Candidate>& candidates, std::vector<IdxType>& pruned_list, 
                                 std::shared_ptr<SearchCache> search_cache);
            void inter_insert(IdxType src, std::vector<IdxType>& src_neighbors, std::shared_ptr<SearchCache> search_cache);

            // for logs
            bool _verbose;
    };
}

#endif // VAMANA_H