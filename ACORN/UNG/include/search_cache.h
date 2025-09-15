#ifndef SEARCH_CACHE_H
#define SEARCH_CACHE_H

#include <mutex>
#include <deque>
#include "visited_set.h"
#include "search_queue.h"



namespace ANNS {

    struct SearchCache {
        SearchQueue search_queue;
        VisitedSet visited_set;
        std::vector<Candidate> expanded_list;
        std::vector<float> occlude_factor;

        SearchCache(IdxType visited_set_size, int32_t search_queue_capacity) {
            search_queue.reserve(search_queue_capacity);
            visited_set.init(visited_set_size);
        }
    };


    class SearchCacheList {
        public:
            SearchCacheList(uint32_t num_cache, IdxType visited_set_size, int32_t search_queue_capacity) {
                _visited_set_size = visited_set_size;
                _search_queue_capacity = search_queue_capacity;
                for (uint32_t i = 0; i < num_cache; i++)
                    pool.emplace_back(std::make_shared<SearchCache>(visited_set_size, search_queue_capacity));
            }

            std::shared_ptr<SearchCache> get_free_cache() {
                std::unique_lock<std::mutex> lock(pool_guard);
                if (pool.empty())
                    return std::make_shared<SearchCache>(_visited_set_size, _search_queue_capacity);
                auto cache = pool.front();
                pool.pop_front();
                return cache;
            }

            void release_cache(std::shared_ptr<SearchCache> cache) {
                std::unique_lock<std::mutex> lock(pool_guard);
                pool.push_back(cache);
            }

            ~SearchCacheList() = default;

        private:
            std::deque<std::shared_ptr<SearchCache>> pool;
            std::mutex pool_guard;
            IdxType _visited_set_size;
            int32_t _search_queue_capacity;
    };
}

#endif // SEARCH_CACHE_H