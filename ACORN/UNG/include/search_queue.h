#ifndef SEARCH_QUQUE
#define SEARCH_QUQUE

#include <vector>
#include <memory>
#include "config.h"


namespace ANNS {

    // for storing each candidate, prefer those with minimal distances
    struct Candidate {
        IdxType id;
        float distance;
        bool expanded;

        Candidate() = default;
        Candidate(IdxType a, float b) : id{a}, distance{b}, expanded(false) {}

        inline bool operator<(const Candidate &other) const {
            return distance < other.distance || (distance == other.distance && id < other.id);
        }
        inline bool operator==(const Candidate &other) const { return (id == other.id); }
    };


    // search queue for ANNS, preserve the closest vectors
    class SearchQueue {
        
        public:
            SearchQueue() : _size(0), _capacity(0), _cur_unexpanded(0) {};
            ~SearchQueue() = default;

            // size
            int32_t size() const { return _size; };
            int32_t capacity() const { return _capacity; };
            void reserve(int32_t capacity);

            // read and write
            Candidate operator[](int32_t idx) const { return _data[idx]; }
            Candidate& operator[](int32_t idx) { return _data[idx]; };
            bool exist(IdxType id) const;
            void insert(IdxType id, float distance);
            void clear() { _size = 0; _cur_unexpanded = 0; };

            // expand
            bool has_unexpanded_node() const { return _cur_unexpanded < _size; };
            const Candidate& get_closest_unexpanded();
            
        private:

            int32_t _size, _capacity, _cur_unexpanded;
            std::vector<Candidate> _data;
    };
}

#endif // SEARCH_QUQUE