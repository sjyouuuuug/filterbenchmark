#include <cstring>
#include "utils.h"
#include "search_queue.h"


namespace ANNS {
            
    // reserve the capacity
    void SearchQueue::reserve(int32_t capacity) {
        if (capacity + 1 > _data.size()) 
            _data.resize(capacity + 1);
        _capacity = capacity;
    }


    // check whether exists an id
    bool SearchQueue::exist(IdxType id) const {
        for (int32_t i = 0; i < _size; i++)
            if (_data[i].id == id)
                return true;
        return false;
    }


    // insert a candidate
    void SearchQueue::insert(IdxType id, float distance) {
        Candidate new_candidate(id, distance);
        if (_size == _capacity && _data[_size - 1] < new_candidate)
            return;

        // binary search
        int32_t lo = 0, hi = _size;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (new_candidate < _data[mid])
                hi = mid;
            else if (UNLIKELY(_data[mid].id == new_candidate.id))   // ensure the same id is not in the set
                return;
            else
                lo = mid + 1;
        }

        // move the elements
        if (lo < _capacity)
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Candidate));
        _data[lo] = new_candidate;

        // update size and currently unexpanded candidate
        if (_size < _capacity)
            _size++;
        if (lo < _cur_unexpanded)
            _cur_unexpanded = lo;
    }


    // get the closest unexpanded node
    const Candidate& SearchQueue::get_closest_unexpanded() {
        _data[_cur_unexpanded].expanded = true;
        auto pre = _cur_unexpanded;
        while (_cur_unexpanded < _size && _data[_cur_unexpanded].expanded)
            _cur_unexpanded++;
        return _data[pre];
    }
}