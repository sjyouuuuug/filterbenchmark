#ifndef VISITED_SET_H
#define VISITED_SET_H

#include <cstring>
#include "config.h"



namespace ANNS {
    class VisitedSet {
        public:
            VisitedSet() = default;

            void init(IdxType num_elements) {
                _curValue = -1;
                _num_elements = num_elements;
                if (_marks != nullptr)
                    delete[] _marks;
                _marks = new MarkType[num_elements];
            }

            void clear() {
                _curValue++;
                if (_curValue == 0) {
                    memset(_marks, 0, sizeof(MarkType) * _num_elements);
                    _curValue++;
                }
            }

            inline void prefetch(IdxType idx) const {
                _mm_prefetch((char *)_marks + idx, _MM_HINT_T0);
            }

            inline void set(IdxType idx) { 
                _marks[idx] = _curValue; 
            }

            inline bool check(IdxType idx) const { 
                return _marks[idx] == _curValue; 
            }

            ~VisitedSet() { 
                delete[] _marks; 
            }

        private:
            MarkType _curValue;
            MarkType* _marks = nullptr;
            IdxType _num_elements;
    };
}

#endif // VISITED_SET_H