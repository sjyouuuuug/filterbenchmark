#ifndef ANNS_STORAGE_H
#define ANNS_STORAGE_H

#include <limits>
#include <string>
#include <vector>
#include <memory>
#include <xmmintrin.h>
#include <immintrin.h>
#include "config.h"
#include "distance.h"
#include <set>


namespace ANNS {

    // interface for storage
    class IStorage {
        public:
            virtual ~IStorage() = default;

            // I/O
            virtual void load_from_file(const std::string& bin_file, const std::string& label_file, 
                                        IdxType max_num_points = std::numeric_limits<IdxType>::max()) = 0;
            virtual void load_from_memory(IdxType input_dim, IdxType input_num, float* vecs_data, 
                                    const std::vector<std::set<int>>& labels, IdxType max_num_points = std::numeric_limits<IdxType>::max()) = 0;
            virtual void write_to_file(const std::string& bin_file, const std::string& label_file) = 0;

            // reorder the vector data
            virtual void reorder_data(const std::vector<IdxType>& new_to_old_ids) = 0;

            // get statistics
            virtual DataType get_data_type() const = 0;
            virtual IdxType get_num_points() const = 0;
            virtual IdxType get_dim() const = 0;

            // get data
            virtual std::vector<LabelType>* get_offseted_label_sets(IdxType idx) = 0;
            virtual char* get_vector(IdxType idx) = 0;
            virtual std::vector<LabelType>& get_label_set(IdxType idx) = 0;
            virtual inline void prefetch_vec_by_id(IdxType idx) const = 0;

            // obtain a point cloest to the center
            virtual IdxType choose_medoid(uint32_t num_threads, std::shared_ptr<DistanceHandler> distance_handler) = 0;

            // clean
            virtual void clean() = 0;
    };
    

    // obtain corresponding storage class
    std::shared_ptr<IStorage> create_storage(const std::string& data_type, bool verbose = true);
    std::shared_ptr<IStorage> create_storage(std::shared_ptr<IStorage> storage, IdxType start, IdxType end);


    // storage class
    template<typename T>
    class Storage : public IStorage {

        public:
            Storage(DataType data_type, bool verbose);
            Storage(std::shared_ptr<IStorage> storage, IdxType start, IdxType end);
            ~Storage() = default;

            // I/O
            void load_from_file(const std::string& bin_file, const std::string& label_file, IdxType max_num_points);
            void load_from_memory(IdxType input_dim, IdxType input_num, float* vecs_data, 
                                    const std::vector<std::set<int>>& labels, IdxType max_num_points);
            void write_to_file(const std::string& bin_file, const std::string& label_file);

            // reorder the vector data
            void reorder_data(const std::vector<IdxType>& new_to_old_ids);

            // get statistics
            DataType get_data_type() const { return data_type; };
            IdxType get_num_points() const { return num_points; };
            IdxType get_dim() const { return dim; };

            // get data
            std::vector<LabelType>* get_offseted_label_sets(IdxType idx) { return label_sets + idx; }
            char* get_vector(IdxType idx) { return reinterpret_cast<char *>(vecs + idx * dim); }
            std::vector<LabelType>& get_label_set(IdxType idx) { return label_sets[idx]; }
            inline void prefetch_vec_by_id(IdxType idx) const {
                for (size_t d = 0; d < prefetch_byte_num; d += 64) _mm_prefetch((const char *)(vecs + idx * dim) + d, _MM_HINT_T0);
            }

            // obtain a point cloest to the center
            IdxType choose_medoid(uint32_t num_threads, std::shared_ptr<DistanceHandler> distance_handler);

            // clean
            void clean() {
                if (vecs)
                    delete[] vecs;
                if (label_sets)
                    delete[] label_sets;
            }

        private:
            DataType data_type;
            IdxType num_points, dim;
            T* vecs = nullptr;
            size_t prefetch_byte_num;
            std::vector<LabelType>* label_sets = nullptr;

            // for logs
            bool verbose;
    };
}



#endif // ANNS_STORAGE_H