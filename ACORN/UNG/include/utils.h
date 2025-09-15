#ifndef UTILS_H
#define UTILS_H

#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "config.h"


#define SEP_LINE "------------------------------------------------------------\n"

// likely and unlikely prediction
#define LIKELY(x) __builtin_expect(x, 1)
#define UNLIKELY(x) __builtin_expect(x, 0)

namespace ANNS {

    // write and load key-value file
    void write_kv_file(const std::string& filename, const std::map<std::string, std::string>& kv_map);
    std::map<std::string, std::string> parse_kv_file(const std::string& filename);

    // write and load groundtruth file
    void write_gt_file(const std::string& filename, const std::pair<IdxType, float>* gt, uint32_t num_queries, uint32_t K);
    void load_gt_file(const std::string& filename, std::pair<IdxType, float>* gt, uint32_t num_queries, uint32_t K);

    // calculated recall
    float calculate_recall(const std::pair<IdxType, float>* gt, const std::pair<IdxType, float>* res, uint32_t num_queries, uint32_t K);

    // write 1D-std::vector
    template<typename T>
    void write_1d_vector(const std::string& filename, const std::vector<T>& vec) {
        std::ofstream out(filename);
        for (auto& idx : vec)
            out << idx << std::endl;
    }

    // load 1D-std::vector
    template<typename T>
    void load_1d_vector(const std::string& filename, std::vector<T>& vec) {
        std::ifstream in(filename);
        T value;
        vec.clear();
        while (in >> value)
            vec.push_back(value);
    }

    // write 2D-std::vector
    template<typename T>
    void write_2d_vectors(const std::string& filename, const std::vector<std::vector<T>>& vecs) {
        std::ofstream out(filename);
        for (auto& vec : vecs) {
            for (auto& idx : vec)
                out << idx << " ";
            out << std::endl;
        }
    }

    // load 2D-std::vector
    template<typename T>
    void load_2d_vectors(const std::string& filename, std::vector<std::vector<T>>& vecs) {
        std::ifstream in(filename);
        std::string line;
        vecs.clear();
        while (std::getline(in, line)) {
            std::istringstream iss(line);
            std::vector<T> vec;
            T value;
            while (iss >> value)
                vec.push_back(value);
            vecs.push_back(vec);
        }
    }

    // write 2D-std::vector
    template<typename T>
    void write_2d_vectors(const std::string& filename, const std::vector<std::pair<T, T>>& vecs) {
        std::ofstream out(filename);
        for (auto& each : vecs)
            out << each.first << " " << each.second << std::endl;
    }

    // load 2D-std::vector
    template<typename T>
    void load_2d_vectors(const std::string& filename, std::vector<std::pair<T, T>>& vecs) {
        std::ifstream in(filename);
        std::string line;
        vecs.clear();
        while (std::getline(in, line)) {
            std::istringstream iss(line);
            T first, second;
            iss >> first >> second;
            vecs.push_back(std::make_pair(first, second));
        }
    }
}

#endif // UTILS_H