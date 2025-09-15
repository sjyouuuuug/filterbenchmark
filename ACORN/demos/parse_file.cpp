#include <arpa/inet.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#include <vector>
#include <chrono>

#include <faiss/Index.h>
#include <faiss/IndexACORN.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/index_io.h>

#include "filtered_bruteforce.h"
#include "utils.h"
#include "utils.cpp"

bool file_exist(const std::string& path) {
    if (FILE* file = fopen(path.c_str(), "r")) {
        fclose(file);
        return true;
    }
    return false;
}

// 搜索参数
const unsigned int nthreads = 16;
std::vector<float> compute_selectivity(
        const std::vector<std::set<int>>& base_label,
        const std::vector<std::set<int>>& query_label,
        const std::string scenario) {
    const size_t nq = query_label.size();
    const size_t N = base_label.size();
    std::vector<float> selectivity(nq, 0.0f);

    for (size_t q = 0; q < nq; ++q) {
        const std::set<int>& q_set = query_label[q];
        size_t count = 0;

        omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t b = 0; b < N; ++b) {
            const std::set<int>& b_set = base_label[b];
            bool satisfy = false;

            if (scenario == "and") {
                // Check if base set includes all elements of query set
                satisfy = std::includes(
                        b_set.begin(), b_set.end(), q_set.begin(), q_set.end());
            } else if (scenario == "or") {
                // Check if there's any intersection
                const auto& smaller =
                        b_set.size() < q_set.size() ? b_set : q_set;
                const auto& larger =
                        b_set.size() < q_set.size() ? q_set : b_set;

                for (int elem : smaller) {
                    if (larger.find(elem) != larger.end()) {
                        satisfy = true;
                        break;
                    }
                }
            } else if (scenario == "equal") {
                // Check if sets are identical
                satisfy = (b_set == q_set);
            } else {
                std::cerr << "Invalid scenario: " << scenario << std::endl;
                exit(1);
            }

            if (satisfy) {
                ++count;
            }
        }

        selectivity[q] = static_cast<float>(count) / static_cast<float>(N);
    }

    return selectivity;
}

// parse queries into 2 parts according to threshold.
std::vector<int> parse_xq(
        float* xq,
        int dim,
        const std::vector<float>& selectivity,
        float thres,
        float* xq_above,
        float* xq_under,
        int& num_above,
        int& num_under) {
    assert(xq != nullptr && xq_above != nullptr && xq_under != nullptr);
    assert(dim > 0 && !selectivity.empty());

    const size_t nq = selectivity.size();
    std::vector<int> parse(nq, 0);

    size_t above_cnt = 0;
    size_t under_cnt = 0;

    for (size_t qid = 0; qid < nq; ++qid) {
        const float* src = xq + qid * dim;

        if (selectivity[qid] >= thres) {
            memcpy(xq_above + above_cnt * dim, src, dim * sizeof(float));
            ++above_cnt;
            parse[qid] = 1;
        } else {
            memcpy(xq_under + under_cnt * dim, src, dim * sizeof(float));
            ++under_cnt;
        }
    }

    num_above = above_cnt;
    num_under = under_cnt;

    std::cout << "Split " << nq << " queries into: " << above_cnt << " above, "
              << under_cnt << " under threshold\n";

    return parse;
}

int main(int argc, char* argv[]) {
    std::cout << "==================== START: running SEARCH_ACORN_INDEX --"
              << nthreads << " cores ====================" << std::endl;
    double t0 = elapsed();

    int efs = 48;
    int k = 10;
    size_t d = 128;
    int gamma;
    std::string dataset;
    size_t N = 0;

    if (argc != 14) {
        std::cerr
                << "Usage: " << argv[0]
                << " <N> <gamma> <dataset> <M> <M_beta> <scenario> <output_path_prefix> "
                << "<base_file> <base_label> <query_file> <query_label> "
                << std::endl;
        return 1;
    }

    N = strtoul(argv[1], NULL, 10);
    gamma = atoi(argv[2]);
    dataset = argv[3];
    const std::string scenario = argv[4];
    const std::string output_path_prefix = argv[5];
    const std::string base_file = argv[6];
    const std::string base_label = argv[7];
    const std::string query_file = argv[8];
    const std::string query_label = argv[9];
   
    std::cout << "Parameters:\n"
              << "  N: " << N << "\n"
              << "  gamma: " << gamma << "\n"
              << "  dataset: " << dataset << "\n"
              << "  index_path: " << index_path << "\n"
              << "  scenario: " << scenario << "\n"
              << "  output_path: " << output_path << std::endl;

    for (const auto& path : {gt_path, base_label, query_label}) {
        if (file_exist(path)) {
            std::cout << path << " exists." << std::endl;
        } else {
            std::cerr << path << " does not exist." << std::endl;
            return 1;
        }
    }

    std::vector<std::set<int>> metadata = load_multi_label(base_label, N);

    size_t nq;
    float* xq;
    std::vector<std::set<int>> aq;
    {
        printf("[%.3f s] Loading query vectors and attributes\n",
               elapsed() - t0);
        size_t d2;
        xq = fvecs_read(query_file.c_str(), &d2, &nq);
        std::cout << "nq: " << nq << ", d: " << d2 << std::endl;

        if (d != d2) {
            d = d2;
        }

        aq = load_multi_label(query_label, nq);
        printf("[%.3f s] Loaded %ld %s queries\n",
               elapsed() - t0,
               nq,
               dataset.c_str());
        // std::cout << "dim: " << d2 << " first xq: " << xq[0] << std::endl;
    }

    std::vector<float> selectivity;
    selectivity = compute_selectivity(metadata, aq, scenario);
    // parse brute force search and graph search
    float* xq_bf = new float[nq * d];
    float* xq_gs = new float[nq * d];
    int num_bf = 0;
    int num_gs = 0;
    std::vector<int> parse(nq, 0);
    double t1_filter = elapsed();
    if (gamma != 1) {
        // printf("dim: %ld\n", d);
        parse = parse_xq(
                xq,
                d,
                selectivity,
                static_cast<float>(1) / static_cast<float>(gamma),
                xq_gs,
                xq_bf,
                num_gs,
                num_bf);
    }
    double t2_filter = elapsed();

    if (gamma == 1){
        // if gamma = 1, use graph search only.
        memcpy(xq_gs, xq, sizeof(nq * d));
        num_bf = 0;
        num_gs = nq;
        for(auto &p:parse){
            // use graph search only;
            p = 1;
        }
    }

    std::cout << "Number of points brute force: " << num_bf
              << " Number of points graph search: " << num_gs << std::endl;

    delete[] xq;
    delete[] xq_bf;
    delete[] xq_gs;

    printf("[%.3f s] ----- DONE -----\n", elapsed() - t0);

    return 0;
}