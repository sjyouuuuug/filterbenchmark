#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include <sys/time.h>

#include <faiss/IndexACORN.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// added these
#include <arpa/inet.h>
#include <assert.h> /* assert */
#include <faiss/Index.h>
#include <faiss/impl/platform_macros.h>
#include <math.h>
// #include <nlohmann/json.hpp>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <cmath> // for std::mean and std::stdev
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <numeric> // for std::accumulate
#include <set>
#include <sstream> // for ostringstream
#include <thread>
#include "utils.cpp"

// indicate the number of threads to use
// nothing will happen when modifying nthreads
// TODO: set nthreads as default parameter for search function
unsigned int nthreads = 16;
// std::vector<int> search_lists =
//         {5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400};
std::vector<int> search_lists = {
        10,
        20,
        50,
        100,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        700,
        800,
        1000,
        1200};
// compute recall@10
double compute_recall(
        const std::vector<faiss::idx_t>& gt,
        const std::vector<faiss::idx_t>& res,
        int num_queries) {
    assert(gt.size() == res.size());
    int correct = 0;
    for (int i = 0; i < num_queries; i++) {
        std::vector<faiss::idx_t> tmp_gt = std::vector<faiss::idx_t>(
                gt.begin() + i * 10, gt.begin() + (i + 1) * 10);
        std::vector<faiss::idx_t> tmp_res = std::vector<faiss::idx_t>(
                res.begin() + i * 10, res.begin() + (i + 1) * 10);
        std::set<faiss::idx_t> gt_set(tmp_gt.begin(), tmp_gt.end());
        std::set<faiss::idx_t> res_set(tmp_res.begin(), tmp_res.end());
        std::vector<faiss::idx_t> intersection;
        std::set_intersection(
                gt_set.begin(),
                gt_set.end(),
                res_set.begin(),
                res_set.end(),
                std::back_inserter(intersection));
        correct += intersection.size();
    }
    return (double)correct / (num_queries * 10);
}

// create indices for debugging, write indices to file, and get recall stats for
// all queries
int main(int argc, char* argv[]) {
    std::cout << "====================\nSTART: running MULTILABELS_TEST --"
              << nthreads << "cores\n"
              << std::endl;
    // printf("====================\nSTART: running MAKE_INDICES for hnsw
    // --...\n");
    double t0 = elapsed();

    int efc = 40;   // default is 40
    int efs = 48;   //  default is 16
    int k = 10;     // search parameter
    size_t d = 128; // dimension of the vectors to index - will be overwritten
                    // by the dimension of the dataset
    int M;          // HSNW param M TODO change M back
    int M_beta;     // param for compression
    // float attr_sel = 0.001;
    // int gamma = (int) 1 / attr_sel;
    int gamma;
    int n_centroids;
    // int filter = 0;
    std::string dataset;
    int test_partitions = 0;
    int step = 10; // 2

    std::string assignment_type = "rand";
    int alpha = 0;

    srand(0); // seed for random number generator
    int num_trials = 60;

    size_t N = 0;

    int opt;
    { // parse arguments
        if (argc < 6 || argc > 8) {
            fprintf(stderr,
                    "Syntax: %s <number vecs> <gamma> [<assignment_type>] [<alpha>] <dataset> <M> <M_beta>\n",
                    argv[0]);
            exit(1);
        }

        N = strtoul(argv[1], NULL, 10);
        printf("N: %ld\n", N);

        gamma = atoi(argv[2]);
        printf("gamma: %d\n", gamma);

        dataset = argv[3];
        printf("dataset: %s\n", dataset.c_str());

        M = atoi(argv[4]);
        printf("M: %d\n", M);

        M_beta = atoi(argv[5]);
        printf("M_beta: %d\n", M_beta);
    }

    const std::string gt_path = "./Datasets/" + dataset + "/128_graph_search_gt.txt";
    const std::string base_label_path = "./Datasets/" + dataset + "/" +
            dataset + "_label_containment_128_base.txt";
    const std::string aq_path = "./Datasets/" + dataset + "/128_query_graph_search_label.txt";

    // load metadata
    n_centroids = gamma;

    std::vector<std::set<int>> metadata;

    // read in metadata
    metadata = load_multi_label(base_label_path, N);

    printf("[%.3f s] Loaded metadata, %ld attr's found\n",
           elapsed() - t0,
           metadata.size());

    size_t nq;
    float* xq;
    std::vector<std::set<int>> aq;
    { // load query vectors and attributes
        printf("[%.3f s] Loading query vectors and attributes\n",
               elapsed() - t0);

        size_t d2;
        bool is_base = 0;
        std::string filename = "./Datasets/" + dataset + "/128_query_graph_search.fvecs";
        xq = fvecs_read(filename.c_str(), &d2, &nq);
        assert(d == d2 ||
               !"query does not have same dimension as expected 128");
        if (d != d2) {
            d = d2;
        }

        std::cout << "query vecs data loaded, with dim: " << d2 << ", nb=" << nq
                  << std::endl;
        printf("[%.3f s] Loaded query vectors from %s\n",
               elapsed() - t0,
               filename.c_str());

        aq = load_multi_label(aq_path, nq);
        printf("[%.3f s] Loaded %ld %s queries\n",
               elapsed() - t0,
               nq,
               dataset.c_str());
    }

    std::vector<faiss::idx_t> gt;
    std::ifstream gt_file(gt_path);
    if (gt_file.is_open()) {
        std::string line;
        while (std::getline(gt_file, line)) {
            std::istringstream iss(line);
            faiss::idx_t val;
            while (iss >> val) {
                gt.push_back(val);
            }
        }
        gt_file.close();
    } else {
        std::cout << "Unable to open gt file" << std::endl;
    }

    // create normal (base) and hybrid index
    printf("[%.3f s] Index Params -- d: %ld, M: %d, N: %ld, gamma: %d\n",
           elapsed() - t0,
           d,
           M,
           N,
           gamma);

    // ACORN-gamma
    std::vector<int> meta(N);
    faiss::IndexACORNFlat hybrid_index(d, M, gamma, meta, M_beta);
    hybrid_index.acorn.efSearch = efs; // default is 16 HybridHNSW.capp
    debug("ACORN index created%s\n", "");

    // ACORN-1
    faiss::IndexACORNFlat hybrid_index_gamma1(d, M, 1, meta, M * 2);
    hybrid_index_gamma1.acorn.efSearch = efs; // default is 16 HybridHNSW.capp

    { // populating the database
        std::cout << "====================Vectors====================\n"
                  << std::endl;

        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        bool is_base = 1;
        std::string filename = get_file_name(dataset, is_base);
        float* xb = fvecs_read(filename.c_str(), &d2, &nb);
        assert(d == d2 || !"dataset does not dim 128 as expected");
        printf("[%.3f s] Loaded base vectors from file: %s\n",
               elapsed() - t0,
               filename.c_str());

        std::cout << "data loaded, with dim: " << d2 << ", nb=" << nb
                  << std::endl;

        printf("[%.3f s] Indexing database, size %ld*%ld from max %ld\n",
               elapsed() - t0,
               N,
               d2,
               nb);

        printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);

        double t1 = elapsed();
        hybrid_index.add(N, xb);
        double t2 = elapsed();
        std::cout << "Create gamma index in time: " << t2 - t1 << std::endl;
        printf("[%.3f s] Vectors added to hybrid index \n", elapsed() - t0);
        std::cout << "Hybrid index vectors added" << nb << std::endl;

        double t1_gamma1 = elapsed();
        hybrid_index_gamma1.add(N, xb);
        double t2_gamma1 = elapsed();
        std::cout << "Create gamma=1 index in time: " << t2_gamma1 - t1_gamma1
                  << std::endl;
        printf("[%.3f s] Vectors added to hybrid index with gamma=1 \n",
               elapsed() - t0);
        std::cout << "Hybrid index with gamma=1 vectors added" << nb
                  << std::endl;

        delete[] xb;
    }

    // write hybrid index and partition indices to files
    {
        std::cout << "====================Write Index====================\n"
                  << std::endl;
        std::stringstream filepath_stream;

        filepath_stream << "./tmp/" << dataset << "/hybrid_containment"
                        << "_M=" << M << "_efc" << efc << "_Mb=" << M_beta
                        << "_gamma=" << gamma << ".json";

        std::string filepath = filepath_stream.str();
        write_index(&hybrid_index, filepath.c_str());
        printf("[%.3f s] Wrote hybrid index to file: %s\n",
               elapsed() - t0,
               filepath.c_str());

        // write hybrid_gamma1 index
        std::stringstream filepath_stream2;
        filepath_stream2 << "./tmp/" << dataset << "/hybrid_containment"
                         << "_M=" << M << "_efc" << efc << "_Mb=" << M_beta
                         << "_gamma=" << 1 << ".json";

        std::string filepath2 = filepath_stream2.str();
        write_index(&hybrid_index_gamma1, filepath2.c_str());
        printf("[%.3f s] Wrote hybrid_gamma1 index to file: %s\n",
               elapsed() - t0,
               filepath2.c_str());
    }

    { // print out stats
        printf("====================================\n");
        printf("============ ACORN INDEX =============\n");
        printf("====================================\n");
        hybrid_index.printStats(false);
    }

    printf("==============================================\n");
    printf("====================Search Results====================\n");
    printf("==============================================\n");
    printf("==============================================\n");
    printf("====================Search====================\n");
    printf("==============================================\n");
    double t1 = elapsed();
    std::vector<double> qps_gamma;
    std::vector<double> qps_gamma_filter;
    std::vector<double> dist_cmps_gamma;
    std::vector<double> recall_at_10_gamma;

    std::vector<double> qps_gamma1;
    std::vector<double> qps_gamma1_filter;
    std::vector<double> dist_cmps_gamma1;
    std::vector<double> recall_at_10_gamma_1;

    std::vector<char> filter_ids_map(nq * N);

    double t1_x = elapsed();
    omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic, 1)
    for (int xq = 0; xq < nq; xq++) {
        for (int xb = 0; xb < N; xb++) {
            filter_ids_map[xq * N + xb] = (bool)(std::includes(
                    metadata[xb].begin(),
                    metadata[xb].end(),
                    aq[xq].begin(),
                    aq[xq].end()));
        }
    }
    double t2_x = elapsed();
    double filter_time = t2_x - t1_x;

    // searching the hybrid database
    for (auto& ef_search : search_lists) {
        hybrid_index.acorn.efSearch = ef_search;
        std::cout << "==================== ACORN INDEX ===================="
                  << std::endl;
        std::cout << "Searching the " << k << " nearest neighbors of " << nq
                  << " vectors in the index, efsearch "
                  << hybrid_index.acorn.efSearch << std::endl;

        std::vector<faiss::idx_t> nns2(k * nq);
        std::vector<float> dis2(k * nq);

        // create filter_ids_map, ie a bitmap of the ids that are in the filter
        // std::includes
        // accept iff the query attributes are a subset of the base attributes

        std::cout << "Starting search" << std::endl;
        double t1_x2 = elapsed();
        hybrid_index.search(
                nq,
                xq,
                k,
                dis2.data(),
                nns2.data(),
                filter_ids_map.data()); // TODO change first argument back to nq
        double t2_x = elapsed();

        double search_time = t2_x - t1_x2;
        double search_time_filter = search_time + filter_time;
        std::cout << "Search time: " << search_time << std::endl;
        std::cout << "Search time containing filter: " << search_time_filter
                  << std::endl;

        std::cout << "QPS: " << nq / search_time << std::endl;
        std::cout << "QPS containing filter: " << nq / search_time_filter
                  << std::endl;

        qps_gamma.push_back(nq / search_time);
        qps_gamma_filter.push_back(nq / search_time_filter);

        double recall = compute_recall(gt, nns2, nq);
        recall_at_10_gamma.push_back(recall);

        const faiss::ACORNStats& stats = faiss::acorn_stats;

        printf("average distance computations per query: %f\n",
               (float)stats.n3 / stats.n1);
        dist_cmps_gamma.push_back((float)stats.n3 / stats.n1);
    }

    // searching the hybrid_gamma1 database
    for (auto& ef_search : search_lists) {
        hybrid_index_gamma1.acorn.efSearch = ef_search;
        std::cout
                << "==================== ACORN INDEX GAMMA=1 ===================="
                << std::endl;
        std::cout << "Searching the " << k << " nearest neighbors of " << nq
                  << " vectors in the index, efsearch "
                  << hybrid_index_gamma1.acorn.efSearch << std::endl;

        std::vector<faiss::idx_t> nns2(k * nq);
        std::vector<float> dis2(k * nq);

        std::cout << "Starting search" << std::endl;
        double t1_x2 = elapsed();
        hybrid_index_gamma1.search(
                nq, xq, k, dis2.data(), nns2.data(), filter_ids_map.data());
        double t2_x = elapsed();

        double search_time = t2_x - t1_x2;
        double search_time_filter = search_time + filter_time;

        std::cout << "Search time: " << search_time << std::endl;
        std::cout << "Search time containing filter: " << search_time_filter
                  << std::endl;

        std::cout << "QPS: " << nq / search_time << std::endl;
        std::cout << "QPS containing filter: " << nq / search_time_filter
                  << std::endl;

        qps_gamma1.push_back(nq / search_time);
        qps_gamma1_filter.push_back(nq / search_time_filter);

        double recall = compute_recall(gt, nns2, nq);
        recall_at_10_gamma_1.push_back(recall);
        const faiss::ACORNStats& stats = faiss::acorn_stats;
        printf("average distance computations per query: %f\n",
               (float)stats.n3 / stats.n1);
        dist_cmps_gamma1.push_back((float)stats.n3 / stats.n1);
    }

    // print overall stats
    std::cout << "====================Overall Stats===================="
              << std::endl;
    std::cout << "====================ACORN INDEX===================="
              << std::endl;
    std::cout
            << "Search lists       QPS      QPS(filter)      dist_cmp    recall@10"
            << std::endl;
    assert(qps_gamma.size() == search_lists.size());
    assert(qps_gamma_filter.size() == search_lists.size());
    for (int i = 0; i < search_lists.size(); i++) {
        std::cout << search_lists[i] << "        " << qps_gamma[i] << "        "
                  << qps_gamma_filter[i] << "        " << dist_cmps_gamma[i]
                  << "    " << recall_at_10_gamma[i] << std::endl;
    }

    std::cout << "====================ACORN INDEX GAMMA=1===================="
              << std::endl;
    std::cout
            << "Search lists       QPS      QPS(filter)      dist_cmp    recall@10"
            << std::endl;
    assert(qps_gamma1.size() == search_lists.size());
    assert(qps_gamma1_filter.size() == search_lists.size());
    for (int i = 0; i < search_lists.size(); i++) {
        std::cout << search_lists[i] << "        " << qps_gamma1[i]
                  << "        " << qps_gamma1_filter[i] << "        "
                  << dist_cmps_gamma1[i] << "    " << recall_at_10_gamma_1[i]
                  << std::endl;
    }

    printf("[%.3f s] -----DONE-----\n", elapsed() - t0);
}