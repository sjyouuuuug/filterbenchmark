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
#include <chrono>
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
#include <bitset>

#include <faiss/Index.h>
#include <faiss/IndexACORN.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/index_io.h>

#include "filtered_bruteforce.h"
#include "utils.cpp"
#include "utils.h"

#define BITSET_MAX_SIZE 1200000

bool file_exist(const std::string& path) {
    if (FILE* file = fopen(path.c_str(), "r")) {
        fclose(file);
        return true;
    }
    return false;
}

// 搜索参数
const unsigned int nthreads = 16;
const std::vector<int> search_lists = {
        100,
        300,
        500,
        700,
        800,
        1200};

double compute_recall(
        const std::vector<faiss::idx_t>& gt,
        const std::vector<faiss::idx_t>& res,
        int num_queries,
        const std::vector<int>& parse,
        int recall_at) {
    if (num_queries == 0) {
        return 0;
    }

    // assert(parse_count * recall_at == gt.size());
    int total_num = parse.size();
    if (total_num * recall_at != gt.size()) {
        std::cerr << "Error in gt" << std::endl;
    }
    int count = std::count(parse.begin(), parse.end(), 1);
    std::cout << "num 1's in parse: " << count << std::endl;

    std::cout << "gt size: " << gt.size() << std::endl;
    std::cout << "parse size: " << parse.size() << std::endl;
    std::cout << "res size: " << res.size() << std::endl;
    std::cout << "num queries: " << num_queries << std::endl;

    int correct = 0;
    int adjusted_query_index = 0;

    for (int i = 0; i < total_num; i++) {
        if (parse[i] == 0) {
            continue;
        }

        std::vector<faiss::idx_t> tmp_gt(
                gt.begin() + i * recall_at, gt.begin() + (i + 1) * recall_at);
        std::vector<faiss::idx_t> tmp_res(
                res.begin() + adjusted_query_index * recall_at,
                res.begin() + (adjusted_query_index + 1) * recall_at);

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
        adjusted_query_index++;
    }

    return static_cast<double>(correct) / (num_queries * recall_at);
}


std::vector<std::bitset<BITSET_MAX_SIZE>> get_label_ivf_from_id2labels(
        std::vector<std::set<int>>& id2labels) {
    int nb = id2labels.size();
    if (nb > BITSET_MAX_SIZE) {
        std::cerr << "Error! number of points should not excceed "
                  << BITSET_MAX_SIZE << std::endl;
    }
    int max_label = -1;
    for (auto& lbls : id2labels) {
        for (auto& lbl : lbls) {
            if (lbl > max_label) {
                max_label = lbl;
            }
        }
    }
    std::vector<std::bitset<BITSET_MAX_SIZE>> bitset_labels(max_label + 1);
    for (int i = 0; i < nb; ++i) {
        for (auto& lbl : id2labels[i]) {
            bitset_labels[lbl].set(i);
        }
    }

    return bitset_labels;
}

void create_filter_map_fake(
        const std::vector<std::bitset<BITSET_MAX_SIZE>>& bitset_labels,
        const std::vector<std::set<int>>& base_labels_set,
        const std::vector<std::set<int>>& query_labels_set,
        size_t nq,
        size_t N,
        std::vector<char>& filter_ids_map,
        const std::string scenario) {
    // filter_ids_map.resize(N * nq, 0);
    if (scenario == "and") {
        omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < nq; ++i) {
            auto lbl = query_labels_set[i];
            std::bitset<BITSET_MAX_SIZE> intersection =
                    bitset_labels[*lbl.begin()];
            // for all labels in the query set, intersect their IDs
            for (const auto& label_id : lbl) {
                intersection &= bitset_labels[label_id];
            }

            for (int id = 0; id < N; ++id)
            {
                if (intersection.test(id))
                {
                    filter_ids_map[i * N + id] = 1;
                }
            }
        }
    }
    else if (scenario == "or"){
        omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < nq; ++i)
        {
            auto lbl = query_labels_set[i];
            std::bitset<BITSET_MAX_SIZE> union_set;
            // for all labels in the query set, union their IDs
            for (const auto &label_id : lbl)
            {
                union_set |= bitset_labels[label_id];
            }

            // for all IDs in the union, set the filter map
            for (int id = 0; id < N; ++id)
            {
                if (union_set.test(id))
                {
                    filter_ids_map[i * N + id] = 1;
                }
            }
        }
    }
    else if (scenario == "equal"){
        omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < nq; ++i)
        {
            auto lbl = query_labels_set[i];
            std::bitset<BITSET_MAX_SIZE> intersection = bitset_labels[*lbl.begin()];
            // for all labels in the query set, intersect their IDs
            for (const auto &label_id : lbl)
            {
                intersection &= bitset_labels[label_id];
            }

            // then for all IDs in the intersection, check if they are equal
            for (int id = 0; id < N; ++id)
            {
                if (intersection.test(id) && base_labels_set[id] == lbl)
                {
                    filter_ids_map[i * N + id] = 1;
                }
            }
        }
    }
}

// 执行搜索操作并收集结果
void perform_search(
        faiss::IndexACORNFlat& index,
        size_t nq,
        float* xq,
        int k,
        std::vector<char>& filter_ids_map,
        const std::vector<faiss::idx_t>& gt,
        double filter_time,
        std::vector<double>& qps_list,
        std::vector<double>& qps_filter_list,
        std::vector<double>& dist_cmps_list,
        std::vector<double>& recall_list,
        const std::vector<int>& parse) {
    for (auto& ef_search : search_lists) {
        index.acorn.efSearch = ef_search;
        std::cout << "==================== ACORN INDEX (efSearch=" << ef_search
                  << ") ====================" << std::endl;

        std::vector<faiss::idx_t> nns2(k * nq);
        std::vector<float> dis2(k * nq);

        double t1 = elapsed();
        index.search(
                nq, xq, k, dis2.data(), nns2.data(), filter_ids_map.data());
        double t2 = elapsed();

        double search_time = t2 - t1;
        double search_time_filter = search_time + filter_time;

        double qps = nq / search_time;
        double qps_filter = nq / search_time_filter;

        std::cout << "Search time: " << search_time
                  << "\nSearch time with filter: " << search_time_filter
                  << "\nQPS: " << qps << "\nQPS with filter: " << qps_filter
                  << std::endl;

        qps_list.push_back(qps);
        qps_filter_list.push_back(qps_filter);

        double recall = compute_recall(gt, nns2, nq, parse, k);
        recall_list.push_back(recall);

        const faiss::ACORNStats& stats = faiss::acorn_stats;
        double dist_comp = static_cast<double>(stats.n3) / stats.n1;
        printf("Average distance computations per query: %.2f\n", dist_comp);
        dist_cmps_list.push_back(dist_comp);
    }
}

double perform_brute_force_search(
        const std::string base_file,
        const std::vector<std::set<int>>& base_label,
        float* xq,
        int nq,
        const std::vector<std::set<int>>& query_label,
        const std::string scenario,
        int recall_at) {
    size_t dim = 0;
    size_t nb = 0;
    float* xb = fvecs_read(base_file.c_str(), &dim, &nb);
    std::string ung_scenario;

    if (scenario == "and") {
        ung_scenario = "containment";
    } else if (scenario == "equal") {
        ung_scenario = "equality";
    } else if (scenario == "or") {
        ung_scenario = "overlap";
    } else {
        std::cout << "(Error) Invalid scenario" << scenario << std::endl;
    }

    std::string data_type = "float";
    std::string dist_fn = "L2";
    std::shared_ptr<ANNS::IStorage> base_storage =
            ANNS::create_storage(data_type);
    std::shared_ptr<ANNS::IStorage> query_storage =
            ANNS::create_storage(data_type);

    base_storage->load_from_memory(dim, nb, xb, base_label);
    query_storage->load_from_memory(dim, nq, xq, query_label);

    std::shared_ptr<ANNS::DistanceHandler> distance_handler =
            ANNS::get_distance_handler(data_type, dist_fn);
    auto groundtruth = new std::pair<
            ANNS::IdxType,
            float>[query_storage->get_num_points() * recall_at];

    std::cout
            << "Computing ground truth using filter then bruteforce search ..."
            << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    ANNS::FilteredBruteforce algo;
    algo.run(
            base_storage,
            query_storage,
            distance_handler,
            ung_scenario,
            nthreads,
            recall_at,
            groundtruth);
    auto time_cost =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start_time)
                    .count();
    std::cout << "Time cost: " << time_cost << "ms" << std::endl;

    return static_cast<double>(time_cost);
}

void create_filter_map(
        const std::vector<std::set<int>>& metadata,
        const std::vector<std::set<int>>& aq,
        size_t nq,
        size_t N,
        std::vector<char>& filter_ids_map,
        const std::string scenario) {
    if (scenario == "and") {
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
    } else if (scenario == "or") {
        omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int xq = 0; xq < nq; xq++) {
            for (int xb = 0; xb < N; xb++) {
                const auto& set1 = metadata[xb];
                const auto& set2 = aq[xq];

                const auto& smaller_set =
                        (set1.size() < set2.size()) ? set1 : set2;
                const auto& larger_set =
                        (set1.size() < set2.size()) ? set2 : set1;

                bool has_intersection = false;
                for (const int& elem : smaller_set) {
                    if (larger_set.find(elem) != larger_set.end()) {
                        has_intersection = true;
                        break;
                    }
                }

                filter_ids_map[xq * N + xb] = has_intersection;
            }
        }
    } else if (scenario == "equal") {
        omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic, 1)
        // exactlly the same as the previous one
        for (int xq = 0; xq < nq; xq++) {
            for (int xb = 0; xb < N; xb++) {
                filter_ids_map[xq * N + xb] = (bool)(metadata[xb] == aq[xq]);
            }
        }
    } else {
        std::cerr << "Error: Invalid scenario: " << scenario << std::endl;
        exit(1);
    }
}

void write_results(
        const std::string& output_path,
        const std::string& output_csv,
        const std::vector<int>& search_lists,
        const std::vector<double>& qps_gamma,
        const std::vector<double>& qps_gamma_filter,
        const std::vector<double>& dist_cmps_gamma,
        const std::vector<double>& recall_at_10_gamma,
        double bf_time,
        int num_bf,
        int num_gs) {
    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Cannot open output file: " << output_path
                  << std::endl;
        return;
    }

    output_file
            << "====================ACORN INDEX====================\n"
            << "Search lists       QPS      QPS(filter)      dist_cmp    recall@10\n";

    for (size_t i = 0; i < search_lists.size(); i++) {
        output_file << search_lists[i] << "        " << qps_gamma[i]
                    << "        " << qps_gamma_filter[i] << "        "
                    << dist_cmps_gamma[i] << "    " << recall_at_10_gamma[i]
                    << "\n";
    }

    output_file.close();

    std::ofstream output_csv_file(output_csv);
    output_csv_file << "L,Cmps,QPS,Recall,QPS_filter" << std::endl;
    for (auto i = 0; i < search_lists.size(); i++) {
        double total_recall = (recall_at_10_gamma[i] * num_gs + 1.0 * num_bf) /
                static_cast<double>(num_gs + num_bf);
        std::cout << "Number of points: "
                  << static_cast<double>(num_gs + num_bf) << std::endl;
        std::cout << "Total recall: " << total_recall << std::endl;
        double total_time = 0;
        if (num_gs != 0)
            total_time = bf_time / 1000.0 + num_gs / qps_gamma_filter[i];
        else
            total_time = bf_time / 1000.0;
        double total_qps = static_cast<double>(num_gs + num_bf) / total_time;
        output_csv_file << search_lists[i] << "," << dist_cmps_gamma[i] << ","
                        << total_qps << "," << total_recall << "," << qps_gamma_filter[i] << std::endl;
    }

    output_csv_file.close();
}

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
    // int k = 10;
    size_t d = 128;
    int M;
    int M_beta;
    int gamma;
    std::string dataset;
    size_t N = 0;

    if (argc != 15) {
        std::cerr
                << "Usage: " << argv[0]
                << " <N> <gamma> <dataset> <M> <M_beta> <index_path> <scenario> <output_path_prefix> "
                << "<base_file> <base_label> <query_file> <query_label> <gt_path> <k>"
                << std::endl;
        return 1;
    }

    N = strtoul(argv[1], NULL, 10);
    gamma = atoi(argv[2]);
    dataset = argv[3];
    M = atoi(argv[4]);
    M_beta = atoi(argv[5]);
    const std::string index_path = argv[6];
    const std::string scenario = argv[7];
    const std::string output_path_prefix = argv[8];
    const std::string base_file = argv[9];
    const std::string base_label = argv[10];
    const std::string query_file = argv[11];
    const std::string query_label = argv[12];
    const std::string gt_path = argv[13];
    int k = atoi(argv[14]);

    const std::string output_path = output_path_prefix +
            "/M=" + std::to_string(M) + "_M_beta=" + std::to_string(M_beta) +
            "_gamma=" + std::to_string(gamma) + ".txt";

    const std::string output_csv = output_path_prefix +
            "/M=" + std::to_string(M) + "_M_beta=" + std::to_string(M_beta) +
            "_gamma=" + std::to_string(gamma) + "_result.csv";

    std::cout << "Parameters:\n"
              << "  N: " << N << "\n"
              << "  gamma: " << gamma << "\n"
              << "  dataset: " << dataset << "\n"
              << "  M: " << M << "\n"
              << "  M_beta: " << M_beta << "\n"
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
    auto bitset_labels = get_label_ivf_from_id2labels(metadata);

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
    // double t1_filter = elapsed();
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
    // double t2_filter = elapsed();

    if (gamma == 1) {
        // if gamma = 1, use graph search only.
        memcpy(xq_gs, xq,  nq * d * sizeof(float));
        num_bf = 0;
        num_gs = nq;
        for (auto& p : parse) {
            // use graph search only;
            p = 1;
        }
    }

    std::vector<std::set<int>> aq_gs;
    for (int i = 0; i < parse.size(); ++i) {
        if (parse[i]) {
            aq_gs.push_back(aq[i]);
        }
    }

    std::cout << "Number of points brute force: " << num_bf
              << " Number of points graph search: " << num_gs << std::endl;

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
        std::cerr << "Unable to open gt file" << std::endl;
        return 1;
    }

    std::string hybrid_index_file = index_path + "/" + dataset +
            "/hybrid_M=" + std::to_string(M) + "_Mb=" + std::to_string(M_beta) +
            "_gamma=" + std::to_string(gamma) + ".json";

    for (const auto& path : {hybrid_index_file}) {
        if (!file_exist(path)) {
            std::cerr << "Error: Index file does not exist: " << path
                      << std::endl;
            return 1;
        }
        std::cout << "Loading index from " << path << std::endl;
    }

    auto& hybrid_index = *dynamic_cast<faiss::IndexACORNFlat*>(
            faiss::read_index(hybrid_index_file.c_str()));

    std::vector<char> filter_ids_map_fake(nq * N);
    double t1_filter = elapsed();
    // only for counting time
    create_filter_map_fake(bitset_labels, metadata, aq, nq, N, filter_ids_map_fake, scenario);
    double t2_filter = elapsed();
    double filter_time = t2_filter - t1_filter;
    std::cout << "Filter map creation time: " << filter_time << "s"
              << std::endl;

    std::vector<char> filter_ids_map(num_gs * N);

    create_filter_map(metadata, aq_gs, num_gs, N, filter_ids_map, scenario);

    // double filter_time = t2_filter - t1_filter;
    // std::cout << "Filter map creation time: " << filter_time << "s"
    //           << std::endl;

    std::vector<double> qps_gamma, qps_gamma_filter, dist_cmps_gamma,
            recall_at_10_gamma;

    std::cout << "=== Performing search with gamma=" << gamma
              << " ===" << std::endl;
    perform_search(
            hybrid_index,
            num_gs,
            xq_gs,
            k,
            filter_ids_map,
            gt,
            filter_time,
            qps_gamma,
            qps_gamma_filter,
            dist_cmps_gamma,
            recall_at_10_gamma,
            parse);

    double bf_time = perform_brute_force_search(
            base_file, metadata, xq_bf, num_bf, aq, scenario, k);

    write_results(
            output_path,
            output_csv,
            search_lists,
            qps_gamma,
            qps_gamma_filter,
            dist_cmps_gamma,
            recall_at_10_gamma,
            bf_time,
            num_bf,
            num_gs);

    delete[] xq;
    delete[] xq_bf;
    delete[] xq_gs;

    printf("[%.3f s] ----- DONE -----\n", elapsed() - t0);

    return 0;
}