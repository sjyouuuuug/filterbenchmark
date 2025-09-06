#include <omp.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <set>

#include <sys/stat.h>
#include <sys/time.h>

#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <bitset>

using idx_t = faiss::idx_t;
using path = const std::string;
using search_result_t =
        std::tuple<std::vector<faiss::idx_t>, std::vector<float>, double>;
        
// nq should not exceed 5M
// use boost dynamic bitset instead
#define BITSET_MAX_SIZE 5000005

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr __attribute__((unused)) = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

std::vector<std::set<int>> load_multi_label(
        const std::string& input_path,
        int N) {
    std::vector<std::set<int>> metadata;
    // load txt file
    std::ifstream file(input_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << input_path << std::endl;
        return metadata;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::set<int> s;
        std::istringstream iss(line);
        // parse line into set by comma
        for (std::string token; std::getline(iss, token, ',');) {
            s.insert(std::stoi(token));
        }
        metadata.push_back(s);
    }
    assert(metadata.size() == N);
    return metadata;
}

bool file_exist(const std::string& path) {
    if (FILE* file = fopen(path.c_str(), "r")) {
        fclose(file);
        return true;
    }
    return false;
}

const unsigned int nthreads = 16;
const std::vector<int> search_nprobe = {1, 5, 10, 15, 20, 40, 60};

double compute_recall(
        const std::vector<faiss::idx_t>& gt,
        const std::vector<faiss::idx_t>& res,
        int num_queries,
        int k) {
    // std::cout << "Recall@: " << k << std::endl;
    // std::cout << "gt size: " << gt.size() << "res size: " << res.size() <<
    // std::endl;
    int correct = 0;
    for (int i = 0; i < num_queries; i++) {
        std::vector<faiss::idx_t> tmp_gt(
                gt.begin() + i * k, gt.begin() + (i + 1) * k);
        std::vector<faiss::idx_t> tmp_res(
                res.begin() + i * k, res.begin() + (i + 1) * k);
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
    return static_cast<double>(correct) / (num_queries * k);
}

search_result_t perform_single_search(
        faiss::IndexIVFPQ& index,
        float* xq,
        int result_size) {
    std::vector<faiss::idx_t> nns(result_size);
    std::vector<float> dis(result_size);
    double t1 = elapsed();
    index.search(1, xq, result_size, dis.data(), nns.data());
    double t2 = elapsed();

    double search_time = t2 - t1;
    return {nns, dis, search_time};
}

search_result_t perform_search(
        faiss::IndexIVFPQ& index,
        size_t nq,
        float* xq,
        int result_size) {
    std::vector<faiss::idx_t> nns2(result_size * nq);
    std::vector<float> dis2(result_size * nq);

    double t1 = elapsed();
    index.search(nq, xq, result_size, dis2.data(), nns2.data());
    double t2 = elapsed();

    double search_time = t2 - t1;
    return {nns2, dis2, search_time};
}

std::vector<faiss::idx_t> post_filter(
        const std::vector<faiss::idx_t>& query_nns,
        const std::vector<std::set<int>>& metadata,
        const std::set<int>& query_labels,
        const std::string& scenario,
        int k) {
    std::vector<faiss::idx_t> filtered_result;
    std::vector<bool> keep_flag(query_nns.size(), false);

    if (scenario == "and") {
#pragma omp parallel for
        for (size_t i = 0; i < query_nns.size(); ++i) {
            const auto idx = query_nns[i];
            if (idx >= metadata.size())
                continue;

            if (std::includes(
                        metadata[idx].begin(),
                        metadata[idx].end(),
                        query_labels.begin(),
                        query_labels.end())) {
                keep_flag[i] = true;
            }
        }
    } else if (scenario == "or") {
#pragma omp parallel for
        for (size_t i = 0; i < query_nns.size(); ++i) {
            const auto idx = query_nns[i];
            if (idx >= metadata.size())
                continue;

            const auto& candidate_labels = metadata[idx];
            bool has_intersection = false;
            if (query_labels.size() < candidate_labels.size()) {
                for (int lbl : query_labels) {
                    if (candidate_labels.count(lbl)) {
                        has_intersection = true;
                        break;
                    }
                }
            } else {
                for (int lbl : candidate_labels) {
                    if (query_labels.count(lbl)) {
                        has_intersection = true;
                        break;
                    }
                }
            }
            if (has_intersection)
                keep_flag[i] = true;
        }
    } else if (scenario == "equal") {
#pragma omp parallel for
        for (size_t i = 0; i < query_nns.size(); ++i) {
            const auto idx = query_nns[i];
            if (idx >= metadata.size())
                continue;

            // EQUAL条件：完全匹配标签
            if (metadata[idx] == query_labels) {
                keep_flag[i] = true;
            }
        }
    } else {
        std::cerr << "Invalid scenario: " << scenario << std::endl;
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < query_nns.size() && filtered_result.size() < k;
         ++i) {
        if (keep_flag[i]) {
            filtered_result.push_back(query_nns[i]);
        }
    }

    return filtered_result;
}


std::vector<faiss::idx_t> post_filter_bitset(
    const std::vector<faiss::idx_t>& query_nns,
    const std::vector<char>& filter_map,
    size_t query_idx,
    size_t N,
    int k) 
{
    std::vector<faiss::idx_t> filtered_result;
    filtered_result.reserve(k);

    for (faiss::idx_t candidate_id : query_nns) {
        if (candidate_id < 0) {
            continue;
        }

        size_t map_index = query_idx * N + candidate_id;
        
        if (filter_map[map_index] == 1) {
            filtered_result.push_back(candidate_id);
        }

        if (filtered_result.size() >= k) {
            break;
        }
    }

    return filtered_result;
}

void write_results(
        const std::string& output_csv,
        const std::vector<int>& search_lists,
        const std::vector<double>& qps,
        const std::vector<double>& dist_cmps,
        const std::vector<double>& recall_at_k) {
    std::ofstream output_csv_file(output_csv);
    output_csv_file << "L,Cmps,QPS,Recall" << std::endl;
    for (auto i = 0; i < search_lists.size(); i++)
        output_csv_file << search_lists[i] << "," << dist_cmps[i] << ","
                        << qps[i] << "," << recall_at_k[i] << std::endl;

    output_csv_file.close();
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


void create_filter_map(
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


int main(int argc, char* argv[]) {
    std::cout << "==================== START: running SEARCH_ivfpq_index --"
              << nthreads << " cores ====================" << std::endl;
    omp_set_num_threads(nthreads);
    double t0 = elapsed();

    int nlist;
    int m;
    int nbits;

    std::string dataset;
    size_t N = 0;
    int k;

    if (argc != 15) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset> <nlist> <m> <nbits> <index_path> <scenario>"
                     " <output_path_prefix> <base_file> <base_label> "
                     "<query_file> <query_label> <gt_path> <k> <N>"
                  << std::endl;
        return 1;
    }

    dataset = argv[1];
    nlist = atoi(argv[2]);
    m = atoi(argv[3]);
    nbits = atoi(argv[4]);

    const std::string index_path = argv[5];
    const std::string scenario = argv[6];
    const std::string output_path_prefix = argv[7];
    const std::string base_file = argv[8];
    const std::string base_label = argv[9];
    const std::string query_file = argv[10];
    const std::string query_label = argv[11];
    const std::string gt_path = argv[12];
    k = atoi(argv[13]);
    N = atoi(argv[14]);

    const std::string output_path = output_path_prefix +
            "/nlist=" + std::to_string(nlist) + "_m=" + std::to_string(m) +
            "_nbits=" + std::to_string(nbits) + ".txt";

    const std::string output_csv = output_path_prefix +
            "/nlist=" + std::to_string(nlist) + "_m=" + std::to_string(m) +
            "_nbits=" + std::to_string(nbits) + ".csv";

    for (const auto& path : {gt_path, base_label, query_label}) {
        if (file_exist(path)) {
            std::cout << path << " exists." << std::endl;
        } else {
            std::cerr << path << " does not exist." << std::endl;
            return 1;
        }
    }

    std::vector<std::set<int>> metadata = load_multi_label(base_label, N);

    // convert metadata to inverted
    auto bitset_labels = get_label_ivf_from_id2labels(metadata);

    size_t nq;
    float* xq;
    size_t d;
    std::vector<std::set<int>> aq;
    {
        printf("[%.3f s] Loading query vectors and attributes\n",
               elapsed() - t0);
        xq = fvecs_read(query_file.c_str(), &d, &nq);
        std::cout << "nq: " << nq << ", d: " << d << std::endl;

        aq = load_multi_label(query_label, nq);
        printf("[%.3f s] Loaded %ld %s queries\n",
               elapsed() - t0,
               nq,
               dataset.c_str());
    }

    std::vector<char> filter_ids_map;
    filter_ids_map.resize(N * nq, 0);
    double start_filter = elapsed();
    create_filter_map(bitset_labels, metadata, aq, nq, N, filter_ids_map, scenario);
    double end_filter = elapsed();
    double filter_time = end_filter - start_filter;

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

    std::stringstream filepath_stream;
    filepath_stream << index_path << "/" << dataset << "/nlist=" << nlist
                   << "_m=" << m << "_nbits=" << nbits << ".index";
    std::string filepath = filepath_stream.str();
    char* cfile_path = new char[filepath.size() + 1];
    std::strcpy(cfile_path, filepath.c_str());

    for (const auto& path : {filepath}) {
        if (!file_exist(path)) {
            std::cerr << "Error: Index file does not exist: " << path
                      << std::endl;
            return 1;
        }
        std::cout << "Loading index from " << path << std::endl;
    }

    auto& ivfpq_index =
            *dynamic_cast<faiss::IndexIVFPQ*>(faiss::read_index(cfile_path));
    std::vector<double> qps_list, recall_at_k_list, dist_cmps_list;

    int result_list = k * 10;
    auto max_possible_depth = N * 3 / 4;
    std::cout << "max_possible_depth: " << max_possible_depth << std::endl;

    for (auto& nprob : search_nprobe) {
        ivfpq_index.nprobe = nprob;
        std::vector<faiss::idx_t> final_results(nq * k, -1);
        // std::vector<int> current_depth(nq, k * 10);
        // std::vector<bool> query_done(nq, false);
        std::set<int> query_done;

        double total_search_time = 0;
        int total_searches = 0;
        double start_search = elapsed();

#pragma omp parallel for
        for (int q = 0; q < nq; ++q) {
            int depth = k * 50;
            float* cur_xq = xq + q * d;
            while (true) {
                // std::cout << "processing query " << q << " with depth " <<
                // depth
                //           << std::endl;
                auto [nns, dis, search_time] =
                        perform_single_search(ivfpq_index, cur_xq, depth);
                auto filtered = post_filter_bitset(nns, filter_ids_map, (size_t)q, N, k);

                if (filtered.size() >= k) {
                    std::copy(
                            filtered.begin(),
                            filtered.begin() + k,
                            final_results.begin() + q * k);
                    // query_done.insert(q);
                    break;
                } else {
                    depth *= 2;
                    if (depth > max_possible_depth) {
                        std::copy(
                                filtered.begin(),
                                filtered.end(),
                                final_results.begin() + q * k);
                        // query_done.insert(q);
                        break;
                    }
                }
            }
        }

        double end_search = elapsed();
        double search_time = end_search - start_search + filter_time;
        double qps = nq / search_time;
        qps_list.push_back(qps);
        double recall = compute_recall(gt, final_results, nq, k);
        recall_at_k_list.push_back(recall);
        dist_cmps_list.push_back(0);

        // faiss::HNSWStats& stats = faiss::hnsw_stats;
        // double dist_comp = static_cast<double>(stats.ndis);
        // double avg_dist_comp = dist_comp / nq;
        // stats.reset();
        // printf("Average distance computations per query: %.2f\n",
        //        avg_dist_comp);
        // dist_cmps_list.push_back(avg_dist_comp);
    }

    write_results(
            output_csv,
            search_nprobe,
            qps_list,
            dist_cmps_list,
            recall_at_k_list);

    delete[] xq;
    delete[] cfile_path;

    printf("[%.3f s] ----- DONE -----\n", elapsed() - t0);
    return 0;
}