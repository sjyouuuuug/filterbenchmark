#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sstream>
#include <fstream>
#include <faiss/IndexACORN.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "utils.cpp"

int main(int argc, char* argv[]) {
    double t0 = elapsed();

    size_t d = 128; // dimension of the vectors to index
    int M, M_beta, gamma;
    std::string dataset, filename, output_path;
    size_t N = 0;

    // Parse command-line arguments
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <N> <gamma> <filename> <M> <M_beta> <output_path> <dataset>" << std::endl;
        return 1;
    }

    N = strtoul(argv[1], NULL, 10);
    gamma = atoi(argv[2]);
    filename = argv[3];
    M = atoi(argv[4]);
    M_beta = atoi(argv[5]);
    output_path = argv[6];
    dataset = argv[7];

    // Create indices
    printf("[%.3f s] Index Params -- M: %d, M_beta: %d, N: %ld, gamma: %d\n",
           elapsed() - t0, M, M_beta, N, gamma);

    int efs = 48;

    // Populate the database

    size_t nb, d2;
    float* xb = fvecs_read(filename.c_str(), &d2, &nb);
    d = d2;

    std::vector<int> meta(N);
    faiss::IndexACORNFlat hybrid_index(d, M, gamma, meta, M_beta);
    hybrid_index.acorn.efSearch = efs;
    faiss::IndexACORNFlat hybrid_index_gamma1(d, M, 1, meta, M_beta);
    hybrid_index_gamma1.acorn.efSearch = efs;

    double t1 = elapsed();
    hybrid_index.add(N, xb);
    double t2 = elapsed();
    std::cout << "Create gamma index in time: " << t2 - t1 << std::endl;

    double t1_gamma1 = elapsed();
    hybrid_index_gamma1.add(N, xb);
    double t2_gamma1 = elapsed();
    std::cout << "Create gamma=1 index in time: " << t2_gamma1 - t1_gamma1 << std::endl;

    delete[] xb;

    // Write indices to files
    {
        std::stringstream filepath_stream;
        filepath_stream << output_path << "/" << dataset << "/hybrid_M=" << M
                        << "_Mb=" << M_beta << "_gamma=" << gamma << ".json";
        std::string filepath = filepath_stream.str();
        write_index(&hybrid_index, filepath.c_str());

        std::stringstream filepath_stream2;
        filepath_stream2 << output_path << "/" << dataset << "/hybrid_M=" << M
                         << "_Mb=" << M_beta << "_gamma=1.json";
        std::string filepath2 = filepath_stream2.str();
        write_index(&hybrid_index_gamma1, filepath2.c_str());
    }

    std::cout << std::endl;
    return 0;
}