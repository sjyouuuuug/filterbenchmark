#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>
#include <iostream>

#include <sys/stat.h>
#include <sys/time.h>

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

using idx_t = faiss::idx_t;
using path = const std::string;

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


int main (int argc, char* argv[]){
    double t0 = elapsed();
    std::string dataset;
    int efc;
    int M;
    std::string output_path;
    std::string base_file;

    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <filename> <M> <efc> <output_path> <dataset>" << std::endl;
        return 1;
    }

    base_file = argv[1];
    M = atoi(argv[2]);
    efc = atoi(argv[3]);
    output_path = argv[4];
    dataset = argv[5];

    printf("[%.3f s] Index Params -- M: %d, efc: %d\n",
           elapsed() - t0, M, efc);

    size_t nb, d;
    float* xb = fvecs_read(base_file.c_str(), &d, &nb);

    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efc;
    double start_build = elapsed();
    index.add(nb, xb);
    double end_build = elapsed();
    std::cout << "create hnsw index in: " << end_build - start_build << std::endl;

    std::stringstream filepath_stream;
    filepath_stream << output_path << "/" << dataset << "/M=" << M
                        << "_efc=" << efc << ".json";
    std::string filepath = filepath_stream.str();
    char* cfile_path = new char[filepath.size() + 1];
    std::strcpy(cfile_path, filepath.c_str());

    faiss::write_index(&index, cfile_path);

    return 0;
}