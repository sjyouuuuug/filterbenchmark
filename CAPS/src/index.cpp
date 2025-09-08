#include <iostream>
#include <fstream>
#include "FilterIndex.h"

#define DATAPATH "/scratch/gg29/data/"
int main(int argc, char** argv)
{   
    string basepath, labelpath, indexpath;
    //default
    string metric = "L2";
    int mode = 1;
    string algo ="kmeans";
    size_t nc =1024;

    int success = argparser(argc, argv, &basepath, &labelpath, &indexpath, &nc, &algo, &mode);

    size_t d, nb; 
    float* data = fvecs_read(basepath.c_str(), &d, &nb);
    vector<vector<string>> properties = getproperties(labelpath,',');
    cout << "Data files read" << endl;
    chrono::time_point<chrono::high_resolution_clock> t1, t2;
    t1 = chrono::high_resolution_clock::now();
    FilterIndex myFilterIndex(data, d, nb, nc, properties, algo, mode);
    cout << "check" <<endl;
    myFilterIndex.get_index(metric, indexpath, mode);
    t2 = chrono::high_resolution_clock::now();
    auto duration_ms = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
    cout << "Index time: " << duration_ms << " ms" << endl;

    cout << "Indexed at: " << indexpath << endl;
    return 0;
}
