#include <iostream>
#include <fstream>
#include "FilterIndex.h"

// #define DATAPATH "/scratch/gg29/data/"

int main(int argc, char** argv)
{
    //default
    string metric = "L2";
    int mode = 0;
    string algo ="kmeans";
    size_t nc =0;
    // size_t buffer_size =0;
    size_t nprobe =0;
    string output_csv;
    vector<size_t> nprobe_lists = {50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000};

    size_t d, nb, nq, num_results = 10; 
    string datapath, Attripath, querypath, queryAttripath, indexpath, GTpath;
    int success = argparser(argc, argv, &datapath, &Attripath, &querypath, &queryAttripath, &indexpath, &GTpath, &output_csv, &nc, &algo, &mode, &nprobe);

    float* data = fvecs_read(datapath.c_str(), &d, &nb);
    vector<vector<string>> properties = getproperties(Attripath,',');
    // nc = atoi(argv[2]); // num clusters
    FilterIndex myFilterIndex(data, d, nb, nc, properties, algo, mode);
    // cout << nprobe << endl;
    myFilterIndex.loadIndex(indexpath);
    cout << "Loaded" << endl;

    float* queryset = fvecs_read(querypath.c_str(), &d, &nq);
    vector<vector<string>> queryprops = getproperties(queryAttripath,',');
    int* queryGTlabel = ivecs_read(GTpath.c_str(), &num_results, &nq);
    cout << "Query files read..." << endl;
    // nq = 10000;
    chrono::time_point<chrono::high_resolution_clock> t1, t2;
    
    vector<double> recalls;
    vector<double> qpss;
    for (size_t i = 0; i < nprobe_lists.size(); i++) {
        nprobe = nprobe_lists[i];
        t1 = chrono::high_resolution_clock::now();
        myFilterIndex.query(queryset, nq, queryprops, num_results, nprobe);
        t2 = chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t2 - t1;

        int32_t* output = myFilterIndex.neighbor_set;
        int output_[num_results*nq];
        copy(output, output+num_results*nq , output_);
        cout<<"numClusters, buffersize, QPS, Recall100@100 :"<<endl;
        //QPS and recall
        double QPS;
        double recall = RecallAtK(queryGTlabel, output_, num_results, nq);
        printf("%d,%d,%f,%f\n",nc, nprobe, nq/diff.count(), recall);

        recalls.push_back(recall);
        qpss.push_back(nq/diff.count());
    }

    // Output results to CSV
    ofstream csvFile(output_csv);
    if (csvFile.is_open()) {
        csvFile << "nprobe,Recall,QPS\n";
        for (size_t i = 0; i < nprobe_lists.size(); i++) {
            csvFile << nprobe_lists[i] << "," << recalls[i] << "," << qpss[i] << "\n";
        }
        csvFile.close();
    }
    else {
        cerr << "Error opening output CSV file: " << output_csv << endl;
    }
}