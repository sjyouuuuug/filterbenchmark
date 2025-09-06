import subprocess
import os
import csv
import time

# Define parameter ranges 
# Ms = list(range(16, 65, 8))
# Ms = [16]
# efcs = [100]

# Define parameter ranges 
Ms = list(range(16, 65, 8)) 
# Ms = [16]
efcs = list(range(10, 351, 20))
# efcs = [100]

# Constants
N = 50000
k = 10
# datasets = ["LAION1M", "tripclick"]
datasets = ["arxiv", "yfcc", "ytb_audio"]
for dataset in datasets:
    scenarios = ["and", "or", "equal"]
    for scenario in scenarios:
        index_path = "../data/small_index_files/hnsw"
        base_file = "../data/" + dataset + "/" + dataset + "_small.fvecs"
        base_label_file = "../data/" + dataset + "/label_base_small.txt"
        query_file = "../data/" + dataset + "/" + dataset + "_query_" + scenario + "_small.fvecs"
        query_label_file = "../data/" + dataset + "/" + dataset + "_query_" + scenario + "_small.txt"
        output_path = "../data/result/hnsw/" + dataset + "/" + scenario
        gt_path = "../data/" + dataset + "/" + dataset + "_gt_" + scenario + "_small.txt"

        # Loop through parameter combinations
        for M in Ms:
            for efc in efcs:
                # Create output directory
                dir_name = f"M={M}_efc={efc}"
                output_dir = os.path.join(output_path, dir_name)
                os.makedirs(output_dir, exist_ok=True)

                # Construct command
                cmd = ['./build/tutorial/cpp/search_HNSW_index', str(dataset), str(M), str(efc), str(index_path), str(scenario), str(output_path), 
                        str(base_file), str(base_label_file), str(query_file), str(query_label_file), str(gt_path), str(k), str(N)]

                env = os.environ.copy()
                env['debugSearchFlag'] = '0'

                try:
                    with open(os.path.join(output_dir, 'output.log'), 'w') as log_file:
                        subprocess.run(cmd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    print(f"Error running M={M}, efc={efc}")
                    print(e)
                    continue
