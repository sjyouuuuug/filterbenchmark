import subprocess
import os
import csv
import time

# Define parameter ranges 
def read_parameters_from_csv(file_path):
    parameters = []
    with open(file_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            M = int(float(row['--M']))
            efc = int(float(row['--efc']))
            parameters.append((M, efc))
    return parameters

# Constants
map_N = {"yfcc": 1000000, "arxiv": 132687}
# dataset = "yfcc"
datasets = ["yfcc", "arxiv"]
scenarios = ["or", "equal", "and"]
# scenarios = ["or"]
for dataset in datasets:
    for scenario in scenarios:
        for k in [1, 25, 50, 100]:
            base_file = "../data/" + dataset + "/" + dataset + "_base.fvecs"
            csv_file_path = f'../data/result/hnsw/{dataset}/representative_parameters.csv'  # 修改为你的 CSV 文件路径
            parameter_combinations = read_parameters_from_csv(csv_file_path)

            index_path = "../data/index_files/hnsw"
            base_label_file = "../data/" + dataset + "/label_base.txt"
            query_file = "../recall_experiment/" + dataset + "/" + dataset + "_query_" + scenario + "_k" + str(k) + ".fvecs"
            query_label_file = "../recall_experiment/" + dataset + "/" + dataset + "_query_" + scenario + "_k" + str(k) + ".txt"
            output_path = "../recall_experiment/result/hnsw_bitset/" + dataset + "/" + scenario + "_recall@" + str(k) 
            gt_path = "../recall_experiment/" + dataset + "/" + dataset + "_gt_" + scenario + "_k" + str(k) + ".txt"

            # Loop through parameter combinations
            for M, efc in parameter_combinations:
                # Create output directory
                dir_name = f"M={M}_efc={efc}"
                output_dir = os.path.join(output_path, dir_name)
                os.makedirs(output_dir, exist_ok=True)

                # Construct command
                cmd = ['./build/tutorial/cpp/search_HNSW_index', str(dataset), str(M), str(efc), str(index_path), str(scenario), str(output_path), 
                            str(base_file), str(base_label_file), str(query_file), str(query_label_file), str(gt_path), str(k), str(map_N[dataset])]

                env = os.environ.copy()
                env['debugSearchFlag'] = '0'

                try:
                    with open(os.path.join(output_dir, 'output.log'), 'w') as log_file:
                        subprocess.run(cmd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    print(f"Error running M={M}, efc={efc}")
                    print(e)
                    continue
