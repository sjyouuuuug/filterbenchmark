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
            M_beta = int(float(row['--M_beta']))
            gamma = int(float(row['--gamma']))
            parameters.append((M, M_beta, gamma))
    return parameters

# Constants
map_N = {"yfcc": 1000000, "arxiv": 132687}
dataset = "arxiv"
scenarios = ["NHQ"]
k=10
# scenarios = ["or"]
datasets = ["yfcc", "arxiv"]
for dataset in datasets:
    for scenario in scenarios:
        base_file = "../data/" + dataset + "/" + dataset + "_base.fvecs"
        csv_file_path = f'../data/result/ACORN/{dataset}/representative_parameters_1.csv'  # 修改为你的 CSV 文件路径
        parameter_combinations = read_parameters_from_csv(csv_file_path)

        index_path = "../data/index_files/ACORN"
        base_label_file = "../equal_length_experiment/" + dataset + "/label_NHQ_base.txt"
        query_file = "../equal_length_experiment/" + dataset + "/" + dataset + "_query_" + scenario + ".fvecs"
        query_label_file = "../equal_length_experiment/" + dataset + "/label_NHQ_query.txt"
        output_path = "../equal_length_experiment/result/ACORN_bitset/" + dataset
        gt_path = "../equal_length_experiment/" + dataset + "/" + dataset + "_gt_" + scenario + ".txt"

        # Loop through parameter combinations
        for M, M_beta, gamma_val in parameter_combinations:
            # Create output directory
            dir_name = f"M={M}_Mb={M_beta}_gamma={gamma_val}"
            output_dir = os.path.join(output_path, dir_name)
            os.makedirs(output_dir, exist_ok=True)

            # Construct command
            cmd = ['./build/demos/search_acorn_index', str(map_N[dataset]), str(gamma_val),
                    str(dataset), str(M), str(M_beta), str(index_path), str("equal"), str(output_path), 
                    str(base_file), str(base_label_file), str(query_file), str(query_label_file), str(gt_path), str(k)]

            env = os.environ.copy()
            env['debugSearchFlag'] = '0'

            try:
                with open(os.path.join(output_dir, 'output.log'), 'w') as log_file:
                    subprocess.run(cmd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print(f"Error running M={M}, M_beta={M_beta}, gamma={gamma_val}")
                print(e)
                continue
