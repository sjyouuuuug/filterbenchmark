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
map_N = {"yfcc": 1000000, "ytb_audio": 5000000}
# dataset = "yfcc"
dataset = "ytb_audio"
scenarios = ["or", "equal", "and"]
# scenarios = ["or"]
k = 10
for scenario in scenarios:
    # for id in [0, 1, 2]:
    for percentage in [10, 25, 50, 100]:
        base_file = "../base_experiment/" + dataset + "/" + dataset + "_base_" + str(percentage) + "p.fvecs"
        csv_file_path = f'../data/result/ACORN/{dataset}/representative_parameters.csv'  # 修改为你的 CSV 文件路径
        parameter_combinations = read_parameters_from_csv(csv_file_path)

        index_path = "../base_experiment/index_files/ACORN_gamma/" + str(percentage) + "p"
        base_label_file = "../base_experiment/" + dataset + "/" + dataset + "_base_" + str(percentage) + "p.txt"
        query_file = "../base_experiment/" + dataset + "/" + dataset + "_query_" + scenario + "_" + str(percentage) + "p.fvecs"
        query_label_file = "../base_experiment/" + dataset + "/" + dataset + "_query_" + scenario + "_" + str(percentage) + "p.txt"
        output_path = "../base_experiment/result/ACORN_gamma/" + dataset + "/" + scenario + "_" + str(percentage)
        gt_path = "../base_experiment/" + dataset + "/" + dataset + "_gt_" + scenario + "_" + str(percentage) + "p.txt"

        # Loop through parameter combinations
        for M, M_beta, gamma_val in parameter_combinations:
            # Create output directory
            dir_name = f"M={M}_Mb={M_beta}_gamma={gamma_val}"
            output_dir = os.path.join(output_path, dir_name)
            os.makedirs(output_dir, exist_ok=True)


            N = map_N[dataset] * percentage // 100
            # Construct command
            cmd = ['./build/demos/search_acorn_index_parse', str(N), str(gamma_val),
                    str(dataset), str(M), str(M_beta), str(index_path), str(scenario), str(output_path), 
                    str(base_file), str(base_label_file), str(query_file), str(query_label_file), str(gt_path)]

            env = os.environ.copy()
            env['debugSearchFlag'] = '0'

            try:
                with open(os.path.join(output_dir, 'output.log'), 'w') as log_file:
                    subprocess.run(cmd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print(f"Error running M={M}, M_beta={M_beta}, gamma={gamma_val}")
                print(e)
                continue
