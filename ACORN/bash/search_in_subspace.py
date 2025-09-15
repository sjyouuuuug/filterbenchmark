import subprocess
import os
import csv
import time

# Define parameter ranges 
Ms = list(range(16, 65, 8)) # 
M_betas = list(range(16, 65, 8)) # 
gammas = list(range(12, 97, 12)) # 10
gammas.append(2)
gammas.append(4)
gammas.append(8)
# gammas = [1]
gammas.append(1)

# Constants
N = 50000
k = 10
datasets = ["LAION1M", "tripclick"]
for dataset in datasets:
    scenarios = ["and", "or", "equal"]
    for scenario in scenarios:
        index_path = "../data/small_index_files/ACORN"
        base_file = "../data/" + dataset + "/" + dataset + "_small.fvecs"
        base_label_file = "../data/" + dataset + "/label_base_small.txt"
        query_file = "../data/" + dataset + "/" + dataset + "_query_" + scenario + "_small.fvecs"
        query_label_file = "../data/" + dataset + "/" + dataset + "_query_" + scenario + "_small.txt"
        output_path = "../data/result/ACORN/" + dataset + "/" + scenario
        gt_path = "../data/" + dataset + "/" + dataset + "_gt_" + scenario + "_small.txt"

        # Loop through parameter combinations
        for M in Ms:
            for M_beta in M_betas:
                if M_beta < M:
                    continue
                for gamma_val in gammas:
                    # Create output directory
                    dir_name = f"M={M}_Mb={M_beta}_gamma={gamma_val}"
                    output_dir = os.path.join(output_path, dir_name)
                    os.makedirs(output_dir, exist_ok=True)

                    # Construct command
                    cmd = ['./build/demos/search_acorn_index', str(N), str(gamma_val),
                            str(dataset), str(M), str(M_beta), str(index_path), str(scenario), str(output_path), 
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
