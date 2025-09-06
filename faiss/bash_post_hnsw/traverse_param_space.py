import subprocess
import os
import csv
import time

# Define parameter ranges 
Ms = list(range(16, 65, 8)) 
# Ms = [16]
efcs = list(range(10, 351, 20))
# efcs = [100]

# Constants
datasets = ["arxiv", "yfcc", "LAION1M", "tripclick", "ytb_audio", "ytb_video"]
# datasets = ["arxiv"]
for dataset in datasets:
    base_file = "../data/" + dataset + "/" + dataset + "_small.fvecs"
    output_path_base = "../data/small_index_files/hnsw"

    # Loop through parameter combinations
    for M in Ms:
        for efc_val in efcs:
            # Create output directory
            dir_name = f"M={M}_efc={efc_val}"
            output_dir = os.path.join(output_path_base, dataset, dir_name)
            os.makedirs(output_dir, exist_ok=True)

            cmd = ['./build/tutorial/cpp/build_HNSW_index', base_file, str(M), str(efc_val), output_path_base, dataset]

            env = os.environ.copy()
            env['debugSearchFlag'] = '0'

            try:
                with open(os.path.join(output_dir, 'output.log'), 'w') as log_file:
                    subprocess.run(cmd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print(f"Error running M={M}, efc={efc_val}")
                print(e)
                continue
