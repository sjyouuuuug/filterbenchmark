import subprocess
import os
import csv

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
map_N = {"yfcc": 1000000, "arxiv": 132687, "LAION1M": 1000448, "tripclick": 1055976, "ytb_video": 1000000}
# datasets = ["LAION1M", "tripclick"]
datasets = ["ytb_video"]

# Loop through datasets
for dataset in datasets:
    base_file = "../data/" + dataset + "/" + dataset + "_base.fvecs"
    csv_file_path = f'../data/result/ACORN/{dataset}/representative_parameters.csv'  # 修改为你的 CSV 文件路径
    parameter_combinations = read_parameters_from_csv(csv_file_path)

    output_path_base = "../data/index_files/ACORN"

    # Loop through parameter combinations
    for M, M_beta, gamma_val in parameter_combinations:
        if M_beta < M:
            continue
        # Create output directory
        dir_name = f"M={M}_Mb={M_beta}_gamma={gamma_val}"
        output_dir = os.path.join(output_path_base, dataset, dir_name)
        os.makedirs(output_dir, exist_ok=True)

        # skip if the directory already exists
        # if os.path.exists(os.path.join(output_dir, 'output.log')):
        #     print(f"Skipping existing directory: {output_dir}")
        #     continue

        # Construct command
        cmd = ['./build/demos/build_gamma_index', str(map_N[dataset]), str(gamma_val),
               base_file, str(M), str(M_beta), output_path_base, dataset]

        env = os.environ.copy()
        env['debugSearchFlag'] = '0'

        # Use taskset to limit the program to 16 CPU threads (cores 0-15)
        cpu_mask = "0xFFFF"  # Binary mask for the first 16 cores

        try:
            with open(os.path.join(output_dir, 'output.log'), 'w') as log_file:
                # Prepend taskset command to restrict CPU cores
                subprocess.run(["taskset", cpu_mask] + cmd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"Error running M={M}, M_beta={M_beta}, gamma={gamma_val}")
            print(e)
            continue