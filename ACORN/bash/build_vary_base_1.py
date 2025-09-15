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

# Read parameters from CSV file

# Constants
map_N = {"ytb_audio": 5000000, "yfcc": 1000000}
datasets = ["yfcc", "ytb_audio"]
for dataset in datasets:
    for p in [10, 25, 50, 100]:
        base_file = "../base_experiment/" + dataset + "/" + dataset + "_base_" + str(p) +"p.fvecs"
        csv_file_path = f'../data/result/ACORN/{dataset}/representative_parameters_1.csv'  # 修改为你的 CSV 文件路径
        parameter_combinations = read_parameters_from_csv(csv_file_path)

        output_path_base = "../base_experiment/index_files/ACORN/" + str(p) + "p"

        # Loop through parameter combinations
        for M, M_beta, gamma_val in parameter_combinations:
            if M_beta < M:
                continue
            # Create output directory
            dir_name = f"M={M}_Mb={M_beta}_gamma={gamma_val}"
            output_dir = os.path.join(output_path_base, dataset, dir_name)
            os.makedirs(output_dir, exist_ok=True)

            # Construct command
            N = map_N[dataset] * p / 100
            cmd = ['./build/demos/build_gamma_index', str(N), str(gamma_val),
                    base_file, str(M), str(M_beta), output_path_base, dataset]

            env = os.environ.copy()
            env['debugSearchFlag'] = '0'

            try:
                with open(os.path.join(output_dir, 'output.log'), 'w') as log_file:
                    subprocess.run(cmd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print(f"Error running M={M}, M_beta={M_beta}, gamma={gamma_val}")
                print(e)
                continue