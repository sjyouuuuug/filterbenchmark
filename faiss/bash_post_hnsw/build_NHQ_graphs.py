import subprocess
import os
import csv

def read_parameters_from_csv(file_path):
    parameters = []
    with open(file_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            M = int(float(row['--M']))
            efc = int(float(row['--efc']))
            parameters.append((M, efc))
    return parameters

# Read parameters from CSV file

# Constants
map_N = {"yfcc": 1000000, "arxiv": 132687, "LAION1M": 1000448, "tripclick": 1055976}
# datasets = ["LAION1M", "tripclick"]
datasets = ["arxiv", "yfcc"]
for dataset in datasets:
    base_file = "../data/" + dataset + "/" + dataset + "_base.fvecs"
    csv_file_path = f'../data/result/hnsw/{dataset}/representative_parameters.csv'  # 修改为你的 CSV 文件路径
    parameter_combinations = read_parameters_from_csv(csv_file_path)


    output_path_base = "../equal_length_experiment/index_files/hnsw"

    # Loop through parameter combinations
    for M, efc in parameter_combinations:
        # Create output directory
        dir_name = f"M={M}_efc={efc}"
        output_dir = os.path.join(output_path_base, dataset, dir_name)
        os.makedirs(output_dir, exist_ok=True)

        # Construct command
        cmd = ['./build/tutorial/cpp/build_HNSW_index', base_file, str(M), str(efc), output_path_base, dataset]

        env = os.environ.copy()
        env['debugSearchFlag'] = '0'

        try:
            with open(os.path.join(output_dir, 'output.log'), 'w') as log_file:
                subprocess.run(cmd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"Error running M={M}, efc={efc}")
            print(e)
            continue