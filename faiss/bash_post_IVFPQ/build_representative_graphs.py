import subprocess
import os
import csv

def read_parameters_from_csv(file_path):
    parameters = []
    with open(file_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            nl = int(float(row['--nlist']))
            m = int(float(row['--m']))
            nb = int(float(row['--nbits']))

            parameters.append((nl, m, nb))
    return parameters

# Read parameters from CSV file

# Constants
# map_N = {"yfcc": 1000000, "arxiv": 132687, "LAION1M": 1000448, "tripclick": 1055976}
map_N = {"ytb_audio": 5000000, "ytb_video": 5000000, "LAION1M": 1000448, "tripclick": 1055976, "yfcc": 1000000, "arxiv": 132687}

# datasets = ["LAION1M", "tripclick"]
datasets = ["ytb_audio", "ytb_video", "LAION1M", "tripclick", "yfcc", "arxiv"]

for dataset in datasets:
    base_file = "../data/" + dataset + "/" + dataset + "_base.fvecs"
    csv_file_path = f'../data/result/ivfpq/{dataset}/representative_parameters.csv'  # 修改为你的 CSV 文件路径
    parameter_combinations = read_parameters_from_csv(csv_file_path)


    output_path_base = "../data/index_files/ivfpq"

    # Loop through parameter combinations
    for nl, m, nb in parameter_combinations:
        # Create output directory
        dir_name = f"nlist_{nl}_m_{m}_nbits_{nb}"
        output_dir = os.path.join(output_path_base, dataset, dir_name)
        os.makedirs(output_dir, exist_ok=True)

        # Construct command
        cmd = ['./build/tutorial/cpp/build_IVFPQ_index', base_file, str(nl), str(m), str(nb), output_path_base, dataset]

        env = os.environ.copy()
        env['debugSearchFlag'] = '0'

        try:
            with open(os.path.join(output_dir, 'output.log'), 'w') as log_file:
                subprocess.run(cmd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"Error running nlist {nl}, m {m}, nbits {nb} for dataset {dataset}:")
            print(e)
            continue