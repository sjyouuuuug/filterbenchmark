import subprocess
import os
import csv
import time

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

# Constants
map_N = {"yfcc": 1000000, "arxiv": 132687}
datasets = ["yfcc", "arxiv"]
# dataset = "yfcc"
for dataset in datasets:
    # scenarios = ["or", "equal", "and"]
    scenarios = ["NHQ"]
    # scenarios = ["or"]
    k = 10
    for scenario in scenarios:
        base_file = "../data/" + dataset + "/" + dataset + "_base.fvecs"
        csv_file_path = f'../data/result/ivfpq/{dataset}/representative_parameters.csv'  # 修改为你的 CSV 文件路径
        parameter_combinations = read_parameters_from_csv(csv_file_path)

        index_path = "../data/index_files/ivfpq"
        base_label_file = "../equal_length_experiment/" + dataset + "/label_NHQ_base.txt"
        query_file = "../equal_length_experiment/" + dataset + "/" + dataset + "_query_" + scenario + ".fvecs"
        query_label_file = "../equal_length_experiment/" + dataset + "/label_NHQ_query.txt"
        output_path = "../equal_length_experiment/result/ivfpq_bitset/" + dataset + "/"
        gt_path = "../equal_length_experiment/" + dataset + "/" + dataset + "_gt_" + scenario + ".txt"

        # Loop through parameter combinations
        for nl, m, nb in parameter_combinations:
            # Create output directory
            dir_name = f"nl={nl}_m={m}_nb={nb}"
            output_dir = os.path.join(output_path, dir_name)
            os.makedirs(output_dir, exist_ok=True)

            # Construct command
            cmd = ['./build/tutorial/cpp/search_IVFPQ_index', str(dataset), str(nl), str(m), str(nb), str(index_path), str("equal"), str(output_path), 
                    str(base_file), str(base_label_file), str(query_file), str(query_label_file), str(gt_path), str(k), str(map_N[dataset])]

            env = os.environ.copy()
            env['debugSearchFlag'] = '0'

            try:
                with open(os.path.join(output_dir, 'output.log'), 'w') as log_file:
                    subprocess.run(cmd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print(f"Error running nl={nl}, m={m}, nb={nb}")
                print(e)
                continue
