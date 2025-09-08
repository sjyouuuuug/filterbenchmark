import subprocess
import os
import csv

def read_parameters_from_csv(file_path):
    parameters = []
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            converted = {}
            for key, val in row.items():
                clean_key = key.lstrip('-')
                try:
                    num = float(val)
                    # 如果 float 是整数，则转为 int
                    if num.is_integer():
                        converted[clean_key] = int(num)
                    else:
                        converted[clean_key] = num
                except ValueError:
                    converted[clean_key] = val
            parameters.append(converted)
    return parameters

# Read parameters from CSV file


def run_all_builds(parameter_combinations, base_file, att_file, output_path_base, dataset, save_graph_base, save_attr_base):
    if not parameter_combinations:
        print("No parameter combinations to run.")
        return

    param_keys = ['K', 'L', 'iter', 'S', 'R', 'RANGE', 'PL', 'B', 'M']

    for params in parameter_combinations:
        dir_name = "_".join(f"{k}={params[k]}" for k in param_keys)
        output_dir = os.path.join(output_path_base, dataset, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        script_path="./build/tests/run_dng_in_bash.sh"

        save_graph = os.path.join(save_graph_base, dir_name, "graph.bin")
        save_attr  = os.path.join(save_attr_base,  dir_name, "attr.txt")

        cmd = [
            str(script_path),
            str(base_file),
            str(att_file),
            save_graph,
            save_attr
        ] + [str(params[k]) for k in param_keys]

        # 可按需设置环境变量
        env = os.environ.copy()
        env["debugSearchFlag"] = "0"

        # 执行并把 stdout/stderr 都重定向到 log 文件
        log_path = os.path.join(output_dir, "output.log")
        try:
            with open(log_path, "w") as log_file:
                subprocess.run(
                    cmd,
                    env=env,
                    check=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT
                )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 执行失败：{dir_name}")
            print(f"命令：{' '.join(cmd)}")
            print(f"返回码：{e.returncode}")
            # 继续下一个组合
            continue

# Constants
map_N = {"yfcc": 1000000, "arxiv": 132687, "LAION1M": 1000448, "tripclick": 1055976}
# datasets = ["LAION1M", "tripclick"]
datasets = ["yfcc"]
for dataset in datasets:
    base_file = "../../data/" + dataset + "/" + dataset + "_base.fvecs"
    att_file = f"../../equal_length_experiment/{dataset}/label_NHQ_base_header.txt"
    csv_file_path = f'../../data/result/NHQ_kgraph/{dataset}/representative_parameters.csv'
    parameter_combinations = read_parameters_from_csv(csv_file_path)

    output_path_base = "../../equal_length_experiment/index_files/NHQ_kgraph"
    save_graph_base=f"../../equal_length_experiment/index_files/NHQ_kgraph/{dataset}"
    save_attr_base =f"../../equal_length_experiment/index_files/NHQ_kgraph/{dataset}"

    run_all_builds(parameter_combinations, base_file, att_file, output_path_base, dataset, save_graph_base, save_attr_base)