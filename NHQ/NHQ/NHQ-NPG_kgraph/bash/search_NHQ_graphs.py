import os
import subprocess
import itertools
import time
import pandas as pd

dataset = 'yfcc'

csv_path = f'../../data/result/NHQ_kgraph/{dataset}/representative_parameters.csv'
representative_params = pd.read_csv(csv_path).to_dict('records')

print(f"Loaded {len(representative_params)} representative parameter sets from {csv_path}")

representative_params = pd.read_csv(csv_path)

required_columns = ['--K','--L','--iter','--S','--R','--RANGE','--PL','--B','--M']
missing_cols = [col for col in required_columns if col not in representative_params.columns]
if missing_cols:
    raise ValueError(f"CSV file miss: {missing_cols}")

try:
    representative_params['--K'] = representative_params['--K'].astype('Int64')
    representative_params['--L'] = representative_params['--L'].astype('Int64')
    representative_params['--iter'] = representative_params['--iter'].astype('Int64')
    representative_params['--S'] = representative_params['--S'].astype('Int64')
    representative_params['--R'] = representative_params['--R'].astype('Int64')
    representative_params['--RANGE'] = representative_params['--RANGE'].astype('Int64')
    representative_params['--PL'] = representative_params['--PL'].astype('Int64')
    representative_params['--B'] = representative_params['--B'].astype(float)
    representative_params['--M'] = representative_params['--M'].astype('Int64')
except Exception as e:
    print(f"type error: {e}")
    exit(1)  


combinations = [(row['--K'], row['--L'], row['--iter'], row['--S'], row['--R'], row['--RANGE'], row['--PL'], row['--B'], row['--M'])
                   for _, row in representative_params.iterrows()]

os.makedirs("../results/", exist_ok=True)

res_dir = f"../results/{dataset}/"
os.makedirs(res_dir, exist_ok=True)
res_path = os.path.join(res_dir, f"small_search_results.txt")

with open(res_path, "w") as res_f:
    res_f.write(f"Search Results for {dataset}:\n\n")

for k, l, iter, s, r, rg, pl, b, m in combinations:
    index_path = f"../../equal_length_experiment/index_files/NHQ_kgraph/{dataset}/K={k}_L={l}_iter={iter}_S={s}_R={r}_RANGE={rg}_PL={pl}_B={b}_M={m}/graph.bin"
    attr_path = f"../../equal_length_experiment/index_files/NHQ_kgraph/{dataset}/K={k}_L={l}_iter={iter}_S={s}_R={r}_RANGE={rg}_PL={pl}_B={b}_M={m}/attr.txt"
    base_path = f"../../data/{dataset}/{dataset}_base.fvecs"
    query_path = f"../../equal_length_experiment/{dataset}/{dataset}_query_NHQ.fvecs"
    query_attr_path = f"../../equal_length_experiment/{dataset}/label_NHQ_query_header.txt"
    gt_path = f"../../equal_length_experiment/{dataset}/{dataset}_gt_NHQ.txt"
    output_path = f"../../equal_length_experiment/result/NHQ_kgraph/{dataset}/K={k}_L={l}_iter={iter}_S={s}_R={r}_RANGE={rg}_PL={pl}_B={b}_M={m}_results.csv"
    output_prefix = f"../../equal_length_experiment/result/NHQ_kgraph/{dataset}/K={k}_L={l}_iter={iter}_S={s}_R={r}_RANGE={rg}_PL={pl}_B={b}_M={m}"

    # check if the index and attribute files exist
    if not os.path.exists(index_path):
        print(f"Index file {index_path} does not exist, skipping this combination.")
        continue
    if not os.path.exists(attr_path):
        print(f"Attribute file {attr_path} does not exist, skipping this combination.")
        continue

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
        
    cmd = [
        './build/tests/test_dng_optimized_search',
        index_path,
        attr_path,
        base_path,
        query_path,
        query_attr_path,
        gt_path,
        output_path
    ]

    log_prefix = os.path.join(output_prefix, "run_log")
    os.makedirs(os.path.dirname(log_prefix), exist_ok=True)
    with open(f"{log_prefix}_stdout.log", "w") as stdout_f, \
            open(f"{log_prefix}_stderr.log", "w") as stderr_f:
        
        try:
            result = subprocess.run(
                cmd,
                stdout=stdout_f,
                stderr=stderr_f,
                text=True,
                check=True,
                timeout=3600
            )
            with open(res_path, "a") as res_f:
                res_f.write(f"Success: K={k}, L={l}, iter={iter}, S={s}, R={r}, RANGE={rg}, PL={pl}, B={b}, M={m}\n")

        except subprocess.CalledProcessError as e:
            print(f"Error in {dataset}: K={k}, L={l}, iter={iter}, S={s}, R={r}, RANGE={rg}, PL={pl}, B={b}, M={m}")
            print(f"Details in: {log_prefix}_stderr.log")
            with open(res_path, "a") as res_f:
                res_f.write(f"Failed: K={k}, L={l}, iter={iter}, S={s}, R={r}, RANGE={rg}, PL={pl}, B={b}, M={m} (exit code {e.returncode})\n")


    print(f"Search for {dataset}: K={k}, L={l}, iter={iter}, S={s}, R={r}, RANGE={rg}, PL={pl}, B={b}, M={m} done!")

print("All parameter combinations processed!")