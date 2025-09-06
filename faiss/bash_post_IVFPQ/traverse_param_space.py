import subprocess
import os
import csv
import time

# Define parameter ranges 
nlists = list(range(100, 1000, 50)) 
# Ms = [16]
ms = [8, 12, 16, 24, 32]
nbits = [2, 4, 6, 8]
# efcs = [100]

# Constants
datasets = ["yfcc"]
# datasets = ["arxiv"]
for dataset in datasets:
    base_file = "../data/" + dataset + "/" + dataset + "_small.fvecs"
    output_path_base = "../data/small_index_files/ivfpq"

    # Loop through parameter combinations
    for nlist in nlists:
        for m in ms:
            for nbit in nbits:
                # Create output directory
                dir_name = f"nlist={nlist}_m={m}_nbits={nbit}"
                output_dir = os.path.join(output_path_base, dataset, dir_name)
                os.makedirs(output_dir, exist_ok=True)

                cmd = ['./build/tutorial/cpp/build_IVFPQ_index', base_file, str(nlist), str(m), str(nbit), output_path_base, dataset]

                env = os.environ.copy()
                env['debugSearchFlag'] = '0'

                try:
                    with open(os.path.join(output_dir, 'output.log'), 'w') as log_file:
                        subprocess.run(cmd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    print(f"Error running nlist={nlist}, M={m}, nbits={nbit}")
                    print(e)
                    continue
