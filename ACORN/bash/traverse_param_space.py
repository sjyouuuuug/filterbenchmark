import subprocess
import os
import csv
import time

# Define parameter ranges 
Ms = list(range(16, 65, 8)) # 
M_betas = list(range(16, 73, 8)) # 
# M_betas = [72]
# gammas = list(range(12, 97, 12)) # 10
# gammas.append(2)
# gammas.append(4)
# gammas.append(8)
# gammas = [2]

gammas = [1]

# Constants
N = 50000
datasets = ["yfcc"]
for dataset in datasets:
    base_file = "../data/" + dataset + "/" + dataset + "_small.fvecs"
    output_path_base = "../data/small_index_files/ACORN"

    # Loop through parameter combinations
    for M in Ms:
        for M_beta in M_betas:
            if M_beta < M:
                continue
            
            for gamma_val in gammas:
                if M_beta > 2*M*gamma_val:
                    continue
                print(f"creating M={M}, Mb={M_beta}, gamma={gamma_val}")
                # Create output directory
                dir_name = f"M={M}_Mb={M_beta}_gamma={gamma_val}"
                output_dir = os.path.join(output_path_base, dataset, dir_name)
                os.makedirs(output_dir, exist_ok=True)

                # # Check if this combination has already been run
                # if os.path.exists(os.path.join(output_dir, 'output.log')):
                #     print(f"Skipping M={M}, M_beta={M_beta}, gamma={gamma_val} (already run)")
                #     continue

                # Construct command
                cmd = ['./build/demos/build_acorn_index', str(N), str(gamma_val),
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
