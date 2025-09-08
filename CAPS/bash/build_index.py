import os
import subprocess

def run_batch_commands(base_path, label_path, index_folder, nb_list, dataset, method="kmeans", flag="1"):
    for nb in nb_list:
        output_path = os.path.join(index_folder, dataset, f"nb_{nb}")
        log_file_path = os.path.join(output_path, "output.log")
      
        command = f"./index {base_path} {label_path} {output_path} {nb} {method} {flag}"
        
        print(f"Running command: {command}")

        # mkdir 
        os.makedirs(output_path, exist_ok=True)
        
        try:
            with open(log_file_path, 'w') as log_file:
                subprocess.run(command, shell=True, check=True, stdout=log_file)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running command: {command}")
            print(e)

if __name__ == "__main__":
    base_path = "../equal_length_experiment/arxiv/arxiv_base.fvecs"
    label_path = "../equal_length_experiment/arxiv/label_NHQ_base.txt"
    index_folder = "../equal_length_experiment/index_files/caps"
    dataset = "arxiv"

    nb_list = list(range(50, 2000, 50))
    # nb_list = [1024]
    
    run_batch_commands(base_path, label_path, index_folder, nb_list, dataset)