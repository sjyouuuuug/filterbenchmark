import os
import subprocess

def run_batch_queries(base_path, label_path, query_path, query_label_path, index_folder, gt_path, nb_list, dataset, method="kmeans", flag="1", top_k="1000"):
    for nb in nb_list:
        index_path = os.path.join(index_folder, dataset, f"nb_{nb}")
        output_csv = "../equal_length_experiment/result/caps/" + dataset + "/" + f"nb_{nb}_results.csv"
        
        command = (
            f"./query {base_path} {label_path} {query_path} {query_label_path} "
            f"{index_path} {gt_path} {output_csv} {nb} {method} {flag} {top_k}"
        )
        
        print(f"Running command: {command}")
        
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running command: {command}")
            print(e)

if __name__ == "__main__":
    base_path = "../equal_length_experiment/arxiv/arxiv_base.fvecs"
    label_path = "../equal_length_experiment/arxiv/label_NHQ_base.txt"
    query_path = "../equal_length_experiment/arxiv/arxiv_query_NHQ.fvecs"
    query_label_path = "../equal_length_experiment/arxiv/label_NHQ_query.txt"
    index_folder = "../equal_length_experiment/index_files/caps"
    gt_path = "../equal_length_experiment/arxiv/arxiv_gt_NHQ.txt"
    dataset = "arxiv"

    nb_list = [128, 256, 512, 1024, 2048, 4096]
    # nb_list = [1024]
 
    run_batch_queries(base_path, label_path, query_path, query_label_path, index_folder, gt_path, nb_list, dataset)