import os
import pandas as pd
import numpy as np

Ms = list(range(16, 65, 8)) 
efcs = list(range(10, 351, 20))

dataset = "ytb_audio"
scenarios = ["and", "or", "equal"]

base_path_template = "../data/result/ivfpq/{dataset}/{scenario}/M={m}_efc={efc}"

for m in Ms:
    for efc in efcs:
        
        all_recalls = []
        all_qps = []
        
        for scenario in scenarios:
            # Construct path for each scenario
            current_path = base_path_template.format(
                dataset=dataset,
                scenario=scenario,
                m=m,
                efc=efc,
            )
            csv_file = current_path + "_result.csv"
            
            if not os.path.exists(csv_file):
                print(f"Warning: File not found - {csv_file}")
                continue
            
            try:
                data = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
            
            if 'Recall' not in data.columns or 'QPS' not in data.columns:
                print(f"Warning: Missing columns in {csv_file}")
                continue
            
            # Remove the first two data points
            recalls = data['Recall'].values[2:]
            qps_values = data['QPS'].values[2:]
            
            all_recalls.append(recalls.tolist())
            all_qps.append(qps_values.tolist())
        
        if not all_recalls or not all_qps:
            print(f"No valid data for M{m}_efc{efc}")
            continue
        
        assert len(all_recalls) == 3, f"Expecting 3 scenarios, got {len(all_recalls)}"
        assert len(all_qps) == 3, f"Expecting 3 scenarios, got {len(all_qps)}"
        # Calculate combined Recall (average) by column
        combined_recall = []
        for i in range(len(all_recalls[0])):
            combined_recall.append((all_recalls[0][i] + all_recalls[1][i] + all_recalls[2][i]) / 3)

        # Calculate combined QPS (average) by column
        combined_qps = []
        # qps = 3/(1/qps1+1/qps2+1/qps3)
        for i in range(len(all_qps[0])):
            combined_qps.append(3 / (1/all_qps[0][i] + 1/all_qps[1][i] + 1/all_qps[2][i]))
        
        # Prepare output directory and file
        output_dir = f"../data/result/ivfpq/{dataset}/combine/M{m}_efc{efc}/"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "small_combine.csv")
        
        # Save results, QPS, recall each column
        result_df = pd.DataFrame({
            'Recall': combined_recall,
            'QPS': combined_qps
        })
        result_df.to_csv(output_file, index=False)