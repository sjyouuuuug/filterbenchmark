import os
import pandas as pd
import numpy as np

# 参数定义
Ms = list(range(16, 65, 8)) # 
M_betas = list(range(16, 65, 8)) # 
gamma = 1

dataset = "ytb_audio"

base_path_template = "../data/result/ACORN/{dataset}/combine/M{m}_M_beta{mb}_gamma{gamma}"

# 目标 Recall
target_recalls = [60, 70, 80]

results = []

# 遍历参数
for m in Ms:
    for mb in M_betas:
        if mb < m:
            continue

        result_path = base_path_template.format(dataset=dataset, m=m, mb=mb, gamma=gamma)
        csv_file = os.path.join(result_path, "small_combine.csv")
        
        if os.path.exists(csv_file):
            data = pd.read_csv(csv_file)

            if 'Recall' not in data.columns or 'QPS' not in data.columns:
                print(f"Warning: 'Recall' or 'QPS' column missing in {csv_file}")
                continue
            
            recall = data['Recall'].values
            qps = data['QPS'].values

            # 按 Recall 排序
            sorted_indices = np.argsort(recall)
            recall = recall[sorted_indices]
            qps = qps[sorted_indices]

            if len(recall) < 2:
                print(f"Warning: Not enough data points in {csv_file} after removing first two points.")
                interpolated_qps = {target: np.nan for target in target_recalls}
            else:
                interpolated_qps = {}
                for target in target_recalls:
                    if target < recall.min():
                        x0, x1 = recall[0], recall[1]
                        y0, y1 = qps[0], qps[1]
                        if x1 == x0:
                            interpolated_qps[target] = np.nan
                        else:
                            slope = (y1 - y0) / (x1 - x0)
                            qps_at_target = y0 + slope * (target - x0)
                            interpolated_qps[target] = qps_at_target
                    elif target > recall.max():
                        interpolated_qps[target] = np.nan
                    else:
                        qps_at_target = np.interp(target, recall, qps)
                        interpolated_qps[target] = qps_at_target
            
            results.append({
                '--M': m,
                '--M_beta': mb,
                '--gamma': gamma,
                'QPS_60': interpolated_qps.get(60, np.nan),
                'QPS_70': interpolated_qps.get(70, np.nan),
                'QPS_80': interpolated_qps.get(80, np.nan)
            })
        else:
            print(f"Warning: File not found - {csv_file}")

# 转换为 DataFrame
df = pd.DataFrame(results)

# 计算 rank
df['rank1'] = df['QPS_60'].rank(method='min', ascending=False)
df['rank2'] = df['QPS_70'].rank(method='min', ascending=False)
df['rank3'] = df['QPS_80'].rank(method='min', ascending=False)

# 处理缺失值
df.loc[df['QPS_60'].isna(), 'rank1'] = np.nan
df.loc[df['QPS_70'].isna(), 'rank2'] = np.nan
df.loc[df['QPS_80'].isna(), 'rank3'] = np.nan

# 计算 rank 总和
df['rank_sum'] = df[['rank1', 'rank2', 'rank3']].sum(axis=1)

# 手动划分三维参数空间为 3x3x3
md_bins = np.linspace(min(Ms), max(Ms), 7)  # 3 区间
lb_bins = np.linspace(min(M_betas), max(M_betas), 7)          # 3 区间

# 为每个参数添加子空间标签
df['m_bin'] = pd.cut(df['--M'], bins=md_bins, labels=False, include_lowest=True)
df['mb_bin'] = pd.cut(df['--M_beta'], bins=lb_bins, labels=False, include_lowest=True)
# df['gamma_bin'] = pd.cut(df['--gamma'], bins=gamma_bins, labels=False, include_lowest=True)

# 按子空间选择 rank_sum 最小的代表元
representatives = (
    df.groupby(['m_bin', 'mb_bin'])
    .apply(lambda group: group.loc[group['rank_sum'].idxmin()])
    .reset_index(drop=True)
)

## remove such representatives that M_beta < M
representatives = representatives[representatives['--M_beta'] >= representatives['--M']]
representatives = representatives.reset_index(drop=True)

representatives[['--M', '--M_beta', '--gamma']].to_csv(f"../data/result/ACORN/{dataset}/representative_parameters_1.csv", index=False)

print("代表元已保存至 representative_parameters_1.csv")