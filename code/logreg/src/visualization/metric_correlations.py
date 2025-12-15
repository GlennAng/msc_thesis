import csv
import pandas as pd
import numpy as np

file = "code/logreg/outputs/example_config_averaged/users_results.csv"

df = pd.read_csv(file)

metrics = ["val_recall", "val_specificity", "val_balanced_accuracy", "val_ndcg_all", "val_mrr_all", "val_hit_rate_at_1_all"]
correlation_matrix = df[metrics].corr()
print(correlation_matrix)

def ndcg(rank):
    return 1.0 / np.log2(rank + 1)

N = [4, 100, 104]

for n in N:
    ranks = list(range(1, n + 2))
    dcg_values = np.array([ndcg(rank) for rank in ranks])
    mean_dcg = np.mean(dcg_values)
    if n == 4:
        print(dcg_values)
    print(f"Mean NDCG for N={n}: {mean_dcg}")

ranks = [1, 2, 3, 4, 5, 10, 20, 50, 100]
ndcg_values = [ndcg(rank) for rank in ranks]
for rank, value in zip(ranks, ndcg_values):
    print(f"NDCG for rank {rank}: {value}")