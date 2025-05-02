from visualize_globally import Global_Visualizer
from visualization_tools import *

outputs_names = "gte_large"
seeds = [1, 2, 25, 26, 42, 43, 75, 76, 150, 151]
seeds_df =  pd.DataFrame()
for seed in seeds:
    outputs_folder = f"outputs/{outputs_names}_s{seed}"
    config, users_info, hyperparameters_combinations, results_before_averaging_over_folds = load_outputs_files(outputs_folder)
    gv = Global_Visualizer(config, users_info, hyperparameters_combinations, results_before_averaging_over_folds, outputs_folder, Score.BALANCED_ACCURACY, False)
    hyperparameter_results_df = gv.results_after_averaging_over_users
    seeds_df = pd.concat([seeds_df, hyperparameter_results_df], axis = 0)
means_df = seeds_df.groupby('combination_idx').mean() * 100
stds_df = seeds_df.groupby('combination_idx').std(ddof = 0) * 100

print(seeds_df["val_mrr_top_4_samples_mean"])

relevant_scores = ["balanced_accuracy", "auroc_ranking", "ndcg", "mrr", "auroc_top_4_samples", "ndcg_top_4_samples", "mrr_top_4_samples"]
for score in relevant_scores:
    score = "val_" + score + "_mean"
    print(f"Score: {score}, Mean: {means_df[score].mean()}, Std: {stds_df[score].mean()}")
    