from visualize_globally import Global_Visualizer
from visualization_tools import *
import json
import os
seeds = [1, 2, 25, 26, 75, 76, 100, 101, 150, 151]

example_config = "experiments/gte_large_384_no_categories.json"
model_name = example_config.split("/")[-1].split(".")[0]
os.makedirs(f"experiments/{model_name}_seeds", exist_ok = True)

config = json.load(open(example_config))

for seed in seeds:
    config["model_random_state"] = seed
    config["cache_random_state"] = seed
    config["ranking_random_state"] = seed
    with open(f"experiments/{model_name}_seeds/{model_name}_s{seed}.json", "w") as f:
        json.dump(config, f, indent = 4)

os.system(f"python run_cross_eval.py --config_path experiments/{model_name}_seeds")

seeds_df =  pd.DataFrame()
for seed in seeds:
    outputs_folder = f"outputs/{model_name}_s{seed}"
    config, users_info, hyperparameters_combinations, results_before_averaging_over_folds = load_outputs_files(outputs_folder)
    gv = Global_Visualizer(config, users_info, hyperparameters_combinations, results_before_averaging_over_folds, outputs_folder, Score.BALANCED_ACCURACY, False)
    hyperparameter_results_df = gv.results_after_averaging_over_users
    seeds_df = pd.concat([seeds_df, hyperparameter_results_df], axis = 0)
means_df = seeds_df.groupby('combination_idx').mean()
stds_df = seeds_df.groupby('combination_idx').std(ddof = 0)

relevant_scores = ["ndcg", "mrr", "info_nce", "ndcg_samples", "mrr_samples", "info_nce_samples", "ndcg_all", "mrr_all", "info_nce_all"]
for score in relevant_scores:
    score_name = "val_" + score + "_mean"
    mean = means_df[score_name].mean()
    std = stds_df[score_name].mean()
    if score not in ["info_nce", "info_nce_samples", "info_nce_all"]:
        mean, std = mean * 100, std * 100
    print(f"Score: {score_name}, Mean: {mean}, Std: {std}")