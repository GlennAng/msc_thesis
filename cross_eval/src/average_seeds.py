from visualize_globally import Global_Visualizer
from visualization_tools import *
import json
import os
seeds = [1, 2, 25, 26, 42, 43, 75, 76, 150, 151]

model_name = "gte_large_256_categories_new"

example_config = "experiments/gte_large_256_no_overlap_categories.json"
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
means_df = seeds_df.groupby('combination_idx').mean() * 100
stds_df = seeds_df.groupby('combination_idx').std(ddof = 0) * 100



relevant_scores = ["ndcg", "mrr", "ndcg_top_4_samples", "mrr_top_4_samples"]
for score in relevant_scores:
    score = "val_" + score + "_mean"
    print(f"Score: {score}, Mean: {means_df[score].mean()}, Std: {stds_df[score].mean()}")
    