import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[1]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_src_paths_to_sys()

import json, os

from visualize_globally import Global_Visualizer
from visualization_tools import *

#seeds = [1, 2, 25, 26, 75, 76, 100, 101, 150, 151]
seeds = [1, 2, 25]
example_config = ProjectPaths.logreg_experiments_path() / "example_config_temporal.json"
config = json.load(open(example_config))
config_stem = example_config.stem
folder_name = ProjectPaths.logreg_experiments_path() / f"{config_stem}_seeds"
os.makedirs(folder_name, exist_ok = True)

for seed in seeds:
    config["model_random_state"] = seed
    config["cache_random_state"] = seed
    config["ranking_random_state"] = seed
    config_name = folder_name / f"{config_stem}_s{seed}.json"
    with open(config_name, "w") as f:
        json.dump(config, f, indent = 4)

os.system(f"python run.py --config_path {folder_name}")

seeds_df = pd.DataFrame()
for seed in seeds:
    outputs_folder = ProjectPaths.logreg_outputs_path() / f"{config_stem}_s{seed}"
    config, users_info, hyperparameters_combinations, results_before_averaging_over_folds = load_outputs_files(outputs_folder)
    gv = Global_Visualizer(config, users_info, hyperparameters_combinations, results_before_averaging_over_folds, outputs_folder, Score.BALANCED_ACCURACY, False)
    hyperparameter_results_df = gv.results_after_averaging_over_users
    seeds_df = pd.concat([seeds_df, hyperparameter_results_df], axis = 0)
means_df = seeds_df.groupby('combination_idx').mean()
stds_df = seeds_df.groupby('combination_idx').std(ddof = 0)

relevant_scores = ["recall", "specificity", "balanced_accuracy", "ndcg", "mrr", "info_nce", "ndcg_samples", "mrr_samples", "info_nce_samples", "ndcg_all", "mrr_all", "info_nce_all"]
for score in relevant_scores:
    score_name = "val_" + score + "_mean"
    mean = means_df[score_name].mean()
    std = stds_df[score_name].mean()
    if score not in ["info_nce", "info_nce_samples", "info_nce_all"]:
        mean, std = mean * 100, std * 100
    print(f"Score: {score_name}, Mean: {mean}, Std: {std}")