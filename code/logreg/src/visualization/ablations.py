from pathlib import Path
import json
import pandas as pd
import numpy as np
import os
import sys
from .visualize_globally import Global_Visualizer
from .visualization_tools import load_outputs_files
from matplotlib.backends.backend_pdf import PdfPages
from ....logreg.src.training.scores_definitions import Score



folder = Path(sys.argv[1])

hyperparameters_combinations = []
for subfolder in folder.iterdir():
    if subfolder.is_dir():
        eval_settings = json.load(open(subfolder / "eval_settings.json"))
        weights_neg_scale = eval_settings["logreg_weights_neg_scale"]
        weights_cache_v = eval_settings["logreg_weights_cache_v"]
        clf_C = eval_settings["logreg_clf_C"]
        tuple = (weights_neg_scale, weights_cache_v, clf_C, subfolder.name)
        hyperparameters_combinations.append(tuple)
hyperparameters_combinations = sorted(hyperparameters_combinations, key=lambda x: (x[0], x[1], x[2]))
hyperparameters_combinations = [(idx, *vals) for idx, vals in enumerate(hyperparameters_combinations)]
hyperparameters_combinations = pd.DataFrame(hyperparameters_combinations, columns=["combination_idx", "weights_neg_scale", "weights_cache_v", "clf_C", "subfolder_name"])
new_folder = Path("code/logreg/outputs/ablations")
os.makedirs(new_folder, exist_ok=True)
# remove subfolder_name column
hyperparameters_combinations_save = hyperparameters_combinations.drop(columns=["subfolder_name"])
hyperparameters_combinations_save.to_csv(new_folder / "hyperparameters_combinations.csv", index=False)

users_results = []
for idx, weights_neg_scale, weights_cache_v, clf_C, subfolder_name in hyperparameters_combinations.values:
    subfolder = folder / subfolder_name
    users_results_subfolder = subfolder / "outputs" / "users_results.csv"
    users_results_df = pd.read_csv(users_results_subfolder)
    users_results_df["combination_idx"] = idx
    users_results.append(users_results_df)
users_results = pd.concat(users_results, ignore_index=True)
users_results.to_csv(new_folder / "users_results.csv", index=False)

first_dir = hyperparameters_combinations.iloc[0]["subfolder_name"]
users_info = pd.read_csv(folder / first_dir / "outputs" / "users_info.csv")
all_configs = folder / first_dir / "configs"
first_config_file = list(all_configs.iterdir())[0]
with open(first_config_file, "r") as f:
    config = json.load(f)
users_info.to_csv(new_folder / "users_info.csv", index=False)
with open(new_folder / "config.json", "w") as f:
    json.dump(config, f, indent=4)

config, users_info, hyperparameters_combinations, results_before_averaging_over_folds = (
    load_outputs_files(new_folder)
)
gv = Global_Visualizer(
    config=config,
    users_info=users_info,
    hyperparameters_combinations=hyperparameters_combinations,
    results_before_averaging_over_folds=results_before_averaging_over_folds,
    folder=new_folder,
    score=Score.MSC_AUC,
)

print(gv.results_after_averaging_over_users)

file_name = new_folder / "plots.pdf"
with PdfPages(file_name) as pdf:
    #gv.generate_fourth_page(pdf=pdf)
    gv.generate_plots(pdf=pdf)


        
        

