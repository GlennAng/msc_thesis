import numpy as np
import json
import os
import pandas as pd
import pickle
from finetuning_evaluation import scores_table_to_df
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
tau_values = [0.01, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0, 100.0]

example_configs = ["experiments/tfidf.json", "experiments/gte_large_256_categories.json", "experiments/gte_large_384_no_categories.json"]
configs_titles = ["TF-IDF", "GTE-Large 256 Categories", "GTE-Large 384 No Categories"]
dir = "experiments/info_nce_tau"
os.makedirs(dir, exist_ok = True)

for example_config in example_configs:
    example_config_name = example_config.split("/")[-1].split(".")[0]
    example_config = json.load(open(example_config))
    for i, tau in enumerate(tau_values):
        tau_str = i
        example_config["info_nce_temperature"] = tau
        with open(f"{dir}/{example_config_name}_tau_{tau_str}.json", "w") as f:
            json.dump(example_config, f, indent = 4)
os.system(f"python run_cross_eval.py --config_path {dir} --save_scores_tables")
example_configs_names = [example_config.split("/")[-1].split(".")[0] for example_config in example_configs]
info_nce_scores = {example_config_name : [] for example_config_name in example_configs_names}
info_nce_scores_samples = {example_config_name : [] for example_config_name in example_configs_names}
info_nce_scores_all = {example_config_name : [] for example_config_name in example_configs_names}
scores_array = [info_nce_scores, info_nce_scores_samples, info_nce_scores_all]
scores_titles = ["InfoNCE (4 Explicit Negatives)", "InfoNCE (100 Random Negatives)", "InfoNCE (4 Explicit Negatives + 100 Random Negatives)"]

for example_config_name in example_configs_names:
    for i, tau in enumerate(tau_values):
        tau_value = tau_values[i]
        outputs_folder = f"outputs/{example_config_name}_tau_{i}"
        with open(f"{outputs_folder}/scores_table_1.pkl", "rb") as f:
            scores_table_1 = pickle.load(f)
        with open(f"{outputs_folder}/scores_table_2.pkl", "rb") as f:
            scores_table_2 = pickle.load(f)
        scores_table_1, scores_table_2 = scores_table_to_df(scores_table_1), scores_table_to_df(scores_table_2)
        scores_df = pd.concat([scores_table_1, scores_table_2[1:]], axis = 0)
        scores_df.set_index("Score", inplace = True)
        info_nce_scores[example_config_name].append(scores_df.loc["INCE", "Validation"])
        info_nce_scores_samples[example_config_name].append(scores_df.loc["INCE\nSmpl", "Validation"])
        info_nce_scores_all[example_config_name].append(scores_df.loc["INCE\nAll", "Validation"])

print(tau_values)
for scores, title in zip(scores_array, scores_titles):
    print(f"\n{title}")
    print(scores)