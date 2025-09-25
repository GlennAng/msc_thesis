import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from ..src.load_files import TEST_RANDOM_STATES
from ..src.project_paths import ProjectPaths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    return parser.parse_args()


args = parse_arguments()
config_path = Path(args.config_path).resolve()
config_stem = config_path.stem
config = json.load(open(config_path, "r"))
assert not config["save_users_predictions"]
assert not config["save_users_coefs"]
configs_folder = ProjectPaths.logreg_experiments_path() / (config_stem + "_seeds")
if configs_folder.exists():
    shutil.rmtree(configs_folder)
os.makedirs(configs_folder, exist_ok=True)


start_time = time.time()
for random_state in TEST_RANDOM_STATES:
    config_random_state = config.copy()
    config_random_state["model_random_state"] = random_state
    config_random_state["cache_random_state"] = random_state
    config_random_state["ranking_random_state"] = random_state
    config_file = configs_folder / f"{config_stem}_s{random_state}.json"
    with open(config_file, "w") as f:
        json.dump(config_random_state, f, indent=4)
    print(f"Running {config_file} ...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "code.logreg.src.main",
            str(config_file),
        ],
        check=True,
    )
time_taken = time.time() - start_time
shutil.rmtree(configs_folder, ignore_errors=True)

results = []
for random_state in TEST_RANDOM_STATES:
    outputs_file = (
        ProjectPaths.logreg_outputs_path()
        / (config_stem + f"_s{random_state}")
        / "users_results.csv"
    )
    results.append(pd.read_csv(outputs_file))

stacked_df = pd.concat(results, ignore_index=True)
group_cols = ["user_id", "fold_idx", "combination_idx"]
grouped = stacked_df.groupby(group_cols)
averaged_df = grouped.mean().reset_index()

first_random_state = TEST_RANDOM_STATES[0]
source_folder = ProjectPaths.logreg_outputs_path() / (config_stem + f"_s{first_random_state}")
outputs_folder = ProjectPaths.logreg_outputs_path() / (config_stem + "_averaged")
if outputs_folder.exists():
    shutil.rmtree(outputs_folder)
shutil.copytree(source_folder, outputs_folder)
averaged_results_file = outputs_folder / "users_results.csv"
if averaged_results_file.exists():
    averaged_results_file.unlink()
averaged_df.to_csv(averaged_results_file, index=False)
config_file = outputs_folder / "config.json"
config = json.load(open(config_file, "r"))
config["time_elapsed"] = time_taken
config["model_random_state"] = TEST_RANDOM_STATES
config["cache_random_state"] = TEST_RANDOM_STATES
config["ranking_random_state"] = TEST_RANDOM_STATES
with open(config_file, "w") as f:
    json.dump(config, f, indent=4)

subprocess.run(
    [
        sys.executable,
        "-m",
        "code.logreg.src.visualization.visualize_globally",
        "--outputs_folder",
        str(outputs_folder),
        "--score",
        "ndcg_all",
    ],
    check=True,
)
files = [
    f.name
    for f in outputs_folder.iterdir()
    if f.name.startswith("global_visu") and f.name.endswith(".pdf")
]
if files:
    print(f"Saved visualization to {outputs_folder / files[0]}")

for random_state in TEST_RANDOM_STATES:
    outputs_folder = ProjectPaths.logreg_outputs_path() / (config_stem + f"_s{random_state}")
    if outputs_folder.exists():
        shutil.rmtree(outputs_folder)
