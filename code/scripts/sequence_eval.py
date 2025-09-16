import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from ..sequence.src.eval.compute_users_embeddings import (
    USERS_SELECTIONS_CHOICES,
    VALID_EMBED_FUNCTIONS,
    VALID_EMBED_FUNCTIONS_RANDOMNESS,
)
from ..src.load_files import TEST_RANDOM_STATES, VAL_RANDOM_STATE
from ..src.project_paths import ProjectPaths
from .create_example_configs import (
    create_example_config,
    create_example_config_sliding_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute users embeddings")

    parser.add_argument("--embed_function", type=str, choices=VALID_EMBED_FUNCTIONS, required=True)
    parser.add_argument("--embedding_path", type=str, required=False, default=None)

    parser.add_argument("--histories_hard_constraint_min_n_train_posrated", type=int, default=10)
    parser.add_argument("--histories_hard_constraint_max_n_train_rated", type=int, default=None)
    parser.add_argument("--histories_soft_constraint_max_n_train_sessions", type=int, default=None)
    parser.add_argument("--histories_soft_constraint_max_n_train_days", type=int, default=None)
    parser.add_argument("--histories_remove_negrated_from_history", action="store_true", default=False)
    parser.add_argument("--single_random_state", action="store_true", default=False)
    parser.add_argument("--single_val_session", action="store_true", default=False)
    parser.add_argument("--use_existing_embeddings", action="store_true", default=False)

    parser.add_argument("--old_ratings", action="store_true", default=False)
    parser.add_argument(
        "--users_selection", type=str, default=None, choices=USERS_SELECTIONS_CHOICES
    )
    args_dict = vars(parser.parse_args())
    return args_dict


def get_output_folder(args_dict: dict) -> Path:
    s = args_dict["embed_function"]
    if args_dict["users_selection"] is not None:
        s += f"_{args_dict['users_selection']}"
    else:
        s += "_all"
    s += f"_pos_{args_dict['histories_hard_constraint_min_n_train_posrated']}"
    s += f"_sess_{args_dict['histories_soft_constraint_max_n_train_sessions']}"
    s += f"_days_{args_dict['histories_soft_constraint_max_n_train_days']}"
    if args_dict["single_val_session"]:
        s += "_single"
    return ProjectPaths.sequence_data_users_embeddings_path() / s


def process_args_dict(args_dict: dict) -> None:
    if args_dict["single_random_state"]:
        args_dict["embed_random_states"] = [VAL_RANDOM_STATE]
        args_dict["eval_random_states"] = [VAL_RANDOM_STATE]
    else:
        args_dict["eval_random_states"] = TEST_RANDOM_STATES
        randomness = VALID_EMBED_FUNCTIONS_RANDOMNESS[args_dict["embed_function"]]
        if randomness:
            args_dict["embed_random_states"] = TEST_RANDOM_STATES
        else:
            args_dict["embed_random_states"] = [VAL_RANDOM_STATE]
    if args_dict["embedding_path"] is None:
        args_dict["embedding_path"] = (
            ProjectPaths.logreg_embeddings_path()
            / "after_pca"
            / "gte_large_256_categories_l2_unit_100"
        )
    else:
        args_dict["embedding_path"] = Path(args_dict["embedding_path"]).resolve()
    args_dict["output_folder"] = get_output_folder(args_dict)
    args_dict["stem"] = args_dict["output_folder"].stem


def compute_users_embeddings(args_dict: dict) -> None:
    config_path = args_dict["output_folder"] / "config.pkl"
    with open(config_path, "wb") as f:
        pickle.dump(args_dict, f)
    for random_state in args_dict["embed_random_states"]:
        args = [
            sys.executable,
            "-m",
            "code.sequence.src.eval.compute_users_embeddings",
            "--config_file",
            str(config_path),
            "--random_state",
            str(random_state),
        ]
        if args_dict["users_selection"] is not None:
            args.extend(["--users_selection", args_dict["users_selection"]])
        subprocess.run(args, check=True)


def run_logreg_configs(args_dict: dict) -> None:
    if args_dict["single_val_session"]:
        example_config = create_example_config()
        example_config["load_users_coefs"] = True
        example_config["users_ratings_selection"] = "session_based_filtering"
    else:
        example_config = create_example_config_sliding_window()
    if args_dict["old_ratings"]:
        example_config["users_ratings_selection"] = "session_based_filtering_old"
    if args_dict["users_selection"] is not None:
        example_config["relevant_users_ids"] = args_dict["users_selection"]
    example_config["embedding_folder"] = str(args_dict["embedding_path"])
    configs_path = args_dict["output_folder"] / "experiments"
    os.makedirs(configs_path, exist_ok=True)
    for random_state in args_dict["eval_random_states"]:
        example_config["model_random_state"] = random_state
        example_config["cache_random_state"] = random_state
        example_config["ranking_random_state"] = random_state
        if len(args_dict["embed_random_states"]) == 1:
            example_config["users_coefs_path"] = str(
                args_dict["output_folder"]
                / "users_embeddings"
                / f"s_{args_dict['embed_random_states'][0]}"
            )
        else:
            example_config["users_coefs_path"] = str(
                args_dict["output_folder"] / "users_embeddings" / f"s_{random_state}"
            )
        config_path = configs_path / f"seq_{args_dict['stem']}_s{random_state}.json"
        with open(config_path, "w") as f:
            json.dump(example_config, f)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "code.logreg.src.main",
                str(config_path),
            ],
            check=True,
        )


def create_output_file(args_dict: dict) -> None:
    results = []
    for random_state in args_dict["eval_random_states"]:
        outputs_file = (
            ProjectPaths.logreg_outputs_path()
            / f"seq_{args_dict['stem']}_s{random_state}"
            / "users_results.csv"
        )
        results.append(pd.read_csv(outputs_file))

    stacked_df = pd.concat(results, ignore_index=True)
    group_cols = ["user_id", "fold_idx", "combination_idx"]
    grouped = stacked_df.groupby(group_cols)
    averaged_df = grouped.mean().reset_index()

    first_random_state = args_dict["eval_random_states"][0]
    source_folder = (
        ProjectPaths.logreg_outputs_path() / f"seq_{args_dict['stem']}_s{first_random_state}"
    )
    outputs_folder = ProjectPaths.logreg_outputs_path() / f"seq_{args_dict['stem']}_averaged"

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
    config["model_random_state"] = args_dict["eval_random_states"]
    config["cache_random_state"] = args_dict["eval_random_states"]
    config["ranking_random_state"] = args_dict["eval_random_states"]
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
    print(f"Saved averaged results to {averaged_results_file}.")
    for random_state in args_dict["eval_random_states"]:
        outputs_folder = (
            ProjectPaths.logreg_outputs_path() / f"seq_{args_dict['stem']}_s{random_state}"
        )
        if outputs_folder.exists():
            shutil.rmtree(outputs_folder)


if __name__ == "__main__":
    args_dict = parse_args()
    process_args_dict(args_dict=args_dict)
    os.makedirs(args_dict["output_folder"], exist_ok=True)
    if args_dict["use_existing_embeddings"]:
        for random_state in args_dict["embed_random_states"]:
            embedding_path = args_dict["output_folder"] / "users_embeddings" / f"s_{random_state}"
            assert embedding_path.exists()
    else:
        compute_users_embeddings(args_dict=args_dict)
    start_time = time.time()
    run_logreg_configs(args_dict=args_dict)
    time_taken = time.time() - start_time
    create_output_file(args_dict=args_dict)
