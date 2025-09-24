import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from ..sequence.src.eval.compute_users_embeddings_utils import (
    EmbedFunction,
    get_embed_function_from_arg,
    get_eval_data_folder,
    get_users_selections_choices,
)
from ..src.load_files import TEST_RANDOM_STATES, VAL_RANDOM_STATE
from ..src.project_paths import ProjectPaths
from .create_example_configs import (
    create_example_config,
    create_example_config_sliding_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute users embeddings")

    parser.add_argument("--embed_function", type=str, required=True)
    parser.add_argument("--users_selection", type=str, required=False, default=None)
    parser.add_argument("--papers_embedding_path", type=str, required=False, default=None)

    parser.add_argument("--single_random_state", action="store_true", default=False)
    parser.add_argument("--single_val_session", action="store_true", default=False)

    parser.add_argument("--use_existing_users_embeddings", action="store_true", default=False)
    parser.add_argument("--existing_users_embeddings_path", type=str, required=False, default=None)

    parser.add_argument("--old_ratings", action="store_true", default=False)

    parser.add_argument("--histories_hard_constraint_min_n_train_posrated", type=int, default=10)
    parser.add_argument("--histories_hard_constraint_max_n_train_rated", type=int, default=None)
    parser.add_argument("--histories_soft_constraint_max_n_train_sessions", type=int, default=None)
    parser.add_argument("--histories_soft_constraint_max_n_train_days", type=int, default=None)
    parser.add_argument(
        "--histories_remove_negrated_from_history", action="store_true", default=False
    )

    args_dict = vars(parser.parse_args())
    return args_dict


def process_neural_precomputed_users_embeddings(args_dict: dict) -> None:
    if args_dict["embed_function"] == EmbedFunction.NEURAL_PRECOMPUTED:
        assert args_dict["users_selection"] == "sequence_test"
        assert args_dict["papers_embedding_path"] is not None
        assert not args_dict["single_random_state"]
        assert not args_dict["single_val_session"]
        assert args_dict["use_existing_users_embeddings"]


def process_papers_embedding_path(args_dict: dict) -> None:
    if args_dict["papers_embedding_path"] is None:
        papers_embedding_path = (
            ProjectPaths.logreg_embeddings_path()
            / "after_pca"
            / "gte_large_256_categories_l2_unit_100"
        )
    else:
        papers_embedding_path = Path(args_dict["papers_embedding_path"]).resolve()
    assert papers_embedding_path.exists()
    if (papers_embedding_path / "abs_X.npz").exists():
        tf_idf = True
    elif (papers_embedding_path / "abs_X.npy").exists():
        tf_idf = False
    else:
        raise ValueError(f"Invalid papers_embedding_path {papers_embedding_path}.")
    if args_dict["embed_function"] == EmbedFunction.NEURAL_PRECOMPUTED:
        assert not tf_idf
    args_dict["papers_embedding_path"] = papers_embedding_path
    args_dict["tf_idf"] = tf_idf


def process_random_states(args_dict: dict) -> None:
    if args_dict["single_random_state"]:
        args_dict["embed_random_states"] = [VAL_RANDOM_STATE]
        args_dict["eval_random_states"] = [VAL_RANDOM_STATE]
    else:
        args_dict["eval_random_states"] = TEST_RANDOM_STATES
        ef = args_dict["embed_function"]
        if ef in [EmbedFunction.MEAN_POS_POOLING, EmbedFunction.NEURAL_PRECOMPUTED]:
            args_dict["embed_random_states"] = [VAL_RANDOM_STATE]
        elif ef == EmbedFunction.LOGISTIC_REGRESSION:
            args_dict["embed_random_states"] = TEST_RANDOM_STATES


def process_existing_users_embeddings(args_dict: dict) -> None:
    if args_dict["use_existing_users_embeddings"]:
        path = args_dict["existing_users_embeddings_path"]
        assert path is not None
        path = Path(path).resolve()
        if path.stem != "users_embeddings":
            path = path / "users_embeddings"
        assert path.exists()
        for random_state in args_dict["embed_random_states"]:
            embedding_path = path / f"s_{random_state}"
            assert embedding_path.exists()
        args_dict["existing_users_embeddings_path"] = path


def process_args_dict(args_dict: dict) -> None:
    args_dict["embed_function"] = get_embed_function_from_arg(args_dict["embed_function"])
    assert args_dict["users_selection"] in get_users_selections_choices()
    process_neural_precomputed_users_embeddings(args_dict)
    process_papers_embedding_path(args_dict)
    process_random_states(args_dict)
    process_existing_users_embeddings(args_dict)
    args_dict["eval_data_folder"] = get_eval_data_folder(
        embed_function=args_dict["embed_function"],
        users_selection=args_dict["users_selection"],
        single_val_session=args_dict["single_val_session"],
    )
    args_dict["eval_settings_path"] = args_dict["eval_data_folder"] / "eval_settings.json"


def save_eval_settings(args_dict: dict) -> None:
    path = args_dict["eval_settings_path"]
    args_dict_for_json = args_dict.copy()
    args_dict_for_json["embed_function"] = args_dict_for_json["embed_function"].name.lower()
    for key, value in args_dict_for_json.items():
        if isinstance(value, Path):
            args_dict_for_json[key] = str(value)
    with open(path, "w") as f:
        json.dump(args_dict_for_json, f, indent=4)


def compute_users_embeddings(args_dict: dict) -> None:
    for random_state in args_dict["embed_random_states"]:
        args = [
            sys.executable,
            "-m",
            "code.sequence.src.eval.compute_users_embeddings",
            "--eval_settings_path",
            str(args_dict["eval_settings_path"]),
            "--random_state",
            str(random_state),
        ]
        subprocess.run(args, check=True)


def create_example_config_sliding_window_eval(args_dict: dict) -> dict:
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
    example_config["embedding_folder"] = str(args_dict["papers_embedding_path"])
    return example_config


def get_users_coefs_path(args_dict: dict, random_state: int) -> str:
    if len(args_dict["embed_random_states"]) > 1:
        users_coefs_path = str(
            args_dict["eval_data_folder"] / "users_embeddings" / f"s_{random_state}"
        )
    else:
        seed = args_dict["embed_random_states"][0]
        users_coefs_path = str(args_dict["eval_data_folder"] / "users_embeddings" / f"s_{seed}")
    return users_coefs_path


def run_logreg_configs(args_dict: dict) -> None:
    example_config = create_example_config_sliding_window_eval(args_dict)
    configs_path = args_dict["eval_data_folder"] / "configs"
    os.makedirs(configs_path, exist_ok=True)
    for random_state in args_dict["eval_random_states"]:
        example_config["model_random_state"] = random_state
        example_config["cache_random_state"] = random_state
        example_config["ranking_random_state"] = random_state
        example_config["users_coefs_path"] = get_users_coefs_path(args_dict, random_state)
        config_path = configs_path / f"s_{random_state}.json"
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


def create_visualization(args_dict: dict, time_taken: float) -> None:
    results = []
    for random_state in args_dict["eval_random_states"]:
        outputs_file = (
            ProjectPaths.logreg_outputs_path() / f"s_{random_state}" / "users_results.csv"
        )
        results.append(pd.read_csv(outputs_file))

    stacked_df = pd.concat(results, ignore_index=True)
    group_cols = ["user_id", "fold_idx", "combination_idx"]
    grouped = stacked_df.groupby(group_cols)
    averaged_df = grouped.mean().reset_index()

    first_random_state = args_dict["eval_random_states"][0]
    source_folder = ProjectPaths.logreg_outputs_path() / f"s_{first_random_state}"
    outputs_folder = args_dict["eval_data_folder"] / "outputs"
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
        outputs_folder = ProjectPaths.logreg_outputs_path() / f"s_{random_state}"
        if outputs_folder.exists():
            shutil.rmtree(outputs_folder)


if __name__ == "__main__":
    args_dict = parse_args()
    process_args_dict(args_dict=args_dict)
    os.makedirs(args_dict["eval_data_folder"], exist_ok=True)
    save_eval_settings(args_dict=args_dict)
    if not args_dict["use_existing_users_embeddings"]:
        compute_users_embeddings(args_dict=args_dict)
    start_time = time.time()
    run_logreg_configs(args_dict=args_dict)
    time_taken = time.time() - start_time
    create_visualization(args_dict=args_dict, time_taken=time_taken)
