import itertools
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ...scripts.create_example_configs import check_config
from ...sequence.src.data.users_embeddings_data import load_users_embeddings, load_users_scores
from ...src.project_paths import ProjectPaths
from .embeddings.embedding import Embedding
from .training.algorithm import (
    Evaluation,
    get_algorithm_from_arg,
    get_evaluation_from_arg,
)
from .training.evaluation import Evaluator
from .training.scores import Score
from .training.users_ratings import (
    get_users_ratings_selection_from_arg,
    load_users_ratings_from_selection,
)
from .training.weights_handler import Weights_Handler, load_hyperparameter_range


def config_assertions(config: Dict[str, Any]) -> None:
    assert config["cache_type"] in [
        "categories_cache",
        "old_cache",
        "random_cache",
    ]
    assert config["k_folds"] == 5, "Config: k_folds must be 5."
    assert config["logreg_solver"] == "lbfgs", "Config: logreg_solver must be 'lbfgs'."
    assert config["max_iter"] == 10000, "Config: max_iter must be 10000."
    assert config["weights"] == "global:cache_v", "Config: weights must be 'global:cache_v'."
    if config["evaluation"] in ["cross_validation", "train_test_split"]:
        assert config["stratified"]


def load_config(config_path: Path) -> Dict[str, Any]:
    try:
        with open(config_path) as file:
            config = json.load(file)
    except FileNotFoundError:
        sys.exit(f"Config File '{config_path}' not found.")
    config_assertions(config)
    if check_config(config):
        print(f"Config File '{config_path}' is valid.")
    config["experiment_name"] = config_path.stem
    embedding_folder_split = config["embedding_folder"].split("_")
    config["categories_dim"] = (
        int(embedding_folder_split[-1]) if "categories" in embedding_folder_split else None
    )
    return config


def convert_enums(config: Dict[str, Any]) -> None:
    config["algorithm"] = get_algorithm_from_arg(config["algorithm"])
    config["evaluation"] = get_evaluation_from_arg(config["evaluation"])
    config["users_ratings_selection"] = get_users_ratings_selection_from_arg(
        config["users_ratings_selection"]
    )


def create_outputs_folder(config: Dict[str, Any]) -> None:
    outputs_dir = ProjectPaths.logreg_outputs_path()
    os.makedirs(outputs_dir, exist_ok=True)
    experiment_dir = outputs_dir / config["experiment_name"]
    os.makedirs(experiment_dir, exist_ok=True)
    config["outputs_dir"] = experiment_dir
    for item in os.listdir(experiment_dir):
        path = experiment_dir / item
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
    os.makedirs(experiment_dir / "tmp", exist_ok=True)
    os.makedirs(experiment_dir / "users_predictions", exist_ok=True)


def load_hyperparameters(config: Dict[str, Any], wh: Weights_Handler) -> list:
    weights_hyperparameters_ranges = wh.load_weights_hyperparameters(config)
    config["weights_hyperparameters"] = list(weights_hyperparameters_ranges.keys())
    non_weights_hyperparameters_ranges = {"clf_C": load_hyperparameter_range(config["clf_C"])}
    hyperparameters_ranges = {
        **weights_hyperparameters_ranges,
        **non_weights_hyperparameters_ranges,
    }
    hyperparameters = {param: index for index, param in enumerate(hyperparameters_ranges.keys())}
    hyperparameters_combinations = list(itertools.product(*list(hyperparameters_ranges.values())))
    config["hyperparameters"] = hyperparameters
    return hyperparameters_combinations


def init_sliding_window(config: Dict[str, Any], users_ratings: pd.DataFrame) -> tuple:
    users_embeddings = load_users_embeddings(config["users_coefs_path"], check=True)
    users_ids = list(users_ratings["user_id"].unique())
    users_embeddings = {
        user_id: user_embeddings
        for user_id, user_embeddings in users_embeddings.items()
        if user_id in users_ids
    }
    val_mask = users_ratings["split"] == "val"
    val_ratings = users_ratings.loc[val_mask]
    for user_id, user_embeddings in users_embeddings.items():
        user_sessions_ids = user_embeddings["sessions_ids"]
        user_val_ratings = val_ratings[val_ratings["user_id"] == user_id]
        assert user_val_ratings["session_id"].unique().tolist() == user_sessions_ids
    return users_embeddings


def init_scores(config: Dict[str, Any]) -> None:
    scores = {}
    for index, score in enumerate(Score):
        scores["train_" + score.name.lower()], scores["val_" + score.name.lower()] = (
            2 * index,
            2 * index + 1,
        )
    config["scores"] = scores


def make_config_serializable(config: Dict[str, Any]) -> dict:
    serializable_config = config.copy()
    for key in serializable_config:
        try:
            json.dumps(serializable_config[key])
        except TypeError:
            serializable_config[key] = serializable_config[key].name
    return serializable_config


def save_config_file(config: Dict[str, Any]) -> None:
    outputs_dir = config["outputs_dir"]
    with open(outputs_dir / "config.json", "w") as file:
        json.dump(make_config_serializable(config), file, indent=4)


def hyperparameters_combinations_to_dataframe(
    hyperparameters: Dict[str, Any], hyperparameters_combinations: List[tuple]
) -> pd.DataFrame:
    columns = sorted(hyperparameters.keys(), key=hyperparameters.get)
    df = pd.DataFrame(hyperparameters_combinations, columns=columns)
    df.insert(0, "combination_idx", range(len(df)))
    return df


def save_hyperparameters_combinations(
    config: Dict[str, Any], hyperparameters_combinations: List[tuple]
) -> None:
    outputs_dir = config["outputs_dir"]
    hyperparameters_combinations_df = hyperparameters_combinations_to_dataframe(
        config["hyperparameters"], hyperparameters_combinations
    )
    hyperparameters_combinations_df.to_csv(
        outputs_dir / "hyperparameters_combinations.csv", index=False
    )


def merge_users_infos(config: Dict[str, Any], users_ids: List[int]) -> None:
    users_infos = []
    columns = []
    outputs_dir = config["outputs_dir"]
    for user_id in users_ids:
        json_file = config["outputs_dir"] / "tmp" / f"user_{user_id}" / "user_info.json"
        if os.path.exists(json_file):
            user_info = json.load(open(json_file))
            if not columns:
                columns = ["user_id"] + list(user_info.keys())
            users_infos.append([user_id] + [user_info[column] for column in columns[1:]])
    users_infos_df = pd.DataFrame(users_infos, columns=columns)
    users_infos_df.to_csv(outputs_dir / "users_info.csv", index=False)


def merge_users_results(config: Dict[str, Any], users_ids: List[int]) -> None:
    users_results = []
    outputs_dir = config["outputs_dir"]
    scores_columns = sorted(config["scores"].keys(), key=config["scores"].get)
    columns = ["user_id", "fold_idx", "combination_idx"] + scores_columns
    for user_id in users_ids:
        json_file = outputs_dir / "tmp" / f"user_{user_id}" / "user_results.json"
        if os.path.exists(json_file):
            user_results = json.load(open(json_file))
            for fold_idx in sorted(list(user_results.keys())):
                fold_results = user_results[fold_idx]
                for combination_idx in sorted(list(fold_results.keys())):
                    row = [user_id, fold_idx, combination_idx] + list(fold_results[combination_idx])
                    users_results.append(row)
    users_results_df = pd.DataFrame(users_results, columns=columns)
    users_results_df.to_csv(outputs_dir / "users_results.csv", index=False)


def merge_users_coefs(config: Dict[str, Any], users_ids: List[int]) -> None:
    if "save_users_coefs" in config and config["save_users_coefs"]:
        users_ids = sorted(users_ids)
        outputs_dir = config["outputs_dir"]
        users_coefs_ids_to_idxs = {}
        for i, user_id in enumerate(users_ids):
            npy_file = outputs_dir / "tmp" / f"user_{user_id}" / "user_coefs.npy"
            if os.path.exists(npy_file):
                user_coefs = np.load(npy_file)
                if i == 0:
                    users_coefs = np.empty((len(users_ids), len(user_coefs)))
                users_coefs[i, :] = user_coefs
                users_coefs_ids_to_idxs[user_id] = i
        with open(outputs_dir / "users_coefs_ids_to_idxs.pkl", "wb") as f:
            pickle.dump(users_coefs_ids_to_idxs, f)
        np.save(outputs_dir / "users_coefs.npy", users_coefs)


if __name__ == "__main__":
    start_time = time.time()
    if len(sys.argv) < 2:
        sys.exit("Usage: python main.py <config_file>")
    config = load_config(Path(sys.argv[1]).resolve())
    convert_enums(config)
    create_outputs_folder(config)

    users_ratings = load_users_ratings_from_selection(
        users_ratings_selection=config["users_ratings_selection"],
        relevant_users_ids=config["relevant_users_ids"],
    )
    users_embeddings, users_scores = None, None
    if config["load_users_scores"]:
        users_scores = load_users_scores(config["users_scores_path"])
    else:
        if config["evaluation"] == Evaluation.SLIDING_WINDOW:
            users_embeddings = init_sliding_window(config=config, users_ratings=users_ratings)
    users_ids = list(users_ratings["user_id"].unique())
    print(f"Number of Users: {len(users_ids)}.")
    assert users_ids == sorted(users_ids)

    init_scores(config)
    wh = Weights_Handler(config)
    hyperparameters_combinations = load_hyperparameters(config, wh)
    embedding = Embedding(config["embedding_folder"], config["embedding_float_precision"])
    config["embedding_is_sparse"], config["embedding_n_dimensions"] = (
        embedding.is_sparse,
        embedding.n_dimensions,
    )
    config_copy = config.copy()
    save_hyperparameters_combinations(config, hyperparameters_combinations)

    evaluator = Evaluator(config, hyperparameters_combinations, wh)
    evaluator.evaluate_embedding(embedding, users_ratings, users_embeddings, users_scores)
    merge_users_infos(config, users_ids)
    merge_users_results(config, users_ids)
    merge_users_coefs(config, users_ids)
    shutil.rmtree(config["outputs_dir"] / "tmp", ignore_errors=True)
    config_copy["time_elapsed"] = time.time() - start_time
    save_config_file(config_copy)
