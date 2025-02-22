from algorithm import get_algorithm_from_arg, get_evaluation_from_arg, Score
from data_handling import get_users_ids_with_sufficient_votes, get_paper_removal_from_arg, get_db_backup_date, get_db_name
from embedding import Embedding
from evaluation import Evaluator
from pathlib import Path
from training_data import get_cache_type_from_arg
from weights_handler import load_hyperparameter_range, Weights_Handler
import itertools
import json
import numpy as np
import os
import pandas as pd
import pickle
import sys
import time

def thesis_assertions(config : dict) -> None:
    assert config["save_tfidf_coefs"] == False, "Config: save_tfidf_coefs must be False."
    assert config["weights"] == "global:cache_v", "Config: weights must be 'global:cache_v'."
    assert config["include_base"] == False, "Config: include_base must be False."
    assert config["include_cache"] == True, "Config: include_cache must be True."
    assert config["max_cache"] == 5000, "Config: max_cache must be 5000."
    assert config["stratified"] == True, "Config: stratified must be True."
    assert config["k_folds"] == 5, "Config: k_folds must be 5."
    assert config["test_size"] == 0.2, "Config: test_size must be 0.2."
    assert config["rated_paper_removal"] == "none", "Config: rated_paper_removal must be 'none'."
    assert config["base_paper_removal"] == "none", "Config: base_paper_removal must be 'none'."
    assert config["algorithm"] == "logreg", "Config: algorithm must be 'logreg'."
    assert config["logreg_solver"] == "lbfgs", "Config: logreg_solver must be 'lbfgs'."
    assert config["max_iter"] == 10000, "Config: max_iter must be 10000."

def load_config(config_file : str) -> dict:
    try:
        with open(config_file) as file:
            config = json.load(file)
            config["experiment_name"] = config_file.split("/")[-1].split(".")[0]
            thesis_assertions(config)
    except FileNotFoundError:
        sys.exit(f"Config File '{config_file}' not found.")
    return config

def convert_enums(config : dict) -> None:
    config["algorithm"] = get_algorithm_from_arg(config["algorithm"])
    config["evaluation"] = get_evaluation_from_arg(config["evaluation"])
    config["rated_paper_removal"] = get_paper_removal_from_arg(config["rated_paper_removal"])
    config["base_paper_removal"] = get_paper_removal_from_arg(config["base_paper_removal"])
    if config["include_cache"]:
        config["cache_type"] = get_cache_type_from_arg(config["cache_type"])

def create_outputs_folder(config : dict, continue_from_previous : bool) -> None:
    project_root = Path(__file__).parent.parent
    outputs_dir = project_root / "outputs"
    os.makedirs(outputs_dir, exist_ok = True)
    experiment_dir = outputs_dir / config["experiment_name"]
    os.makedirs(experiment_dir, exist_ok = True)
    config["outputs_dir"] = experiment_dir
    for item in os.listdir(experiment_dir):
        path = os.path.join(experiment_dir, item)
        if continue_from_previous:
            if item == "tmp" or (item == "users_predictions" and config["save_users_predictions"]):
                continue
        if os.path.isfile(path):
            os.remove(path)
        else:
            os.system(f"rm -r {path}")
    os.makedirs(experiment_dir / "tmp", exist_ok = True)
    if config["save_users_predictions"]:
        os.makedirs(experiment_dir / "users_predictions", exist_ok = True)

def get_users_ids(users_selection : str, max_users : int = None, min_n_posrated : int = 20, min_n_negrated : int = 20, take_complement : bool = False, random_state : int = None) -> pd.DataFrame:
    users_selection = users_selection.lower()
    if users_selection not in ["random", "largest_n", "smallest_n"]:
        raise ValueError("Users Selection: users_selection must be one of ['random', 'largest_n', 'smallest_n'].")

    users_ids_with_sufficient_votes = get_users_ids_with_sufficient_votes(min_n_posrated = min_n_posrated, min_n_negrated = min_n_negrated, sort_ids = False)
    n_users_with_sufficient_votes = len(users_ids_with_sufficient_votes)
    max_users = n_users_with_sufficient_votes if max_users is None else min(max_users, n_users_with_sufficient_votes)
    if max_users >= n_users_with_sufficient_votes:
        assert not take_complement, "Users Selection: take_complement must be False when all users are selected."
    else:
        if take_complement:
            users_ids_with_sufficient_votes_complement = users_ids_with_sufficient_votes.copy()
        if users_selection == "random":
            users_ids_with_sufficient_votes = users_ids_with_sufficient_votes.sort_values(by = "user_id")
            users_ids_with_sufficient_votes = users_ids_with_sufficient_votes.sample(n = max_users, random_state = random_state)
        elif users_selection in ["largest_n", "smallest_n"]:
            smallest_n_bool = (users_selection == "smallest_n")
            users_ids_with_sufficient_votes = users_ids_with_sufficient_votes.sort_values(["n_rated", "n_posrated", "user_id"], 
                                                                            ascending = [smallest_n_bool, smallest_n_bool, False]).head(max_users)
        if take_complement:
            users_ids_with_sufficient_votes = users_ids_with_sufficient_votes_complement[~users_ids_with_sufficient_votes_complement["user_id"].isin(users_ids_with_sufficient_votes["user_id"])]
    return users_ids_with_sufficient_votes.sort_values(by = "user_id")

def get_users_not_yet_evaluated(config : dict, users_ids : list, continue_from_previous : bool) -> list:
    if not continue_from_previous:
        return users_ids
    outputs_dir = config["outputs_dir"]
    users_already_evaluated = []
    for user_id in users_ids:
        exists_user_info = os.path.exists(outputs_dir / f"tmp/user_{user_id}/user_info.json")
        exists_user_results = os.path.exists(outputs_dir / f"tmp/user_{user_id}/user_results.json")
        exists_user_predictions = os.path.exists(outputs_dir / f"users_predictions/user_{user_id}/user_predictions.json") if config["save_users_predictions"] else True
        if exists_user_info and exists_user_results and exists_user_predictions:
            users_already_evaluated.append(user_id)
    return [user_id for user_id in users_ids if user_id not in users_already_evaluated]

def load_hyperparameters(config : dict, wh : Weights_Handler) -> list:
    weights_hyperparameters_ranges = wh.load_weights_hyperparameters(config)
    config["weights_hyperparameters"] = list(weights_hyperparameters_ranges.keys())
    non_weights_hyperparameters_ranges = {"clf_C": load_hyperparameter_range(config["clf_C"])}
    hyperparameters_ranges = {**weights_hyperparameters_ranges, **non_weights_hyperparameters_ranges}
    hyperparameters = {param: index for index, param in enumerate(hyperparameters_ranges.keys())}
    hyperparameters_combinations = list(itertools.product(*list(hyperparameters_ranges.values())))
    config["hyperparameters"] = hyperparameters
    print(hyperparameters_combinations)
    return hyperparameters_combinations

def init_scores(config : dict) -> None:
    scores = {}
    for index, score in enumerate(Score):
        scores["train_" + score.name.lower()], scores["val_" + score.name.lower()] = 2 * index, 2 * index + 1
    config["scores"] = scores

def make_config_serializable(config : dict) -> dict:
    serializable_config = config.copy()
    for key in serializable_config:
        try:
            json.dumps(serializable_config[key])
        except TypeError:
            serializable_config[key] = serializable_config[key].name
    return serializable_config

def save_config_file(config : dict) -> None:
    outputs_dir = config["outputs_dir"]
    with open(outputs_dir / "config.json", 'w') as file:
        json.dump(make_config_serializable(config), file, indent = 4)

def hyperparameters_combinations_to_dataframe(hyperparameters : dict, hyperparameters_combinations : list) -> pd.DataFrame:
    columns = sorted(hyperparameters.keys(), key = hyperparameters.get)
    df = pd.DataFrame(hyperparameters_combinations, columns = columns)
    df.insert(0, 'combination_idx', range(len(df)))
    return df

def save_hyperparameters_combinations(config : dict, hyperparameters_combinations : list) -> None:
    outputs_dir = config["outputs_dir"]
    hyperparameters_combinations_df = hyperparameters_combinations_to_dataframe(config["hyperparameters"], hyperparameters_combinations)
    hyperparameters_combinations_df.to_csv(outputs_dir / "hyperparameters_combinations.csv", index = False)

def merge_users_infos(config : dict, users_ids : list) -> None:
    users_infos = []
    columns = []
    outputs_dir = config["outputs_dir"]
    for user_id in users_ids:
        user_info = json.load(open(f"{config['outputs_dir']}/tmp/user_{user_id}/user_info.json"))
        if not columns:
            columns = ["user_id"] + list(user_info.keys())
        users_infos.append([user_id] + [user_info[column] for column in columns[1:]])
    users_infos_df = pd.DataFrame(users_infos, columns = columns)
    users_infos_df.to_csv(outputs_dir / "users_info.csv", index = False)

def merge_users_results(config : dict, users_ids : list) -> None:
    users_results = []
    outputs_dir = config["outputs_dir"]
    scores_columns = sorted(config["scores"].keys(), key = config["scores"].get)
    columns = ["user_id", "fold_idx", "combination_idx"] + scores_columns
    for user_id in users_ids:
        user_results = json.load(open(f"{outputs_dir}/tmp/user_{user_id}/user_results.json"))
        for fold_idx in sorted(list(user_results.keys())):
            fold_results = user_results[fold_idx]
            for combination_idx in sorted(list(fold_results.keys())):
                row = [user_id, fold_idx, combination_idx] + list(fold_results[combination_idx])
                users_results.append(row)
    users_results_df = pd.DataFrame(users_results, columns = columns)
    users_results_df.to_csv(outputs_dir / "users_results.csv", index = False)

def merge_users_coefs(config : dict, users_ids : list) -> None:
    if "save_coefs" in config and config["save_coefs"]:
        outputs_dir = config["outputs_dir"]
        users_coefs_ids_to_idxs = {}
        for i, user_id in enumerate(users_ids):
            user_coefs = np.load(f"{outputs_dir}/tmp/user_{user_id}/user_coefs.npy")
            if i == 0:
                users_coefs = np.empty((len(users_ids), len(user_coefs)))
            users_coefs[i, :] = user_coefs
            users_coefs_ids_to_idxs[user_id] = i
        with open(outputs_dir / "users_coefs_ids_to_idxs.pkl", 'wb') as f:
            pickle.dump(users_coefs_ids_to_idxs, f)
        np.save(outputs_dir / "users_coefs.npy", users_coefs)

if __name__ == "__main__":
    start_time = time.time()
    if len(sys.argv) < 2:
        sys.exit("Usage: python main.py <config_file>")
    config = load_config(sys.argv[1])
    config["db_backup_date"], config["db_name"] = get_db_backup_date(), get_db_name()
    convert_enums(config)

    if len(sys.argv) > 2:
        continue_from_previous = (sys.argv[2] == "continue_from_previous")
    else:
        continue_from_previous = False
    create_outputs_folder(config, continue_from_previous)
    users_ids = get_users_ids(users_selection = config["users_selection"], max_users = config["max_users"], min_n_posrated = config["min_n_posrated"], min_n_negrated = config["min_n_negrated"], 
                              take_complement = config["take_complement_of_users"], random_state = config["random_state"])
    users_ids = users_ids["user_id"].tolist()
    remaining_users_ids = get_users_not_yet_evaluated(config, users_ids, continue_from_previous)

    init_scores(config)
    wh = Weights_Handler(config)
    hyperparameters_combinations = load_hyperparameters(config, wh)
    embedding = Embedding(config["embedding_folder"], config["embedding_float_precision"])
    config["embedding_is_sparse"], config["embedding_n_dimensions"] = embedding.is_sparse, embedding.n_dimensions
    config_copy = config.copy()
    save_hyperparameters_combinations(config, hyperparameters_combinations)
    evaluator = Evaluator(config, remaining_users_ids, hyperparameters_combinations, wh)
    evaluator.evaluate_embedding(embedding)
    merge_users_infos(config, users_ids)
    merge_users_results(config, users_ids)
    merge_users_coefs(config, users_ids)
    os.system(f"rm -r {config['outputs_dir']}/tmp")
    config_copy["time_elapsed"] = time.time() - start_time
    save_config_file(config_copy)