from algorithm import get_algorithm_from_arg, get_evaluation_from_arg, Score
from data_handling import get_db_name, get_db_backup_date, get_users_ids_with_sufficient_votes
from embedding import Embedding
#from evaluation import Evaluator
from pathlib import Path
from weights_handler import load_hyperparameter_range, Weights_Handler
import itertools
import json
import numpy as np
import os
import pandas as pd
import pickle
import sys
import time

def load_config(config_file : str) -> dict:
    try:
        with open(config_file) as file:
            config = json.load(file)
            config["experiment_name"] = config_file.split("/")[-1].split(".")[0]
            config["weights"] = "pos:both_constant_hyp,neg:both_linear_hyp"
    except FileNotFoundError:
        sys.exit(f"Config File '{config_file}' not found.")
    return config

def convert_enums(config : dict) -> None:
    return
    config["algorithm"] = get_algorithm_from_arg(config["algorithm"])
    config["evaluation"] = get_evaluation_from_arg(config["evaluation"])

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

def get_users_ids(min_n_posrated : int = 0, min_n_negrated : int = 0, max_users : int = None, random_state : int = None, take_complement : bool = False) -> list:
    users_ids_with_sufficient_votes = get_users_ids_with_sufficient_votes(min_n_negrated, min_n_posrated)["user_id"]
    if take_complement:
        users_ids_with_sufficient_votes_complement = users_ids_with_sufficient_votes.copy()
    n_users_with_sufficient_votes = len(users_ids_with_sufficient_votes)
    if max_users is not None and max_users < n_users_with_sufficient_votes:
        users_ids_with_sufficient_votes.sort()
        users_ids_with_sufficient_votes.sample(n = max_users, random_state = random_state)
    if take_complement:
        users_ids_with_sufficient_votes = np.setdiff1d(users_ids_with_sufficient_votes_complement, users_ids_with_sufficient_votes)
    return sorted([int(user_id) for user_id in users_ids])

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
    users_ids = get_users_ids(config["min_n_posrated"], config["min_n_negrated"], config["max_users"], config["random_state"], config["take_complement_of_users"])
    remaining_users_ids = get_users_not_yet_evaluated(config, users_ids, continue_from_previous)
    """

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
    """