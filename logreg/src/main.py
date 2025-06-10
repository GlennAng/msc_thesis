import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[2]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_src_paths_to_sys()

import itertools, json, os, pickle, time
import numpy as np, pandas as pd

from algorithm import get_algorithm_from_arg, get_evaluation_from_arg, Score
from create_example_configs import check_config
from get_users_ratings import get_users_ratings
from embedding import Embedding
from evaluation import Evaluator
from weights_handler import load_hyperparameter_range, Weights_Handler

def config_assertions(config: dict) -> None:
    assert config["save_tfidf_coefs"] == False, "Config: save_tfidf_coefs must be False."
    assert config["include_base"] == False, "Config: include_base must be False."
    assert config["include_zerorated"] == False, "Config: include_zerorated must be False."
    assert config["include_cache"] == True, "Config: include_cache must be True."
    assert config["cache_type"] in ["global", "user_filtered"]
    assert config["k_folds"] == 5, "Config: k_folds must be 5."
    assert config["algorithm"] == "logreg", "Config: algorithm must be 'logreg'."
    assert config["logreg_solver"] == "lbfgs", "Config: logreg_solver must be 'lbfgs'."
    assert config["max_iter"] == 10000, "Config: max_iter must be 10000."
    assert config["weights"] == "global:cache_v", "Config: weights must be 'global:cache_v'."

def load_config(config_path: Path) -> dict:
    try:
        with open(config_path) as file:
            config = json.load(file)
    except FileNotFoundError:
        sys.exit(f"Config File '{config_path}' not found.")
    config_assertions(config)
    if check_config(config):
        print("Config File is valid.")
    config["experiment_name"] = config_path.stem
    return config

def convert_enums(config: dict) -> None:
    config["algorithm"] = get_algorithm_from_arg(config["algorithm"])
    config["evaluation"] = get_evaluation_from_arg(config["evaluation"])

def create_outputs_folder(config: dict) -> None:
    outputs_dir = ProjectPaths.logreg_outputs_path()
    os.makedirs(outputs_dir, exist_ok = True)
    experiment_dir = outputs_dir / config["experiment_name"]
    os.makedirs(experiment_dir, exist_ok = True)
    config["outputs_dir"] = experiment_dir
    for item in os.listdir(experiment_dir):
        path = experiment_dir / item
        if os.path.isfile(path):
            os.remove(path)
        else:
            os.system(f"rm -r {path}")
    os.makedirs(experiment_dir / "tmp", exist_ok = True)
    os.makedirs(experiment_dir / "users_predictions", exist_ok = True)

def load_hyperparameters(config: dict, wh: Weights_Handler) -> list:
    weights_hyperparameters_ranges = wh.load_weights_hyperparameters(config)
    config["weights_hyperparameters"] = list(weights_hyperparameters_ranges.keys())
    non_weights_hyperparameters_ranges = {"clf_C": load_hyperparameter_range(config["clf_C"])}
    hyperparameters_ranges = {**weights_hyperparameters_ranges, **non_weights_hyperparameters_ranges}
    hyperparameters = {param: index for index, param in enumerate(hyperparameters_ranges.keys())}
    hyperparameters_combinations = list(itertools.product(*list(hyperparameters_ranges.values())))
    config["hyperparameters"] = hyperparameters
    return hyperparameters_combinations

def init_scores(config: dict) -> None:
    scores = {}
    for index, score in enumerate(Score):
        scores["train_" + score.name.lower()], scores["val_" + score.name.lower()] = 2 * index, 2 * index + 1
    config["scores"] = scores

def make_config_serializable(config: dict) -> dict:
    serializable_config = config.copy()
    for key in serializable_config:
        try:
            json.dumps(serializable_config[key])
        except TypeError:
            serializable_config[key] = serializable_config[key].name
    return serializable_config

def save_config_file(config: dict) -> None:
    outputs_dir = config["outputs_dir"]
    with open(outputs_dir / "config.json", 'w') as file:
        json.dump(make_config_serializable(config), file, indent = 4)

def hyperparameters_combinations_to_dataframe(hyperparameters: dict, hyperparameters_combinations: list) -> pd.DataFrame:
    columns = sorted(hyperparameters.keys(), key = hyperparameters.get)
    df = pd.DataFrame(hyperparameters_combinations, columns = columns)
    df.insert(0, 'combination_idx', range(len(df)))
    return df

def save_hyperparameters_combinations(config: dict, hyperparameters_combinations: list) -> None:
    outputs_dir = config["outputs_dir"]
    hyperparameters_combinations_df = hyperparameters_combinations_to_dataframe(config["hyperparameters"], hyperparameters_combinations)
    hyperparameters_combinations_df.to_csv(outputs_dir / "hyperparameters_combinations.csv", index = False)

def merge_users_infos(config: dict, users_ids: list) -> None:
    users_infos = []
    columns = []
    outputs_dir = config["outputs_dir"]
    for user_id in users_ids:
        json_file = config["outputs_dir"] / "tmp" / f"user_{user_id}" / "user_info.json"
        user_info = json.load(open(json_file))
        if not columns:
            columns = ["user_id"] + list(user_info.keys())
        users_infos.append([user_id] + [user_info[column] for column in columns[1:]])
    users_infos_df = pd.DataFrame(users_infos, columns = columns)
    users_infos_df.to_csv(outputs_dir / "users_info.csv", index = False)

def merge_users_results(config: dict, users_ids: list) -> None:
    users_results = []
    outputs_dir = config["outputs_dir"]
    scores_columns = sorted(config["scores"].keys(), key = config["scores"].get)
    columns = ["user_id", "fold_idx", "combination_idx"] + scores_columns
    for user_id in users_ids:
        json_file = outputs_dir / "tmp" / f"user_{user_id}" / "user_results.json"
        user_results = json.load(open(json_file))
        for fold_idx in sorted(list(user_results.keys())):
            fold_results = user_results[fold_idx]
            for combination_idx in sorted(list(fold_results.keys())):
                row = [user_id, fold_idx, combination_idx] + list(fold_results[combination_idx])
                users_results.append(row)
    users_results_df = pd.DataFrame(users_results, columns = columns)
    users_results_df.to_csv(outputs_dir / "users_results.csv", index = False)

def merge_users_coefs(config: dict, users_ids: list) -> None:
    if "save_users_coefs" in config and config["save_users_coefs"]:
        users_ids = sorted(users_ids)
        outputs_dir = config["outputs_dir"]
        users_coefs_ids_to_idxs = {}
        for i, user_id in enumerate(users_ids):
            npy_file = outputs_dir / "tmp" / f"user_{user_id}" / "user_coefs.npy"
            user_coefs = np.load(npy_file)
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
    config = load_config(Path(sys.argv[1]).resolve())
    convert_enums(config)
    create_outputs_folder(config)

    users_ratings, users_ids = get_users_ratings(users_selection = config["users_selection"], evaluation = config["evaluation"], test_size = config["test_size"],
                    max_users = config["max_users"], users_random_state = config["users_random_state"], model_random_state = config["model_random_state"],
                    stratify = config["stratified"], min_n_posrated = config["min_n_posrated"], min_n_negrated = config["min_n_negrated"],
                    take_complement = config["take_complement_of_users"], users_mapped = config["users_mapped"],
                    min_n_posrated_train = config["min_n_posrated_train"], min_n_negrated_train = config["min_n_negrated_train"],
                    min_n_posrated_val = config["min_n_posrated_val"], min_n_negrated_val = config["min_n_negrated_val"])
                    
    init_scores(config)
    wh = Weights_Handler(config)
    hyperparameters_combinations = load_hyperparameters(config, wh)
    embedding = Embedding(config["embedding_folder"], config["embedding_float_precision"])
    config["embedding_is_sparse"], config["embedding_n_dimensions"] = embedding.is_sparse, embedding.n_dimensions
    config_copy = config.copy()
    save_hyperparameters_combinations(config, hyperparameters_combinations)
    
    evaluator = Evaluator(config, hyperparameters_combinations, wh)
    evaluator.evaluate_embedding(embedding, users_ratings)
    merge_users_infos(config, users_ids)
    merge_users_results(config, users_ids)
    merge_users_coefs(config, users_ids)
    os.system(f"rm -r {config['outputs_dir'] / 'tmp'}")
    config_copy["time_elapsed"] = time.time() - start_time
    save_config_file(config_copy)