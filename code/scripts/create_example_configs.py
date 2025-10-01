import json
import os
from pathlib import Path

from ..src.load_files import ProjectPaths


def create_example_config(embeddings_folder: Path = None) -> dict:
    if embeddings_folder is None:
        embeddings_folder = (
            ProjectPaths.logreg_embeddings_path()
            / "after_pca"
            / "gte_large_256_categories_l2_unit_100"
        )
    example_config = {}
    example_config.update(
        {
            "model_random_state": 42,
            "cache_random_state": 42,
            "ranking_random_state": 42,
        }
    )
    example_config.update(
        {"save_users_predictions": False, "save_users_coefs": False, "save_tfidf_coefs": False}
    )
    example_config.update(
        {
            "users_ratings_selection": "session_based_no_filtering",
            "relevant_users_ids": None,
            "load_users_coefs": False,
            "users_coefs_path": None,
            "load_users_scores": False,
            "users_scores_path": None,
        }
    )
    example_config.update(
        {
            "cache_type": "categories_cache",
            "n_cache": 5000,
            "n_categories_cache": 0,
            "categories_scale": 1.0,
        }
    )
    example_config.update(
        {
            "evaluation": "session_based",
            "n_negative_samples": 100,
            "same_negrated_for_all_pos": False,
            "stratified": True,
            "k_folds": 5,
        }
    )
    example_config.update(
        {
            "algorithm": "logreg",
            "logreg_solver": "lbfgs",
            "svm_kernel": None,
            "max_iter": 10000,
            "n_jobs": 1,
        }
    )
    example_config.update(
        {
            "weights": "global:cache_v",
            "clf_C": 0.1,
            "weights_cache_v": 0.9,
            "weights_neg_scale": 5.0,
        }
    )
    example_config.update(
        {
            "embedding_folder": str(embeddings_folder),
            "embedding_float_precision": None,
        }
    )
    return example_config


def create_example_config_cross_val(embeddings_folder: Path = None) -> dict:
    example_config = create_example_config(embeddings_folder)
    example_config.update({"evaluation": "cross_validation", "n_jobs": -1})
    return example_config


def create_example_config_tfidf(embeddings_folder: Path = None) -> dict:
    if embeddings_folder is None:
        embeddings_folder = ProjectPaths.logreg_embeddings_path() / "tfidf" / "tfidf_10k"
    example_config = create_example_config(embeddings_folder).copy()
    example_config.update({"clf_C": 0.4, "weights_cache_v": 0.9, "weights_neg_scale": 1.0})
    return example_config


def create_example_config_tfidf_cross_val(embeddings_folder: Path = None) -> dict:
    example_config_tfidf = create_example_config_tfidf(embeddings_folder)
    example_config_tfidf.update({"evaluation": "cross_validation", "n_jobs": -1})
    return example_config_tfidf


def create_example_config_sliding_window(users_embeddings_dict_path: Path = None) -> dict:
    example_config = create_example_config()
    if users_embeddings_dict_path is None:
        users_embeddings_dict_path = ProjectPaths.sequence_data_sliding_window_eval_path() / "logreg"
    update_dict = {
        "users_ratings_selection": "session_based_filtering",
        "evaluation": "sliding_window",
        "users_coefs_path": str(users_embeddings_dict_path.resolve()),
        "n_cache": 0,
    }
    example_config.update(update_dict)
    return example_config


def check_config(config: dict) -> bool:
    example_config = create_example_config()
    config_keys, example_config_keys = set(config.keys()), set(example_config.keys())
    if config_keys != example_config_keys:
        raise ValueError("Config Keys do not match with the Example Config Keys.")
    return True


if __name__ == "__main__":
    logreg_experiments_path = ProjectPaths.logreg_experiments_path() / "example_configs"
    os.makedirs(logreg_experiments_path, exist_ok=True)
    example_config = create_example_config()
    with open(logreg_experiments_path / "example_config.json", "w") as f:
        json.dump(example_config, f, indent=4)
    example_config_cross_val = create_example_config_cross_val()
    with open(logreg_experiments_path / "example_config_cross_val.json", "w") as f:
        json.dump(example_config_cross_val, f, indent=4)
    example_config_tfidf = create_example_config_tfidf()
    with open(logreg_experiments_path / "example_config_tfidf.json", "w") as f:
        json.dump(example_config_tfidf, f, indent=4)
    example_config_tfidf_cross_val = create_example_config_tfidf_cross_val()
    with open(logreg_experiments_path / "example_config_tfidf_cross_val.json", "w") as f:
        json.dump(example_config_tfidf_cross_val, f, indent=4)
