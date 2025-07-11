import json
import os

from pathlib import Path
from ..src.load_files import ProjectPaths


def create_example_config(embeddings_folder: Path = None) -> dict:
    if embeddings_folder is None:
        embeddings_folder = ProjectPaths.logreg_embeddings_path() / "after_pca" / "gte_large_256_categories_l2_unit_100"
    example_config = {}
    example_config.update(
        {
            "users_random_state": 42,
            "model_random_state": 42,
            "cache_random_state": 42,
            "ranking_random_state": 42,
        }
    )
    example_config.update(
        {"save_users_predictions": False, "save_users_coefs": False, "save_tfidf_coefs": False}
    )
    example_config.update({"load_users_coefs": False, "users_coefs_path": None})
    example_config.update(
        {"users_selection": "finetuning_test", "max_users": 500, "take_complement_of_users": False}
    )
    example_config.update(
        {"min_n_posrated": 20, "min_n_negrated": 20, "same_negrated_for_all_pos": False}
    )
    example_config.update(
        {
            "min_n_posrated_train": 16,
            "min_n_negrated_train": 16,
            "min_n_posrated_val": 4,
            "min_n_negrated_val": 4,
        }
    )
    example_config.update(
        {"include_base": False, "include_zerorated": False, "categories_scale": 1.0}
    )
    example_config.update(
        {
            "include_cache": True,
            "cache_type": "user_filtered",
            "max_cache": 5000,
            "n_cache_attached": 5000,
            "n_negative_samples": 100,
        }
    )
    example_config.update(
        {"evaluation": "cross_validation", "train_size": 0.8, "stratified": True, "k_folds": 5}
    )
    example_config.update(
        {
            "algorithm": "logreg",
            "logreg_solver": "lbfgs",
            "svm_kernel": None,
            "max_iter": 10000,
            "n_jobs": -1,
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


def create_example_config_temporal(embeddings_folder: Path = None) -> dict:
    example_config = create_example_config(embeddings_folder)
    example_config.update({"evaluation": "session_based", "train_size": 1.0})
    return example_config


def create_example_config_tfidf(embeddings_folder: Path = None) -> dict:
    if embeddings_folder is None:
        embeddings_folder = ProjectPaths.logreg_embeddings_path() / "tfidf" / "tfidf_10k"
    example_config = create_example_config(embeddings_folder).copy()
    example_config.update({"clf_C": 0.4, "weights_cache_v": 0.9, "weights_neg_scale": 1.0})
    return example_config


def create_example_config_tfidf_temporal(embeddings_folder: Path = None) -> dict:
    example_config_tfidf = create_example_config_tfidf(embeddings_folder)
    example_config_tfidf.update({"evaluation": "session_based", "train_size": 1.0})
    return example_config_tfidf


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
    example_config_temporal = create_example_config_temporal()
    with open(logreg_experiments_path / "example_config_temporal.json", "w") as f:
        json.dump(example_config_temporal, f, indent=4)
