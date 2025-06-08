import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[2]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_src_paths_to_sys()

import json, os

def create_example_config() -> dict:
    example_config = {}
    example_config.update({"users_random_state": 42, "model_random_state": 42, "cache_random_state": 42, "ranking_random_state": 42})
    example_config.update({"save_users_predictions": False, "save_users_coefs": False, "save_tfidf_coefs": False})
    example_config.update({"load_users_coefs": False, "users_coefs_path": None})
    example_config.update({"users_selection": "random", "max_users": 500, "take_complement_of_users": False, "remove_null_dates": True})
    example_config.update({"min_n_posrated": 20, "min_n_negrated": 20})
    example_config.update({"min_n_posrated_train": 16, "min_n_negrated_train": 16, "min_n_posrated_val": 4, "min_n_negrated_val": 4})
    example_config.update({"include_base": False, "include_zerorated": False})
    example_config.update({"include_cache": True, "cache_type": "user_filtered", "max_cache": 5000, "n_cache_attached": 5000})
    example_config.update({"n_negative_samples": 100, "info_nce_temperature": 1.0})
    example_config.update({"evaluation": "cross_validation", "test_size": 0.2, "stratified": True, "k_folds": 5})
    example_config.update({"algorithm": "logreg", "logreg_solver": "lbfgs", "svm_kernel": None, "max_iter": 10000, "n_jobs": -1})
    example_config.update({"weights": "global:cache_v", "clf_C": 0.1, "weights_cache_v": 0.8, "weights_neg_scale": 8.0})
    example_config.update({"embedding_folder": str(ProjectPaths.logreg_data_embeddings_path() / "after_pca" / "gte_large_2025-02-23_256_categories_100_l2_unit"),
                           "embedding_float_precision": None})
    return example_config

def check_config(config: dict) -> bool:
    example_config = create_example_config()
    config_keys, example_config_keys = set(config.keys()), set(example_config.keys())
    if config_keys != example_config_keys:
        raise ValueError("Config Keys do not match with the Example Config Keys.")
    return True

if __name__ == "__main__":
    logreg_experiments_path = ProjectPaths.logreg_experiments_path()
    os.makedirs(logreg_experiments_path, exist_ok = True)
    example_config = create_example_config()
    with open(logreg_experiments_path / "example_config.json", "w") as f:
        json.dump(example_config, f, indent = 4)  