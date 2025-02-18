import json
import os
import sys
from pathlib import Path

def generate_example_config(overwrite_dict : dict = None) -> dict:
    example_config = {}
    example_config["embedding_float_precision"] = None
    example_config["save_users_predictions"] = False
    example_config["save_coefs"] = False
    example_config["max_users"] = 10
    example_config["min_n_posrated"] = 20
    example_config["min_n_negrated"] = 20
    example_config["take_complement_of_users"] = False
    example_config["random_state"] = 42
    example_config["max_cache"] = 5000
    example_config["evaluation"] = "cross_validation"
    example_config["stratified"] = True
    example_config["k_folds"] = 5
    example_config["test_size"] = 0.2
    example_config["algorithm"] = "logreg"
    example_config["logreg_solver"] = "lbfgs"
    example_config["svm_kernel"] = "linear"
    example_config["max_iter"] = 10000
    example_config["n_jobs"] = -1
    example_config["clf_C"] = "np.logspace(-3, 5, 9)"
    example_config["weights_negrated_importance"] = [0.8, 0.99]
    for key, value in overwrite_dict.items():
        example_config[key] = value
    return example_config
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python generate_example_config.py <embedding_folder>")
    embedding_folder = sys.argv[1].rstrip("/")
    if not (os.path.exists(f"{embedding_folder}/abs_X.npy") or os.path.exists(f"{embedding_folder}/abs_X.npz")):
        sys.exit(f"Missing file abs_X.npy or abs_X.npz in {embedding_folder}")
    if not os.path.exists(f"{embedding_folder}/abs_paper_ids_to_idx.pkl"):
        sys.exit(f"Missing file abs_paper_ids_to_idx.pkl in {embedding_folder}")

    project_root = Path(__file__).parent.parent
    experiments_directory = project_root / "experiments"
    os.makedirs(experiments_directory, exist_ok = True)
    overwrite_dict = {"embedding_folder": embedding_folder}
    example_config = generate_example_config(overwrite_dict)

    json_path = experiments_directory / "example_config.json"
    with open(json_path, 'w') as f:
        json.dump(example_config, f, indent = 4)