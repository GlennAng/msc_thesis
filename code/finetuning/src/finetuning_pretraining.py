import json
import os
import pickle
import subprocess
import sys

import numpy as np

from ...scripts.create_example_configs import create_example_config
from ...src.load_files import load_finetuning_users
from ...src.project_paths import ProjectPaths


def create_finetuning_config_train(example_config: dict, train_users: list) -> dict:
    finetuning_config_train = example_config.copy()
    finetuning_config_train.update(
        {
            "users_selection": train_users,
            "min_n_posrated": 20,
            "min_n_negrated": 20,
            "evaluation": "train_test_split",
            "train_size": 1.0,
            "save_users_coefs": True,
        }
    )
    return finetuning_config_train


def create_finetuning_config_val(example_config: dict, val_users: list) -> dict:
    finetuning_config_val = example_config.copy()
    finetuning_config_val.update(
        {
            "users_selection": val_users,
            "min_n_posrated_train": 16,
            "min_n_negrated_train": 16,
            "min_n_posrated_val": 4,
            "min_n_negrated_val": 4,
            "evaluation": "session_based",
            "train_size": 1.0,
            "save_users_coefs": True,
        }
    )
    return finetuning_config_val


def create_finetuning_configs(train_users: list, val_users: list) -> None:
    example_config = create_example_config()
    finetuning_config_train = create_finetuning_config_train(example_config, train_users)
    finetuning_config_val = create_finetuning_config_val(example_config, val_users)
    finetuning_config_path = ProjectPaths.logreg_experiments_path() / "finetuning_pretraining"
    os.makedirs(finetuning_config_path, exist_ok=True)
    with open(finetuning_config_path / "finetuning_config_train.json", "w") as f:
        json.dump(finetuning_config_train, f, indent=4)
    with open(finetuning_config_path / "finetuning_config_val.json", "w") as f:
        json.dump(finetuning_config_val, f, indent=4)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "code.scripts.logreg_run",
            "--config_path",
            str(finetuning_config_path),
        ],
        check=True,
    )


def load_outputs() -> tuple:
    finetuning_outputs_path = ProjectPaths.logreg_outputs_path()
    finetuning_outputs_path_train = finetuning_outputs_path / "finetuning_config_train"
    finetuning_outputs_path_val = finetuning_outputs_path / "finetuning_config_val"
    assert os.path.exists(finetuning_outputs_path_train) and os.path.exists(
        finetuning_outputs_path_val
    )
    train_coefs = np.load(finetuning_outputs_path_train / "users_coefs.npy")
    val_coefs = np.load(finetuning_outputs_path_val / "users_coefs.npy")
    assert train_coefs.shape[0] == len(train_users) and val_coefs.shape[0] == len(val_users)
    with open(finetuning_outputs_path_train / "users_coefs_ids_to_idxs.pkl", "rb") as f:
        train_users_ids_to_idxs = pickle.load(f)
    with open(finetuning_outputs_path_val / "users_coefs_ids_to_idxs.pkl", "rb") as f:
        val_users_ids_to_idxs = pickle.load(f)
    assert (
        list(train_users_ids_to_idxs.keys()) == train_users
        and list(val_users_ids_to_idxs.keys()) == val_users
    )
    return train_coefs, val_coefs, train_users_ids_to_idxs, val_users_ids_to_idxs


def save_outputs(
    train_coefs: np.ndarray,
    val_coefs: np.ndarray,
    train_users_ids_to_idxs: dict,
    val_users_ids_to_idxs: dict,
) -> None:
    coefs_merged = np.concatenate((train_coefs, val_coefs), axis=0)
    assert coefs_merged.shape[0] == len(train_users) + len(val_users)
    val_users_ids_to_idxs_updated = {
        user_id: idx + len(train_users) for user_id, idx in val_users_ids_to_idxs.items()
    }
    users_ids_to_idxs = {**train_users_ids_to_idxs, **val_users_ids_to_idxs_updated}
    assert list(users_ids_to_idxs.keys()) == train_users + val_users
    os.makedirs(ProjectPaths.finetuning_data_path(), exist_ok=True)
    state_dicts_path = ProjectPaths.finetuning_data_model_state_dicts_path()
    os.makedirs(state_dicts_path, exist_ok=True)
    np.save(
        state_dicts_path / "users_coefs.npy",
        coefs_merged,
    )
    print(
        f"Saved users_coefs.npy with shape {coefs_merged.shape} at "
        f"{state_dicts_path / 'users_coefs.npy'}"
    )
    with open(
        state_dicts_path / "users_coefs_ids_to_idxs.pkl",
        "wb",
    ) as f:
        pickle.dump(users_ids_to_idxs, f)
    print(
        f"Saved users_coefs_ids_to_idxs.pkl with {len(users_ids_to_idxs)} entries at "
        f"{state_dicts_path / "users_coefs_ids_to_idxs.pkl"}"
    )


if __name__ == "__main__":
    finetuning_users = load_finetuning_users()
    train_users, val_users = finetuning_users["train"], finetuning_users["val"]
    create_finetuning_configs(train_users, val_users)
    train_coefs, val_coefs, train_users_ids_to_idxs, val_users_ids_to_idxs = load_outputs()
    save_outputs(train_coefs, val_coefs, train_users_ids_to_idxs, val_users_ids_to_idxs)
