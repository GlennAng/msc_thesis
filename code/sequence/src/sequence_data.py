import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ...logreg.src.training.get_users_ratings import (
    USERS_SELECTIONS,
    sequence_load_users_ratings,
)
from ...src.project_paths import ProjectPaths


def get_users_val_sessions_ids(users_ratings: pd.DataFrame) -> dict:
    users_ids = users_ratings["user_id"].unique().tolist()
    sessions_ids = {}
    users_ratings_val = users_ratings[users_ratings["split"] == "val"].reset_index(drop=True)
    for user_id in users_ids:
        user_ratings_val = users_ratings_val[users_ratings_val["user_id"] == user_id].reset_index(
            drop=True
        )
        user_sessions_ids_val = user_ratings_val["session_id"].unique().tolist()
        assert len(user_sessions_ids_val) > 0
        assert user_sessions_ids_val == sorted(user_sessions_ids_val)
        sessions_ids[user_id] = user_sessions_ids_val
    return sessions_ids


def get_embedding_path(users_selection: str) -> Path:
    if users_selection not in USERS_SELECTIONS:
        raise ValueError(f"Unknown users selection: {users_selection}")
    if users_selection == "finetuning_test":
        return (
            ProjectPaths.logreg_embeddings_path()
            / "after_pca"
            / "gte_large_256_test_categories_l2_unit_100"
        )
    elif users_selection == "finetuning_val":
        return (
            ProjectPaths.logreg_embeddings_path()
            / "after_pca"
            / "gte_large_256_val_categories_l2_unit_100"
        )
    elif users_selection == "session_based":
        return (
            ProjectPaths.logreg_embeddings_path()
            / "after_pca"
            / "gte_large_256_session_based_categories_l2_unit_100"
        )


def save_users_embeddings_dict(
    users_embeddings: dict, users_selection: str, save_path: Path, embedding_path: Path = None
) -> None:
    if embedding_path is None:
        embedding_path = get_embedding_path(users_selection)
    users_embeddings_dict = {
        "users_embeddings": users_embeddings,
        "users_selection": users_selection,
        "embedding_path": embedding_path,
    }
    users_embeddings_dict = load_users_embeddings_dict(
        users_embeddings_dict=users_embeddings_dict, check=True
    )
    with open(save_path, "wb") as f:
        pickle.dump(users_embeddings_dict, f)
    print(f"Users embeddings dictionary saved to {save_path}.")


def load_users_embeddings_dict(
    path: Path = None, users_embeddings_dict: dict = None, check: bool = True
) -> dict:
    if users_embeddings_dict is None:
        assert path is not None
        with open(path, "rb") as f:
            users_embeddings_dict = pickle.load(f)
    assert isinstance(users_embeddings_dict, dict)
    assert users_embeddings_dict["users_selection"] in USERS_SELECTIONS
    assert Path(users_embeddings_dict["embedding_path"]).resolve().exists()
    users_embeddings = users_embeddings_dict.get("users_embeddings")
    users_ids = list(users_embeddings.keys())
    assert users_ids == sorted(users_ids)
    for user_id in users_ids:
        user_embeddings = users_embeddings[user_id]
        assert isinstance(user_embeddings, dict)
        sessions_ids = user_embeddings["sessions_ids"]
        assert isinstance(sessions_ids, list)
        assert sessions_ids == sorted(sessions_ids) and len(sessions_ids) == len(set(sessions_ids))
        sessions_embeddings = user_embeddings["sessions_embeddings"]
        assert isinstance(sessions_embeddings, np.ndarray)
        assert len(sessions_ids) == sessions_embeddings.shape[0]
        sessions_ids_to_idxs = {session_id: idx for idx, session_id in enumerate(sessions_ids)}
        user_embeddings["sessions_ids_to_idxs"] = sessions_ids_to_idxs
    if check:
        check_users_embeddings_dict(users_embeddings_dict)
    return users_embeddings_dict


def check_users_embeddings_dict(
    users_embeddings_dict: dict, users_ratings: pd.DataFrame = None
) -> bool:
    if users_ratings is None:
        users_ratings = sequence_load_users_ratings(users_embeddings_dict["users_selection"])
    users_embeddings = users_embeddings_dict["users_embeddings"]
    users_ids = users_ratings["user_id"].unique().tolist()
    assert users_ids == list(users_embeddings.keys())
    users_ratings_val = users_ratings[users_ratings["split"] == "val"].reset_index(drop=True)

    for user_id in users_ids:
        user_ratings_val = users_ratings_val[users_ratings_val["user_id"] == user_id]
        user_sessions_ids_val = user_ratings_val["session_id"].unique().tolist()
        assert user_sessions_ids_val == users_embeddings[user_id]["sessions_ids"]
    return True


def load_users_embeddings_dict_logreg(path: Path) -> dict:
    with open(path / "users_coefs_ids_to_idxs.pkl", "rb") as f:
        users_ids_to_idxs = pickle.load(f)
    users_embeddings_matrix = np.load(path / "users_coefs.npy")
    config = json.load(open(path / "config.json"))
    users_selection = config["users_selection"]
    embedding_path = config["embedding_folder"]
    users_ids = list(users_ids_to_idxs.keys())
    assert len(users_ids) == users_embeddings_matrix.shape[0]

    users_embeddings = {}
    users_ratings = sequence_load_users_ratings(selection=users_selection)
    val_mask = users_ratings["split"] == "val"
    if val_mask.any():
        min_session_ids = users_ratings[val_mask].groupby("user_id")["session_id"].min()
        assert len(min_session_ids) == len(users_ids)
        for user_id, min_session_id in min_session_ids.items():
            user_val_mask = (users_ratings["user_id"] == user_id) & (
                users_ratings["split"] == "val"
            )
            users_ratings.loc[user_val_mask, "session_id"] = min_session_id
            user_embedding = users_embeddings_matrix[users_ids_to_idxs[user_id]]
            users_embeddings[user_id] = {
                "sessions_ids": [min_session_id],
                "sessions_embeddings": user_embedding.reshape(1, -1),
            }
    users_embeddings_dict = {
        "users_embeddings": users_embeddings,
        "users_selection": users_selection,
        "embedding_path": embedding_path,
    }
    users_embeddings_dict = load_users_embeddings_dict(
        users_embeddings_dict=users_embeddings_dict, check=False
    )
    check_users_embeddings_dict(users_embeddings_dict, users_ratings)
    return users_embeddings_dict


def compare_users_embeddings_dicts(
    users_embeddings_dict_1: dict, users_embeddings_dict_2: dict
) -> None:
    assert users_embeddings_dict_1["users_selection"] == users_embeddings_dict_2["users_selection"]
    assert users_embeddings_dict_1["embedding_path"] == users_embeddings_dict_2["embedding_path"]
    for user_id, user_embedding_1 in users_embeddings_dict_1["users_embeddings"].items():
        user_embedding_2 = users_embeddings_dict_2["users_embeddings"][user_id]
        assert user_embedding_1["sessions_ids"] == user_embedding_2["sessions_ids"]
        np.testing.assert_array_equal(
            user_embedding_1["sessions_embeddings"],
            user_embedding_2["sessions_embeddings"],
            err_msg=f"User {user_id} embeddings do not match",
        )
    print("Users embeddings dictionaries match.")


if __name__ == "__main__":
    logreg_path = ProjectPaths.logreg_outputs_path() / "example_config"
    users_embeddings_dict_logreg = load_users_embeddings_dict_logreg(
        path=logreg_path,
    )
    users_embeddings_dict_logreg_2 = load_users_embeddings_dict_logreg(
        path=logreg_path,
    )
    compare_users_embeddings_dicts(users_embeddings_dict_logreg, users_embeddings_dict_logreg_2)
