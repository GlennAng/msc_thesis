import pickle
from pathlib import Path

import numpy as np
import pandas as pd


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


def save_users_embeddings(users_embeddings: dict, args_dict: dict) -> None:
    output_folder = args_dict["output_folder"]
    output_folder.mkdir(parents=True, exist_ok=True)
    if args_dict["single_val_session"]:
        users_ids = list(users_embeddings.keys())
        assert users_ids == sorted(users_ids)
        assert len(users_ids) > 0
        users_coefs = np.zeros(
            (len(users_ids), users_embeddings[users_ids[0]]["sessions_embeddings"].shape[1])
        )
        users_coefs_ids_to_idxs = {}
        for idx, user_id in enumerate(users_ids):
            users_coefs_ids_to_idxs[user_id] = idx
            assert len(users_embeddings[user_id]["sessions_ids"]) == 1
            users_coefs[idx] = users_embeddings[user_id]["sessions_embeddings"][0]
        np.save(output_folder / "users_coefs.npy", users_coefs)
        with open(output_folder / "users_coefs_ids_to_idxs.pkl", "wb") as f:
            pickle.dump(users_coefs_ids_to_idxs, f)
    else:
        with open(output_folder / "users_embeddings.pkl", "wb") as f:
            pickle.dump(users_embeddings, f)
    print(f"Users embeddings saved to {output_folder}.")


def load_users_embeddings(path: Path, check: bool = True) -> dict:
    if not isinstance(path, Path):
        path = Path(path).resolve()
    with open(path / "users_embeddings.pkl", "rb") as f:
        users_embeddings = pickle.load(f)
    if check:
        check_users_embeddings(users_embeddings)
    return users_embeddings


def check_users_embeddings(users_embeddings: dict) -> None:
    assert isinstance(users_embeddings, dict)
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


def compare_users_embeddings(
    users_embeddings_1: dict, users_embeddings_2: dict, config_1: dict = None, config_2: dict = None
) -> None:
    if config_1 is not None and config_2 is not None:
        for key, value in config_1.items():
            assert key in config_2 and config_2[key] == value, f"Config mismatch for key: {key}"
    for user_id, user_embedding_1 in users_embeddings_1.items():
        user_embedding_2 = users_embeddings_2[user_id]
        assert user_embedding_1["sessions_ids"] == user_embedding_2["sessions_ids"]
        np.testing.assert_array_equal(
            user_embedding_1["sessions_embeddings"],
            user_embedding_2["sessions_embeddings"],
            err_msg=f"User {user_id} embeddings do not match",
        )
    print("Users embeddings match.")
