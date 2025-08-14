import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ...logreg.src.embeddings.embedding import Embedding
from ...logreg.src.training.get_users_ratings import (
    USERS_SELECTIONS,
    sequence_load_users_ratings,
)
from ...src.project_paths import ProjectPaths

pd.set_option("display.max_rows", None)


def compute_mean_pos_user_embedding(
    user_train_set_embeddings: np.ndarray, user_train_set_ratings: np.ndarray, n_last: int = None
) -> np.ndarray:
    assert user_train_set_embeddings.shape[0] == user_train_set_ratings.shape[0]
    pos_ratings = user_train_set_embeddings[user_train_set_ratings > 0]
    if pos_ratings.shape[0] == 0:
        return np.zeros(user_train_set_embeddings.shape[1])
    if n_last is not None:
        pos_ratings = pos_ratings[-n_last:]
    return pos_ratings.mean(axis=0)


def compute_users_embeddings(
    users_ratings: pd.DataFrame,
    users_val_sessions_ids: dict,
    embedding: Embedding,
    embed_function: callable,
    embed_function_params: dict = {},
) -> dict:
    users_embeddings = {}
    users_ids = users_ratings["user_id"].unique().tolist()
    assert users_ids == list(users_val_sessions_ids.keys())
    for user_id in users_ids:
        user_ratings = users_ratings[users_ratings["user_id"] == user_id]
        user_sessions_ids = users_val_sessions_ids[user_id]
        user_embeddings = np.zeros((len(user_sessions_ids), embedding.matrix.shape[1]))
        for i, session_id in enumerate(user_sessions_ids):
            user_train_set = user_ratings[user_ratings["session_id"] < session_id].reset_index(
                drop=True
            )
            user_train_set_papers_ids = user_train_set["paper_id"].tolist()
            user_train_set_ratings = user_train_set["rating"].to_numpy(dtype=np.int64)
            user_train_set_embeddings = embedding.matrix[
                embedding.get_idxs(user_train_set_papers_ids)
            ]
            user_embeddings[i] = embed_function(
                user_train_set_embeddings, user_train_set_ratings, **embed_function_params
            )
        users_embeddings[user_id] = {
            "sessions_ids": user_sessions_ids,
            "sessions_embeddings": user_embeddings,
        }
    return users_embeddings


def save_users_embeddings_dict(
    users_embeddings: dict, users_selection: str, embedding_path: Path, save_path: Path
) -> None:
    users_embeddings_dict = {
        "users_embeddings": users_embeddings,
        "users_selection": users_selection,
        "embedding_path": embedding_path,
    }
    with open(save_path, "wb") as f:
        pickle.dump(users_embeddings_dict, f)


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


if __name__ == "__main__":

    users_selection = "finetuning_test"
    users_ratings, users_ids, users_negrated_ranking = sequence_load_users_ratings(
        selection=users_selection
    )
    users_val_sessions_ids = get_users_val_sessions_ids(users_ratings=users_ratings)
    assert users_ids == list(users_val_sessions_ids.keys())

    embedding_path = get_embedding_path(users_selection)
    embedding = Embedding(embedding_path)
    users_embeddings = compute_users_embeddings(
        users_ratings=users_ratings,
        users_val_sessions_ids=users_val_sessions_ids,
        embedding=embedding,
        embed_function=compute_mean_pos_user_embedding,
        embed_function_params={"n_last": None},
    )
    save_users_embeddings_dict(
        users_embeddings=users_embeddings,
        users_selection=users_selection,
        embedding_path=embedding_path,
        save_path=ProjectPaths.sequence_data_users_embeddings_path() / "finetuning_test_mean.pkl",
    )
