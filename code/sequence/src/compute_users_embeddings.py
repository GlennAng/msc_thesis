import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ...logreg.src.embeddings.embedding import Embedding
from ...logreg.src.training.algorithm import Evaluation
from ...logreg.src.training.get_users_ratings import (
    get_users_ratings,
)
from .compute_users_embeddings_logreg import (
    compute_logreg_user_embedding,
    logreg_get_embed_function_params,
    logreg_transform_embed_function_params,
)
from .sequence_data import (
    get_users_val_sessions_ids,
    save_users_embeddings,
)

EMBEDDING_DIM = 357
VALID_EMBED_FUNCTIONS_RANDOMNESS = {
    "mean_pos": False,
    "logreg": True,
    "mean_pos_minus_neg": False,
    "mean_pos_minus_mean_neg": False,
}
VALID_EMBED_FUNCTIONS = list(VALID_EMBED_FUNCTIONS_RANDOMNESS.keys())


def parse_args() -> tuple:
    parser = argparse.ArgumentParser(description="Compute users embeddings")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--random_state", type=int, required=False, default=None)
    args_dict = vars(parser.parse_args())
    args_dict["config_file"] = Path(args_dict["config_file"]).resolve()
    return args_dict["config_file"], args_dict["random_state"]


def compute_mean_pos_user_embedding(
    user_train_set_embeddings: np.ndarray, user_train_set_ratings: np.ndarray
) -> np.ndarray:
    assert user_train_set_embeddings.shape[0] == user_train_set_ratings.shape[0]
    pos_ratings = user_train_set_embeddings[user_train_set_ratings > 0]
    if pos_ratings.shape[0] == 0:
        return np.zeros(user_train_set_embeddings.shape[1] + 1)
    pos_ratings = pos_ratings.mean(axis=0)
    pos_ratings = np.hstack([pos_ratings, 0])
    return pos_ratings


def compute_mean_pos_minus_neg_user_embedding(
    user_train_set_embeddings: np.ndarray, user_train_set_ratings: np.ndarray
) -> np.ndarray:
    assert user_train_set_embeddings.shape[0] == user_train_set_ratings.shape[0]
    pos_ratings = user_train_set_embeddings[user_train_set_ratings > 0]
    neg_ratings = user_train_set_embeddings[user_train_set_ratings == 0]
    if pos_ratings.shape[0] == 0 and neg_ratings.shape[0] == 0:
        return np.zeros(user_train_set_embeddings.shape[1] + 1)
    # scale everything except the last 100 columns by -0.1
    neg_ratings *= -0.3
    if neg_ratings.shape[0] > 0:
        pos_ratings = np.vstack([pos_ratings, neg_ratings])
    pos_ratings = pos_ratings.mean(axis=0)
    pos_ratings = np.hstack([pos_ratings, 0])
    return pos_ratings


def compute_mean_pos_minus_mean_neg_user_embedding(
    user_train_set_embeddings: np.ndarray, user_train_set_ratings: np.ndarray
) -> np.ndarray:
    assert user_train_set_embeddings.shape[0] == user_train_set_ratings.shape[0]
    pos_ratings = user_train_set_embeddings[user_train_set_ratings > 0]
    neg_ratings = user_train_set_embeddings[user_train_set_ratings == 0]
    if pos_ratings.shape[0] == 0 and neg_ratings.shape[0] == 0:
        return np.zeros(user_train_set_embeddings.shape[1] + 1)
    pos_ratings = pos_ratings.mean(axis=0)
    neg_ratings = (
        neg_ratings.mean(axis=0) if neg_ratings.shape[0] > 0 else np.zeros_like(pos_ratings)
    )
    pos_ratings = np.hstack([pos_ratings, 0])
    neg_ratings = np.hstack([neg_ratings, 0])
    return pos_ratings - neg_ratings


def get_user_train_set_starting_session_id_max_n_train_sessions(
    user_train_set: pd.DataFrame, session_id: int, max_n_train_sessions: int
) -> int:
    min_session_id = user_train_set["session_id"].min()
    if max_n_train_sessions is None:
        return min_session_id
    assert max_n_train_sessions >= 1
    return max(min_session_id, session_id - max_n_train_sessions)


def get_user_train_set_starting_session_id_max_n_train_days(
    user_train_set: pd.DataFrame, min_val_time: pd.Timestamp, max_n_train_days: int
) -> int:
    min_session_id = user_train_set["session_id"].min()
    if max_n_train_days is None:
        return min_session_id
    cutoff_time = min_val_time - pd.Timedelta(days=max_n_train_days)
    sessions_start_times = user_train_set.groupby("session_id")["time"].min()
    valid_sessions = sessions_start_times[sessions_start_times >= cutoff_time]
    if valid_sessions.empty:
        return None
    return valid_sessions.index.min()


def get_user_train_set(
    user_ratings: pd.DataFrame,
    session_id: int,
    hard_constraint_min_n_train_posrated: int,
    soft_constraint_max_n_train_sessions: int = None,
    soft_constraint_max_n_train_days: int = None,
) -> pd.DataFrame:
    user_train_set = user_ratings[user_ratings["session_id"] < session_id].reset_index(drop=True)
    n_pos_train = user_train_set[user_train_set["rating"] > 0].shape[0]
    if n_pos_train < hard_constraint_min_n_train_posrated:
        raise ValueError(
            f"Fewer than {hard_constraint_min_n_train_posrated} positive ratings in training set."
        )
    if soft_constraint_max_n_train_days is None and soft_constraint_max_n_train_sessions is None:
        return user_train_set

    user_train_set_starting_session_id_max_n_train_sessions = (
        get_user_train_set_starting_session_id_max_n_train_sessions(
            user_train_set, session_id, soft_constraint_max_n_train_sessions
        )
    )
    min_val_time = user_ratings[user_ratings["session_id"] == session_id]["time"].min()
    user_train_set_starting_session_id_max_n_train_days = (
        get_user_train_set_starting_session_id_max_n_train_days(
            user_train_set, min_val_time, soft_constraint_max_n_train_days
        )
    )
    if user_train_set_starting_session_id_max_n_train_days is not None:
        user_train_set_session_id = max(
            user_train_set_starting_session_id_max_n_train_sessions,
            user_train_set_starting_session_id_max_n_train_days,
        )
        user_train_set_c = user_train_set[user_train_set["session_id"] >= user_train_set_session_id]
        n_pos_train_c = user_train_set_c[user_train_set_c["rating"] > 0].shape[0]
        if n_pos_train_c >= hard_constraint_min_n_train_posrated:
            return user_train_set_c

    user_train_set_session_id = user_train_set["session_id"].max()
    while user_train_set_session_id >= user_train_set["session_id"].min():
        user_train_set_c = user_train_set[user_train_set["session_id"] >= user_train_set_session_id]
        n_pos_train_c = user_train_set_c[user_train_set_c["rating"] > 0].shape[0]
        if n_pos_train_c >= hard_constraint_min_n_train_posrated:
            return user_train_set_c
        user_train_set_session_id -= 1


def compute_users_embeddings_general(
    users_ratings: pd.DataFrame,
    users_val_sessions_ids: dict,
    embedding: Embedding,
    embed_function: callable,
    hard_constraint_min_n_train_posrated: int,
    soft_constraint_max_n_train_sessions: int = None,
    soft_constraint_max_n_train_days: int = None,
    embed_function_params: dict = {},
    embed_function_params_transform: callable = None,
) -> dict:
    users_embeddings = {}
    users_ids = users_ratings["user_id"].unique().tolist()
    assert users_ids == list(users_val_sessions_ids.keys())
    for user_id in tqdm(users_ids):
        user_ratings = users_ratings[users_ratings["user_id"] == user_id]
        if embed_function_params_transform is not None:
            user_embed_function_params = embed_function_params_transform(
                user_id, user_ratings, embedding, **embed_function_params
            )
        else:
            user_embed_function_params = embed_function_params
        user_sessions_ids = users_val_sessions_ids[user_id]
        user_embeddings = np.zeros((len(user_sessions_ids), embedding.matrix.shape[1] + 1))
        for i, session_id in enumerate(user_sessions_ids):
            user_train_set = get_user_train_set(
                user_ratings=user_ratings,
                session_id=session_id,
                hard_constraint_min_n_train_posrated=hard_constraint_min_n_train_posrated,
                soft_constraint_max_n_train_sessions=soft_constraint_max_n_train_sessions,
                soft_constraint_max_n_train_days=soft_constraint_max_n_train_days,
            )
            user_train_set_papers_ids = user_train_set["paper_id"].tolist()
            user_train_set_ratings = user_train_set["rating"].to_numpy(dtype=np.int64)
            user_train_set_embeddings = embedding.matrix[
                embedding.get_idxs(user_train_set_papers_ids)
            ]
            user_embeddings[i] = embed_function(
                user_train_set_embeddings, user_train_set_ratings, **user_embed_function_params
            )
        users_embeddings[user_id] = {
            "sessions_ids": user_sessions_ids,
            "sessions_embeddings": user_embeddings,
        }
    return users_embeddings


def init_users_ratings(args_dict: dict) -> tuple:
    users_ratings = get_users_ratings(
        users_selection=args_dict["users_selection"],
        evaluation=Evaluation.SESSION_BASED,
        train_size=args_dict["train_size"],
        min_n_posrated_train=args_dict["min_n_posrated_train"],
        min_n_negrated_train=args_dict["min_n_negrated_train"],
        min_n_posrated_val=args_dict["min_n_posrated_val"],
        min_n_negrated_val=args_dict["min_n_negrated_val"],
        filter_for_negrated_ranking=args_dict["filter_for_negrated_ranking"],
    )
    if args_dict["single_val_session"]:
        users_ids = users_ratings["user_id"].unique().tolist()
        users_ratings = users_ratings.copy()
        val_mask = users_ratings["split"] == "val"
        min_session_ids = users_ratings[val_mask].groupby("user_id")["session_id"].min()
        assert len(min_session_ids) == len(users_ids)
        for user_id, min_session_id in min_session_ids.items():
            user_val_mask = (users_ratings["user_id"] == user_id) & (
                users_ratings["split"] == "val"
            )
            users_ratings.loc[user_val_mask, "session_id"] = min_session_id
    return users_ratings


def compute_users_embeddings(args_dict: dict, random_state: int = None) -> dict:
    users_ratings = init_users_ratings(args_dict)
    users_val_sessions_ids = get_users_val_sessions_ids(users_ratings)
    embedding = Embedding(args_dict["embedding_path"])
    embed_function = args_dict["embed_function"]

    users_embeddings = None
    if embed_function in ["mean_pos", "mean_pos_minus_neg", "mean_pos_minus_mean_neg"]:
        if embed_function == "mean_pos":
            embed_function = compute_mean_pos_user_embedding
        elif embed_function == "mean_pos_minus_neg":
            embed_function = compute_mean_pos_minus_neg_user_embedding
        elif embed_function == "mean_pos_minus_mean_neg":
            embed_function = compute_mean_pos_minus_mean_neg_user_embedding
        users_embeddings = compute_users_embeddings_general(
            users_ratings=users_ratings,
            users_val_sessions_ids=users_val_sessions_ids,
            embedding=embedding,
            embed_function=embed_function,
            hard_constraint_min_n_train_posrated=args_dict["hard_constraint_min_n_train_posrated"],
            soft_constraint_max_n_train_sessions=args_dict["soft_constraint_max_n_train_sessions"],
            soft_constraint_max_n_train_days=args_dict["soft_constraint_max_n_train_days"],
        )
    elif embed_function == "logreg":
        embed_function_params = logreg_get_embed_function_params(
            users_ids=users_ratings["user_id"].unique().tolist(),
            random_state=random_state,
        )
        users_embeddings = compute_users_embeddings_general(
            users_ratings=users_ratings,
            users_val_sessions_ids=users_val_sessions_ids,
            embedding=embedding,
            embed_function=compute_logreg_user_embedding,
            hard_constraint_min_n_train_posrated=args_dict["hard_constraint_min_n_train_posrated"],
            soft_constraint_max_n_train_sessions=args_dict["soft_constraint_max_n_train_sessions"],
            soft_constraint_max_n_train_days=args_dict["soft_constraint_max_n_train_days"],
            embed_function_params=embed_function_params,
            embed_function_params_transform=logreg_transform_embed_function_params,
        )

    for user_id in users_embeddings:
        assert users_embeddings[user_id]["sessions_embeddings"].shape[1] == EMBEDDING_DIM
    return users_embeddings


if __name__ == "__main__":
    config_file, random_state = parse_args()
    with open(config_file, "rb") as f:
        args_dict = pickle.load(f)

    args_dict["output_folder"] = args_dict["output_folder"] / "users_embeddings"
    args_dict["output_folder"] = args_dict["output_folder"] / f"s_{random_state}"
    users_embeddings = compute_users_embeddings(args_dict, random_state)
    save_users_embeddings(users_embeddings=users_embeddings, args_dict=args_dict)
