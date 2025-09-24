import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ....logreg.src.embeddings.embedding import Embedding
from ....logreg.src.training.users_ratings import (
    UsersRatingsSelection,
    load_users_ratings_from_selection,
)
from ..data.users_embeddings_data import (
    get_users_val_sessions_ids,
    save_users_embeddings,
)
from .compute_users_embeddings_logreg import (
    compute_logreg_user_embedding,
    logreg_get_embed_function_params,
    logreg_transform_embed_function_params,
)

EMBEDDING_DIM = 357
USERS_SELECTIONS_CHOICES = [None, "sequence_val", "sequence_test", "sequence_high_sessions"]
VALID_EMBED_FUNCTIONS_RANDOMNESS = {
    "mean_pos": False,
    "logreg": True,
    "neural": False,
}
VALID_EMBED_FUNCTIONS = list(VALID_EMBED_FUNCTIONS_RANDOMNESS.keys())


def parse_args() -> tuple:
    parser = argparse.ArgumentParser(description="Compute users embeddings")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--random_state", type=int, required=False, default=None)
    parser.add_argument(
        "--users_selection", type=str, default=None, choices=USERS_SELECTIONS_CHOICES
    )
    args_dict = vars(parser.parse_args())
    args_dict["config_file"] = Path(args_dict["config_file"]).resolve()
    return args_dict["config_file"], args_dict["random_state"]


def compute_mean_pos_user_embedding(
    user_train_set_embeddings: np.ndarray,
    user_train_set_ratings: np.ndarray,
    user_train_set_time_diffs: np.ndarray = None,
) -> np.ndarray:
    assert user_train_set_embeddings.shape[0] == user_train_set_ratings.shape[0]
    assert np.all(user_train_set_ratings >= 0)
    if user_train_set_embeddings.shape[0] == 0:
        return np.zeros(user_train_set_embeddings.shape[1] + 1)
    mean_pos_user_embedding = user_train_set_embeddings.mean(axis=0)
    mean_pos_user_embedding = np.hstack([mean_pos_user_embedding, 0])
    return mean_pos_user_embedding


def get_user_train_set_starting_session_id(
    user_train_set: pd.DataFrame,
    session_id: int,
    session_min_time: pd.Timestamp,
    max_n_train_sessions: int = None,
    max_n_train_days: int = None,
) -> int:
    starting_session_id_sessions = user_train_set["session_id"].min()
    if max_n_train_sessions is not None:
        assert max_n_train_sessions >= 1
        diff = session_id - max_n_train_sessions
        starting_session_id_sessions = max(starting_session_id_sessions, diff)
    starting_session_id_days = user_train_set["session_id"].min()
    if max_n_train_days is not None:
        assert max_n_train_days >= 1
        cutoff_time = session_min_time - pd.Timedelta(days=max_n_train_days)
        sessions_end_times = user_train_set.groupby("session_id")["time"].max()
        valid_sessions = sessions_end_times[sessions_end_times >= cutoff_time]
        if not valid_sessions.empty:
            starting_session_id_days = valid_sessions.index.min()
    return max(starting_session_id_sessions, starting_session_id_days)


def compute_session_min_time(user_ratings: pd.DataFrame, session_id: int) -> pd.Timestamp:
    session_times = user_ratings[user_ratings["session_id"] == session_id]["time"]
    if session_times.empty:
        raise ValueError(f"No ratings found for session_id {session_id}")
    return session_times.min()


def get_user_train_set(
    user_ratings: pd.DataFrame,
    session_id: int,
    hard_constraint_min_n_train_posrated: int,
    hard_constraint_max_n_train_rated: int = None,
    soft_constraint_max_n_train_sessions: int = None,
    soft_constraint_max_n_train_days: int = None,
    remove_negrated_from_history: bool = False,
    session_min_time: pd.Timestamp = None,
) -> pd.DataFrame:
    if session_min_time is None:
        session_min_time = compute_session_min_time(user_ratings, session_id)
    user_train_set = user_ratings[user_ratings["session_id"] < session_id].reset_index(drop=True)
    if remove_negrated_from_history:
        user_train_set = user_train_set[user_train_set["rating"] > 0].reset_index(drop=True)
    if hard_constraint_max_n_train_rated is not None:
        if len(user_train_set) > hard_constraint_max_n_train_rated:
            user_train_set = user_train_set.tail(hard_constraint_max_n_train_rated)
            user_train_set = user_train_set.reset_index(drop=True)
    n_pos_train = user_train_set[user_train_set["rating"] > 0].shape[0]
    if n_pos_train < hard_constraint_min_n_train_posrated:
        raise ValueError(
            f"Fewer than {hard_constraint_min_n_train_posrated} positive ratings in training set."
        )
    if soft_constraint_max_n_train_days is None and soft_constraint_max_n_train_sessions is None:
        return user_train_set

    user_train_set_session_id = get_user_train_set_starting_session_id(
        user_train_set=user_train_set,
        session_id=session_id,
        session_min_time=session_min_time,
        max_n_train_sessions=soft_constraint_max_n_train_sessions,
        max_n_train_days=soft_constraint_max_n_train_days,
    )
    positive_sessions_ids = user_train_set[user_train_set["rating"] > 0]["session_id"].unique()
    sessions_to_try = positive_sessions_ids[positive_sessions_ids <= user_train_set_session_id]
    sessions_to_try = sorted(sessions_to_try, reverse=True)
    for session_id in sessions_to_try:
        user_train_set_c = user_train_set[user_train_set["session_id"] >= session_id]
        n_pos_train_c = (user_train_set_c["rating"] > 0).sum()
        if n_pos_train_c >= hard_constraint_min_n_train_posrated:
            return user_train_set_c
    raise ValueError(
        f"Fewer than {hard_constraint_min_n_train_posrated} positive ratings in training set."
    )


def compute_users_embeddings_general(
    users_ratings: pd.DataFrame,
    users_val_sessions_ids: dict,
    embedding: Embedding,
    embed_function: callable,
    hard_constraint_min_n_train_posrated: int,
    hard_constraint_max_n_train_rated: int = None,
    soft_constraint_max_n_train_sessions: int = None,
    soft_constraint_max_n_train_days: int = None,
    remove_negrated_from_history: bool = False,
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
            session_min_time = compute_session_min_time(user_ratings, session_id)
            user_train_set = get_user_train_set(
                user_ratings=user_ratings,
                session_id=session_id,
                hard_constraint_min_n_train_posrated=hard_constraint_min_n_train_posrated,
                hard_constraint_max_n_train_rated=hard_constraint_max_n_train_rated,
                soft_constraint_max_n_train_sessions=soft_constraint_max_n_train_sessions,
                soft_constraint_max_n_train_days=soft_constraint_max_n_train_days,
                remove_negrated_from_history=remove_negrated_from_history,
            )
            user_train_set_papers_ids = user_train_set["paper_id"].tolist()
            user_train_set_ratings = user_train_set["rating"].to_numpy(dtype=np.int64)
            user_train_set_embeddings = embedding.matrix[
                embedding.get_idxs(user_train_set_papers_ids)
            ]
            user_train_set_time_diffs = (
                session_min_time - user_train_set["time"]
            ).dt.days.to_numpy()
            assert np.all(user_train_set_time_diffs >= 0)
            user_embeddings[i] = embed_function(
                user_train_set_embeddings=user_train_set_embeddings,
                user_train_set_ratings=user_train_set_ratings,
                user_train_set_time_diffs=user_train_set_time_diffs,
                **user_embed_function_params,
            )
        users_embeddings[user_id] = {
            "sessions_ids": user_sessions_ids,
            "sessions_embeddings": user_embeddings,
        }
    return users_embeddings


def init_users_ratings(args_dict: dict) -> tuple:
    if args_dict["old_ratings"]:
        urs = UsersRatingsSelection.SESSION_BASED_FILTERING_OLD
    else:
        urs = UsersRatingsSelection.SESSION_BASED_FILTERING
    users_ratings = load_users_ratings_from_selection(
        users_ratings_selection=urs, relevant_users_ids=args_dict["users_selection"]
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
    if embed_function == "mean_pos":
        embed_function = compute_mean_pos_user_embedding
        users_embeddings = compute_users_embeddings_general(
            users_ratings=users_ratings,
            users_val_sessions_ids=users_val_sessions_ids,
            embedding=embedding,
            embed_function=embed_function,
            hard_constraint_min_n_train_posrated=args_dict[
                "histories_hard_constraint_min_n_train_posrated"
            ],
            hard_constraint_max_n_train_rated=args_dict[
                "histories_hard_constraint_max_n_train_rated"
            ],
            soft_constraint_max_n_train_sessions=args_dict[
                "histories_soft_constraint_max_n_train_sessions"
            ],
            soft_constraint_max_n_train_days=args_dict[
                "histories_soft_constraint_max_n_train_days"
            ],
            remove_negrated_from_history=True,
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
            hard_constraint_min_n_train_posrated=args_dict[
                "histories_hard_constraint_min_n_train_posrated"
            ],
            hard_constraint_max_n_train_rated=args_dict[
                "histories_hard_constraint_max_n_train_rated"
            ],
            soft_constraint_max_n_train_sessions=args_dict[
                "histories_soft_constraint_max_n_train_sessions"
            ],
            soft_constraint_max_n_train_days=args_dict[
                "histories_soft_constraint_max_n_train_days"
            ],
            embed_function_params=embed_function_params,
            embed_function_params_transform=logreg_transform_embed_function_params,
            remove_negrated_from_history=False,
        )

    for user_id in users_embeddings:
        assert users_embeddings[user_id]["sessions_embeddings"].shape[1] == EMBEDDING_DIM
    return users_embeddings


if __name__ == "__main__":
    config_file, random_state = parse_args()
    with open(config_file, "r") as f:
        args_dict = json.load(f)
    
    args_dict["output_folder"] = args_dict["output_folder"] / "users_embeddings"
    args_dict["output_folder"] = args_dict["output_folder"] / f"s_{random_state}"
    users_embeddings = compute_users_embeddings(args_dict, random_state)
    save_users_embeddings(users_embeddings=users_embeddings, args_dict=args_dict)
