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
from .compute_users_embeddings_mean_pos_pooling import (
    compute_mean_pos_pooling_user_embedding,
)
from .compute_users_embeddings_utils import EmbedFunction, get_embed_function_from_arg


def parse_args() -> tuple:
    parser = argparse.ArgumentParser(description="Compute users embeddings")
    parser.add_argument("--eval_settings_path", type=str, required=True)
    parser.add_argument("--random_state", type=int, required=False, default=None)
    args_dict = vars(parser.parse_args())
    eval_settings_path = Path(args_dict["eval_settings_path"]).resolve()
    with open(eval_settings_path, "r") as f:
        eval_settings = json.load(f)
    return eval_settings, args_dict["random_state"]


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
    ignore_hard_constraint_min_n_train_posrated: bool = False,
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
        if not ignore_hard_constraint_min_n_train_posrated:
            raise ValueError(
                f"Fewer than {hard_constraint_min_n_train_posrated} positive ratings in training set. Only {n_pos_train}."
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
    if user_train_set_session_id not in sessions_to_try:
        sessions_to_try = [user_train_set_session_id] + sessions_to_try
    for session_id in sessions_to_try:
        user_train_set_c = user_train_set[user_train_set["session_id"] >= session_id]
        n_pos_train_c = (user_train_set_c["rating"] > 0).sum()
        if n_pos_train_c >= hard_constraint_min_n_train_posrated:
            return user_train_set_c
    if ignore_hard_constraint_min_n_train_posrated:
        return user_train_set[user_train_set["session_id"] >= sessions_to_try[-1]]
    raise ValueError(
        f"Fewer than {hard_constraint_min_n_train_posrated} positive ratings in training set. Only {n_pos_train}."
    )


def compute_users_embeddings_general(
    users_ratings: pd.DataFrame,
    users_val_sessions_ids: dict,
    embedding: Embedding,
    embed_function: callable,
    embed_dim: int,
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
        user_embeddings = np.zeros((len(user_sessions_ids), embed_dim))
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


def init_users_ratings(eval_settings: dict) -> tuple:
    if eval_settings["old_ratings"]:
        urs = UsersRatingsSelection.SESSION_BASED_FILTERING_OLD
    else:
        urs = UsersRatingsSelection.SESSION_BASED_FILTERING
    users_ratings = load_users_ratings_from_selection(
        users_ratings_selection=urs, relevant_users_ids=eval_settings["users_selection"]
    )
    if eval_settings["single_val_session"]:
        users_ids = users_ratings["user_id"].unique().tolist()
        users_ratings = users_ratings.copy()
        users_ratings["old_session_id"] = users_ratings["session_id"]
        val_mask = users_ratings["split"] == "val"
        min_session_ids = users_ratings[val_mask].groupby("user_id")["session_id"].min()
        assert len(min_session_ids) == len(users_ids)
        for user_id, min_session_id in min_session_ids.items():
            user_val_mask = (users_ratings["user_id"] == user_id) & (
                users_ratings["split"] == "val"
            )
            users_ratings.loc[user_val_mask, "session_id"] = min_session_id
    return users_ratings


def compute_users_embeddings(eval_settings: dict, random_state: int = None) -> dict:
    users_ratings = init_users_ratings(eval_settings)
    users_val_sessions_ids = get_users_val_sessions_ids(users_ratings)
    embedding = Embedding(eval_settings["papers_embedding_path"])
    embed_function = get_embed_function_from_arg(eval_settings["embed_function"])

    users_embeddings = None
    if embed_function == EmbedFunction.MEAN_POS_POOLING:
        embed_function = compute_mean_pos_pooling_user_embedding
        embed_dim = embedding.matrix.shape[1]
        users_embeddings = compute_users_embeddings_general(
            users_ratings=users_ratings,
            users_val_sessions_ids=users_val_sessions_ids,
            embedding=embedding,
            embed_function=embed_function,
            embed_dim=embed_dim,
            hard_constraint_min_n_train_posrated=eval_settings[
                "histories_hard_constraint_min_n_train_posrated"
            ],
            hard_constraint_max_n_train_rated=eval_settings[
                "histories_hard_constraint_max_n_train_rated"
            ],
            soft_constraint_max_n_train_sessions=eval_settings[
                "histories_soft_constraint_max_n_train_sessions"
            ],
            soft_constraint_max_n_train_days=eval_settings[
                "histories_soft_constraint_max_n_train_days"
            ],
            remove_negrated_from_history=True,
        )
    elif embed_function == EmbedFunction.LOGISTIC_REGRESSION:
        embed_function_params = logreg_get_embed_function_params(
            users_ids=users_ratings["user_id"].unique().tolist(),
            random_state=random_state,
            eval_settings=eval_settings,
        )
        embed_dim = embedding.matrix.shape[1] + 1
        users_embeddings = compute_users_embeddings_general(
            users_ratings=users_ratings,
            users_val_sessions_ids=users_val_sessions_ids,
            embedding=embedding,
            embed_function=compute_logreg_user_embedding,
            embed_dim=embed_dim,
            hard_constraint_min_n_train_posrated=eval_settings[
                "histories_hard_constraint_min_n_train_posrated"
            ],
            hard_constraint_max_n_train_rated=eval_settings[
                "histories_hard_constraint_max_n_train_rated"
            ],
            soft_constraint_max_n_train_sessions=eval_settings[
                "histories_soft_constraint_max_n_train_sessions"
            ],
            soft_constraint_max_n_train_days=eval_settings[
                "histories_soft_constraint_max_n_train_days"
            ],
            embed_function_params=embed_function_params,
            embed_function_params_transform=logreg_transform_embed_function_params,
            remove_negrated_from_history=False,
        )

    for user_id in users_embeddings:
        assert (
            users_embeddings[user_id]["sessions_embeddings"].shape[1] == embed_dim
        )
    return users_embeddings


if __name__ == "__main__":
    eval_settings, random_state = parse_args()
    users_embeddings = compute_users_embeddings(eval_settings, random_state)
    eval_data_folder = Path(eval_settings["eval_data_folder"]).resolve()
    users_embeddings_folder = eval_data_folder / "users_embeddings" / f"s_{random_state}"
    save_users_embeddings(
        users_embeddings, users_embeddings_folder, eval_settings["single_val_session"]
    )
