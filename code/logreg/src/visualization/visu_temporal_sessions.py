import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from ....logreg.src.training.users_ratings import (
    filter_users_ratings_with_sufficient_votes_session_based,
    get_users_ratings_selection_from_arg,
    load_users_ratings_from_selection,
)
from ....src.load_files import load_users_ratings
from .visu_temporal import (
    filter_users_ratings,
    get_sessions_df,
    get_visu_types,
    get_window_df_included_users,
    load_embedding,
)


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users_selection", type=str, default="session_based_no_filtering")
    parser.add_argument("--users_n_min_sessions", type=int, default=None)
    parser.add_argument("--users_n_min_days", type=int, default=None)
    args = vars(parser.parse_args())
    args["users_selection"] = get_users_ratings_selection_from_arg(args["users_selection"])
    return args


def extract_sessions_scores(
    sessions_df: pd.DataFrame, users_embeddings_df: pd.DataFrame, visu_types: dict
) -> None:
    visu_types_scores = {
        visu_type: {"first_session": [], "other_sessions": []} for visu_type in visu_types
    }
    users_ids = sessions_df["user_id"].unique().tolist()
    for user_id in tqdm(users_ids):
        user_sessions = sessions_df[sessions_df["user_id"] == user_id].reset_index(drop=True)
        print(user_sessions)
        for session_idx, session in user_sessions.iterrows():
            for visu_type, visu_info in visu_types.items():
                use_session = len(get_window_df_included_users(session, visu_type)) > 0
                print(session_idx, use_session)
        break


if __name__ == "__main__":
    args = parse_args()
    visu_types = get_visu_types()
    embedding = load_embedding()
    users_ids = load_users_ratings_from_selection(
        users_ratings_selection=args["users_selection"], ids_only=True
    )
    users_ratings = load_users_ratings(relevant_users_ids=users_ids, include_neutral_ratings=True)
    users_ratings = filter_users_ratings(
        users_ratings,
        n_min_sessions=args["users_n_min_sessions"],
        n_min_days=args["users_n_min_days"],
    )
    users_ratings = filter_users_ratings_with_sufficient_votes_session_based(
        users_ratings=users_ratings,
        min_n_posrated_train=20,
        min_n_negrated_train=0,
        min_n_posrated_val=0,
        min_n_negrated_val=0,
        min_n_sessions=0,
        test_size=1.0,
    )
    sessions_df, users_embeddings_df = get_sessions_df(users_ratings, embedding)
    for visu_type in visu_types.keys():
        visu_types[visu_type]["split"] = visu_type in [
            "cosine_start_all",
            "cosine_start_rated",
            "cosine_start_pos",
        ]
    extract_sessions_scores(sessions_df, users_embeddings_df, visu_types)
