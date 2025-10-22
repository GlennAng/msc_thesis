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
    get_window_scores,
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


def extract_sessions_scores_user(
    user_sessions: pd.DataFrame, user_embeddings: pd.DataFrame, visu_types: dict
) -> dict:
    user_scores = {
        visu_type: {"first_session": np.nan, "other_sessions": []} for visu_type in visu_types
    }
    sessions_ids = user_sessions["session_id"].tolist()
    assert sessions_ids[0] == 0
    for session_id in sessions_ids:
        session_df = user_sessions.loc[user_sessions["session_id"] == session_id]
        for visu_type in visu_types.keys():
            window_df = get_window_df_included_users(window_df=session_df, visu_type=visu_type)
            if len(window_df) == 0:
                continue
            score = get_window_scores(
                window_df=window_df,
                visu_type=visu_type,
                true_window_size=1,
                embeddings_df=user_embeddings,
            ).values[0]
            if session_id == 0:
                user_scores[visu_type]["first_session"] = score
            else:
                user_scores[visu_type]["other_sessions"].append(score)
    for visu_type, scores in user_scores.items():
        if len(scores["other_sessions"]) > 0:
            user_scores[visu_type]["other_sessions"] = np.mean(scores["other_sessions"])
        else:
            user_scores[visu_type]["other_sessions"] = np.nan
    return user_scores


def print_results(visu_types: dict, visu_types_scores: dict) -> None:
    for visu_type, scores in visu_types_scores.items():
        assert not np.any(np.isnan(scores["other_sessions"]))
        if not visu_types[visu_type]["split"]:
            assert not np.any(np.isnan(scores["first_session"]))
            assert len(scores["first_session"]) == len(scores["other_sessions"])
        print("_________________________________")
        print(f"Results for Visu Type: {visu_type}, N = {len(scores['other_sessions'])}")
        for s in ["first_session", "other_sessions"]:
            if visu_types[visu_type]["split"] and s == "first_session":
                continue
            scores_s = np.array(scores[s], dtype=np.float64)
            print(
                f" {s}: Mean: {np.mean(scores_s):.4f}, "
                f"Std: {np.std(scores_s):.4f}, "
                f"Median: {np.median(scores_s):.4f}."
            )
        print("")


def extract_sessions_scores(
    sessions_df: pd.DataFrame, users_embeddings_df: pd.DataFrame, visu_types: dict
) -> None:
    visu_types_scores = {
        visu_type: {"first_session": [], "other_sessions": []} for visu_type in visu_types
    }
    users_ids = sessions_df["user_id"].unique().tolist()
    for user_id in tqdm(users_ids):
        user_sessions = sessions_df[sessions_df["user_id"] == user_id].reset_index(drop=True)
        user_embeddings = users_embeddings_df[users_embeddings_df["user_id"] == user_id]
        user_scores = extract_sessions_scores_user(
            user_sessions=user_sessions, user_embeddings=user_embeddings, visu_types=visu_types
        )
        for visu_type, scores in user_scores.items():
            if visu_types[visu_type]["split"]:
                if not np.isnan(scores["other_sessions"]):
                    visu_types_scores[visu_type]["other_sessions"].append(scores["other_sessions"])
            else:
                if np.isnan(scores["first_session"]) or np.isnan(scores["other_sessions"]):
                    continue
                visu_types_scores[visu_type]["first_session"].append(scores["first_session"])
                visu_types_scores[visu_type]["other_sessions"].append(scores["other_sessions"])
    print_results(visu_types=visu_types, visu_types_scores=visu_types_scores)


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
