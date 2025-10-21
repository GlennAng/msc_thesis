import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from ....finetuning.src.finetuning_compare_embeddings import (
    compute_sims,
    compute_sims_same_set,
)
from ....logreg.src.training.users_ratings import (
    filter_users_ratings_with_sufficient_votes_session_based,
    get_users_ratings_selection_from_arg,
    load_users_ratings_from_selection,
)
from ....src.load_files import NEUTRAL_RATING, load_users_ratings
from ....src.project_paths import ProjectPaths
from ..embeddings.embedding import Embedding


def get_visu_types() -> dict:
    visu_types = {
        "n_votes_all": {
            "agg_func": "median",
            "title": "Median Number of All Papers (including Unrated)",
            "y_label": "Number of Papers",
            "y_upper_bound": None,
            "y_lower_bound": 0,
        },
        "n_votes_rated": {
            "agg_func": "mean",
            "title": "Median Number of Rated Papers",
            "y_label": "Number of Papers",
            "y_upper_bound": None,
            "y_lower_bound": 0,
        },
        "n_votes_pos": {
            "agg_func": "median",
            "title": "Median Number of Upvoted Papers",
            "y_label": "Number of Papers",
            "y_upper_bound": None,
            "y_lower_bound": 0,
        },
        "n_votes_neg": {
            "agg_func": "median",
            "title": "Median Number of Downvoted Papers",
            "y_label": "Number of Papers",
            "y_upper_bound": None,
            "y_lower_bound": 0,
        },
        "pos_portion_all": {
            "agg_func": "mean",
            "title": "Mean Portion of Upvoted Papers (among All Papers)",
            "y_label": "Portion of Upvoted Papers",
            "y_upper_bound": None,
            "y_lower_bound": 0.0,
        },
        "pos_portion_rated": {
            "agg_func": "mean",
            "title": "Mean Portion of Upvoted Papers (among Rated Papers)",
            "y_label": "Portion of Upvoted Papers",
            "y_upper_bound": None,
            "y_lower_bound": 0.0,
        },
        "cosine_with_self_all": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities of All Papers (including Unrated)",
            "y_label": "Cosine Similarity",
            "y_upper_bound": None,
            "y_lower_bound": 0.0,
        },
        "cosine_with_self_rated": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities of Rated Papers",
            "y_label": "Cosine Similarity",
            "y_upper_bound": None,
            "y_lower_bound": 0.0,
        },
        "cosine_with_self_pos": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities of Upvoted Papers",
            "y_label": "Cosine Similarity",
            "y_upper_bound": None,
            "y_lower_bound": 0.0,
        },
        "cosine_with_self_neg": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities of Downvoted Papers",
            "y_label": "Cosine Similarity",
            "y_upper_bound": None,
            "y_lower_bound": 0.0,
        },
        "cosine_pos_neg": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities between Upvoted and Downvoted Papers",
            "y_label": "Cosine Similarity",
            "y_upper_bound": None,
            "y_lower_bound": 0.0,
        },
        "cosine_start_all": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities with Initial Onboarding (All Papers)",
            "y_label": "Cosine Similarity",
            "y_upper_bound": None,
            "y_lower_bound": 0.0,
        },
        "cosine_start_rated": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities with Initial Onboarding (Rated Papers)",
            "y_label": "Cosine Similarity",
            "y_upper_bound": None,
            "y_lower_bound": 0.0,
        },
        "cosine_start_pos": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities with Initial Onboarding (Upvoted Papers)",
            "y_label": "Cosine Similarity",
            "y_upper_bound": None,
            "y_lower_bound": 0.0,
        },
        "ndcg": {
            "agg_func": "mean",
            "title": "Mean nDCG Score",
            "y_label": "nDCG",
            "y_upper_bound": 1.0,
            "y_lower_bound": 0.6,
        },
    }
    return visu_types


def get_visu_type_entry(visu_type: str) -> dict:
    visu_types = get_visu_types()
    if visu_type not in visu_types:
        raise ValueError(
            f"Invalid visu_type: {visu_type}. Valid types are: {list(visu_types.keys())}"
        )
    return visu_types[visu_type]


def get_default_window_size(temporal_type: str) -> int:
    if temporal_type == "sessions":
        return 5
    elif temporal_type == "days":
        return 30


def get_last_iter_included(temporal_type: str) -> int:
    if temporal_type == "sessions":
        return 100
    elif temporal_type == "days":
        return 500


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visu_type", type=str, choices=list(get_visu_types().keys()))
    parser.add_argument("--users_selection", type=str, default="session_based_no_filtering")
    parser.add_argument(
        "--temporal_type", type=str, default="sessions", choices=["sessions", "days"]
    )
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--first_iter_included", type=int, default=0)
    parser.add_argument("--last_iter_included", type=int, default=None)
    parser.add_argument(
        "--not_plot_regression", action="store_false", dest="plot_regression", default=True
    )
    parser.add_argument("--users_n_min_sessions", type=int, default=None)
    parser.add_argument("--users_n_min_days", type=int, default=None)
    parser.add_argument("--scores_path", type=str, default=None)
    parser.add_argument("--scores_path_before", type=str, default=None)

    args = vars(parser.parse_args())
    args["users_selection"] = get_users_ratings_selection_from_arg(args["users_selection"])
    args["visu_type_entry"] = get_visu_type_entry(args["visu_type"])
    if args["window_size"] is None:
        args["window_size"] = get_default_window_size(args["temporal_type"])
    if args["last_iter_included"] is None:
        args["last_iter_included"] = get_last_iter_included(args["temporal_type"])
    if args["scores_path"] is not None:
        scores_path = Path(args["scores_path"])
        if scores_path.stem != "users_embeddings":
            scores_path = scores_path / "users_embeddings"
        args["scores_path"] = scores_path
    if args["scores_path_before"] is not None:
        scores_path_before = Path(args["scores_path_before"])
        if scores_path_before.stem != "users_embeddings":
            scores_path_before = scores_path_before / "users_embeddings"
        args["scores_path_before"] = scores_path_before
    return args


def filter_users_ratings(
    users_ratings: pd.DataFrame, n_min_sessions: int, n_min_days: int
) -> pd.DataFrame:
    if n_min_sessions is not None:
        user_sessions = users_ratings.groupby("user_id")["session_id"].nunique()
        valid_users = user_sessions[user_sessions >= n_min_sessions].index
        users_ratings = users_ratings[users_ratings["user_id"].isin(valid_users)]
    if n_min_days is not None:
        user_time_ranges = users_ratings.groupby("user_id")["time"].agg(
            lambda x: (x.max() - x.min()).days
        )
        valid_users = user_time_ranges[user_time_ranges >= n_min_days].index
        users_ratings = users_ratings[users_ratings["user_id"].isin(valid_users)]
    users_ratings = users_ratings.reset_index(drop=True)
    print(
        f"Number of Ratings: {len(users_ratings)}, Number of Users: {users_ratings['user_id'].nunique()}"
    )
    return users_ratings


def load_embedding(
    embedding_path: Path = ProjectPaths().logreg_embeddings_path() / "after_pca" / "gte_large_256",
) -> Embedding:
    return Embedding(embedding_folder=embedding_path)


def sessions_df_attach_embeddings(
    sessions_df: pd.DataFrame, users_ratings: pd.DataFrame, embedding: Embedding, rating: int = 1
) -> pd.DataFrame:
    if rating == 0:
        n_compare = sessions_df["n_neg"]
        s = "embeddings_neg"
    elif rating == 1:
        n_compare = sessions_df["n_pos"]
        s = "embeddings_pos"
    elif rating == NEUTRAL_RATING:
        n_compare = sessions_df["n_neutral"]
        s = "embeddings_neutral"
    embeddings = (
        users_ratings[users_ratings["rating"] == rating]
        .groupby(["user_id", "session_id"])["paper_id"]
        .apply(lambda paper_ids: embedding.matrix[embedding.get_idxs(list(paper_ids))])
        .reset_index()
        .rename(columns={"paper_id": s})
    )
    sessions_df = sessions_df.merge(embeddings, on=["user_id", "session_id"], how="left")
    sessions_df[s] = sessions_df[s].apply(
        lambda x: (
            x
            if isinstance(x, np.ndarray) and x.ndim == 2
            else np.array([], dtype=embedding.matrix.dtype).reshape(0, embedding.matrix.shape[1])
        )
    )
    assert (sessions_df[s].apply(lambda x: x.shape[0]) == n_compare).all()
    return sessions_df


def sessions_df_attach_extended_embeddings(sessions_df: pd.DataFrame) -> pd.DataFrame:
    sessions_df["embeddings_all"] = sessions_df.apply(
        lambda row: np.vstack(
            [row["embeddings_pos"], row["embeddings_neg"], row["embeddings_neutral"]]
        ),
        axis=1,
    )
    sessions_df["embeddings_rated"] = sessions_df.apply(
        lambda row: np.vstack([row["embeddings_pos"], row["embeddings_neg"]]), axis=1
    )
    assert (sessions_df["embeddings_all"].apply(lambda x: x.shape[0]) == sessions_df["n_all"]).all()
    assert (
        sessions_df["embeddings_rated"].apply(lambda x: x.shape[0]) == sessions_df["n_rated"]
    ).all()
    return sessions_df


def sessions_df_attach_n_days_passed(
    sessions_df: pd.DataFrame, users_ratings: pd.DataFrame
) -> pd.DataFrame:
    sessions_times = (
        users_ratings.groupby(["user_id", "session_id"])["time"]
        .min()
        .reset_index()
        .rename(columns={"time": "session_min_time"})
    )
    sessions_df = sessions_df.merge(sessions_times, on=["user_id", "session_id"], how="left")
    users_first_session = (
        sessions_df.groupby("user_id")["session_min_time"]
        .min()
        .reset_index()
        .rename(columns={"session_min_time": "user_first_time"})
    )
    sessions_df = sessions_df.merge(users_first_session, on="user_id", how="left")
    sessions_df["n_days_passed"] = (
        sessions_df["session_min_time"] - sessions_df["user_first_time"]
    ).dt.days
    sessions_df = sessions_df.drop(columns=["session_min_time", "user_first_time"])
    return sessions_df


def sessions_df_attach_split(sessions_df: pd.DataFrame, users_ratings: pd.DataFrame) -> tuple:
    users_embeddings_df = None
    if "split" in users_ratings.columns:
        assert (users_ratings.groupby(["user_id", "session_id"])["split"].nunique() == 1).all()
        sessions_splits = (
            users_ratings.groupby(["user_id", "session_id"])["split"]
            .first()
            .reset_index()
            .rename(columns={"split": "split"})
        )
        sessions_df = sessions_df.merge(sessions_splits, on=["user_id", "session_id"], how="left")
        sessions_df_train = sessions_df[sessions_df["split"] == "train"]
        users_embeddings_df = (
            sessions_df_train.groupby("user_id")["embeddings_all"]
            .apply(lambda x: np.vstack(x))
            .reset_index()
            .rename(columns={"embeddings_all": "train_embeddings_all"})
        )
        users_embeddings_df["train_embeddings_rated"] = (
            sessions_df_train.groupby("user_id")["embeddings_rated"]
            .apply(lambda x: np.vstack(x))
            .reset_index()
            .rename(columns={"embeddings_rated": "train_embeddings_rated"})[
                "train_embeddings_rated"
            ]
        )
        users_embeddings_df["train_embeddings_pos"] = (
            sessions_df_train.groupby("user_id")["embeddings_pos"]
            .apply(lambda x: np.vstack(x))
            .reset_index()
            .rename(columns={"embeddings_pos": "train_embeddings_pos"})["train_embeddings_pos"]
        )
    return sessions_df, users_embeddings_df


def get_sessions_df(users_ratings: pd.DataFrame, embedding: Embedding) -> tuple:
    n_sessions = len(users_ratings[["user_id", "session_id"]].drop_duplicates())
    sessions_df = (
        users_ratings.groupby(["user_id", "session_id"])["rating"]
        .agg(
            n_neg=lambda x: (x == 0).sum(),
            n_pos=lambda x: (x == 1).sum(),
            n_neutral=lambda x: (x == 2).sum(),
        )
        .reset_index()
    )
    assert len(sessions_df) == n_sessions
    assert sessions_df["n_neg"].sum() == (users_ratings["rating"] == 0).sum()
    assert sessions_df["n_pos"].sum() == (users_ratings["rating"] == 1).sum()
    assert sessions_df["n_neutral"].sum() == (users_ratings["rating"] == NEUTRAL_RATING).sum()
    sessions_df["n_rated"] = sessions_df["n_pos"] + sessions_df["n_neg"]
    assert sessions_df["n_rated"].sum() == (users_ratings["rating"] != NEUTRAL_RATING).sum()
    sessions_df["n_all"] = sessions_df["n_pos"] + sessions_df["n_neg"] + sessions_df["n_neutral"]
    assert sessions_df["n_all"].sum() == len(users_ratings)
    for rating in [0, 1, NEUTRAL_RATING]:
        sessions_df = sessions_df_attach_embeddings(sessions_df, users_ratings, embedding, rating)
    sessions_df = sessions_df_attach_extended_embeddings(sessions_df)
    sessions_df = sessions_df_attach_n_days_passed(sessions_df, users_ratings)
    sessions_df, users_embeddings_df = sessions_df_attach_split(sessions_df, users_ratings)
    print(f"Number of Sessions: {len(sessions_df)}")
    return sessions_df, users_embeddings_df


def get_window_df_included_users(window_df: pd.DataFrame, visu_type: str) -> pd.DataFrame:
    if visu_type == "cosine_with_self_all":
        return window_df[window_df["n_all"] > 1]
    elif visu_type == "cosine_with_self_rated":
        return window_df[window_df["n_rated"] > 1]
    elif visu_type == "cosine_with_self_pos":
        return window_df[window_df["n_pos"] > 1]
    elif visu_type == "cosine_with_self_neg":
        return window_df[window_df["n_neg"] > 1]
    elif visu_type == "cosine_pos_neg":
        window_df = window_df[(window_df["n_pos"] > 0) & (window_df["n_neg"] > 0)]
    elif visu_type == "cosine_start_all":
        window_df = window_df[window_df["split"] == "val"]
        window_df = window_df[window_df["n_all"] > 0]
    elif visu_type == "cosine_start_rated":
        window_df = window_df[window_df["split"] == "val"]
        window_df = window_df[window_df["n_rated"] > 0]
    elif visu_type == "cosine_start_pos":
        window_df = window_df[window_df["split"] == "val"]
        window_df = window_df[window_df["n_pos"] > 0]
    elif visu_type in ["ndcg", "ndcg_before"]:
        window_df = window_df[window_df["split"] == "val"]
        window_df = window_df[window_df["n_pos"] > 0]
    return window_df


def aggregate_window_scores(scores: pd.Series, agg_func: str) -> tuple:
    if agg_func == "median":
        score = scores.median()
        spread_lower = scores.quantile(0.25)
        spread_upper = scores.quantile(0.75)
    elif agg_func == "mean":
        if len(scores) == 0:
            return np.nan, np.nan, np.nan
        score = scores.mean()
        std = scores.std() if pd.notna(scores.std()) else 0
        spread_lower = score - std
        spread_upper = score + std
    return score, spread_lower, spread_upper


def get_window_scores(
    window_df: pd.DataFrame,
    visu_type: str,
    true_window_size: int = None,
    embeddings_df: pd.DataFrame = None,
) -> pd.Series:
    if visu_type == "n_votes_all":
        scores = window_df.groupby("user_id")["n_all"].sum()
        scores = scores / true_window_size
    if visu_type == "n_votes_rated":
        scores = window_df.groupby("user_id")["n_rated"].sum()
        scores = scores / true_window_size
    elif visu_type == "n_votes_pos":
        scores = window_df.groupby("user_id")["n_pos"].sum()
        scores = scores / true_window_size
    elif visu_type == "n_votes_neg":
        scores = window_df.groupby("user_id")["n_neg"].sum()
        scores = scores / true_window_size
    elif visu_type == "pos_portion_all":
        scores = window_df.groupby("user_id").apply(
            lambda df: df["n_pos"].sum() / df["n_all"].sum(),
            include_groups=False,
        )
    elif visu_type == "pos_portion_rated":
        scores = window_df.groupby("user_id").apply(
            lambda df: df["n_pos"].sum() / df["n_rated"].sum(),
            include_groups=False,
        )
    elif visu_type == "cosine_with_self_all":
        scores = window_df.groupby("user_id").apply(
            lambda df: (compute_sims_same_set(np.vstack(df["embeddings_all"]))),
            include_groups=False,
        )
    elif visu_type == "cosine_with_self_rated":
        scores = window_df.groupby("user_id").apply(
            lambda df: (compute_sims_same_set(np.vstack(df["embeddings_rated"]))),
            include_groups=False,
        )
    elif visu_type == "cosine_with_self_pos":
        scores = window_df.groupby("user_id").apply(
            lambda df: (compute_sims_same_set(np.vstack(df["embeddings_pos"]))),
            include_groups=False,
        )
    elif visu_type == "cosine_with_self_neg":
        scores = window_df.groupby("user_id").apply(
            lambda df: (compute_sims_same_set(np.vstack(df["embeddings_neg"]))),
            include_groups=False,
        )
    elif visu_type == "cosine_pos_neg":
        scores = window_df.groupby("user_id").apply(
            lambda df: (
                compute_sims(
                    np.vstack(df["embeddings_pos"]),
                    np.vstack(df["embeddings_neg"]),
                )
            ),
            include_groups=False,
        )
    elif visu_type == "cosine_start_all":
        window_df_merged = window_df.merge(
            embeddings_df[["user_id", "train_embeddings_all"]],
            on="user_id",
            how="left",
        )
        scores = window_df_merged.groupby("user_id").apply(
            lambda df: compute_sims(
                np.vstack(df["embeddings_all"]),
                df["train_embeddings_all"].iloc[0],
            ),
            include_groups=False,
        )
    elif visu_type == "cosine_start_rated":
        window_df_merged = window_df.merge(
            embeddings_df[["user_id", "train_embeddings_rated"]],
            on="user_id",
            how="left",
        )
        scores = window_df_merged.groupby("user_id").apply(
            lambda df: compute_sims(
                np.vstack(df["embeddings_rated"]),
                df["train_embeddings_rated"].iloc[0],
            ),
            include_groups=False,
        )
    elif visu_type == "cosine_start_pos":
        window_df_merged = window_df.merge(
            embeddings_df[["user_id", "train_embeddings_pos"]],
            on="user_id",
            how="left",
        )
        scores = window_df_merged.groupby("user_id").apply(
            lambda df: compute_sims(
                np.vstack(df["embeddings_pos"]),
                df["train_embeddings_pos"].iloc[0],
            ),
            include_groups=False,
        )
    elif visu_type in ["ndcg", "ndcg_before"]:
        scores = window_df.groupby("user_id")[visu_type].apply(
            lambda x: np.mean(np.concatenate(x.tolist()))
        )
    return scores


def get_mean_and_std(
    sessions_df: pd.DataFrame, args: dict, embeddings_df: pd.DataFrame = None
):
    temp_column = "session_id" if args["temporal_type"] == "sessions" else "n_days_passed"
    first_iter_included = max(args["first_iter_included"], 0)
    last_iter_included = min(args["last_iter_included"], sessions_df[temp_column].max())
    sessions_df = sessions_df[sessions_df[temp_column] >= first_iter_included]
    sessions_df = sessions_df[sessions_df[temp_column] <= last_iter_included]
    window_size = args["window_size"]

    users_ids = sessions_df["user_id"].unique().tolist()
    users_means = []
    users_stds = []
    for user_id in tqdm(users_ids):
        user_scores = []
        user_sessions = sessions_df[sessions_df["user_id"] == user_id]
        for i in range(first_iter_included, last_iter_included + 1):
            start_idx = max(first_iter_included, i - window_size)
            end_idx = min(last_iter_included, i + window_size)
            true_window_size = end_idx - start_idx + 1
            window_df = user_sessions[user_sessions[temp_column] >= start_idx]
            window_df = window_df[window_df[temp_column] <= end_idx]
            window_df = get_window_df_included_users(window_df, args["visu_type"])
            if len(window_df) == 0:
                continue
            window_scores = get_window_scores(
                window_df, args["visu_type"], true_window_size, embeddings_df
            )
            assert len(window_scores) == 1
            user_scores.append(window_scores.iloc[0])
        assert len(user_scores) > 0
        users_means.append(np.mean(user_scores))
        users_stds.append(np.std(user_scores, ddof=0))
    print(f"Mean: {np.mean(users_means)}, Std: {np.mean(users_stds)}")

def extract_data_sessions(
    sessions_df: pd.DataFrame, args: dict, embeddings_df: pd.DataFrame = None
) -> dict:
    n_users = sessions_df["user_id"].nunique()
    temp_column = "session_id" if args["temporal_type"] == "sessions" else "n_days_passed"
    first_iter_included = max(args["first_iter_included"], 0)
    last_iter_included = min(args["last_iter_included"], sessions_df[temp_column].max())
    sessions_df = sessions_df[sessions_df[temp_column] >= first_iter_included]
    sessions_df = sessions_df[sessions_df[temp_column] <= last_iter_included]
    window_size = args["window_size"]
    users_ids = sessions_df["user_id"].unique().tolist()
    users_scores = {user_id: [] for user_id in users_ids}

    percentages_users_included, scores, spreads_lower, spreads_upper = [], [], [], []
    for i in tqdm(range(first_iter_included, last_iter_included + 1)):
        start_idx = max(first_iter_included, i - window_size)
        end_idx = min(last_iter_included, i + window_size)
        true_window_size = end_idx - start_idx + 1
        window_df = sessions_df[sessions_df[temp_column] >= start_idx]
        window_df = window_df[window_df[temp_column] <= end_idx]
        window_df = get_window_df_included_users(window_df, args["visu_type"])
        if len(window_df) == 0:
            percentages_users_included.append(0.0)
            scores.append(np.nan)
            spreads_lower.append(np.nan)
            spreads_upper.append(np.nan)
            continue
        percentages_users_included.append(window_df["user_id"].nunique() / n_users)
        window_scores = get_window_scores(
            window_df, args["visu_type"], true_window_size, embeddings_df
        )
        for user_id in window_df["user_id"].unique():
            users_scores[user_id].append(window_scores.loc[user_id])
        score, spread_lower, spread_upper = aggregate_window_scores(
            window_scores, args["visu_type_entry"]["agg_func"]
        )
        scores.append(score)
        spreads_lower.append(spread_lower)
        spreads_upper.append(spread_upper)
    users_means = [np.mean(u_scores) for u_scores in users_scores.values() if len(u_scores) > 0]
    print(f"Mean: {np.mean(users_means)}, Std: {np.std(users_means)}")
    return {
        "percentages_users_included": np.array(percentages_users_included),
        "scores": np.array(scores),
        "spreads_lower": np.array(spreads_lower),
        "spreads_upper": np.array(spreads_upper),
    }


def plot_data_sessions(
    sessions_df: pd.DataFrame, args: dict, embeddings_df: pd.DataFrame = None, path: str = None
) -> None:
    if path is None:
        path = "visu_temporal.png"
    plot_components = extract_data_sessions(
        sessions_df=sessions_df, args=args, embeddings_df=embeddings_df
    )
    plt.figure(figsize=(10, 5))
    x = np.arange(
        args["first_iter_included"],
        args["first_iter_included"] + len(plot_components["scores"]),
    )
    if args["plot_regression"]:
        y = plot_components["scores"]
        mask = ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]
        if len(x_clean) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            regression_line = slope * x + intercept
            plt.plot(x, regression_line, "-", color="orange", linewidth=1.75, alpha=0.7)
    for i in range(len(x) - 1):
        min_width = 0.5
        max_width = 4.0
        thickness = (
            min_width + (max_width - min_width) * plot_components["percentages_users_included"][i]
        )
        plt.plot(
            x[i : i + 2], plot_components["scores"][i : i + 2], color="blue", linewidth=thickness
        )
    plt.fill_between(
        x,
        plot_components["spreads_lower"],
        plot_components["spreads_upper"],
        color="blue",
        alpha=0.2,
    )
    plt.xlim(left=0, right=x[-1])
    if args["visu_type_entry"]["y_upper_bound"] is not None:
        plt.ylim(
            bottom=args["visu_type_entry"]["y_lower_bound"],
            top=args["visu_type_entry"]["y_upper_bound"],
        )
    else:
        plt.ylim(bottom=args["visu_type_entry"]["y_lower_bound"])
    plt.xlabel(f"{args['temporal_type'].capitalize()}")
    plt.ylabel(args["visu_type_entry"]["y_label"])
    title = f"{args['visu_type_entry']['title']} over {args['temporal_type'].capitalize()}"
    title += f" (Window Size {2* args['window_size'] + 1})."
    plt.title(title)
    plt.grid()
    plt.savefig(path)
    plt.close()


def plot_data_sessions_compare_with_before(
    sessions_df: pd.DataFrame, args: dict, embeddings_df: pd.DataFrame = None, path: str = None
) -> None:
    if path is None:
        path = "visu_temporal_compare.png"
    visu_type = args["visu_type"]
    plot_components = extract_data_sessions(
        sessions_df=sessions_df, args=args, embeddings_df=embeddings_df
    )
    args_before = args.copy()
    if visu_type == "ndcg":
        args_before["visu_type"] = "ndcg_before"
    plot_components_before = extract_data_sessions(
        sessions_df=sessions_df, args=args_before, embeddings_df=embeddings_df
    )
    plt.figure(figsize=(10, 5))
    x = np.arange(
        args["first_iter_included"],
        args["first_iter_included"] + len(plot_components["scores"]),
    )
    for i in range(len(x) - 1):
        min_width = 0.5
        max_width = 4.0
        thickness = (
            min_width + (max_width - min_width) * plot_components["percentages_users_included"][i]
        )
        plt.plot(
            x[i : i + 2],
            plot_components["scores"][i : i + 2],
            color="blue",
            linewidth=thickness,
            label="After" if i == 0 else None,
        )
        plt.plot(
            x[i : i + 2],
            plot_components_before["scores"][i : i + 2],
            color="green",
            linewidth=thickness,
            label="Before" if i == 0 else None,
        )
    plt.fill_between(
        x,
        plot_components["spreads_lower"],
        plot_components["spreads_upper"],
        color="blue",
        alpha=0.2,
    )
    plt.fill_between(
        x,
        plot_components_before["spreads_lower"],
        plot_components_before["spreads_upper"],
        color="green",
        alpha=0.2,
    )
    plt.xlim(left=0, right=x[-1])
    if args["visu_type_entry"]["y_upper_bound"] is not None:
        plt.ylim(
            bottom=args["visu_type_entry"]["y_lower_bound"],
            top=args["visu_type_entry"]["y_upper_bound"],
        )
    else:
        plt.ylim(bottom=args["visu_type_entry"]["y_lower_bound"])
    plt.xlabel(f"{args['temporal_type'].capitalize()}")
    plt.ylabel(args["visu_type_entry"]["y_label"])
    title = f"{args['visu_type_entry']['title']} over {args['temporal_type'].capitalize()}"
    title += f" (Window Size {2* args['window_size'] + 1})."
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(path)
    plt.close()


def attach_ndcg_column(sessions_df: pd.DataFrame, scores_path: str, name: str = "ndcg") -> pd.DataFrame:
    import os
    import pickle

    sessions_df[name] = None
    sessions_df[name] = sessions_df[name].astype(object)
    sessions_df["split"] = "train"

    seeds = os.listdir(scores_path)
    all_scores = []
    for seed in seeds:
        seed_path = Path(os.path.join(scores_path, seed))
        with open(seed_path / "users_scores.pkl", "rb") as f:
            all_scores.append(pickle.load(f))

    sessions_df["n_rated_cum"] = sessions_df.groupby("user_id")["n_rated"].cumsum()
    sessions_df["n_pos_cum"] = sessions_df.groupby("user_id")["n_pos"].cumsum()
    users_ids = sessions_df["user_id"].unique()
    assert set(users_ids) <= set(all_scores[0].keys())
    
    for user_id in tqdm(users_ids):
        user_ndcgs = []
        for i in range(len(all_scores)):
            user_scores_seed = all_scores[i][user_id]
            val_logits_pos = user_scores_seed["y_val_logits_pos"].reshape(-1, 1)
            val_negrated_ranking_logits = user_scores_seed["y_val_negrated_ranking_logits"]
            val_negative_samples_logits = user_scores_seed["y_negative_samples_logits"]
            user_scores_seed = np.hstack(
                [val_logits_pos, val_negrated_ranking_logits, val_negative_samples_logits]
            )
            pos_vals = user_scores_seed[:, 0].reshape(-1, 1)
            pos_ranks = np.sum(user_scores_seed[:, 1:] >= pos_vals, axis=1) + 1
            user_ndcgs_seed = 1 / np.log2(pos_ranks + 1)
            user_ndcgs.append(user_ndcgs_seed)
        user_ndcgs_mean = np.mean(user_ndcgs, axis=0)
        n_train_rated = all_scores[0][user_id]["y_train_rated_logits"].shape[0]
        mask = (sessions_df["user_id"] == user_id) & (sessions_df["n_rated_cum"] > n_train_rated)
        user_sessions = sessions_df[mask].copy()
        ndcgs_counter = 0
        
        for idx, row in user_sessions.iterrows():
            if ndcgs_counter >= len(user_ndcgs_mean):
                assert ndcgs_counter == len(user_ndcgs_mean)
                break
            n_pos = row["n_pos"]
            if n_pos > 0:
                ndcgs = user_ndcgs_mean[ndcgs_counter : ndcgs_counter + n_pos]
                idx_pos = sessions_df.index.get_loc(idx)
                sessions_df[name].values[idx_pos] = ndcgs
                sessions_df.at[idx, "split"] = "val"
                ndcgs_counter += n_pos    
    return sessions_df


if __name__ == "__main__":
    args = parse_args()
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
    if args["visu_type"] in ["cosine_start_all", "cosine_start_rated", "cosine_start_pos"]:
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
    if args["visu_type"] == "ndcg":
        sessions_df = attach_ndcg_column(sessions_df, args["scores_path"])
        if args["scores_path_before"] is not None:
            sessions_df = attach_ndcg_column(sessions_df, args["scores_path_before"], name="ndcg_before")
            plot_data_sessions_compare_with_before(sessions_df, args, users_embeddings_df)
        else:
            plot_data_sessions(sessions_df, args, users_embeddings_df)
    else:
        plot_data_sessions(sessions_df, args, users_embeddings_df)
