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
        },
        "n_votes_rated": {
            "agg_func": "median",
            "title": "Median Number of Rated Papers",
            "y_label": "Number of Papers",
        },
        "n_votes_pos": {
            "agg_func": "median",
            "title": "Median Number of Upvoted Papers",
            "y_label": "Number of Papers",
        },
        "n_votes_neg": {
            "agg_func": "median",
            "title": "Median Number of Downvoted Papers",
            "y_label": "Number of Papers",
        },
        "pos_portion_all": {
            "agg_func": "mean",
            "title": "Mean Portion of Upvoted Papers (among All Papers)",
            "y_label": "Portion of Upvoted Papers",
        },
        "pos_portion_rated": {
            "agg_func": "mean",
            "title": "Mean Portion of Upvoted Papers (among Rated Papers)",
            "y_label": "Portion of Upvoted Papers",
        },
        "cosine_with_self_all": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities of All Papers (including Unrated)",
            "y_label": "Cosine Similarity",
        },
        "cosine_with_self_rated": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities of Rated Papers",
            "y_label": "Cosine Similarity",
        },
        "cosine_with_self_pos": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities of Upvoted Papers",
            "y_label": "Cosine Similarity",
        },
        "cosine_with_self_neg": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities of Downvoted Papers",
            "y_label": "Cosine Similarity",
        },
        "cosine_pos_neg": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities between Upvoted and Downvoted Papers",
            "y_label": "Cosine Similarity",
        },
        "cosine_start_all": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities with Initial Onboarding (All Papers)",
            "y_label": "Cosine Similarity",
        },
        "cosine_start_rated": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities with Initial Onboarding (Rated Papers)",
            "y_label": "Cosine Similarity",
        },
        "cosine_start_pos": {
            "agg_func": "mean",
            "title": "Mean Cosine Similarities with Initial Onboarding (Upvoted Papers)",
            "y_label": "Cosine Similarity",
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
        return 2
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
    parser.add_argument("--y_lower_bound", type=float, default=0)
    parser.add_argument("--y_upper_bound", type=float, default=None)

    args = vars(parser.parse_args())
    args["users_selection"] = get_users_ratings_selection_from_arg(args["users_selection"])
    args["visu_type_entry"] = get_visu_type_entry(args["visu_type"])
    if args["window_size"] is None:
        args["window_size"] = get_default_window_size(args["temporal_type"])
    if args["last_iter_included"] is None:
        args["last_iter_included"] = get_last_iter_included(args["temporal_type"])
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


def get_sessions_df(users_ratings: pd.DataFrame, embedding: Embedding = None) -> tuple:
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
    if embedding is not None:
        for rating in [0, 1, NEUTRAL_RATING]:
            sessions_df = sessions_df_attach_embeddings(
                sessions_df, users_ratings, embedding, rating
            )
        sessions_df = sessions_df_attach_extended_embeddings(sessions_df)
    sessions_df = sessions_df_attach_n_days_passed(sessions_df, users_ratings)
    sessions_df, users_embeddings_df = sessions_df_attach_split(sessions_df, users_ratings)
    print(f"Number of Sessions: {len(sessions_df)}")
    return sessions_df, users_embeddings_df


def get_window_df_included_users(window_df: pd.DataFrame, visu_type: str) -> pd.DataFrame:
    if visu_type == "n_votes_all":
        return window_df[window_df["n_all"] > 0]
    elif visu_type == "n_votes_rated":
        return window_df[window_df["n_rated"] > 0]
    elif visu_type == "n_votes_pos":
        return window_df[window_df["n_pos"] > 0]
    elif visu_type == "n_votes_neg":
        return window_df[window_df["n_neg"] > 0]
    elif visu_type == "pos_portion_all":
        return window_df[window_df["n_all"] > 0]
    elif visu_type == "pos_portion_rated":
        return window_df[window_df["n_rated"] > 0]
    elif visu_type == "cosine_with_self_all":
        return window_df[window_df["n_all"] > 1]
    elif visu_type == "cosine_with_self_rated":
        return window_df[window_df["n_rated"] > 1]
    elif visu_type == "cosine_with_self_pos":
        return window_df[window_df["n_pos"] > 1]
    elif visu_type == "cosine_with_self_neg":
        return window_df[window_df["n_neg"] > 1]
    elif visu_type == "cosine_pos_neg":
        grouped = window_df.groupby(["user_id", "session_id"]).agg({"n_pos": "sum", "n_neg": "sum"})
        valid_combinations = grouped[(grouped["n_pos"] > 0) & (grouped["n_neg"] > 0)].index
        window_df = window_df[
            window_df.set_index(["user_id", "session_id"]).index.isin(valid_combinations)
        ]
    elif visu_type == "cosine_start_all":
        window_df = window_df[window_df["split"] == "val"]
        window_df = window_df[window_df["n_all"] > 0]
    elif visu_type == "cosine_start_rated":
        window_df = window_df[window_df["split"] == "val"]
        window_df = window_df[window_df["n_rated"] > 0]
    elif visu_type == "cosine_start_pos":
        window_df = window_df[window_df["split"] == "val"]
        window_df = window_df[window_df["n_pos"] > 0]
    else:
        raise ValueError(f"Unknown visu_type: {visu_type}")
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
    true_window_size: int,
    embeddings_df: pd.DataFrame,
    **kwargs: any,
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
    return scores


def extract_data_sessions(
    sessions_df: pd.DataFrame,
    args: dict,
    embeddings_df: pd.DataFrame = None,
    included_users_func: callable = None,
    scores_func: callable = None,
    compare: bool = False,
) -> dict:
    if included_users_func is None:
        included_users_func = get_window_df_included_users
    if scores_func is None:
        scores_func = get_window_scores
    n_users = sessions_df["user_id"].nunique()
    temp_column = "session_id" if args["temporal_type"] == "sessions" else "n_days_passed"
    first_iter_included = max(args["first_iter_included"], 0)
    last_iter_included = min(args["last_iter_included"], sessions_df[temp_column].max())
    sessions_df = sessions_df[sessions_df[temp_column] >= first_iter_included]
    sessions_df = sessions_df[sessions_df[temp_column] <= last_iter_included]
    window_size = args["window_size"]

    percentages_users_included, scores, spreads_lower, spreads_upper = [], [], [], []
    for i in tqdm(range(first_iter_included, last_iter_included + 1)):
        start_idx = max(first_iter_included, i - window_size)
        end_idx = min(last_iter_included, i + window_size)
        true_window_size = end_idx - start_idx + 1
        window_df = sessions_df[sessions_df[temp_column] >= start_idx]
        window_df = window_df[window_df[temp_column] <= end_idx]
        window_df = included_users_func(window_df, args["visu_type"])
        if len(window_df) == 0:
            percentages_users_included.append(0.0)
            scores.append(np.nan)
            spreads_lower.append(np.nan)
            spreads_upper.append(np.nan)
            continue
        percentages_users_included.append(window_df["user_id"].nunique() / n_users)
        scores_func_args = {
            "window_df": window_df,
            "visu_type": args["visu_type"],
            "true_window_size": true_window_size,
            "embeddings_df": embeddings_df,
            "compare": compare,
        }
        window_scores = scores_func(**scores_func_args)
        score, spread_lower, spread_upper = aggregate_window_scores(
            window_scores, args["visu_type_entry"]["agg_func"]
        )
        scores.append(score)
        spreads_lower.append(spread_lower)
        spreads_upper.append(spread_upper)
    return {
        "percentages_users_included": np.array(percentages_users_included),
        "scores": np.array(scores),
        "spreads_lower": np.array(spreads_lower),
        "spreads_upper": np.array(spreads_upper),
    }


def plot_linear_regression(plt: object, x: np.ndarray, y: np.ndarray) -> None:
    mask = ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) > 1:
        slope, intercept, _, _, _ = stats.linregress(x_clean, y_clean)
        regression_line = slope * x + intercept
        plt.plot(
            x,
            regression_line,
            "-",
            color="orange",
            linewidth=2.4,
            alpha=0.9,
            label="Linear Regression",
        )


def plot_line_fill(
    plt: object,
    x: np.ndarray,
    plot_components: dict,
    line_label: str,
    fill_label: str,
) -> None:
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
            label=line_label if i == 0 else None,
        )
    plt.fill_between(
        x,
        plot_components["spreads_lower"],
        plot_components["spreads_upper"],
        color="blue",
        alpha=0.15,
        label=fill_label,
    )


def plot_lims_ticks(
    plt: object,
    ax: object,
    plot_components: dict,
    x: np.ndarray,
    sessions_df: pd.DataFrame,
    args: dict,
) -> None:
    plt.ylim(bottom=args["y_lower_bound"])
    if args["y_upper_bound"] is not None:
        plt.ylim(top=args["y_upper_bound"])
    plt.xlabel(f"{args['temporal_type'].capitalize()}")
    plt.ylabel(args["visu_type_entry"]["y_label"])

    first_iter, last_iter = x[0], x[-1]
    plt.xlim(left=0, right=last_iter)
    tick_positions = np.linspace(0, last_iter, 6)
    tick_positions[0] = first_iter
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{int(pos)}" for pos in tick_positions])
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    users_percentages = plot_components["percentages_users_included"][
        tick_positions.astype(int) - first_iter
    ]
    users_counts = (users_percentages * sessions_df["user_id"].nunique()).astype(int)
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{count} Users" for count in users_counts], fontsize=8)


def plot_data_sessions(
    sessions_df: pd.DataFrame,
    args: dict,
    embeddings_df: pd.DataFrame = None,
    path: str = None,
    included_users_func: callable = None,
    scores_func: callable = None,
    compare: bool = False,
) -> None:
    if path is None:
        path = "visu_temporal.pdf"
    plot_components = extract_data_sessions(
        sessions_df=sessions_df,
        args=args,
        embeddings_df=embeddings_df,
        included_users_func=included_users_func,
        scores_func=scores_func,
        compare=compare,
    )
    _, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#f0f0f0")
    ax.grid(alpha=0.65, linewidth=0.5)
    x = np.arange(
        args["first_iter_included"],
        args["first_iter_included"] + len(plot_components["scores"]),
    )
    if args["plot_regression"]:
        plot_linear_regression(plt=plt, x=x, y=plot_components["scores"])

    if args["visu_type_entry"]["agg_func"] == "median":
        line_label, fill_label = "Median", "Spread (25th-75th percentile)"
    else:
        line_label, fill_label = "Mean", "Spread (±1 Std Dev)"
    plot_line_fill(
        plt=plt,
        x=x,
        plot_components=plot_components,
        line_label=line_label,
        fill_label=fill_label,
    )
    plot_lims_ticks(
        plt=plt,
        ax=ax,
        plot_components=plot_components,
        x=x,
        sessions_df=sessions_df,
        args=args,
    )
    if plot_components["scores"][0] > plot_components["scores"][-1]:
        legend_loc = "upper right"
    else:
        legend_loc = "lower right"
    ax.legend(loc=legend_loc, fontsize=8.5)
    plt.savefig(path)
    plt.close()


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
    plot_data_sessions(sessions_df, args, users_embeddings_df)
