import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ....logreg.src.training.users_ratings import (
    UsersRatingsSelection,
    load_users_ratings_from_selection,
)
from ....src.load_files import load_users_ratings
from .visu_temporal_models import (
    attach_non_ranking_metric_column,
    attach_ranking_metric_column,
    filter_users_ratings,
    get_sessions_df,
    load_users_scores,
)

valid_bin_columns = ["session_id", "n_pos_cum"]


def concat_lists(series: pd.Series) -> list:
    result = []
    for item in series:
        if item is not None:
            result.extend(item)
    return result if result else None


def sessions_df_separated_by_bins(
    sessions_df: pd.DataFrame, bin_column: str, elements_per_bin: int, max_elements: int
) -> pd.DataFrame:
    assert bin_column in valid_bin_columns

    def filter_late_train(group):
        first_val_idx = group[group["split"] != "train"].index.min()
        if pd.isna(first_val_idx):
            return group
        return group[(group.index < first_val_idx) | (group["split"] != "train")]

    sessions_df = sessions_df.groupby("user_id", group_keys=False).apply(filter_late_train)
    sessions_df["bin"] = sessions_df[bin_column] // elements_per_bin
    max_bin = max_elements // elements_per_bin
    sessions_df.loc[sessions_df["bin"] >= max_bin, "bin"] = max_bin
    sessions_df["n_pos_val"] = sessions_df.apply(
        lambda row: row["n_pos"] if row["split"] != "train" else 0, axis=1
    )
    agg_dict = {
        "n_neg": "sum",
        "n_pos": "sum",
        "n_neutral": "sum",
        "n_pos_cum": "last",
        "n_rated": "sum",
        "n_all": "sum",
        "n_days_passed": "max",
        "ndcg": concat_lists,
        "split": "last",
        "n_rated_cum": "last",
        "recall": concat_lists,
        "n_pos_val": "sum",
    }
    grouped = sessions_df.groupby(["user_id", "bin"]).agg(agg_dict).reset_index()
    for idx, row in grouped.iterrows():
        if row["ndcg"] is not None:
            if len(row["ndcg"]) != row["n_pos_val"]:
                print(f"Row {idx}: ndcg length {len(row['ndcg'])} != n_pos_val {row['n_pos_val']}")
        if row["recall"] is not None:
            if len(row["recall"]) != row["n_pos_val"]:
                print(
                    f"Row {idx}: recall length {len(row['recall'])} != n_pos_val {row['n_pos_val']}"
                )
        else:
            assert row["ndcg"] is None
            assert row["n_pos_val"] == 0
    return grouped


def get_bins_df(sessions_df: pd.DataFrame) -> pd.DataFrame:
    sessions_df["ndcg_mean"] = sessions_df["ndcg"].apply(lambda x: np.mean(x) if x is not None else None)
    sessions_df["recall_mean"] = sessions_df["recall"].apply(
        lambda x: np.mean(x) if x is not None else None
    )
    bins_df = sessions_df.groupby('bin').agg(
        ndcg_mean=('ndcg_mean', lambda x: x.dropna().mean()),
        recall_mean=('recall_mean', lambda x: x.dropna().mean()),
        n_users=('ndcg_mean', lambda x: x.notna().sum())  # Count from either one
    ).reset_index()
    return bins_df


if __name__ == "__main__":
    folder = sys.argv[1]
    dirs = os.listdir(folder)
    k_scores = {}
    for dir in dirs:
        dir_path = Path(folder) / dir
        eval_settings = json.load(open(dir_path / "eval_settings.json", "r"))
        k = eval_settings["clustering_k_means_n_clusters"]
        users_scores = load_users_scores(dir_path)
        k_scores[k] = users_scores

    users_ids = load_users_ratings_from_selection(
        users_ratings_selection=UsersRatingsSelection.SESSION_BASED_FILTERING, ids_only=True
    )
    users_ratings = load_users_ratings(relevant_users_ids=users_ids, include_neutral_ratings=True)
    users_ratings = filter_users_ratings(users_ratings, n_min_sessions=None, n_min_days=None)
    sessions_df, _ = get_sessions_df(users_ratings)
    sessions_df_per_k = {}
    for k, users_scores in k_scores.items():
        sessions_df_k = sessions_df.copy()
        sessions_df_k = attach_ranking_metric_column(
            sessions_df_k, visu_type="ndcg", users_scores=k_scores[k]
        )
        sessions_df_k = attach_non_ranking_metric_column(
            sessions_df_k, visu_type="recall", users_scores=k_scores[k]
        )
        sessions_df_k = sessions_df_separated_by_bins(
            sessions_df_k,
            bin_column="n_pos_cum",
            elements_per_bin=25,
            max_elements=250,
        )
        sessions_df_per_k[k] = sessions_df_k
    k_sorted = sorted(sessions_df_per_k.keys())
    all_bins_dfs = []
    for k in k_sorted:
        sessions_df_k = sessions_df_per_k[k]
        bins_df_k = get_bins_df(sessions_df_k)
        bins_df_k['k'] = k  # Add k as a column
        all_bins_dfs.append(bins_df_k)

    # Combine all dataframes
    combined_df = pd.concat(all_bins_dfs, ignore_index=True)

    # Pivot to get each metric across different k values
    for metric in ['ndcg_mean', 'recall_mean', 'n_users']:
        print(f"\n{metric}:")
        pivot_df = combined_df.pivot(index='bin', columns='k', values=metric)
        print(pivot_df)
