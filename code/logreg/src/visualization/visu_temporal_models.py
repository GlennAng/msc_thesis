import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ....logreg.src.training.users_ratings import (
    get_users_ratings_selection_from_arg,
    load_users_ratings_from_selection,
)
from ....src.load_files import load_users_ratings
from .visu_temporal import (
    extract_data_sessions,
    filter_users_ratings,
    get_default_window_size,
    get_last_iter_included,
    get_sessions_df,
    plot_data_sessions,
    plot_lims_ticks,
    plot_line_fill,
)

RANKING_METRICS = ["ndcg", "mrr", "hr@1", "ince"]
NON_RANKING_METRICS = ["recall", "specificity", "balanced_accuracy"]
UPPER_BOUND_METRICS = ["best_k", "difference_to_k_1"]
PERCENTAGE_GLOBAL_MODEL_METRICS = [
    "percentage_global_model_pos",
    "percentage_global_model_neg",
    "percentage_global_model_rated",
    "n_clusters_with_sufficient_size",
]


def get_visu_types_models() -> dict:
    visu_types = {
        "recall": {
            "agg_func": "mean",
            "y_label": "Recall",
        },
        "specificity": {
            "agg_func": "mean",
            "y_label": "Specificity",
        },
        "balanced_accuracy": {
            "agg_func": "mean",
            "y_label": "Balanced Accuracy",
        },
        "ndcg": {
            "agg_func": "mean",
            "y_label": "nDCG",
        },
        "mrr": {
            "agg_func": "mean",
            "y_label": "MRR",
        },
        "hr@1": {
            "agg_func": "mean",
            "y_label": "Hit Rate @1",
        },
        "ince": {
            "agg_func": "mean",
            "y_label": "InfoNCE",
        },
        "norm": {
            "agg_func": "mean",
            "y_label": "L2 Norm",
        },
        "mean_abs_coef": {
            "agg_func": "mean",
            "y_label": "Mean Absolute Coefficient",
        },
        "max_abs_coef": {
            "agg_func": "mean",
            "y_label": "Max Absolute Coefficient",
        },
        "std_coef": {
            "agg_func": "mean",
            "y_label": "Standard Deviation",
        },
        "ratio_nonzero": {
            "agg_func": "mean",
            "y_label": "Ratio of Non-Zero Coefficients",
        },
        "num_nonzero": {
            "agg_func": "mean",
            "y_label": "Number of Non-Zero Coefficients",
        },
        "distance_to_prev": {
            "agg_func": "mean",
            "y_label": "L2 Distance\nto Previous Model Coefficients",
        },
        "sim_to_prev": {
            "agg_func": "mean",
            "y_label": "Cosine Similarity\nto Previous Coefficients",
        },
        "distance_to_init": {
            "agg_func": "mean",
            "y_label": "L2 Distance\nto Initial Coefficients",
        },
        "sim_to_init": {
            "agg_func": "mean",
            "y_label": "Cosine Similarity\nto Initial Coefficients",
        },
        "top10_overlap_prev": {
            "agg_func": "mean",
            "y_label": "Top 10 Overlap\nwith Previous Coefficients",
        },
        "top10_overlap_init": {
            "agg_func": "mean",
            "y_label": "Top 10 Overlap\nwith Initial Coefficients",
        },
        "best_k": {
            "agg_func": "mean",
            "y_label": "Number of Clusters",
        },
        "difference_to_k_1": {
            "agg_func": "mean",
            "y_label": "Difference between\n ideal k and 1",
        },
        "percentage_global_model_pos": {
            "agg_func": "mean",
            "y_label": "Percentage of Positive Samples\nassigned to Global Model",
        },
        "percentage_global_model_neg": {
            "agg_func": "mean",
            "y_label": "Percentage of Negative Samples\nassigned to Global Model",
        },
        "percentage_global_model_rated": {
            "agg_func": "mean",
            "y_label": "Percentage of Rated Samples\nassigned to Global Model",
        },
        "n_clusters_with_sufficient_size": {
            "agg_func": "mean",
            "y_label": "Number of Clusters with Sufficient Size",
        },
    }
    return visu_types


def get_visu_type_entry(visu_type: str) -> dict:
    visu_types = get_visu_types_models()
    if visu_type not in visu_types:
        raise ValueError(
            f"Invalid visu_type: {visu_type}. Valid types are: {list(visu_types.keys())}"
        )
    return visu_types[visu_type]


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visu_type", type=str, choices=list(get_visu_types_models().keys()))
    parser.add_argument("--users_selection", type=str, default="session_based_filtering")
    parser.add_argument(
        "--temporal_type", type=str, default="sessions", choices=["sessions", "days", "n_posrated"]
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

    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--model_path_compare", type=str, required=False)
    parser.add_argument("--model_name", type=str, default="Model 1")
    parser.add_argument("--model_name_compare", type=str, default="Model 2")
    parser.add_argument("--include_categories", action="store_true", default=False)

    args = vars(parser.parse_args())
    args["users_selection"] = get_users_ratings_selection_from_arg(args["users_selection"])
    args["visu_type_entry"] = get_visu_type_entry(args["visu_type"])
    if args["window_size"] is None:
        args["window_size"] = get_default_window_size(args["temporal_type"])
    if args["last_iter_included"] is None:
        args["last_iter_included"] = get_last_iter_included(args["temporal_type"])
    return args


def load_users_scores(model_path: Path) -> dict:
    if model_path.stem != "users_embeddings":
        model_path = model_path / "users_embeddings"
    seeds = os.listdir(model_path)
    users_scores = {}
    for seed in seeds:
        seed_int = int(seed.split("_")[-1])
        seed_path = Path(os.path.join(model_path, seed))
        with open(seed_path / "users_scores.pkl", "rb") as f:
            users_scores[seed_int] = pickle.load(f)
    return users_scores


def parse_args_models() -> tuple:
    args = parse_args()
    model_path = Path(args["model_path"])
    users_scores = load_users_scores(model_path)
    model_path_compare = args.get("model_path_compare", None)
    args["compare_models"] = model_path_compare is not None
    if args["visu_type"] not in RANKING_METRICS:
        args["include_last_train_iter"] = not (
            "init" in args["visu_type"] or "prev" in args["visu_type"]
        )
    if model_path_compare is not None:
        users_scores_compare = load_users_scores(Path(model_path_compare))
    else:
        users_scores_compare = None
    return args, users_scores, users_scores_compare


def get_window_df_included_users(window_df: pd.DataFrame, visu_type: str) -> pd.DataFrame:
    window_df = window_df[window_df["split"] == "val"]
    if visu_type in RANKING_METRICS:
        return window_df[window_df["n_pos"] > 0]
    elif visu_type in UPPER_BOUND_METRICS:
        return window_df[window_df["n_pos"] > 0]
    elif visu_type in PERCENTAGE_GLOBAL_MODEL_METRICS:
        if visu_type == "percentage_global_model_pos":
            return window_df[window_df["n_pos"] > 0]
        elif visu_type == "percentage_global_model_neg":
            return window_df[(window_df["n_pos"] > 0) & (window_df["n_neg"] > 0)]
        elif visu_type == "percentage_global_model_rated":
            return window_df[window_df["n_pos"] > 0]
        elif visu_type == "n_clusters_with_sufficient_size":
            return window_df[window_df["n_pos"] > 0]
    elif visu_type == "recall":
        return window_df[window_df["n_pos"] > 0]
    elif visu_type == "specificity":
        return window_df[window_df["n_neg"] > 0]
    elif visu_type == "balanced_accuracy":
        return window_df[window_df["n_rated"] > 0]
    else:
        return window_df


def compute_ranking_metric(user_scores_seed: np.ndarray, ranking_metric: str) -> np.ndarray:
    if ranking_metric == "ince":
        max_scores = np.max(user_scores_seed, axis=1, keepdims=True)
        exp_scores = np.exp(user_scores_seed - max_scores)
        softmax_pos = exp_scores[:, 0] / np.sum(exp_scores, axis=1)
        user_ranking_metric = -np.log(softmax_pos + 1e-10)
        return user_ranking_metric
    pos_vals = user_scores_seed[:, 0].reshape(-1, 1)
    pos_ranks = np.sum(user_scores_seed[:, 1:] >= pos_vals, axis=1) + 1
    if ranking_metric == "ndcg":
        user_ranking_metric = 1 / np.log2(pos_ranks + 1)
    elif ranking_metric == "mrr":
        user_ranking_metric = 1 / pos_ranks
    elif ranking_metric == "hr@1":
        user_ranking_metric = (pos_ranks <= 1).astype(float)
    return user_ranking_metric


def attach_ranking_metric_column(
    sessions_df: pd.DataFrame,
    visu_type: str,
    users_scores: dict,
    compare: bool = False,
) -> pd.DataFrame:
    ranking_metric = visu_type + ("_compare" if compare else "")
    sessions_df[ranking_metric] = None
    sessions_df[ranking_metric] = sessions_df[ranking_metric].astype(object)

    if not compare:
        sessions_df["split"] = "train"
        sessions_df["n_rated_cum"] = sessions_df.groupby("user_id")["n_rated"].cumsum()
    users_ids = sessions_df["user_id"].unique().tolist()

    random_states = list(users_scores.keys())
    for user_id in tqdm(users_ids):
        user_metrics = []
        for random_state in random_states:
            user_scores_seed = users_scores[random_state][user_id]
            val_logits_pos = user_scores_seed["y_val_logits_pos"].reshape(-1, 1)
            val_negrated_ranking_logits = user_scores_seed["y_val_negrated_ranking_logits"]
            val_negative_samples_logits = user_scores_seed["y_negative_samples_logits"]
            user_scores_seed = np.hstack(
                [val_logits_pos, val_negrated_ranking_logits, val_negative_samples_logits]
            )
            user_metric_seed = compute_ranking_metric(user_scores_seed, visu_type)
            user_metrics.append(user_metric_seed)
        user_metrics_mean = np.mean(user_metrics, axis=0)
        n_train_rated = users_scores[random_states[0]][user_id]["y_train_rated_logits"].shape[0]
        mask = (sessions_df["user_id"] == user_id) & (sessions_df["n_rated_cum"] > n_train_rated)

        metric_counter = 0
        user_sessions = sessions_df[mask].copy()
        for idx, row in user_sessions.iterrows():
            if metric_counter >= len(user_metrics_mean):
                assert metric_counter == len(user_metrics_mean)
                break
            n_pos = row["n_pos"]
            if n_pos > 0:
                metrics = user_metrics_mean[metric_counter : metric_counter + n_pos]
                idx_pos = sessions_df.index.get_loc(idx)
                sessions_df[ranking_metric].values[idx_pos] = metrics
                sessions_df.at[idx, "split"] = "val"
                metric_counter += n_pos
    return sessions_df


def attach_non_ranking_metric_column(
    sessions_df: pd.DataFrame,
    visu_type: str,
    users_scores: dict,
    compare: bool = False,
) -> pd.DataFrame:
    metric_column = visu_type + ("_compare" if compare else "")
    sessions_df[metric_column] = None
    sessions_df[metric_column] = sessions_df[metric_column].astype(object)
    if not compare:
        sessions_df["split"] = "train"
        sessions_df["n_rated_cum"] = sessions_df.groupby("user_id")["n_rated"].cumsum()
    users_ids = sessions_df["user_id"].unique().tolist()
    random_states = list(users_scores.keys())
    for user_id in tqdm(users_ids):
        user_metrics = []
        for random_state in random_states:
            user_scores_seed = users_scores[random_state][user_id]
            if visu_type == "recall":
                val_logits = user_scores_seed["y_val_logits_pos"]
                user_scores_seed = np.zeros(val_logits.shape)
                user_scores_seed[val_logits >= 0.5] = 1
            elif visu_type == "specificity":
                val_logits = user_scores_seed["y_val_logits_neg"]
                user_scores_seed = np.zeros(val_logits.shape)
                user_scores_seed[val_logits < 0.5] = 1
            user_metrics.append(user_scores_seed)
        user_metrics_mean = np.mean(user_metrics, axis=0)
        n_train_rated = users_scores[random_states[0]][user_id]["y_train_rated_logits"].shape[0]
        mask = (sessions_df["user_id"] == user_id) & (sessions_df["n_rated_cum"] > n_train_rated)

        metric_counter = 0
        user_sessions = sessions_df[mask].copy()
        for idx, row in user_sessions.iterrows():
            if metric_counter >= len(user_metrics_mean):
                assert metric_counter == len(user_metrics_mean)
                break
            n_samples = row["n_pos"] if visu_type == "recall" else row["n_neg"]
            if n_samples > 0:
                metrics = user_metrics_mean[metric_counter : metric_counter + n_samples]
                idx_pos = sessions_df.index.get_loc(idx)
                sessions_df[metric_column].values[idx_pos] = metrics
                sessions_df.at[idx, "split"] = "val"
                metric_counter += n_samples
    return sessions_df



def attach_upper_bound_column(
    sessions_df: pd.DataFrame, visu_type: str, users_scores: dict, compare: bool = False
) -> pd.DataFrame:
    if not compare:
        sessions_df["split"] = "train"
        sessions_df["n_rated_cum"] = sessions_df.groupby("user_id")["n_rated"].cumsum()
    col_name = visu_type + ("_compare" if compare else "")
    sessions_df[col_name] = np.nan
    users_ids = sessions_df["user_id"].unique().tolist()
    random_states = list(users_scores.keys())
    if visu_type == "best_k":
        lookup_name = "val_best_k_list"
    elif visu_type == "difference_to_k_1":
        lookup_name = "val_difference_best_k_to_1_list"
    else:
        raise ValueError(f"Invalid visu_type for upper bound: {visu_type}")
    for user_id in tqdm(users_ids):
        n_train_rated = users_scores[random_states[0]][user_id]["y_train_rated_logits"].shape[0]
        n_val_rated = users_scores[random_states[0]][user_id]["y_val_logits"].shape[0]
        val_mask = (sessions_df["user_id"] == user_id) & (
            sessions_df["n_rated_cum"].between(n_train_rated + 1, n_train_rated + n_val_rated)
        )
        sessions_df.loc[val_mask, "split"] = "val"
        sessions_df.loc[val_mask, col_name] = np.array(
            users_scores[random_states[0]][user_id][lookup_name]
        )
    return sessions_df


def attach_percentage_global_model_column(
    sessions_df: pd.DataFrame, visu_type: str, users_scores: dict, compare: bool = False
) -> pd.DataFrame:
    percentage_column = visu_type + ("_compare" if compare else "")
    sessions_df[percentage_column] = np.nan
    if not compare:
        sessions_df["split"] = "train"
        sessions_df["n_rated_cum"] = sessions_df.groupby("user_id")["n_rated"].cumsum()
    users_ids = sessions_df["user_id"].unique().tolist()
    random_states = list(users_scores.keys())

    for user_id in tqdm(users_ids):
        n_train_rated = users_scores[random_states[0]][user_id]["y_train_rated_logits"].shape[0]
        n_val_rated = users_scores[random_states[0]][user_id]["y_val_logits"].shape[0]
        val_mask = (sessions_df["user_id"] == user_id) & (
            sessions_df["n_rated_cum"].between(n_train_rated + 1, n_train_rated + n_val_rated)
        )
        sessions_df.loc[val_mask, "split"] = "val"

        if visu_type == "percentage_global_model_pos":
            arr = users_scores[random_states[0]][user_id]["n_global_model_pos"]
        elif visu_type == "percentage_global_model_neg":
            arr = users_scores[random_states[0]][user_id]["n_global_model_neg"]
        elif visu_type == "percentage_global_model_rated":
            arr = (
                users_scores[random_states[0]][user_id]["n_global_model_pos"]
                + users_scores[random_states[0]][user_id]["n_global_model_neg"]
            )
        elif visu_type == "n_clusters_with_sufficient_size":
            arr = users_scores[random_states[0]][user_id]["n_clusters_with_sufficient_size"]
        else:
            raise ValueError(f"Invalid visu_type for percentage global model: {visu_type}")
        assert len(arr) == val_mask.sum()
        sessions_df.loc[val_mask, percentage_column] = arr
    return sessions_df


def attach_model_column(
    sessions_df: pd.DataFrame,
    visu_type: str,
    users_scores: dict,
    include_train_last_iter: bool,
    include_categories: bool,
    compare: bool = False,
) -> pd.DataFrame:
    model_column = visu_type + ("_compare" if compare else "")
    sessions_df[model_column] = np.nan
    if not compare:
        sessions_df["split"] = "train"
        sessions_df["n_rated_cum"] = sessions_df.groupby("user_id")["n_rated"].cumsum()
    users_ids = sessions_df["user_id"].unique().tolist()
    visu_type_cat = visu_type + ("_cat" if include_categories else "_no_cat")
    random_states = list(users_scores.keys())
    for user_id in tqdm(users_ids):
        if not compare:
            n_train_rated = users_scores[random_states[0]][user_id]["y_train_rated_logits"].shape[0]
            n_val_rated = users_scores[random_states[0]][user_id]["y_val_logits"].shape[0]
            lower_bound = n_train_rated if include_train_last_iter else n_train_rated + 1
            val_mask = (sessions_df["user_id"] == user_id) & (
                sessions_df["n_rated_cum"].between(lower_bound, n_train_rated + n_val_rated)
            )
            sessions_df.loc[val_mask, "split"] = "val"
        else:
            val_mask = (sessions_df["user_id"] == user_id) & (sessions_df["split"] == "val")
        user_model_values_seeds = [users_scores[rs][user_id][visu_type_cat] for rs in random_states]
        user_model_values_mean = np.mean(user_model_values_seeds, axis=0)
        assert len(user_model_values_mean) == val_mask.sum()
        sessions_df.loc[val_mask, model_column] = user_model_values_mean
    return sessions_df


def get_window_scores(
    window_df: pd.DataFrame, visu_type: str, compare: bool, **kwargs: any
) -> pd.DataFrame:
    column = visu_type + ("_compare" if compare else "")
    if visu_type in RANKING_METRICS:
        scores = window_df.groupby("user_id")[column].apply(
            lambda x: np.mean(np.concatenate(x.tolist()))
        )
    elif visu_type in NON_RANKING_METRICS:
        scores = window_df.groupby("user_id")[column].apply(
            lambda x: np.mean(np.concatenate(x.tolist()))
        )
    elif (
        visu_type in PERCENTAGE_GLOBAL_MODEL_METRICS
        and visu_type != "n_clusters_with_sufficient_size"
    ):
        if visu_type == "percentage_global_model_pos":
            div_column = "n_pos"
        elif visu_type == "percentage_global_model_neg":
            div_column = "n_neg"
        elif visu_type == "percentage_global_model_rated":
            div_column = "n_rated"
        grouped = window_df.groupby("user_id")
        scores = grouped[column].sum() / grouped[div_column].sum()
    else:
        scores = window_df.groupby("user_id")[column].mean()
    return scores


def plot_data_sessions_compare(sessions_df: pd.DataFrame, args: dict, path: str = None) -> None:
    if path is None:
        path = "visu_temporal.pdf"
    plot_components_1 = extract_data_sessions(
        sessions_df=sessions_df,
        args=args,
        included_users_func=get_window_df_included_users,
        scores_func=get_window_scores,
        compare=False,
    )
    plot_components_2 = extract_data_sessions(
        sessions_df=sessions_df,
        args=args,
        included_users_func=get_window_df_included_users,
        scores_func=get_window_scores,
        compare=True,
    )
    assert len(plot_components_1["scores"]) == len(plot_components_2["scores"])
    _, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#f0f0f0")
    ax.grid(alpha=0.7, linewidth=0.5)
    x = np.arange(
        args["first_iter_included"],
        args["first_iter_included"] + len(plot_components_1["scores"]),
    )
    model_1_label = args.get("model_name", "K=1")
    model_2_label = args.get("model_name_compare", "K=3")
    plot_line_fill(
        plt=plt,
        plot_components=plot_components_1,
        x=x,
        line_label=model_1_label,
        fill=False,
    )
    plot_line_fill(
        plt=plt,
        plot_components=plot_components_2,
        x=x,
        line_label=model_2_label,
        color="orange",
        fill=False,
    )
    plot_lims_ticks(
        plt=plt,
        ax=ax,
        plot_components=plot_components_1,
        x=x,
        sessions_df=sessions_df,
        args=args,
    )
    if plot_components_1["scores"][0] > plot_components_1["scores"][-1]:
        legend_loc = "upper right"
    else:
        legend_loc = "lower right"
    ax.legend(loc=legend_loc, fontsize=8.5)
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    args, users_scores, users_scores_compare = parse_args_models()
    users_ids = load_users_ratings_from_selection(
        users_ratings_selection=args["users_selection"], ids_only=True
    )
    users_ratings = load_users_ratings(relevant_users_ids=users_ids, include_neutral_ratings=True)
    users_ratings = load_users_ratings(relevant_users_ids=users_ids, include_neutral_ratings=True)
    users_ratings = filter_users_ratings(
        users_ratings,
        n_min_sessions=args["users_n_min_sessions"],
        n_min_days=args["users_n_min_days"],
    )
    sessions_df, _ = get_sessions_df(users_ratings)
    if args["visu_type"] in RANKING_METRICS:
        sessions_df = attach_ranking_metric_column(
            sessions_df, args["visu_type"], users_scores, compare=False
        )
        if args["compare_models"]:
            sessions_df = attach_ranking_metric_column(
                sessions_df, args["visu_type"], users_scores_compare, compare=True
            )
    elif args["visu_type"] in NON_RANKING_METRICS:
        sessions_df = attach_non_ranking_metric_column(
            sessions_df, args["visu_type"], users_scores, compare=False
        )
        if args["compare_models"]:
            sessions_df = attach_non_ranking_metric_column(
                sessions_df, args["visu_type"], users_scores_compare, compare=True
            )

    elif args["visu_type"] in UPPER_BOUND_METRICS:
        sessions_df = attach_upper_bound_column(
            sessions_df=sessions_df,
            visu_type=args["visu_type"],
            users_scores=users_scores,
            compare=False,
        )
        if args["compare_models"]:
            sessions_df = attach_upper_bound_column(
                sessions_df=sessions_df,
                visu_type=args["visu_type"],
                users_scores=users_scores_compare,
                compare=True,
            )
    elif args["visu_type"] in PERCENTAGE_GLOBAL_MODEL_METRICS:
        sessions_df = attach_percentage_global_model_column(
            sessions_df=sessions_df,
            visu_type=args["visu_type"],
            users_scores=users_scores,
            compare=False,
        )
        if args["compare_models"]:
            sessions_df = attach_percentage_global_model_column(
                sessions_df=sessions_df,
                visu_type=args["visu_type"],
                users_scores=users_scores_compare,
                compare=True,
            )
    else:
        sessions_df = attach_model_column(
            sessions_df=sessions_df,
            visu_type=args["visu_type"],
            users_scores=users_scores,
            include_train_last_iter=args["include_last_train_iter"],
            include_categories=args["include_categories"],
            compare=False,
        )
        if args["compare_models"]:
            sessions_df = attach_model_column(
                sessions_df=sessions_df,
                visu_type=args["visu_type"],
                users_scores=users_scores_compare,
                include_train_last_iter=args["include_last_train_iter"],
                include_categories=args["include_categories"],
                compare=True,
            )

    if args["compare_models"]:
        plot_data_sessions_compare(sessions_df=sessions_df, args=args)
    else:
        plot_data_sessions(
            sessions_df=sessions_df,
            args=args,
            included_users_func=get_window_df_included_users,
            scores_func=get_window_scores,
            compare=False,
        )
