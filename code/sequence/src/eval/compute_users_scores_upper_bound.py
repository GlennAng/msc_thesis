import numpy as np
import pandas as pd


def merge_train_logits_k(train_components_dict_k: dict) -> np.ndarray:
    y_train_rated_logits_k = train_components_dict_k["y_train_rated_logits"]
    y_train_negrated_ranking_logits_k = train_components_dict_k["y_train_negrated_ranking_logits"]
    y_negative_samples_logits_after_train_k = train_components_dict_k[
        "y_negative_samples_logits_after_train"
    ]
    rated_expanded = y_train_rated_logits_k[:, np.newaxis]
    n_pos_rated = rated_expanded.shape[0]
    assert n_pos_rated == y_train_negrated_ranking_logits_k.shape[0]
    negative_samples_tiled = np.tile(y_negative_samples_logits_after_train_k, (n_pos_rated, 1))
    merged = np.concatenate(
        [rated_expanded, y_train_negrated_ranking_logits_k, negative_samples_tiled], axis=1
    )
    assert merged.shape[0] == n_pos_rated
    assert (
        merged.shape[1]
        == rated_expanded.shape[1]
        + y_train_negrated_ranking_logits_k.shape[1]
        + negative_samples_tiled.shape[1]
    )
    return merged


def compute_metric_k(logits_k: np.ndarray, metric: str = "ndcg") -> float:
    pos_vals = logits_k[:, 0].reshape(-1, 1)
    pos_ranks = np.sum(logits_k[:, 1:] >= pos_vals, axis=1) + 1
    if metric == "ndcg":
        user_ranking_metric = 1 / np.log2(pos_ranks + 1)
    return user_ranking_metric.mean()


def find_highest_k(metrics_different_k: dict) -> float:
    best_k = max(metrics_different_k, key=metrics_different_k.get)
    return best_k


def get_train_scores_dict(
    train_ratings: pd.DataFrame,
    train_rated_logits_dict: dict,
) -> dict:
    pos_mask = train_ratings["rating"] == 1
    different_k = list(train_rated_logits_dict["y_train_rated_logits"].keys())
    metrics_different_k = {}
    for k in different_k:
        train_rated_logits_dict_k = {
            key: value[k] for key, value in train_rated_logits_dict.items()
        }
        assert pos_mask.shape == train_rated_logits_dict_k["y_train_rated_logits"].shape
        train_rated_logits_dict_k["y_train_rated_logits"] = train_rated_logits_dict_k[
            "y_train_rated_logits"
        ][pos_mask]
        train_logits_k = merge_train_logits_k(train_rated_logits_dict_k)
        metric_k = compute_metric_k(logits_k=train_logits_k)
        metrics_different_k[k] = metric_k
    best_k = find_highest_k(metrics_different_k=metrics_different_k)
    return {
        "train_best_k": best_k,
        "train_difference_best_k_to_1": metrics_different_k[best_k] - metrics_different_k[1],
        "y_train_rated_logits": train_rated_logits_dict["y_train_rated_logits"][best_k],
        "y_train_negrated_ranking_logits": train_rated_logits_dict[
            "y_train_negrated_ranking_logits"
        ][best_k],
        "y_negative_samples_logits_after_train": train_rated_logits_dict[
            "y_negative_samples_logits_after_train"
        ][best_k],
    }


def get_val_components_dict(
    val_ratings: pd.DataFrame,
    y_val_logits: dict,
    y_val_negrated_ranking_logits: dict,
    y_negative_samples_logits: dict,
) -> dict:
    n_val_ratings = val_ratings.shape[0]
    if y_val_logits is not None:
        assert n_val_ratings == y_val_logits[1].shape[0]
    n_pos_val_ratings = val_ratings[val_ratings["rating"] == 1].shape[0]
    if y_val_negrated_ranking_logits is not None:
        assert n_pos_val_ratings == y_val_negrated_ranking_logits[1].shape[0]
    if y_negative_samples_logits is not None:
        assert n_pos_val_ratings == y_negative_samples_logits[1].shape[0]

    val_sessions_df = (
        val_ratings.groupby(["session_id"])["rating"]
        .agg(
            n_neg=lambda x: (x == 0).sum(),
            n_pos=lambda x: (x == 1).sum(),
        )
        .reset_index()
    )
    val_sessions_df["n_rated"] = val_sessions_df["n_pos"] + val_sessions_df["n_neg"]
    val_sessions_df["n_pos_cumcount"] = val_sessions_df["n_pos"].cumsum()
    val_sessions_df["n_rated_cumcount"] = val_sessions_df["n_rated"].cumsum()
    y_val_logits_ub = np.zeros_like(y_val_logits[1])
    y_val_negrated_ranking_logits_ub = np.zeros_like(y_val_negrated_ranking_logits[1])
    y_negative_samples_logits_ub = np.zeros_like(y_negative_samples_logits[1])

    return {
        "val_sessions_df": val_sessions_df,
        "y_val_logits_ub": y_val_logits_ub,
        "y_val_negrated_ranking_logits_ub": y_val_negrated_ranking_logits_ub,
        "y_negative_samples_logits_ub": y_negative_samples_logits_ub,
    }


def update_logits_no_pos_in_session(
    y_val_logits: dict,
    rated_counter: int,
    n_rated_in_session: int,
    n_clusters_with_sufficient_sizes_dict: dict,
    y_val_logits_ub: np.ndarray,
    best_k_list: list,
    difference_best_k_to_1_list: list,
    n_clusters_with_sufficient_sizes_ub: list,
) -> None:
    different_k = list(y_val_logits.keys())
    metrics_different_k = {}
    for k in different_k:
        y_val_logits_k = y_val_logits[k][rated_counter : rated_counter + n_rated_in_session]
        metrics_different_k[k] = np.mean(y_val_logits_k)
    best_k = min(metrics_different_k, key=metrics_different_k.get)
    best_y_val_logits = y_val_logits[best_k][rated_counter : rated_counter + n_rated_in_session]
    y_val_logits_ub[rated_counter : rated_counter + n_rated_in_session] = best_y_val_logits
    best_k_list.append(best_k)
    difference_best_k_to_1_list.append(0.0)
    n_clusters_with_sufficient_sizes_ub.append(n_clusters_with_sufficient_sizes_dict[best_k])


def update_logits_pos_in_session(
    y_val_logits: dict,
    y_val_negrated_ranking_logits: dict,
    y_negative_samples_logits: dict,
    rated_counter: int,
    pos_counter: int,
    n_rated_in_session: int,
    n_pos_in_session: int,
    n_clusters_with_sufficient_sizes_dict: dict,
    y_val_logits_ub: np.ndarray,
    y_val_negrated_ranking_logits_ub: np.ndarray,
    y_negative_samples_logits_ub: np.ndarray,
    best_k_list: list,
    difference_best_k_to_1_list: list,
    n_clusters_with_sufficient_sizes_ub: list,
    pos_mask: pd.Series,
) -> None:
    different_k = list(y_val_logits.keys())
    metrics_different_k = {}
    for k in different_k:
        y_val_logits_k = y_val_logits[k][rated_counter : rated_counter + n_rated_in_session]
        y_val_logits_k = y_val_logits_k[pos_mask.values]
        y_val_negrated_ranking_logits_k = y_val_negrated_ranking_logits[k][
            pos_counter : pos_counter + n_pos_in_session
        ]
        y_negative_samples_logits_k = y_negative_samples_logits[k][
            pos_counter : pos_counter + n_pos_in_session
        ]
        merged_logits_k = np.concatenate(
            [
                y_val_logits_k[:, np.newaxis],
                y_val_negrated_ranking_logits_k,
                y_negative_samples_logits_k,
            ],
            axis=1,
        )
        metric_k = compute_metric_k(logits_k=merged_logits_k)
        metrics_different_k[k] = metric_k
    best_k = find_highest_k(metrics_different_k=metrics_different_k)
    best_y_val_logits = y_val_logits[best_k][rated_counter : rated_counter + n_rated_in_session]
    best_y_val_negrated_ranking_logits = y_val_negrated_ranking_logits[best_k][
        pos_counter : pos_counter + n_pos_in_session
    ]
    best_y_negative_samples_logits = y_negative_samples_logits[best_k][
        pos_counter : pos_counter + n_pos_in_session
    ]
    y_val_logits_ub[rated_counter : rated_counter + n_rated_in_session] = best_y_val_logits
    y_val_negrated_ranking_logits_ub[pos_counter : pos_counter + n_pos_in_session] = (
        best_y_val_negrated_ranking_logits
    )
    y_negative_samples_logits_ub[pos_counter : pos_counter + n_pos_in_session] = (
        best_y_negative_samples_logits
    )
    best_k_list.append(best_k)
    difference_best_k_to_1_list.append(metrics_different_k[best_k] - metrics_different_k[1])
    n_clusters_with_sufficient_sizes_ub.append(n_clusters_with_sufficient_sizes_dict[best_k])


def get_val_scores_dict(
    val_ratings: pd.DataFrame,
    user_sessions_ids: list,
    val_components_dict: dict,
    y_val_logits: dict,
    y_val_negrated_ranking_logits: dict,
    y_negative_samples_logits: dict,
    n_clusters_with_sufficient_sizes_dict: dict,
) -> dict:
    val_sessions_df = val_components_dict["val_sessions_df"]
    y_val_logits_ub = val_components_dict["y_val_logits_ub"]
    y_val_negrated_ranking_logits_ub = val_components_dict["y_val_negrated_ranking_logits_ub"]
    y_negative_samples_logits_ub = val_components_dict["y_negative_samples_logits_ub"]
    best_k_list = []
    difference_best_k_to_1_list = []
    n_clusters_with_sufficient_sizes_ub = []

    rated_counter, pos_counter = 0, 0
    for session_id in user_sessions_ids:
        session_ratings = val_ratings[val_ratings["session_id"] == session_id]
        session_df = val_sessions_df[val_sessions_df["session_id"] == session_id]
        n_rated_in_session = session_df["n_rated"].values[0]
        n_pos_in_session = session_df["n_pos"].values[0]
        if n_pos_in_session == 0:
            assert session_ratings["rating"].sum() == 0
            update_logits_no_pos_in_session(
                y_val_logits=y_val_logits,
                rated_counter=rated_counter,
                n_rated_in_session=n_rated_in_session,
                n_clusters_with_sufficient_sizes_dict=n_clusters_with_sufficient_sizes_dict,
                y_val_logits_ub=y_val_logits_ub,
                best_k_list=best_k_list,
                difference_best_k_to_1_list=difference_best_k_to_1_list,
                n_clusters_with_sufficient_sizes_ub=n_clusters_with_sufficient_sizes_ub,
            )
        else:
            pos_mask = session_ratings["rating"] == 1
            update_logits_pos_in_session(
                y_val_logits=y_val_logits,
                y_val_negrated_ranking_logits=y_val_negrated_ranking_logits,
                y_negative_samples_logits=y_negative_samples_logits,
                rated_counter=rated_counter,
                pos_counter=pos_counter,
                n_rated_in_session=n_rated_in_session,
                n_pos_in_session=n_pos_in_session,
                n_clusters_with_sufficient_sizes_dict=n_clusters_with_sufficient_sizes_dict,
                y_val_logits_ub=y_val_logits_ub,
                y_val_negrated_ranking_logits_ub=y_val_negrated_ranking_logits_ub,
                y_negative_samples_logits_ub=y_negative_samples_logits_ub,
                best_k_list=best_k_list,
                difference_best_k_to_1_list=difference_best_k_to_1_list,
                n_clusters_with_sufficient_sizes_ub=n_clusters_with_sufficient_sizes_ub,
                pos_mask=pos_mask,
            )
        rated_counter += n_rated_in_session
        pos_counter += n_pos_in_session
    assert rated_counter == val_ratings.shape[0]
    assert pos_counter == val_ratings[val_ratings["rating"] == 1].shape[0]

    return {
        "y_val_logits": y_val_logits_ub,
        "y_val_negrated_ranking_logits": y_val_negrated_ranking_logits_ub,
        "y_negative_samples_logits": y_negative_samples_logits_ub,
        "val_best_k_list": best_k_list,
        "val_difference_best_k_to_1_list": difference_best_k_to_1_list,
        "val_n_clusters_with_sufficient_sizes_ub": n_clusters_with_sufficient_sizes_ub,
    }


def get_user_scores_upper_bound(
    user_ratings: pd.DataFrame,
    user_sessions_ids: list,
    train_rated_logits_dict: dict,
    y_val_logits: dict,
    y_val_negrated_ranking_logits: dict,
    y_negative_samples_logits: dict,
    n_clusters_with_sufficient_sizes_dict: dict,
) -> dict:
    train_ratings = user_ratings[user_ratings["split"] == "train"]
    train_scores_dict = get_train_scores_dict(
        train_ratings=train_ratings,
        train_rated_logits_dict=train_rated_logits_dict,
    )
    val_ratings = user_ratings[user_ratings["split"] == "val"]
    val_components_dict = get_val_components_dict(
        val_ratings=val_ratings,
        y_val_logits=y_val_logits,
        y_val_negrated_ranking_logits=y_val_negrated_ranking_logits,
        y_negative_samples_logits=y_negative_samples_logits,
    )
    val_scores_dict = get_val_scores_dict(
        val_ratings=val_ratings,
        user_sessions_ids=user_sessions_ids,
        val_components_dict=val_components_dict,
        y_val_logits=y_val_logits,
        y_val_negrated_ranking_logits=y_val_negrated_ranking_logits,
        y_negative_samples_logits=y_negative_samples_logits,
        n_clusters_with_sufficient_sizes_dict=n_clusters_with_sufficient_sizes_dict,
    )
    return {**train_scores_dict, **val_scores_dict}
