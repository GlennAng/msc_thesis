import numpy as np
import pandas as pd
from scipy import sparse

from ....logreg.src.embeddings.embedding import Embedding
from ....logreg.src.training.algorithm import Algorithm, get_model
from ....logreg.src.training.evaluation import (
    get_cache_papers_ids_full,
    get_user_cache_papers,
    get_user_val_negative_samples,
    get_val_negative_samples_ids,
)
from ....logreg.src.training.training_data import get_user_categories_ratios
from ....src.load_files import load_papers, load_users_significant_categories
from .logreg_temporal_decay import (
    TemporalDecay,
    TemporalDecayNormalization,
    get_sample_weights_temporal_decay,
    get_temporal_decay_from_arg,
    get_temporal_decay_normalization_from_arg,
)

LOGREG_HYPERPARAMETERS = {"weights_neg_scale": 0, "weights_cache_v": 1, "clf_C": 2}


def logreg_get_embed_function_params(
    users_ids: list, random_state: int, eval_settings: dict
) -> dict:
    users_significant_categories = load_users_significant_categories(relevant_users_ids=users_ids)
    papers = load_papers(relevant_columns=["paper_id", "in_ratings", "in_cache", "l1"])
    val_negative_samples_ids = get_val_negative_samples_ids(
        papers=papers,
        n_categories_samples=eval_settings["logreg_n_val_negative_samples"],
        random_state=random_state,
        papers_ids_to_exclude=papers[papers["in_ratings"]]["paper_id"].tolist(),
    )
    cache_papers_categories_ids, cache_papers_ids = get_cache_papers_ids_full(
        papers=papers,
        cache_type=eval_settings["logreg_cache_type"],
        n_cache=eval_settings["logreg_n_cache"],
        random_state=random_state,
        n_categories_cache=eval_settings["logreg_n_categories_cache"],
    )
    return {
        "random_state": random_state,
        "users_significant_categories": users_significant_categories,
        "val_negative_samples_ids": val_negative_samples_ids,
        "cache_papers_categories_ids": cache_papers_categories_ids,
        "cache_papers_ids": cache_papers_ids,
        "eval_settings": eval_settings,
    }


def logreg_transform_embed_function_params(
    user_id: int,
    user_ratings: pd.DataFrame,
    embedding: Embedding,
    random_state: int,
    users_significant_categories: pd.DataFrame,
    val_negative_samples_ids: list,
    cache_papers_categories_ids: list,
    cache_papers_ids: list,
    eval_settings: dict,
    compute_val_negative_samples_embeddings: bool = False,
    n_negative_samples: int = 100,
) -> dict:
    user_significant_categories = users_significant_categories[
        users_significant_categories["user_id"] == user_id
    ]
    user_categories_ratios = get_user_categories_ratios(
        categories_to_exclude=user_significant_categories
    )
    user_val_negative_samples_ids = get_user_val_negative_samples(
        val_negative_samples_ids=val_negative_samples_ids,
        n_negative_samples=n_negative_samples,
        random_state=random_state,
        user_categories_ratios=user_categories_ratios,
        embedding=None,
    )["val_negative_samples_ids"]
    if compute_val_negative_samples_embeddings:
        val_negative_samples_embeddings = embedding.matrix[
            embedding.get_idxs(user_val_negative_samples_ids)
        ]
    papers_ids_to_exclude_from_cache = (
        user_ratings["paper_id"].tolist() + user_val_negative_samples_ids
    )
    user_cache_papers = get_user_cache_papers(
        cache_type=eval_settings["logreg_cache_type"],
        cache_papers_ids=cache_papers_ids,
        cache_papers_categories_ids=cache_papers_categories_ids,
        n_categories_cache=eval_settings["logreg_n_categories_cache"],
        random_state=random_state,
        papers_ids_to_exclude_from_cache=papers_ids_to_exclude_from_cache,
        user_categories_ratios=user_categories_ratios,
        embedding=embedding,
    )
    X_cache = embedding.matrix[user_cache_papers["cache_embedding_idxs"]]
    user_data = {"X_cache": X_cache, "random_state": random_state, "eval_settings": eval_settings}
    if compute_val_negative_samples_embeddings:
        user_data["val_negative_samples_embeddings"] = val_negative_samples_embeddings
    return user_data


def get_hyperparameters_combination(eval_settings: dict) -> tuple:
    combi = [0] * len(LOGREG_HYPERPARAMETERS)
    for param, idx in LOGREG_HYPERPARAMETERS.items():
        hyperparameter_string = f"logreg_{param}"
        assert hyperparameter_string in eval_settings
        combi[idx] = eval_settings[hyperparameter_string]
    return tuple(combi)


def get_weights_cache(
    correction: int,
    neg_scale: float,
    cache_v: float,
    n_negrated: int,
    n_cache: int,
) -> tuple:
    cache_denom = cache_v * n_negrated + (1.0 - cache_v) * n_cache
    assert cache_denom > 0
    w_cache = correction * neg_scale * (1.0 - cache_v) / cache_denom
    return w_cache, cache_denom


def get_weights_cluster_pos(
    correction: int,
    neg_scale: float,
    pos_scheme: str,
    n_posrated: int,
    n_pos_cluster_in: int,
    cluster_alpha: float,
) -> tuple:
    n_pos_cluster_out = n_posrated - n_pos_cluster_in
    assert n_pos_cluster_out >= 0
    if pos_scheme == "absolute":
        original_ratio_cluster_all = n_pos_cluster_in / n_posrated
        desired_ratio = n_pos_cluster_in / (n_pos_cluster_in + cluster_alpha)
        desired_ratio = max(desired_ratio, original_ratio_cluster_all)
        w_pos_in_cluster = correction * (1.0 - neg_scale) * desired_ratio / n_pos_cluster_in
        w_pos_out_cluster = (
            correction * (1.0 - neg_scale) * (1.0 - desired_ratio) / n_pos_cluster_out
        )
    elif pos_scheme == "relative":
        denom = cluster_alpha * n_pos_cluster_in + (1.0 - cluster_alpha) * n_pos_cluster_out
        assert denom > 0
        w_pos_in_cluster = correction * (1.0 - neg_scale) * cluster_alpha / denom
        w_pos_out_cluster = correction * (1.0 - neg_scale) * (1.0 - cluster_alpha) / denom
        desired_ratio = cluster_alpha / denom
    return w_pos_in_cluster, w_pos_out_cluster, desired_ratio


def get_weights_cluster_neg_none(
    correction: int,
    neg_scale: float,
    cache_v: float,
    n_posrated: int,
    n_negrated: int,
    n_cache: int,
    pos_scheme: str,
    n_pos_cluster_in: int,
    cluster_alpha: float,
) -> tuple:
    w_cache, cache_denom = get_weights_cache(
        correction=correction,
        neg_scale=neg_scale,
        cache_v=cache_v,
        n_negrated=n_negrated,
        n_cache=n_cache,
    )
    w_pos_in_cluster, w_pos_out_cluster, _ = get_weights_cluster_pos(
        correction=correction,
        neg_scale=neg_scale,
        pos_scheme=pos_scheme,
        n_posrated=n_posrated,
        n_pos_cluster_in=n_pos_cluster_in,
        cluster_alpha=cluster_alpha,
    )
    w_neg_in_cluster = correction * neg_scale * cache_v / cache_denom
    return w_pos_in_cluster, w_neg_in_cluster, w_cache, w_pos_out_cluster, w_neg_in_cluster


def get_weights_cluster_neg_middle(
    correction: int,
    neg_scale: float,
    cache_v: float,
    n_posrated: int,
    n_negrated: int,
    n_cache: int,
    pos_scheme: str,
    n_pos_cluster_in: int,
    cluster_alpha: float,
) -> tuple:
    n_pos_cluster_out = n_posrated - n_pos_cluster_in
    assert n_pos_cluster_out >= 0
    if pos_scheme == "absolute":
        raise NotImplementedError
    elif pos_scheme == "relative":
        after_const = 0.6
        alpha_denom = cluster_alpha * n_pos_cluster_in + (1.0 - cluster_alpha) * n_pos_cluster_out
        assert alpha_denom > 0
        capital_alpha = cluster_alpha / alpha_denom
        ratio_neg_pos_before = neg_scale / ((1.0 - neg_scale) * n_pos_cluster_in / n_posrated)
        ratio_neg_pos_after = neg_scale / ((1.0 - neg_scale) * n_pos_cluster_in * capital_alpha)
        capital_beta = (
            after_const * ratio_neg_pos_after + (1.0 - after_const) * ratio_neg_pos_before
        )
        numer = capital_beta * n_pos_cluster_in * capital_alpha
        neg_scale_prime = numer / (1.0 + numer)
        w_pos_in_cluster = correction * (1.0 - neg_scale_prime) * cluster_alpha / alpha_denom
        w_pos_out_cluster = (
            correction * (1.0 - neg_scale_prime) * (1.0 - cluster_alpha) / alpha_denom
        )
    w_cache, cache_denom = get_weights_cache(
        correction=correction,
        neg_scale=neg_scale_prime,
        cache_v=cache_v,
        n_negrated=n_negrated,
        n_cache=n_cache,
    )
    w_neg_in_cluster = correction * neg_scale_prime * cache_v / cache_denom
    return w_pos_in_cluster, w_neg_in_cluster, w_cache, w_pos_out_cluster, w_neg_in_cluster


def get_weights_cluster_neg_same_alpha(
    correction: int,
    neg_scale: float,
    cache_v: float,
    n_posrated: int,
    n_negrated: int,
    n_cache: int,
    pos_scheme: str,
    n_pos_cluster_in: int,
    n_neg_cluster_in: int,
    cluster_alpha: float,
) -> tuple:
    w_cache, cache_denom = get_weights_cache(
        correction=correction,
        neg_scale=neg_scale,
        cache_v=cache_v,
        n_negrated=n_negrated,
        n_cache=n_cache,
    )
    n_pos_cluster_out = n_posrated - n_pos_cluster_in
    assert n_pos_cluster_out >= 0
    n_neg_cluster_out = n_negrated - n_neg_cluster_in
    assert n_neg_cluster_out >= 0
    if pos_scheme == "absolute":
        original_ratio_cluster_all = n_pos_cluster_in / n_posrated
        desired_ratio = n_pos_cluster_in / (n_pos_cluster_in + cluster_alpha)
        desired_ratio = max(desired_ratio, original_ratio_cluster_all)
        w_pos_in_cluster = correction * (1.0 - neg_scale) * desired_ratio / n_pos_cluster_in
        w_pos_out_cluster = (
            correction * (1.0 - neg_scale) * (1.0 - desired_ratio) / n_pos_cluster_out
        )
        if n_neg_cluster_in == 0:
            w_neg_in_cluster = 0.0
            w_neg_out_cluster = correction * neg_scale * cache_v / cache_denom
        elif n_neg_cluster_out == 0:
            w_neg_out_cluster = 0.0
            w_neg_in_cluster = correction * neg_scale * cache_v / cache_denom
        else:
            alpha = "same_as_pasdos"
            original_ratio_neg = n_neg_cluster_in / n_negrated
            if alpha == "same_as_pos":
                desired_ratio_neg = max(original_ratio_neg, desired_ratio)
            else:
                desired_ratio_neg = n_neg_cluster_in / (n_neg_cluster_in + cluster_alpha)
                desired_ratio_neg = max(desired_ratio_neg, original_ratio_neg)
            neg_correction = n_negrated * cache_v / cache_denom
            w_neg_in_cluster = (
                correction * neg_scale * neg_correction * desired_ratio_neg / n_neg_cluster_in
            )
            w_neg_out_cluster = (
                correction
                * neg_scale
                * neg_correction
                * (1.0 - desired_ratio_neg)
                / n_neg_cluster_out
            )
    elif pos_scheme == "relative":
        pos_denom = cluster_alpha * n_pos_cluster_in + (1.0 - cluster_alpha) * n_pos_cluster_out
        assert pos_denom > 0
        w_pos_in_cluster = correction * (1.0 - neg_scale) * cluster_alpha / pos_denom
        w_pos_out_cluster = correction * (1.0 - neg_scale) * (1.0 - cluster_alpha) / pos_denom
        neg_denom = cluster_alpha * n_neg_cluster_in + (1.0 - cluster_alpha) * n_neg_cluster_out
        assert neg_denom > 0
        correction_num = n_negrated * cache_v / cache_denom
        correction_denom_1 = n_neg_cluster_in * cluster_alpha / neg_denom
        correction_denom_2 = n_neg_cluster_out * (1.0 - cluster_alpha) / neg_denom
        correction_factor = correction_num / (correction_denom_1 + correction_denom_2)
        w_neg_in_cluster = correction * neg_scale * correction_factor * cluster_alpha / neg_denom
        w_neg_out_cluster = (
            correction * neg_scale * correction_factor * (1.0 - cluster_alpha) / neg_denom
        )
    return w_pos_in_cluster, w_neg_in_cluster, w_cache, w_pos_out_cluster, w_neg_out_cluster


def get_weights_cluster_neg_same_ratio(
    correction: int,
    neg_scale: float,
    cache_v: float,
    n_posrated: int,
    n_negrated: int,
    n_cache: int,
    pos_scheme: str,
    n_pos_cluster_in: int,
    n_neg_cluster_in: int,
    cluster_alpha: float,
) -> tuple:
    w_pos_in_cluster, w_pos_out_cluster, _ = get_weights_cluster_pos(
        correction=correction,
        neg_scale=neg_scale,
        pos_scheme=pos_scheme,
        n_posrated=n_posrated,
        n_pos_cluster_in=n_pos_cluster_in,
        cluster_alpha=cluster_alpha,
    )
    pos_sum_in = w_pos_in_cluster * n_pos_cluster_in
    pos_sum_out = w_pos_out_cluster * (n_posrated - n_pos_cluster_in)
    desired_ratio = pos_sum_in / (pos_sum_in + pos_sum_out)
    w_cache, cache_denom = get_weights_cache(
        correction=correction,
        neg_scale=neg_scale,
        cache_v=cache_v,
        n_negrated=n_negrated,
        n_cache=n_cache,
    )
    n_neg_cluster_out = n_negrated - n_neg_cluster_in
    cache_term = (n_negrated * cache_v) / cache_denom
    if n_neg_cluster_in == 0:
        w_neg_in_cluster = 0.0
        w_neg_out_cluster = correction * neg_scale * cache_v / cache_denom
    elif n_neg_cluster_out == 0:
        w_neg_out_cluster = 0.0
        w_neg_in_cluster = correction * neg_scale * cache_v / cache_denom
    else:
        w_neg_in_cluster = correction * neg_scale * desired_ratio * cache_term / n_neg_cluster_in
        w_neg_out_cluster = (
            correction * neg_scale * (1.0 - desired_ratio) * cache_term / n_neg_cluster_out
        )
    return w_pos_in_cluster, w_neg_in_cluster, w_cache, w_pos_out_cluster, w_neg_out_cluster


def get_weights_cluster_exponential(
    correction: int,
    neg_scale: float,
    cache_v: float,
    n_posrated: int,
    n_negrated: int,
    n_cache: int,
    cluster_alpha: float,
    cluster_label: int,
    pos_clusters_idxs: dict,
    neg_clusters_idxs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple:
    # Generate the masks and original indices
    n_rated = n_posrated + n_negrated
    pos_rated_mask = y_train[:n_rated] == 1
    neg_rated_mask = y_train[:n_rated] == 0
    pos_original_idxs = np.where(pos_rated_mask)[0]
    neg_original_idxs = np.where(neg_rated_mask)[0]

    # Map cluster indices to original X_train indices
    pos_cluster_original_idxs = pos_original_idxs[pos_clusters_idxs[cluster_label]]
    X_pos_cluster = X_train[pos_cluster_original_idxs]
    mean_pos_cluster = np.mean(X_pos_cluster, axis=0)

    mean_norm = np.linalg.norm(mean_pos_cluster)
    if mean_norm > 0:
        mean_pos_cluster_normalized = mean_pos_cluster / mean_norm
    else:
        mean_pos_cluster_normalized = mean_pos_cluster

    all_sims = []
    all_cluster_labels = []
    cluster_sizes = {}
    for k, idxs in pos_clusters_idxs.items():
        if len(idxs) == 0:
            cluster_sizes[k] = 0
            continue
        # Map cluster indices to original X_train indices
        original_idxs = pos_original_idxs[idxs]
        samples = X_train[original_idxs]
        samples_normalized = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        sims = np.dot(samples_normalized, mean_pos_cluster_normalized)
        all_sims.extend(sims)
        all_cluster_labels.extend([k] * len(sims))
        cluster_sizes[k] = len(idxs)

    all_sims = np.array(all_sims)
    exp_sims = np.exp(cluster_alpha * all_sims)
    total_sim = np.sum(exp_sims)
    normalized_weights = exp_sims / total_sim
    assert len(normalized_weights) == n_posrated

    w_pos_dict = {}
    idx = 0
    for k, idxs in pos_clusters_idxs.items():
        n = cluster_sizes[k]
        if n > 0:
            cluster_weights = normalized_weights[idx : idx + n]
            cluster_weights_scaled = correction * (1.0 - neg_scale) * cluster_weights
            w_pos_dict[k] = cluster_weights_scaled
            idx += n
        else:
            w_pos_dict[k] = np.array([])

    w_neg_dict = {}
    w_cache, cache_denom = get_weights_cache(
        correction=correction,
        neg_scale=neg_scale,
        cache_v=cache_v,
        n_negrated=n_negrated,
        n_cache=n_cache,
    )

    neg_weight_per_sample = correction * neg_scale * cache_v / cache_denom
    for k, idxs in neg_clusters_idxs.items():
        if len(idxs) > 0:
            w_neg_dict[k] = np.full(len(idxs), neg_weight_per_sample)
        else:
            w_neg_dict[k] = np.array([])

    return w_pos_dict, w_neg_dict, w_cache


def get_weights_cluster_merge(
    correction: int,
    neg_scale: float,
    cache_v: float,
    n_posrated: int,
    n_negrated: int,
    n_cache: int,
    cluster_alpha: float,
    cluster_label: int,
    pos_clusters_idxs: dict,
    neg_clusters_idxs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple:
    # Generate the masks and original indices
    n_rated = n_posrated + n_negrated
    pos_rated_mask = y_train[:n_rated] == 1
    neg_rated_mask = y_train[:n_rated] == 0
    pos_original_idxs = np.where(pos_rated_mask)[0]
    neg_original_idxs = np.where(neg_rated_mask)[0]

    # Map cluster indices to original X_train indices
    pos_cluster_original_idxs = pos_original_idxs[pos_clusters_idxs[cluster_label]]
    X_pos_cluster = X_train[pos_cluster_original_idxs]
    mean_pos_cluster = np.mean(X_pos_cluster, axis=0)

    mean_norm = np.linalg.norm(mean_pos_cluster)
    if mean_norm > 0:
        mean_pos_cluster_normalized = mean_pos_cluster / mean_norm
    else:
        mean_pos_cluster_normalized = mean_pos_cluster

    cluster_idxs = pos_original_idxs[pos_clusters_idxs[cluster_label]]
    cluster_samples = X_train[cluster_idxs]
    cluster_samples_normalized = cluster_samples / np.linalg.norm(
        cluster_samples, axis=1, keepdims=True
    )
    cluster_sims = np.dot(cluster_samples_normalized, mean_pos_cluster_normalized)
    cluster_sim = np.mean(cluster_sims)

    in_clusters = [cluster_label]
    out_clusters = []

    THRESHOLD = -0.03

    cluster_sizes = {}
    for k, idxs in pos_clusters_idxs.items():
        cluster_sizes[k] = len(idxs)
        if k == cluster_label:
            continue
        original_idxs = pos_original_idxs[idxs]
        samples = X_train[original_idxs]
        samples_normalized = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        sims = np.dot(samples_normalized, mean_pos_cluster_normalized)
        sim = np.mean(sims)
        if sim - cluster_sim >= THRESHOLD:
            print(f"Cluster {k} assigned to IN clusters (sim={sim:.4f}, ref={cluster_sim:.4f})")
            in_clusters.append(k)
        else:
            out_clusters.append(k)

    n_in_samples = sum(cluster_sizes[k] for k in in_clusters)
    n_out_samples = sum(cluster_sizes[k] for k in out_clusters)
    denom = cluster_alpha * n_in_samples + (1 - cluster_alpha) * n_out_samples
    w_in = correction * (1.0 - neg_scale) * cluster_alpha / denom
    w_out = correction * (1.0 - neg_scale) * (1 - cluster_alpha) / denom

    w_pos_dict = {}
    for k in pos_clusters_idxs.keys():
        n = cluster_sizes[k]
        if n > 0:
            if k in in_clusters:
                w_pos_dict[k] = np.full(n, w_in)
            else:
                w_pos_dict[k] = np.full(n, w_out)
        else:
            w_pos_dict[k] = np.array([])
    desired_ratio = cluster_alpha * n_in_samples / denom
    w_neg_dict = {}
    w_cache, cache_denom = get_weights_cache(
        correction=correction,
        neg_scale=neg_scale,
        cache_v=cache_v,
        n_negrated=n_negrated,
        n_cache=n_cache,
    )
    cluster_sizes_neg = {}
    for k, idxs in neg_clusters_idxs.items():
        cluster_sizes_neg[k] = len(idxs)
    n_neg_cluster_in = sum(cluster_sizes_neg[k] for k in in_clusters)
    n_neg_cluster_out = n_negrated - n_neg_cluster_in
    cache_term = (n_negrated * cache_v) / cache_denom
    if n_neg_cluster_in == 0:
        w_neg_in_cluster = 0.0
        w_neg_out_cluster = correction * neg_scale * cache_v / cache_denom
    elif n_neg_cluster_out == 0:
        w_neg_out_cluster = 0.0
        w_neg_in_cluster = correction * neg_scale * cache_v / cache_denom
    else:
        w_neg_in_cluster = correction * neg_scale * desired_ratio * cache_term / n_neg_cluster_in
        w_neg_out_cluster = (
            correction * neg_scale * (1.0 - desired_ratio) * cache_term / n_neg_cluster_out
        )
    for k, idxs in neg_clusters_idxs.items():
        if len(idxs) > 0:
            if k in in_clusters:
                w_neg_dict[k] = np.full(len(idxs), w_neg_in_cluster)
            else:
                w_neg_dict[k] = np.full(len(idxs), w_neg_out_cluster)
        else:
            w_neg_dict[k] = np.array([])
    return w_pos_dict, w_neg_dict, w_cache


def get_weights_cluster_softmax(
    correction: int,
    neg_scale: float,
    cache_v: float,
    n_posrated: int,
    n_negrated: int,
    n_cache: int,
    cluster_alpha: float,
    cluster_label: int,
    pos_clusters_idxs: dict,
    neg_clusters_idxs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple:
    # Generate the masks and original indices
    n_rated = n_posrated + n_negrated
    pos_rated_mask = y_train[:n_rated] == 1
    neg_rated_mask = y_train[:n_rated] == 0
    pos_original_idxs = np.where(pos_rated_mask)[0]
    neg_original_idxs = np.where(neg_rated_mask)[0]

    # Map cluster indices to original X_train indices
    pos_cluster_original_idxs = pos_original_idxs[pos_clusters_idxs[cluster_label]]
    X_pos_cluster = X_train[pos_cluster_original_idxs]
    mean_pos_cluster = np.mean(X_pos_cluster, axis=0)

    mean_norm = np.linalg.norm(mean_pos_cluster)
    if mean_norm > 0:
        mean_pos_cluster_normalized = mean_pos_cluster / mean_norm
    else:
        mean_pos_cluster_normalized = mean_pos_cluster
    clusters_norms = {}
    clusters_lengths = {}
    for k, idxs in pos_clusters_idxs.items():
        original_idxs = pos_original_idxs[idxs]
        samples = X_train[original_idxs]
        samples_normalized = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        sims = np.dot(samples_normalized, mean_pos_cluster_normalized)
        clusters_norms[k] = np.mean(sims)
        clusters_lengths[k] = len(idxs)

    clusters_keys = list(clusters_norms.keys())
    clusters_sims = np.array([clusters_norms[k] for k in clusters_keys])
    clusters_sims_scaled = clusters_sims / cluster_alpha
    clusters_sims_exp = np.exp(clusters_sims_scaled - np.max(clusters_sims_scaled))
    clusters_softmax = clusters_sims_exp / np.sum(clusters_sims_exp)
    clusters_softmax_dict = {k: clusters_softmax[i] for i, k in enumerate(clusters_keys)}
    denom = np.sum(clusters_softmax_dict[k] * clusters_lengths[k] for k in clusters_keys)
    w_pos_dict = {}
    for k in clusters_keys:
        w_pos_dict[k] = correction * (1.0 - neg_scale) * clusters_softmax_dict[k] / denom
    w_neg_dict = {}
    w_cache, cache_denom = get_weights_cache(
        correction=correction,
        neg_scale=neg_scale,
        cache_v=cache_v,
        n_negrated=n_negrated,
        n_cache=n_cache,
    )
    neg_weight_per_sample = correction * neg_scale * cache_v / cache_denom
    for k, idxs in neg_clusters_idxs.items():
        if len(idxs) > 0:
            w_neg_dict[k] = np.full(len(idxs), neg_weight_per_sample)
        else:
            w_neg_dict[k] = np.array([])
    return w_pos_dict, w_neg_dict, w_cache


def get_weights_cluster_single(
    correction: int,
    neg_scale: float,
    cache_v: float,
    n_posrated: int,
    n_negrated: int,
    n_cache: int,
    cluster_alpha: float,
    cluster_label: int,
    pos_clusters_idxs: dict,
    neg_clusters_idxs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple:
    n_clusters = len(pos_clusters_idxs)
    w_pos_dict = {}
    for k, idxs in pos_clusters_idxs.items():
        cluster_size = len(idxs)
        if cluster_size > 0:
            cluster_weight = correction * (1.0 - neg_scale) / n_clusters / cluster_size
            w_pos_dict[k] = np.full(cluster_size, cluster_weight)
        else:
            w_pos_dict[k] = np.array([])
    w_neg_dict = {}
    w_cache, cache_denom = get_weights_cache(
        correction=correction,
        neg_scale=neg_scale,
        cache_v=cache_v,
        n_negrated=n_negrated,
        n_cache=n_cache,
    )
    neg_weight_per_sample = correction * neg_scale * cache_v / cache_denom
    for k, idxs in neg_clusters_idxs.items():
        if len(idxs) > 0:
            w_neg_dict[k] = np.full(len(idxs), neg_weight_per_sample)
        else:
            w_neg_dict[k] = np.array([])
    return w_pos_dict, w_neg_dict, w_cache


def get_weights_cluster_scheme(
    correction: int,
    neg_scale: float,
    cache_v: float,
    n_posrated: int,
    n_negrated: int,
    n_cache: int,
    pos_scheme: str,
    neg_scheme: str,
    n_pos_cluster_in: int,
    n_neg_cluster_in: int,
    cluster_alpha: float,
    cluster_label: int,
    pos_clusters_idxs: dict,
    neg_clusters_idxs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple:
    if pos_scheme == "exponential":
        return get_weights_cluster_exponential(
            correction=correction,
            neg_scale=neg_scale,
            cache_v=cache_v,
            n_posrated=n_posrated,
            n_negrated=n_negrated,
            n_cache=n_cache,
            cluster_alpha=cluster_alpha,
            cluster_label=cluster_label,
            pos_clusters_idxs=pos_clusters_idxs,
            neg_clusters_idxs=neg_clusters_idxs,
            X_train=X_train,
            y_train=y_train,
        )
    elif pos_scheme == "softmax":
        return get_weights_cluster_softmax(
            correction=correction,
            neg_scale=neg_scale,
            cache_v=cache_v,
            n_posrated=n_posrated,
            n_negrated=n_negrated,
            n_cache=n_cache,
            cluster_alpha=cluster_alpha,
            cluster_label=cluster_label,
            pos_clusters_idxs=pos_clusters_idxs,
            neg_clusters_idxs=neg_clusters_idxs,
            X_train=X_train,
            y_train=y_train,
        )
    elif pos_scheme == "merge":
        return get_weights_cluster_merge(
            correction=correction,
            neg_scale=neg_scale,
            cache_v=cache_v,
            n_posrated=n_posrated,
            n_negrated=n_negrated,
            n_cache=n_cache,
            cluster_alpha=cluster_alpha,
            cluster_label=cluster_label,
            pos_clusters_idxs=pos_clusters_idxs,
            neg_clusters_idxs=neg_clusters_idxs,
            X_train=X_train,
            y_train=y_train,
        )
    elif pos_scheme == "single":
        return get_weights_cluster_single(
            correction=correction,
            neg_scale=neg_scale,
            cache_v=cache_v,
            n_posrated=n_posrated,
            n_negrated=n_negrated,
            n_cache=n_cache,
            cluster_alpha=cluster_alpha,
            cluster_label=cluster_label,
            pos_clusters_idxs=pos_clusters_idxs,
            neg_clusters_idxs=neg_clusters_idxs,
            X_train=X_train,
            y_train=y_train,
        )
    if neg_scheme == "none":
        return get_weights_cluster_neg_none(
            correction=correction,
            neg_scale=neg_scale,
            cache_v=cache_v,
            n_posrated=n_posrated,
            n_negrated=n_negrated,
            n_cache=n_cache,
            pos_scheme=pos_scheme,
            n_pos_cluster_in=n_pos_cluster_in,
            cluster_alpha=cluster_alpha,
        )
    elif neg_scheme == "fixed_neg_scale":
        return get_weights_cluster_neg_none(
            correction=correction,
            neg_scale=0.88,
            cache_v=cache_v,
            n_posrated=n_posrated,
            n_negrated=n_negrated,
            n_cache=n_cache,
            pos_scheme=pos_scheme,
            n_pos_cluster_in=n_pos_cluster_in,
            cluster_alpha=cluster_alpha,
        )
    elif neg_scheme == "middle":
        return get_weights_cluster_neg_middle(
            correction=correction,
            neg_scale=neg_scale,
            cache_v=cache_v,
            n_posrated=n_posrated,
            n_negrated=n_negrated,
            n_cache=n_cache,
            pos_scheme=pos_scheme,
            n_pos_cluster_in=n_pos_cluster_in,
            cluster_alpha=cluster_alpha,
        )
    elif neg_scheme == "same_alpha":
        return get_weights_cluster_neg_same_alpha(
            correction=correction,
            neg_scale=neg_scale,
            cache_v=cache_v,
            n_posrated=n_posrated,
            n_negrated=n_negrated,
            n_cache=n_cache,
            pos_scheme=pos_scheme,
            n_pos_cluster_in=n_pos_cluster_in,
            n_neg_cluster_in=n_neg_cluster_in,
            cluster_alpha=cluster_alpha,
        )
    elif neg_scheme == "same_ratio":
        return get_weights_cluster_neg_same_ratio(
            correction=correction,
            neg_scale=neg_scale,
            cache_v=cache_v,
            n_posrated=n_posrated,
            n_negrated=n_negrated,
            n_cache=n_cache,
            pos_scheme=pos_scheme,
            n_pos_cluster_in=n_pos_cluster_in,
            n_neg_cluster_in=n_neg_cluster_in,
            cluster_alpha=cluster_alpha,
        )
    else:
        raise ValueError(f"Unknown neg_scheme: {neg_scheme}")


def get_weights(
    hyperparameters_combination: tuple,
    n_posrated: int,
    n_negrated: int,
    n_cache: int,
    is_cluster: bool = False,
    pos_scheme: str = None,
    neg_scheme: str = None,
    cluster_label: int = None,
    pos_clusters_idxs: dict = None,
    neg_clusters_idxs: dict = None,
    cluster_alpha: float = None,
    X_train: np.ndarray = None,
    y_train: np.ndarray = None,
) -> tuple:
    n_pos_cluster_in = len(pos_clusters_idxs[cluster_label]) if is_cluster else None
    n_neg_cluster_in = len(neg_clusters_idxs[cluster_label]) if is_cluster else None
    n_rated = n_posrated + n_negrated
    correction = n_rated + n_cache
    neg_scale = hyperparameters_combination[LOGREG_HYPERPARAMETERS["weights_neg_scale"]]
    cache_v = hyperparameters_combination[LOGREG_HYPERPARAMETERS["weights_cache_v"]]
    if is_cluster or pos_scheme == "single":
        if is_cluster:
            assert n_pos_cluster_in is not None and n_neg_cluster_in is not None
            n_pos_cluster_out = n_posrated - n_pos_cluster_in
            n_neg_cluster_out = n_negrated - n_neg_cluster_in
            assert n_pos_cluster_out >= 0 and n_neg_cluster_out >= 0
            assert cluster_alpha is not None
        return get_weights_cluster_scheme(
            correction=correction,
            neg_scale=neg_scale,
            cache_v=cache_v,
            n_posrated=n_posrated,
            n_negrated=n_negrated,
            n_cache=n_cache,
            pos_scheme=pos_scheme,
            neg_scheme=neg_scheme,
            n_pos_cluster_in=n_pos_cluster_in,
            n_neg_cluster_in=n_neg_cluster_in,
            cluster_alpha=cluster_alpha,
            cluster_label=cluster_label,
            pos_clusters_idxs=pos_clusters_idxs,
            neg_clusters_idxs=neg_clusters_idxs,
            X_train=X_train,
            y_train=y_train,
        )
    else:
        w_pos_in_cluster = correction * (1.0 - neg_scale) / n_posrated
        neg_denom = cache_v * n_negrated + (1.0 - cache_v) * n_cache
        assert neg_denom > 0
        w_neg_in_cluster = correction * neg_scale * cache_v / neg_denom
        w_cache = correction * neg_scale * (1.0 - cache_v) / neg_denom
        return w_pos_in_cluster, w_neg_in_cluster, w_cache, None, None


def get_sample_weights_temporal_decay_none(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_rated: int,
    hyperparameters_combination: tuple,
    is_cluster: bool = False,
    pos_scheme: str = None,
    neg_scheme: str = None,
    cluster_label: int = None,
    pos_clusters_idxs: dict = None,
    neg_clusters_idxs: dict = None,
    cluster_alpha: float = None,
) -> np.ndarray:
    n_total = y_train.shape[0]
    sample_weights = np.empty(n_total, dtype=np.float64)
    y_rated = y_train[:n_rated]
    weights = get_weights(
        hyperparameters_combination=hyperparameters_combination,
        n_posrated=np.sum(y_rated == 1),
        n_negrated=np.sum(y_rated == 0),
        n_cache=n_total - n_rated,
        is_cluster=is_cluster,
        pos_scheme=pos_scheme,
        neg_scheme=neg_scheme,
        cluster_label=cluster_label,
        pos_clusters_idxs=pos_clusters_idxs,
        neg_clusters_idxs=neg_clusters_idxs,
        cluster_alpha=cluster_alpha,
        X_train=X_train,
        y_train=y_train,
    )
    if not is_cluster and pos_scheme not in ["exponential", "softmax", "merge", "single"]:
        w_pos_in_cluster, w_neg_in_cluster, w_cache, w_pos_out_cluster, w_neg_out_cluster = weights
        sample_weights[y_train == 1] = w_pos_in_cluster
        sample_weights[y_train == 0] = w_neg_in_cluster
    else:
        if pos_scheme in ["exponential", "softmax", "merge", "single"]:
            w_pos, w_neg, w_cache = weights
            pos_rated_mask = y_train[:n_rated] == 1
            neg_rated_mask = y_train[:n_rated] == 0
            pos_original_idxs = np.where(pos_rated_mask)[0]
            neg_original_idxs = np.where(neg_rated_mask)[0]
            for k in pos_clusters_idxs.keys():
                original_pos_idxs = pos_original_idxs[pos_clusters_idxs[k]]
                original_neg_idxs = neg_original_idxs[neg_clusters_idxs[k]]
                sample_weights[original_pos_idxs] = w_pos[k]
                sample_weights[original_neg_idxs] = w_neg[k]

            n_posrated = np.sum(y_rated == 1)
            n_negrated = np.sum(y_rated == 0)
            n_cache = n_total - n_rated
            assert n_posrated + n_negrated == n_rated and n_rated + n_cache == n_total
            cache_v = hyperparameters_combination[LOGREG_HYPERPARAMETERS["weights_cache_v"]]
            neg_scale = hyperparameters_combination[LOGREG_HYPERPARAMETERS["weights_neg_scale"]]
            pos_goal = n_total * (1.0 - neg_scale)
            neg_denom = cache_v * n_negrated + (1.0 - cache_v) * (n_total - n_rated)
            neg_goal = n_negrated * n_total * neg_scale * cache_v / neg_denom
            cache_goal = (n_total - n_rated) * n_total * neg_scale * (1.0 - cache_v) / neg_denom
            assert np.isclose(pos_goal + neg_goal + cache_goal, n_total)

            actual_pos = np.sum(sample_weights[:n_rated][y_rated == 1])
            pos_sum = 0
            for k in pos_clusters_idxs.keys():
                pos_sum += np.sum(w_pos[k])
            actual_neg = np.sum(sample_weights[:n_rated][y_rated == 0])
            assert np.isclose(actual_pos, pos_goal), f"{actual_pos} vs {pos_goal}"
            assert np.isclose(actual_neg, neg_goal), f"{actual_neg} vs {neg_goal}"
        else:
            pos_rated_mask = y_train[:n_rated] == 1
            neg_rated_mask = y_train[:n_rated] == 0
            pos_original_idxs = np.where(pos_rated_mask)[0]
            neg_original_idxs = np.where(neg_rated_mask)[0]
            w_pos_in_cluster, w_neg_in_cluster, w_cache, w_pos_out_cluster, w_neg_out_cluster = (
                weights
            )
            pos_cluster_in_idxs = pos_clusters_idxs[cluster_label]
            neg_cluster_in_idxs = neg_clusters_idxs[cluster_label]
            pos_cluster_out_idxs = [
                idx
                for lbl, idxs in pos_clusters_idxs.items()
                if lbl != cluster_label
                for idx in idxs
            ]
            pos_cluster_out_idxs = np.array(pos_cluster_out_idxs, dtype=np.int64)
            neg_cluster_out_idxs = [
                idx
                for lbl, idxs in neg_clusters_idxs.items()
                if lbl != cluster_label
                for idx in idxs
            ]
            neg_cluster_out_idxs = np.array(neg_cluster_out_idxs, dtype=np.int64)
            sample_weights[pos_original_idxs[pos_cluster_in_idxs]] = w_pos_in_cluster
            sample_weights[neg_original_idxs[neg_cluster_in_idxs]] = w_neg_in_cluster
            sample_weights[pos_original_idxs[pos_cluster_out_idxs]] = w_pos_out_cluster
            sample_weights[neg_original_idxs[neg_cluster_out_idxs]] = w_neg_out_cluster
            """
            sample_weights[pos_cluster_in_idxs] = w_pos_in_cluster
            sample_weights[neg_cluster_in_idxs] = w_neg_in_cluster
            sample_weights[pos_cluster_out_idxs] = w_pos_out_cluster
            sample_weights[neg_cluster_out_idxs] = w_neg_out_cluster
            """

            n_posrated = np.sum(y_rated == 1)
            n_negrated = np.sum(y_rated == 0)
            n_cache = n_total - n_rated
            assert n_posrated + n_negrated == n_rated and n_rated + n_cache == n_total
            cache_v = hyperparameters_combination[LOGREG_HYPERPARAMETERS["weights_cache_v"]]
            neg_scale = hyperparameters_combination[LOGREG_HYPERPARAMETERS["weights_neg_scale"]]
            pos_goal = n_total * (1.0 - neg_scale)
            neg_denom = cache_v * n_negrated + (1.0 - cache_v) * (n_total - n_rated)
            neg_goal = n_negrated * n_total * neg_scale * cache_v / neg_denom
            cache_goal = (n_total - n_rated) * n_total * neg_scale * (1.0 - cache_v) / neg_denom
            assert np.isclose(pos_goal + neg_goal + cache_goal, n_total)
            assert np.isclose(pos_goal, w_pos_in_cluster * len(pos_cluster_in_idxs) + w_pos_out_cluster * len(pos_cluster_out_idxs)), f"pos_goal: {pos_goal}, computed: {w_pos_in_cluster * len(pos_cluster_in_idxs) + w_pos_out_cluster * len(pos_cluster_out_idxs)}"

            actual_pos = np.sum(sample_weights[:n_rated][y_rated == 1])
            actual_neg = np.sum(sample_weights[:n_rated][y_rated == 0])
            assert np.isclose(actual_pos, pos_goal), f"{actual_pos} vs {pos_goal}"
            assert np.isclose(actual_neg, neg_goal), f"{actual_neg} vs {neg_goal}"

    sample_weights[n_rated:] = w_cache
    assert np.isclose(np.sum(sample_weights), n_total), f"{np.sum(sample_weights)} vs {n_total}"
    return sample_weights


def get_sample_weights_temporal_decay_cluster(
    rated_time_diffs: np.ndarray,
    temporal_decay_param: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_rated: int,
    hyperparameters_combination: tuple,
    cluster_label: int,
    pos_clusters_idxs: dict,
    neg_clusters_idxs: dict,
    cluster_alpha: float,
) -> np.ndarray:
    n_total = y_train.shape[0]
    sample_weights = np.empty(n_total, dtype=np.float64)
    y_rated = y_train[:n_rated]
    weights = get_weights(
        hyperparameters_combination=hyperparameters_combination,
        n_posrated=np.sum(y_rated == 1),
        n_negrated=np.sum(y_rated == 0),
        n_cache=n_total - n_rated,
        is_cluster=True,
        pos_scheme="relative",
        neg_scheme="same_ratio",
        cluster_label=cluster_label,
        pos_clusters_idxs=pos_clusters_idxs,
        neg_clusters_idxs=neg_clusters_idxs,
        cluster_alpha=cluster_alpha,
        X_train=X_train,
        y_train=y_train,
    )
    w_pos_in_cluster, w_neg_in_cluster, w_cache, w_pos_out_cluster, w_neg_out_cluster = weights
    pos_original_idxs = np.where(y_rated == 1)[0]
    neg_original_idxs = np.where(y_rated == 0)[0]
    pos_cluster_in_idxs = pos_clusters_idxs[cluster_label]
    neg_cluster_in_idxs = neg_clusters_idxs[cluster_label]


    pos_cluster_out_idxs = [
        idx
        for lbl, idxs in pos_clusters_idxs.items()
        if lbl != cluster_label
        for idx in idxs
    ]
    pos_cluster_out_idxs = np.array(pos_cluster_out_idxs, dtype=np.int64)
    neg_cluster_out_idxs = [
        idx
        for lbl, idxs in neg_clusters_idxs.items()
        if lbl != cluster_label
        for idx in idxs
    ]
    neg_cluster_out_idxs = np.array(neg_cluster_out_idxs, dtype=np.int64)
    sample_weights[neg_original_idxs[neg_cluster_in_idxs]] = w_neg_in_cluster
    sample_weights[neg_original_idxs[neg_cluster_out_idxs]] = w_neg_out_cluster
    n_pos_in_cluster, n_pos_out_cluster = len(pos_cluster_in_idxs), len(pos_cluster_out_idxs)
    pos_sum_in_cluster = w_pos_in_cluster * n_pos_in_cluster
    pos_sum_out_cluster = w_pos_out_cluster * n_pos_out_cluster
    weights_neg_scale = hyperparameters_combination[LOGREG_HYPERPARAMETERS["weights_neg_scale"]]
    assert np.isclose(pos_sum_in_cluster + pos_sum_out_cluster, n_total * (1.0 - weights_neg_scale))
    pos_decays = get_sample_weights_temporal_decay(
        user_train_set_ratings=y_rated[:n_rated],
        user_train_set_time_diffs=rated_time_diffs,
        n_cache=n_total - n_rated,
        weights_neg_scale=weights_neg_scale,
        weights_cache_v=hyperparameters_combination[LOGREG_HYPERPARAMETERS["weights_cache_v"]],
        temporal_decay=TemporalDecay.EXPONENTIAL,
        temporal_decay_normalization=TemporalDecayNormalization.POSITIVES,
        temporal_decay_param=temporal_decay_param,
        pos_decays_only=True,
    )
    pos_decays *= n_total
    assert np.isclose(np.sum(pos_decays), n_total * (1.0 - weights_neg_scale))
    pos_decays_in_cluster = pos_decays[pos_cluster_in_idxs]
    pos_decays_out_cluster = pos_decays[pos_cluster_out_idxs]
    pos_decays_in_cluster = pos_decays_in_cluster / np.sum(pos_decays_in_cluster) * pos_sum_in_cluster
    pos_decays_out_cluster = pos_decays_out_cluster / np.sum(pos_decays_out_cluster) * pos_sum_out_cluster
    assert np.isclose(np.sum(pos_decays_in_cluster) + np.sum(pos_decays_out_cluster), n_total * (1.0 - weights_neg_scale))
    sample_weights[pos_original_idxs[pos_cluster_in_idxs]] = pos_decays_in_cluster
    sample_weights[pos_original_idxs[pos_cluster_out_idxs]] = pos_decays_out_cluster
    sample_weights[n_rated:] = w_cache
    assert np.isclose(np.sum(sample_weights), n_total), f"{np.sum(sample_weights)} vs {n_total}"
    return sample_weights


def get_sample_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_rated: int,
    rated_time_diffs: np.ndarray,
    eval_settings: dict,
    is_cluster: bool = False,
    cluster_label: int = None,
    pos_clusters_idxs: dict = None,
    neg_clusters_idxs: dict = None,
) -> np.ndarray:
    hyperparameters_combination = get_hyperparameters_combination(eval_settings)
    temporal_decay = get_temporal_decay_from_arg(eval_settings["logreg_temporal_decay"])
    temporal_decay_normalization = get_temporal_decay_normalization_from_arg(
        eval_settings["logreg_temporal_decay_normalization"]
    )
    if temporal_decay == TemporalDecay.NONE:
        return get_sample_weights_temporal_decay_none(
            X_train=X_train,
            y_train=y_train,
            n_rated=n_rated,
            hyperparameters_combination=hyperparameters_combination,
            is_cluster=is_cluster,
            pos_scheme=eval_settings.get("clustering_pos_weighting_scheme", None),
            neg_scheme=eval_settings.get("clustering_neg_weighting_scheme", None),
            cluster_label=cluster_label,
            pos_clusters_idxs=pos_clusters_idxs,
            neg_clusters_idxs=neg_clusters_idxs,
            cluster_alpha=eval_settings.get("clustering_cluster_alpha", None),
        )
    else:
        return get_sample_weights_temporal_decay(
            user_train_set_ratings=y_train[:n_rated],
            user_train_set_time_diffs=rated_time_diffs,
            n_cache=y_train.shape[0] - n_rated,
            weights_neg_scale=hyperparameters_combination[
                LOGREG_HYPERPARAMETERS["weights_neg_scale"]
            ],
            weights_cache_v=hyperparameters_combination[LOGREG_HYPERPARAMETERS["weights_cache_v"]],
            temporal_decay=temporal_decay,
            temporal_decay_normalization=temporal_decay_normalization,
            temporal_decay_param=eval_settings["logreg_temporal_decay_param"],
        )


def compute_logreg_user_embedding(
    user_train_set_embeddings: np.ndarray,
    user_train_set_ratings: np.ndarray,
    user_train_set_time_diffs: np.ndarray,
    user_train_set_sessions_ids: np.ndarray,
    X_cache: np.ndarray,
    random_state: int,
    eval_settings: dict,
) -> np.ndarray:
    is_sparse = sparse.isspmatrix(user_train_set_embeddings) or sparse.isspmatrix(X_cache)
    if is_sparse:
        X_train = sparse.vstack([user_train_set_embeddings, X_cache])
    else:
        X_train = np.vstack([user_train_set_embeddings, X_cache])
    y_cache = np.zeros(X_cache.shape[0], dtype=np.int64)
    y_train = np.hstack([user_train_set_ratings, y_cache])
    sample_weights = get_sample_weights(
        user_train_set_ratings=user_train_set_ratings,
        user_train_set_time_diffs=user_train_set_time_diffs,
        n_cache=X_cache.shape[0],
        eval_settings=eval_settings,
        user_train_set_embeddings=user_train_set_embeddings,
        user_train_set_sessions_ids=user_train_set_sessions_ids,
    )
    model = get_model(
        algorithm=Algorithm.LOGREG,
        max_iter=eval_settings["logreg_max_iter"],
        clf_C=eval_settings["logreg_clf_C"],
        random_state=random_state,
        logreg_solver=eval_settings["logreg_solver"],
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    embedding = np.hstack([model.coef_[0], model.intercept_[0]])
    return embedding
