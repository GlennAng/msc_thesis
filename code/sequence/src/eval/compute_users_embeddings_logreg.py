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
    papers = load_papers(relevant_columns=["paper_id", "in_ratings", "l1"])
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
        w_neg_out_cluster = correction * neg_scale * (1.0 - desired_ratio) * cache_term / n_neg_cluster_out
    return w_pos_in_cluster, w_neg_in_cluster, w_cache, w_pos_out_cluster, w_neg_out_cluster


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
) -> tuple:
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
    n_pos_cluster_in: int = None,
    n_neg_cluster_in: int = None,
    cluster_alpha: float = None,
) -> tuple:
    correction = n_posrated + n_negrated + n_cache
    neg_scale = hyperparameters_combination[LOGREG_HYPERPARAMETERS["weights_neg_scale"]]
    cache_v = hyperparameters_combination[LOGREG_HYPERPARAMETERS["weights_cache_v"]]
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
        )
    else:
        w_pos_in_cluster = correction * (1.0 - neg_scale) / n_posrated
        neg_denom = cache_v * n_negrated + (1.0 - cache_v) * n_cache
        assert neg_denom > 0
        w_neg_in_cluster = correction * neg_scale * cache_v / neg_denom
        w_cache = correction * neg_scale * (1.0 - cache_v) / neg_denom
        return w_pos_in_cluster, w_neg_in_cluster, w_cache, None, None


def get_sample_weights_temporal_decay_none(
    y_train: np.ndarray,
    n_rated: int,
    hyperparameters_combination: tuple,
    is_cluster: bool = False,
    pos_scheme: str = None,
    neg_scheme: str = None,
    pos_cluster_in_idxs: np.ndarray = None,
    pos_cluster_out_idxs: np.ndarray = None,
    neg_cluster_in_idxs: np.ndarray = None,
    neg_cluster_out_idxs: np.ndarray = None,
    cluster_alpha: float = None,
) -> np.ndarray:
    n_total = y_train.shape[0]
    sample_weights = np.empty(n_total, dtype=np.float64)
    y_rated = y_train[:n_rated]
    w_pos_in_cluster, w_neg_in_cluster, w_cache, w_pos_out_cluster, w_neg_out_cluster = get_weights(
        hyperparameters_combination=hyperparameters_combination,
        n_posrated=np.sum(y_rated == 1),
        n_negrated=np.sum(y_rated == 0),
        n_cache=n_total - n_rated,
        is_cluster=is_cluster,
        pos_scheme=pos_scheme,
        neg_scheme=neg_scheme,
        n_pos_cluster_in=pos_cluster_in_idxs.shape[0] if is_cluster else None,
        n_neg_cluster_in=neg_cluster_in_idxs.shape[0] if is_cluster else None,
        cluster_alpha=cluster_alpha,
    )
    if not is_cluster:
        sample_weights[y_train == 1] = w_pos_in_cluster
        sample_weights[y_train == 0] = w_neg_in_cluster
    else:
        sample_weights[pos_cluster_in_idxs] = w_pos_in_cluster
        sample_weights[neg_cluster_in_idxs] = w_neg_in_cluster
        sample_weights[pos_cluster_out_idxs] = w_pos_out_cluster
        sample_weights[neg_cluster_out_idxs] = w_neg_out_cluster
    sample_weights[n_rated:] = w_cache
    assert np.isclose(np.sum(sample_weights), n_total)
    return sample_weights


def get_sample_weights(
    y_train: np.ndarray,
    n_rated: int,
    rated_time_diffs: np.ndarray,
    eval_settings: dict,
    is_cluster: bool = False,
    pos_cluster_in_idxs: np.ndarray = None,
    pos_cluster_out_idxs: np.ndarray = None,
    neg_cluster_in_idxs: np.ndarray = None,
    neg_cluster_out_idxs: np.ndarray = None,
) -> np.ndarray:
    hyperparameters_combination = get_hyperparameters_combination(eval_settings)
    temporal_decay = get_temporal_decay_from_arg(eval_settings["logreg_temporal_decay"])
    temporal_decay_normalization = get_temporal_decay_normalization_from_arg(
        eval_settings["logreg_temporal_decay_normalization"]
    )
    if temporal_decay == TemporalDecay.NONE:
        return get_sample_weights_temporal_decay_none(
            y_train=y_train,
            n_rated=n_rated,
            hyperparameters_combination=hyperparameters_combination,
            is_cluster=is_cluster,
            pos_scheme=eval_settings.get("clustering_pos_weighting_scheme", None),
            neg_scheme=eval_settings.get("clustering_neg_weighting_scheme", None),
            pos_cluster_in_idxs=pos_cluster_in_idxs,
            pos_cluster_out_idxs=pos_cluster_out_idxs,
            neg_cluster_in_idxs=neg_cluster_in_idxs,
            neg_cluster_out_idxs=neg_cluster_out_idxs,
            cluster_alpha=eval_settings.get("clustering_cluster_alpha", None),
        )
    else:
        return get_sample_weights_temporal_decay(
            user_train_set_ratings=y_train[:n_rated],
            user_train_set_time_diffs=rated_time_diffs,
            n_cache=y_train.shape[0] - n_rated,
            weights_neg_scale=hyperparameters_combination[LOGREG_HYPERPARAMETERS["weights_neg_scale"]],
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
