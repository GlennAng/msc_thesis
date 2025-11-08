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


def get_weights(
    hyperparameters_combination: tuple,
    n_posrated: int,
    n_negrated: int,
    n_cache: int,
    is_cluster: bool = False,
    n_cluster_in: int = None,
    cluster_alpha: float = None,
    cluster_correction: bool = False,
) -> tuple:
    correction = n_posrated + n_negrated + n_cache
    neg_scale = hyperparameters_combination[LOGREG_HYPERPARAMETERS["weights_neg_scale"]]
    if is_cluster:
        assert n_cluster_in is not None and cluster_alpha is not None
        n_cluster_out = n_posrated - n_cluster_in
        assert n_cluster_out >= 0
        alpha_denom = cluster_alpha * n_cluster_in + (1.0 - cluster_alpha) * n_cluster_out
        assert alpha_denom > 0
        if cluster_correction:
            capital_alpha = cluster_alpha / alpha_denom
            ratio_before = neg_scale / ((1.0 - neg_scale) * n_cluster_in / n_posrated)
            ratio_after = neg_scale / ((1.0 - neg_scale) * n_cluster_in * capital_alpha)
            capital_beta = 0.6 * ratio_after + 0.4 * ratio_before
            numer = capital_beta * n_cluster_in * capital_alpha
            neg_scale_prime = numer / (1.0 + numer)
        else:
            neg_scale_prime = neg_scale
        w_p = correction * (1.0 - neg_scale_prime) * cluster_alpha / alpha_denom
        w_a = correction * (1.0 - neg_scale_prime) * (1.0 - cluster_alpha) / alpha_denom
    else:
        neg_scale_prime = neg_scale
        w_p = correction * (1.0 - neg_scale_prime) / n_posrated
        w_a = None
    cache_v = hyperparameters_combination[LOGREG_HYPERPARAMETERS["weights_cache_v"]]
    neg_denom = cache_v * n_negrated + (1.0 - cache_v) * n_cache
    assert neg_denom > 0
    w_n = correction * neg_scale_prime * cache_v / neg_denom
    w_c = correction * neg_scale_prime * (1.0 - cache_v) / neg_denom
    return w_p, w_a, w_n, w_c


def get_sample_weights_temporal_decay_none(
    y_train: np.ndarray,
    n_rated: int,
    hyperparameters_combination: tuple,
    is_cluster: bool = False,
    cluster_in_idxs: np.ndarray = None,
    cluster_out_idxs: np.ndarray = None,
    cluster_alpha: float = None,
) -> np.ndarray:
    n_total = y_train.shape[0]
    sample_weights = np.empty(n_total, dtype=np.float64)
    y_rated = y_train[:n_rated]
    w_p, w_a, w_n, w_c = get_weights(
        hyperparameters_combination=hyperparameters_combination,
        n_posrated=np.sum(y_rated == 1),
        n_negrated=np.sum(y_rated == 0),
        n_cache=n_total - n_rated,
        is_cluster=is_cluster,
        n_cluster_in=cluster_in_idxs.shape[0] if is_cluster else None,
        cluster_alpha=cluster_alpha,
    )
    sample_weights[y_train == 0] = w_n
    sample_weights[n_rated:] = w_c
    if not is_cluster:
        sample_weights[y_train == 1] = w_p
    else:
        sample_weights[cluster_in_idxs] = w_p
        sample_weights[cluster_out_idxs] = w_a
    return sample_weights


def get_sample_weights_temporal_decay(
    y_train: np.ndarray,
    n_rated: int,
    rated_time_diffs: np.ndarray,
    hyperparameters_combination: tuple,
    temporal_decay: TemporalDecay,
    temporal_decay_normalization: TemporalDecayNormalization,
    temporal_decay_param: float,
    is_cluster: bool = False,
    cluster_in_idxs: np.ndarray = None,
    cluster_out_idxs: np.ndarray = None,
) -> np.ndarray:
    assert temporal_decay_normalization == TemporalDecayNormalization.POSITIVES
    n_total = y_train.shape[0]
    sample_weights = np.empty(n_total, dtype=np.float64)
    y_rated = y_train[:n_rated]
    w_p, w_a, w_n, w_c = get_weights(
        hyperparameters_combination=hyperparameters_combination,
        n_posrated=np.sum(y_rated == 1),
        n_negrated=np.sum(y_rated == 0),
        n_cache=n_total - n_rated,
        is_cluster=is_cluster,
        n_cluster_in=cluster_in_idxs.shape[0] if is_cluster else None,
        cluster_alpha=None,
    )
    sample_weights[y_train == 0] = w_n
    sample_weights[n_rated:] = w_c

    pos_time_diffs = rated_time_diffs[y_rated == 1]
    pos_decays = np.exp(-temporal_decay_param * pos_time_diffs)
    pos_decays /= np.sum(pos_decays) if pos_decays.shape[0] > 0 else pos_decays
    


def get_sample_weights(
    y_train: np.ndarray,
    n_rated: int,
    rated_time_diffs: np.ndarray,
    eval_settings: dict,
    is_cluster: bool = False,
    cluster_in_idxs: np.ndarray = None,
    cluster_out_idxs: np.ndarray = None,
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
            cluster_in_idxs=cluster_in_idxs,
            cluster_out_idxs=cluster_out_idxs,
            cluster_alpha=eval_settings.get("clustering_cluster_alpha", None),
        )
    else:
        assert temporal_decay_normalization == TemporalDecayNormalization.POSITIVES


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
