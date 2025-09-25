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
from ....logreg.src.training.weights_handler import Weights_Handler
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
) -> dict:
    user_significant_categories = users_significant_categories[
        users_significant_categories["user_id"] == user_id
    ]
    user_categories_ratios = get_user_categories_ratios(
        categories_to_exclude=user_significant_categories
    )
    user_val_negative_samples_ids = get_user_val_negative_samples(
        val_negative_samples_ids=val_negative_samples_ids,
        n_negative_samples=100,
        random_state=random_state,
        user_categories_ratios=user_categories_ratios,
        embedding=None,
    )["val_negative_samples_ids"]
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
    return {"X_cache": X_cache, "random_state": random_state, "eval_settings": eval_settings}


def get_hyperparameters_combination(eval_settings: dict) -> tuple:
    combi = [0] * len(LOGREG_HYPERPARAMETERS)
    for param, idx in LOGREG_HYPERPARAMETERS.items():
        hyperparameter_string = f"logreg_{param}"
        assert hyperparameter_string in eval_settings
        combi[idx] = eval_settings[hyperparameter_string]
    return tuple(combi)


def get_sample_weights_temporal_decay_none(
    user_train_set_ratings: np.ndarray,
    n_cache: int,
    hyperparameters_combination: tuple,
) -> np.ndarray:
    n_total = user_train_set_ratings.shape[0] + n_cache
    sample_weights = np.empty(n_total, dtype=np.float64)
    pos_idxs = np.where(user_train_set_ratings > 0)[0]
    neg_idxs = np.where(user_train_set_ratings == 0)[0]
    assert pos_idxs.shape[0] + neg_idxs.shape[0] == user_train_set_ratings.shape[0]
    cache_idxs = np.arange(user_train_set_ratings.shape[0], n_total)
    train_posrated_n = pos_idxs.shape[0]
    train_negrated_n = neg_idxs.shape[0]
    wh = Weights_Handler(config={"weights": "global:cache_v"})
    w_p, w_n, _, _, w_c = wh.load_weights_for_user(
        hyperparameters=LOGREG_HYPERPARAMETERS,
        hyperparameters_combination=hyperparameters_combination,
        train_posrated_n=train_posrated_n,
        train_negrated_n=train_negrated_n,
        cache_n=n_cache,
    )
    sample_weights[pos_idxs] = w_p
    sample_weights[neg_idxs] = w_n
    sample_weights[cache_idxs] = w_c
    return sample_weights


def get_sample_weights(
    user_train_set_ratings: np.ndarray,
    user_train_set_time_diffs: np.ndarray,
    n_cache: int,
    eval_settings: dict,
) -> np.ndarray:
    hyperparameters_combination = get_hyperparameters_combination(eval_settings)
    temporal_decay = get_temporal_decay_from_arg(eval_settings["logreg_temporal_decay"])
    temporal_decay_normalization = get_temporal_decay_normalization_from_arg(
        eval_settings["logreg_temporal_decay_normalization"]
    )
    if temporal_decay == TemporalDecay.NONE:
        sample_weights = get_sample_weights_temporal_decay_none(
            user_train_set_ratings, n_cache, hyperparameters_combination
        )
    else:
        sample_weights = get_sample_weights_temporal_decay(
            user_train_set_ratings=user_train_set_ratings,
            user_train_set_time_diffs=user_train_set_time_diffs,
            n_cache=n_cache,
            weights_neg_scale=eval_settings["logreg_weights_neg_scale"],
            weights_cache_v=eval_settings["logreg_weights_cache_v"],
            temporal_decay=temporal_decay,
            temporal_decay_normalization=temporal_decay_normalization,
            temporal_decay_param=eval_settings["logreg_temporal_decay_param"],
        )
    return sample_weights


def compute_logreg_user_embedding(
    user_train_set_embeddings: np.ndarray,
    user_train_set_ratings: np.ndarray,
    user_train_set_time_diffs: np.ndarray,
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
        user_train_set_ratings, user_train_set_time_diffs, X_cache.shape[0], eval_settings
    )
    model = get_model(
        algorithm=Algorithm.LOGREG,
        max_iter=eval_settings["logreg_max_iter"],
        clf_C=eval_settings["logreg_clf_C"],
        random_state=random_state,
        logreg_solver=eval_settings["logreg_solver"],
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return np.hstack([model.coef_[0], model.intercept_[0]])
