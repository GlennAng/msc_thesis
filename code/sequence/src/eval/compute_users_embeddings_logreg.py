from enum import Enum, auto

import numpy as np
import pandas as pd

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

LOGREG_CACHE_TYPE = "categories_cache"
LOGREG_N_CACHE = 5000
LOGREG_N_CATEGORIES_CACHE = 0
LOGREG_N_VAL_NEGATIVE_SAMPLES = 100
LOGREG_HYPERPARAMETERS = {"weights_neg_scale": 0, "weights_cache_v": 1, "clf_C": 2}
LOGREG_HYPERPARAMETERS_COMBINATION = (5.0, 0.9, 0.1)
LOGREG_MAX_ITER = 10000
LOGREG_SOLVER = "lbfgs"


class TemporalDecay(Enum):
    NONE = auto()
    LINEAR = auto()
    EXPONENTIAL = auto()
    CUT_OFF = auto()


def get_temporal_decay_from_arg(temporal_decay_arg: str) -> TemporalDecay:
    valid_temporal_decay_args = [decay.name.lower() for decay in TemporalDecay]
    if temporal_decay_arg.lower() not in valid_temporal_decay_args:
        raise ValueError(
            f"Invalid argument {temporal_decay_arg} 'temporal_decay'. Possible values: {valid_temporal_decay_args}."
        )
    return TemporalDecay[temporal_decay_arg.upper()]


def logreg_get_embed_function_params(users_ids: list, random_state: int) -> dict:
    users_significant_categories = load_users_significant_categories(relevant_users_ids=users_ids)
    papers = load_papers(relevant_columns=["paper_id", "in_ratings", "l1"])
    val_negative_samples_ids = get_val_negative_samples_ids(
        papers=papers,
        n_categories_samples=LOGREG_N_VAL_NEGATIVE_SAMPLES,
        random_state=random_state,
        papers_ids_to_exclude=papers[papers["in_ratings"]]["paper_id"].tolist(),
    )
    cache_papers_categories_ids, cache_papers_ids = get_cache_papers_ids_full(
        papers=papers,
        cache_type=LOGREG_CACHE_TYPE,
        n_cache=LOGREG_N_CACHE,
        random_state=random_state,
        n_categories_cache=LOGREG_N_CATEGORIES_CACHE,
    )
    return {
        "random_state": random_state,
        "users_significant_categories": users_significant_categories,
        "val_negative_samples_ids": val_negative_samples_ids,
        "cache_papers_categories_ids": cache_papers_categories_ids,
        "cache_papers_ids": cache_papers_ids,
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
        cache_type=LOGREG_CACHE_TYPE,
        cache_papers_ids=cache_papers_ids,
        cache_papers_categories_ids=cache_papers_categories_ids,
        n_categories_cache=LOGREG_N_CATEGORIES_CACHE,
        random_state=random_state,
        papers_ids_to_exclude_from_cache=papers_ids_to_exclude_from_cache,
        user_categories_ratios=user_categories_ratios,
        embedding=embedding,
    )
    X_cache = embedding.matrix[user_cache_papers["cache_embedding_idxs"]]
    return {"X_cache": X_cache, "random_state": random_state}


def get_sample_weights(user_train_set_ratings: np.ndarray, n_cache: int) -> np.ndarray:
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
        hyperparameters_combination=LOGREG_HYPERPARAMETERS_COMBINATION,
        train_posrated_n=train_posrated_n,
        train_negrated_n=train_negrated_n,
        cache_n=n_cache,
    )
    sample_weights[pos_idxs] = w_p
    sample_weights[neg_idxs] = w_n
    sample_weights[cache_idxs] = w_c
    return sample_weights


def compute_logreg_user_embedding(
    user_train_set_embeddings: np.ndarray,
    user_train_set_ratings: np.ndarray,
    user_train_set_time_diffs: np.ndarray,
    X_cache: np.ndarray,
    random_state: int,
) -> np.ndarray:
    X_train = np.vstack([user_train_set_embeddings, X_cache])
    y_cache = np.zeros(X_cache.shape[0], dtype=np.int64)
    y_train = np.hstack([user_train_set_ratings, y_cache])
    sample_weights = get_sample_weights(user_train_set_ratings, X_cache.shape[0])
    model = get_model(
        algorithm=Algorithm.LOGREG,
        max_iter=LOGREG_MAX_ITER,
        clf_C=LOGREG_HYPERPARAMETERS_COMBINATION[LOGREG_HYPERPARAMETERS["clf_C"]],
        random_state=random_state,
        logreg_solver=LOGREG_SOLVER,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return np.hstack([model.coef_[0], model.intercept_[0]])
