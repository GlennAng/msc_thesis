from turtle import shape

import numpy as np
import pandas as pd

from ..embeddings.embedding import Embedding
from .algorithm import (
    DERIVABLE_SCORES,
    NON_DERIVABLE_SCORES,
    SCORES_DICT,
    derive_score,
    get_category_scores,
    get_ranking_scores,
    get_score,
)


def fill_user_predictions_dict(val_data_dict: dict) -> dict:
    return {
        "train_ids": val_data_dict["X_train_rated_papers_ids"],
        "train_labels": val_data_dict["y_train_rated"].tolist(),
        "val_ids": val_data_dict["X_val_papers_ids"],
        "val_labels": val_data_dict["y_val"].tolist(),
    }


def update_user_predictions_dict(user_predictions: dict) -> dict:
    return {
        "train_predictions": user_predictions["train_predictions"],
        "val_predictions": user_predictions["val_predictions"],
        "negative_samples_predictions": user_predictions["negative_samples_predictions"],
        "tfidf_coefs": user_predictions["tfidf_coefs"],
    }


def load_user_data_dicts(
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    train_negrated_ranking: pd.DataFrame,
    val_negrated_ranking: pd.DataFrame,
    embedding: Embedding,
    load_user_train_data_dict_bool=True,
    cache_embedding_idxs: np.ndarray = None,
    y_cache: np.ndarray = None,
    random_state: int = None,
) -> tuple:
    train_data_dict = None
    if load_user_train_data_dict_bool:
        train_data_dict = load_user_train_data_dict(
            train_ratings=train_ratings,
            cache_embedding_idxs=cache_embedding_idxs,
            y_cache=y_cache,
            random_state=random_state,
            embedding=embedding,
        )
    val_data_dict = load_user_val_data_dict(
        train_ratings=train_ratings,
        val_ratings=val_ratings,
        embedding=embedding,
        train_negrated_ranking=train_negrated_ranking,
        val_negrated_ranking=val_negrated_ranking,
    )
    return train_data_dict, val_data_dict


def load_user_train_data_dict(
    train_ratings: pd.DataFrame,
    cache_embedding_idxs: np.ndarray,
    y_cache: np.ndarray,
    random_state: int,
    embedding: Embedding,
) -> dict:
    LABEL_NEGRATED, LABEL_POSRATED, LABEL_CACHE = 0, 1, 2
    rng = np.random.default_rng(random_state)
    train_ratings_ids, y_train_ratings = (
        train_ratings["paper_id"].tolist(),
        train_ratings["rating"].values,
    )
    train_rated_embedding_idxs = embedding.get_idxs(train_ratings_ids)
    train_source_labels = np.where(y_train_ratings > 0, LABEL_POSRATED, LABEL_NEGRATED)
    cache_source_labels = np.full(len(cache_embedding_idxs), LABEL_CACHE)

    embedding_idxs_full = np.concatenate((train_rated_embedding_idxs, cache_embedding_idxs))
    y_full = np.concatenate((y_train_ratings, y_cache))
    source_full = np.concatenate((train_source_labels, cache_source_labels))
    permuted_idxs = rng.permutation(len(embedding_idxs_full))
    embedding_idxs_full = embedding_idxs_full[permuted_idxs]
    y_full = y_full[permuted_idxs]
    source_full = source_full[permuted_idxs]

    pos_idxs = np.where(source_full == LABEL_POSRATED)[0]
    neg_idxs = np.where(source_full == LABEL_NEGRATED)[0]
    cache_idxs = np.where(source_full == LABEL_CACHE)[0]
    X_train = embedding.matrix[embedding_idxs_full]
    train_data_dict = {
        "X_train": X_train,
        "y_train": y_full,
        "pos_idxs": pos_idxs,
        "neg_idxs": neg_idxs,
        "cache_idxs": cache_idxs,
    }
    return train_data_dict


def load_user_val_data_dict(
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    embedding: Embedding,
    train_negrated_ranking: pd.DataFrame = None,
    val_negrated_ranking: pd.DataFrame = None,
) -> dict:
    X_train_rated_papers_ids, X_val_papers_ids = (
        train_ratings["paper_id"].tolist(),
        val_ratings["paper_id"].tolist(),
    )
    X_train_rated = embedding.matrix[embedding.get_idxs(X_train_rated_papers_ids)]
    y_train_rated = train_ratings["rating"].values
    X_val = embedding.matrix[embedding.get_idxs(X_val_papers_ids)]
    y_val = val_ratings["rating"].values
    categories_dict = {
        "l1_train_rated": train_ratings["l1"].values,
        "l1_val": val_ratings["l1"].values,
        "l2_train_rated": train_ratings["l2"].values,
        "l2_val": val_ratings["l2"].values,
    }
    val_data_dict = {
        "X_train_rated": X_train_rated,
        "y_train_rated": y_train_rated,
        "X_train_rated_papers_ids": X_train_rated_papers_ids,
        "X_val": X_val,
        "y_val": y_val,
        "X_val_papers_ids": X_val_papers_ids,
        "categories_dict": categories_dict,
    }
    if train_negrated_ranking is not None:
        X_train_negrated_ranking = embedding.matrix[
            embedding.get_idxs(train_negrated_ranking["paper_id"].tolist())
        ]
        val_data_dict["X_train_negrated_ranking"] = X_train_negrated_ranking
    if val_negrated_ranking is not None:
        X_val_negrated_ranking = embedding.matrix[
            embedding.get_idxs(val_negrated_ranking["paper_id"].tolist())
        ]
        val_data_dict["X_val_negrated_ranking"] = X_val_negrated_ranking
    return val_data_dict


def get_user_outputs_dict(
    model: object,
    val_data_dict: dict,
    train_negrated_ranking_idxs: np.ndarray = None,
    val_negrated_ranking_idxs: np.ndarray = None,
    negative_samples_embeddings: np.ndarray = None,
) -> dict:
    user_outputs_dict = {}
    if "X_train_rated" in val_data_dict:
        X_train_rated = val_data_dict["X_train_rated"]
        user_outputs_dict["y_train_rated_pred"] = model.predict(X_train_rated)
        user_outputs_dict["y_train_rated_proba"] = model.predict_proba(X_train_rated)[:, 1]
        user_outputs_dict["y_train_rated_logits"] = model.decision_function(X_train_rated)
    if "X_val" in val_data_dict:
        X_val = val_data_dict["X_val"]
        user_outputs_dict["y_val_pred"] = model.predict(X_val)
        user_outputs_dict["y_val_proba"] = model.predict_proba(X_val)[:, 1]
        user_outputs_dict["y_val_logits"] = model.decision_function(X_val)
    if "X_train_negrated_ranking" in val_data_dict:
        idxs = train_negrated_ranking_idxs
        train_neg_rank = val_data_dict["X_train_negrated_ranking"]
        y_train_neg_rank_logits = model.decision_function(train_neg_rank)
        user_outputs_dict["y_train_negrated_ranking_logits"] = y_train_neg_rank_logits[idxs]
    if "X_val_negrated_ranking" in val_data_dict:
        idxs = val_negrated_ranking_idxs
        val_neg_rank = val_data_dict["X_val_negrated_ranking"]
        y_val_neg_rank_logits = model.decision_function(val_neg_rank)
        user_outputs_dict["y_val_negrated_ranking_logits"] = y_val_neg_rank_logits[idxs]
    if negative_samples_embeddings is not None:
        embed = negative_samples_embeddings
        user_outputs_dict["y_negative_samples_pred"] = model.predict(embed)
        user_outputs_dict["y_negative_samples_proba"] = model.predict_proba(embed)[:, 1]
        user_outputs_dict["y_negative_samples_logits"] = model.decision_function(embed)
    return user_outputs_dict


def get_user_outputs_dict_sliding_window_multiple_models(
    val_data_dict: dict,
    user_models: list,
    train_negrated_ranking_idxs: np.ndarray,
    val_negrated_ranking_idxs: np.ndarray,
    negative_samples_embeddings: np.ndarray,
) -> dict:
    user_outputs_dict = {}
    train_model = user_models[0]
    train_model_val_data_dict = {
        "X_train_rated": val_data_dict["X_train_rated"],
        "X_train_negrated_ranking": val_data_dict["X_train_negrated_ranking"],
    }
    user_outputs_dict = get_user_outputs_dict(
        model=train_model,
        val_data_dict=train_model_val_data_dict,
        train_negrated_ranking_idxs=train_negrated_ranking_idxs,
    )
    for model in user_models:
        model_data_dict = {
            "X_val": val_data_dict["X_val"],
            "X_val_negrated_ranking": val_data_dict["X_val_negrated_ranking"],
        }
        model_outputs_dict = get_user_outputs_dict(
            model=model,
            val_data_dict=model_data_dict,
            val_negrated_ranking_idxs=val_negrated_ranking_idxs,
            negative_samples_embeddings=negative_samples_embeddings,
        )
        for key, value in model_outputs_dict.items():
            if key not in user_outputs_dict:
                user_outputs_dict[key] = [value]
            else:
                user_outputs_dict[key].append(value)
    return user_outputs_dict


def extract_relevant_scores_from_multiple_models_val(
    val_idxs_to_val_sessions_idxs: list,
    val_pos_idxs_to_val_sessions_idxs: list,
    key: str,
    scores_list: list,
) -> np.ndarray:
    assert len(scores_list) > 0
    relevant_scores = np.zeros_like(scores_list[0])
    if key.startswith("y_val_negrated_ranking"):
        idxs = val_pos_idxs_to_val_sessions_idxs
    else:
        idxs = val_idxs_to_val_sessions_idxs
    assert len(idxs) == relevant_scores.shape[0]
    for idx, session_idx in enumerate(idxs):
        relevant_scores[idx] = scores_list[session_idx][idx]
    return relevant_scores


def extract_relevant_scores_from_multiple_models_negative_samples(
    val_pos_idxs_to_val_sessions_idxs: list,
    scores_list: list,
) -> np.ndarray:
    assert len(scores_list) > 0
    first_elem = scores_list[0]
    arr_shape = (len(val_pos_idxs_to_val_sessions_idxs), first_elem.shape[0])
    relevant_scores = np.zeros_like(first_elem, shape=arr_shape)
    for idx, session_idx in enumerate(val_pos_idxs_to_val_sessions_idxs):
        relevant_scores[idx] = scores_list[session_idx]
    return relevant_scores


def extract_relevant_scores_from_multiple_models(
    val_idxs_to_val_sessions_idxs: list,
    val_pos_idxs_to_val_sessions_idxs: list,
    user_outputs_dict: dict,
    n_models: int,
    n_train_rated: int,
    n_train_rated_pos: int,
) -> dict:
    user_outputs_dict_copy = user_outputs_dict.copy()
    for key, value in user_outputs_dict.items():
        if key.startswith("y_train"):
            assert isinstance(value, np.ndarray)
            if key.startswith("y_train_negrated_ranking"):
                assert value.ndim == 2
                assert value.shape[0] == n_train_rated_pos
            else:
                assert value.ndim == 1
                assert value.shape[0] == n_train_rated
        elif key.startswith("y_val"):
            assert isinstance(value, list)
            assert len(value) == n_models
            user_outputs_dict_copy[key] = extract_relevant_scores_from_multiple_models_val(
                val_idxs_to_val_sessions_idxs=val_idxs_to_val_sessions_idxs,
                val_pos_idxs_to_val_sessions_idxs=val_pos_idxs_to_val_sessions_idxs,
                key=key,
                scores_list=value,
            )
        elif key.startswith("y_negative_samples"):
            assert isinstance(value, list)
            assert len(value) == n_models
            user_outputs_dict_copy[key] = (
                extract_relevant_scores_from_multiple_models_negative_samples(
                    val_pos_idxs_to_val_sessions_idxs=val_pos_idxs_to_val_sessions_idxs,
                    scores_list=value,
                )
            )
            user_outputs_dict_copy[f"{key}_after_train"] = user_outputs_dict_copy[key][0]
        else:
            raise ValueError(f"Unexpected key in user_outputs_dict: {key}.")
    return user_outputs_dict_copy


def get_user_outputs_dict_sliding_window(
    val_idxs_to_val_sessions_idxs: list,
    val_pos_idxs_to_val_sessions_idxs: list,
    val_data_dict: dict,
    user_models: list,
    train_negrated_ranking_idxs: np.ndarray,
    val_negrated_ranking_idxs: np.ndarray,
    negative_samples_embeddings: np.ndarray,
) -> dict:
    user_outputs_dict_multiple_models = get_user_outputs_dict_sliding_window_multiple_models(
        val_data_dict=val_data_dict,
        user_models=user_models,
        train_negrated_ranking_idxs=train_negrated_ranking_idxs,
        val_negrated_ranking_idxs=val_negrated_ranking_idxs,
        negative_samples_embeddings=negative_samples_embeddings,
    )
    n_models = len(user_models)
    n_train_rated = val_data_dict["y_train_rated"].shape[0]
    n_train_rated_pos = np.sum(val_data_dict["y_train_rated"])
    n_val = val_data_dict["y_val"].shape[0]
    n_val_pos = np.sum(val_data_dict["y_val"])
    assert n_val == len(val_idxs_to_val_sessions_idxs)
    assert n_val_pos == len(val_pos_idxs_to_val_sessions_idxs)

    user_outputs_dict = extract_relevant_scores_from_multiple_models(
        val_idxs_to_val_sessions_idxs=val_idxs_to_val_sessions_idxs,
        val_pos_idxs_to_val_sessions_idxs=val_pos_idxs_to_val_sessions_idxs,
        user_outputs_dict=user_outputs_dict_multiple_models,
        n_models=n_models,
        n_train_rated=n_train_rated,
        n_train_rated_pos=n_train_rated_pos,
    )
    return user_outputs_dict


def score_user_models(
    scores: dict,
    val_data_dict: dict,
    user_models: list,
    negative_samples_embeddings: np.ndarray,
    train_negrated_ranking_idxs: np.ndarray,
    val_negrated_ranking_idxs: np.ndarray,
    save_users_predictions: bool = False,
) -> tuple:
    user_results = {}
    user_predictions = {
        "train_predictions": {},
        "val_predictions": {},
        "negative_samples_predictions": {},
    }
    for i, model in enumerate(user_models):
        if len(val_data_dict["y_val"]) > 0:
            user_outputs_dict = get_user_outputs_dict(
                model=model,
                val_data_dict=val_data_dict,
                train_negrated_ranking_idxs=train_negrated_ranking_idxs,
                val_negrated_ranking_idxs=val_negrated_ranking_idxs,
                negative_samples_embeddings=negative_samples_embeddings,
            )
            scores = get_user_scores(
                scores=scores, val_data_dict=val_data_dict, user_outputs_dict=user_outputs_dict
            )
            user_results[i] = scores
            if save_users_predictions:
                user_predictions["train_predictions"][i] = user_outputs_dict[
                    "y_train_rated_proba"
                ].tolist()
                user_predictions["val_predictions"][i] = user_outputs_dict["y_val_proba"].tolist()
                user_predictions["negative_samples_predictions"][i] = user_outputs_dict[
                    "y_negative_samples_proba"
                ].tolist()
    return user_results, user_predictions


def score_user_models_sliding_window(
    val_idxs_to_val_sessions_idxs: list,
    val_pos_idxs_to_val_sessions_idxs: list,
    scores: dict,
    val_data_dict: dict,
    user_models: list,
    negative_samples_embeddings: np.ndarray,
    train_negrated_ranking_idxs: np.ndarray,
    val_negrated_ranking_idxs: np.ndarray,
) -> tuple:
    user_results = {}
    user_predictions = {
        "train_predictions": {},
        "val_predictions": {},
        "negative_samples_predictions": {},
    }
    user_outputs_dict = get_user_outputs_dict_sliding_window(
        val_idxs_to_val_sessions_idxs=val_idxs_to_val_sessions_idxs,
        val_pos_idxs_to_val_sessions_idxs=val_pos_idxs_to_val_sessions_idxs,
        val_data_dict=val_data_dict,
        user_models=user_models,
        negative_samples_embeddings=negative_samples_embeddings,
        train_negrated_ranking_idxs=train_negrated_ranking_idxs,
        val_negrated_ranking_idxs=val_negrated_ranking_idxs,
    )
    scores = get_user_scores(
        scores=scores, val_data_dict=val_data_dict, user_outputs_dict=user_outputs_dict
    )
    user_results[0] = scores
    return user_results, user_predictions


def get_user_scores(
    scores: dict,
    val_data_dict: dict,
    user_outputs_dict: dict,
) -> tuple:
    user_scores = [0] * len(scores)
    if "y_negative_samples_pred_after_train" in user_outputs_dict:
        y_negative_samples_pred = user_outputs_dict["y_negative_samples_pred_after_train"]
    else:
        y_negative_samples_pred = user_outputs_dict["y_negative_samples_pred"]
    if "y_negative_samples_proba_after_train" in user_outputs_dict:
        y_negative_samples_proba = user_outputs_dict["y_negative_samples_proba_after_train"]
    else:
        y_negative_samples_proba = user_outputs_dict["y_negative_samples_proba"]

    for score in NON_DERIVABLE_SCORES:
        if not SCORES_DICT[score]["ranking"]:
            user_scores[scores[f"train_{score.name.lower()}"]] = get_score(
                score=score,
                y_true=val_data_dict["y_train_rated"],
                y_pred=user_outputs_dict["y_train_rated_pred"],
                y_proba=user_outputs_dict["y_train_rated_proba"],
                y_negative_samples_pred=y_negative_samples_pred,
                y_negative_samples_proba=y_negative_samples_proba,
            )
            user_scores[scores[f"val_{score.name.lower()}"]] = get_score(
                score=score,
                y_true=val_data_dict["y_val"],
                y_pred=user_outputs_dict["y_val_pred"],
                y_proba=user_outputs_dict["y_val_proba"],
                y_negative_samples_pred=y_negative_samples_pred,
                y_negative_samples_proba=y_negative_samples_proba,
            )
    ranking_scores = get_ranking_scores(
        y_train_rated=val_data_dict["y_train_rated"],
        y_train_rated_logits=user_outputs_dict["y_train_rated_logits"],
        y_val=val_data_dict["y_val"],
        y_val_logits=user_outputs_dict["y_val_logits"],
        y_train_negrated_ranking_logits=user_outputs_dict["y_train_negrated_ranking_logits"],
        y_val_negrated_ranking_logits=user_outputs_dict["y_val_negrated_ranking_logits"],
        y_negative_samples_logits=user_outputs_dict["y_negative_samples_logits"],
        y_negative_samples_logits_after_train=user_outputs_dict.get(
            "y_negative_samples_logits_after_train"
        ),
    )
    for ranking_score in ranking_scores:
        user_scores[scores[ranking_score]] = ranking_scores[ranking_score]
    category_scores = get_category_scores(
        y_train_rated=val_data_dict["y_train_rated"],
        y_val=val_data_dict["y_val"],
        categories_dict=val_data_dict["categories_dict"],
    )
    for category_score in category_scores:
        user_scores[scores[category_score]] = category_scores[category_score]
    user_scores_copy = user_scores.copy()
    for score in DERIVABLE_SCORES:
        user_scores[scores[f"train_{score.name.lower()}"]] = derive_score(
            score, user_scores_copy, scores, validation=False
        )
        user_scores[scores[f"val_{score.name.lower()}"]] = derive_score(
            score, user_scores_copy, scores, validation=True
        )
    return tuple(user_scores)
