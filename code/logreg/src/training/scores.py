import numpy as np
import pandas as pd
from scipy.special import softmax

from ..embeddings.embedding import Embedding
from .scores_definitions import (
    SCORES_BY_TYPE,
    SCORES_DICT,
    Score,
    Score_Type,
    get_score_category,
    get_score_default,
    get_score_default_derivable,
    get_score_ranking,
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


def gather_user_predictions(user_predictions: dict, user_outputs_dict: dict, i: int = 0) -> dict:
    user_predictions = user_predictions.copy()
    y_train_rated_proba = user_outputs_dict["y_train_rated_proba"].tolist()
    if "train_predictions" not in user_predictions:
        user_predictions["train_predictions"] = {}
    user_predictions["train_predictions"][i] = y_train_rated_proba

    y_val_proba = user_outputs_dict["y_val_proba"].tolist()
    if "val_predictions" not in user_predictions:
        user_predictions["val_predictions"] = {}
    user_predictions["val_predictions"][i] = y_val_proba

    if user_outputs_dict["y_negative_samples_proba"].ndim == 2:
        y_negative_samples_proba = user_outputs_dict["y_negative_samples_proba_after_train"]
    else:
        y_negative_samples_proba = user_outputs_dict["y_negative_samples_proba"]
    y_negative_samples_proba = y_negative_samples_proba.tolist()
    if "negative_samples_predictions" not in user_predictions:
        user_predictions["negative_samples_predictions"] = {}
    user_predictions["negative_samples_predictions"][i] = y_negative_samples_proba
    return user_predictions


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
        "time_train_rated": train_ratings["time"].values,
        "X_val": X_val,
        "y_val": y_val,
        "X_val_papers_ids": X_val_papers_ids,
        "time_val": val_ratings["time"].values,
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
    scores_to_indices_dict: dict,
    val_data_dict: dict,
    user_models: list,
    negative_samples_embeddings: np.ndarray,
    train_negrated_ranking_idxs: np.ndarray,
    val_negrated_ranking_idxs: np.ndarray,
    save_users_predictions_bool: bool = False,
) -> tuple:
    user_results, user_predictions = {}, {}
    for i, model in enumerate(user_models):
        if len(val_data_dict["y_val"]) <= 0:
            break
        user_outputs_dict = get_user_outputs_dict(
            model=model,
            val_data_dict=val_data_dict,
            train_negrated_ranking_idxs=train_negrated_ranking_idxs,
            val_negrated_ranking_idxs=val_negrated_ranking_idxs,
            negative_samples_embeddings=negative_samples_embeddings,
        )
        user_results[i] = get_user_scores(
            scores_to_indices_dict=scores_to_indices_dict,
            val_data_dict=val_data_dict,
            user_outputs_dict=user_outputs_dict,
        )
        if save_users_predictions_bool:
            user_predictions = gather_user_predictions(user_predictions, user_outputs_dict, i)
    return user_results, user_predictions


def score_user_models_sliding_window(
    val_idxs_to_val_sessions_idxs: list,
    val_pos_idxs_to_val_sessions_idxs: list,
    scores_to_indices_dict: dict,
    val_data_dict: dict,
    user_models: list,
    negative_samples_embeddings: np.ndarray,
    train_negrated_ranking_idxs: np.ndarray,
    val_negrated_ranking_idxs: np.ndarray,
    save_users_predictions_bool: bool,
) -> tuple:
    user_results, user_predictions = {}, {}
    user_outputs_dict = get_user_outputs_dict_sliding_window(
        val_idxs_to_val_sessions_idxs=val_idxs_to_val_sessions_idxs,
        val_pos_idxs_to_val_sessions_idxs=val_pos_idxs_to_val_sessions_idxs,
        val_data_dict=val_data_dict,
        user_models=user_models,
        negative_samples_embeddings=negative_samples_embeddings,
        train_negrated_ranking_idxs=train_negrated_ranking_idxs,
        val_negrated_ranking_idxs=val_negrated_ranking_idxs,
    )
    user_results[0] = get_user_scores(
        scores_to_indices_dict=scores_to_indices_dict,
        val_data_dict=val_data_dict,
        user_outputs_dict=user_outputs_dict,
    )
    if save_users_predictions_bool:
        user_predictions = gather_user_predictions(user_predictions, user_outputs_dict, 0)
    return user_results, user_predictions


def fill_user_scores_with_score(
    score: Score,
    user_scores: list,
    scores_to_indices_dict: dict,
    train_score: float = None,
    val_score: float = None,
) -> None:
    if train_score is not None:
        train_idx = scores_to_indices_dict[f"train_{score.name.lower()}"]
        user_scores[train_idx] = train_score
    if val_score is not None:
        val_idx = scores_to_indices_dict[f"val_{score.name.lower()}"]
        user_scores[val_idx] = val_score


def y_negative_samples_pred_proba(user_outputs_dict: dict) -> np.ndarray:
    if "y_negative_samples_pred_after_train" in user_outputs_dict:
        y_negative_samples_pred = user_outputs_dict["y_negative_samples_pred_after_train"]
    else:
        y_negative_samples_pred = user_outputs_dict["y_negative_samples_pred"]
    assert y_negative_samples_pred.ndim == 1
    if "y_negative_samples_proba_after_train" in user_outputs_dict:
        y_negative_samples_proba = user_outputs_dict["y_negative_samples_proba_after_train"]
    else:
        y_negative_samples_proba = user_outputs_dict["y_negative_samples_proba"]
    assert y_negative_samples_proba.ndim == 1
    return y_negative_samples_pred, y_negative_samples_proba


def get_scores_default(
    user_scores: list,
    scores_to_indices_dict: dict,
    val_data_dict: dict,
    user_outputs_dict: dict,
) -> None:
    y_negative_samples_pred, y_negative_samples_proba = y_negative_samples_pred_proba(
        user_outputs_dict=user_outputs_dict
    )
    for score in SCORES_BY_TYPE[Score_Type.DEFAULT]:
        train_score, val_score = [
            get_score_default(
                score=score,
                y_true=val_data_dict[f"y_{split}"],
                y_pred=user_outputs_dict[f"y_{split}_pred"],
                y_proba=user_outputs_dict[f"y_{split}_proba"],
                y_negative_samples_pred=y_negative_samples_pred,
                y_negative_samples_proba=y_negative_samples_proba,
            )
            for split in ["train_rated", "val"]
        ]
        fill_user_scores_with_score(
            score, user_scores, scores_to_indices_dict, train_score, val_score
        )


def get_scores_default_derivable(
    user_scores: list,
    scores_to_indices_dict: dict,
) -> None:
    for score in SCORES_BY_TYPE[Score_Type.DEFAULT_DERIVABLE]:
        train_score, val_score = [
            get_score_default_derivable(
                score=score,
                user_scores=user_scores,
                scores_to_indices_dict=scores_to_indices_dict,
                val=is_validation,
            )
            for is_validation in [False, True]
        ]
        fill_user_scores_with_score(
            score, user_scores, scores_to_indices_dict, train_score, val_score
        )


def get_scores_category_dfs(
    y_train_rated: np.ndarray, y_val: np.ndarray, categories_dict: dict
) -> dict:
    train_rated_pos_mask, val_pos_mask = (y_train_rated == 1), (y_val == 1)
    dfs = {
        "l1_train_rated_pos": pd.Series(categories_dict["l1_train_rated"][train_rated_pos_mask]),
        "l1_train_rated_neg": pd.Series(categories_dict["l1_train_rated"][~train_rated_pos_mask]),
        "l2_train_rated_pos": pd.Series(categories_dict["l2_train_rated"][train_rated_pos_mask]),
        "l2_train_rated_neg": pd.Series(categories_dict["l2_train_rated"][~train_rated_pos_mask]),
        "l1_val_pos": pd.Series(categories_dict["l1_val"][val_pos_mask]),
        "l1_val_neg": pd.Series(categories_dict["l1_val"][~val_pos_mask]),
        "l2_val_pos": pd.Series(categories_dict["l2_val"][val_pos_mask]),
        "l2_val_neg": pd.Series(categories_dict["l2_val"][~val_pos_mask]),
    }
    dfs["l1l2_train_rated_pos"] = pd.Series(
        [f"{l1}_{l2}" for l1, l2 in zip(dfs["l1_train_rated_pos"], dfs["l2_train_rated_pos"])]
    )
    dfs["l1l2_train_rated_neg"] = pd.Series(
        [f"{l1}_{l2}" for l1, l2 in zip(dfs["l1_train_rated_neg"], dfs["l2_train_rated_neg"])]
    )
    dfs["l1l2_val_pos"] = pd.Series(
        [f"{l1}_{l2}" for l1, l2 in zip(dfs["l1_val_pos"], dfs["l2_val_pos"])]
    )
    dfs["l1l2_val_neg"] = pd.Series(
        [f"{l1}_{l2}" for l1, l2 in zip(dfs["l1_val_neg"], dfs["l2_val_neg"])]
    )
    return dfs


def get_scores_category(
    user_scores: list,
    scores_to_indices_dict: dict,
    y_train_rated: np.ndarray,
    y_val: np.ndarray,
    categories_dict: dict,
) -> None:
    dfs = get_scores_category_dfs(
        y_train_rated=y_train_rated,
        y_val=y_val,
        categories_dict=categories_dict,
    )
    for score in SCORES_BY_TYPE[Score_Type.CATEGORY]:
        train_score, val_score = [
            get_score_category(
                score=score,
                l1_pos=dfs[f"l1_{split}_pos"],
                l1_neg=dfs[f"l1_{split}_neg"],
                l1l2_pos=dfs[f"l1l2_{split}_pos"],
                l1l2_neg=dfs[f"l1l2_{split}_neg"],
            )
            for split in ["train_rated", "val"]
        ]
        fill_user_scores_with_score(
            score, user_scores, scores_to_indices_dict, train_score, val_score
        )


def get_scores_ranking_preprocessing(
    y_train_rated: np.ndarray,
    y_train_rated_logits: np.ndarray,
    y_val: np.ndarray,
    y_val_logits: np.ndarray,
    y_negative_samples_logits: np.ndarray,
    y_negative_samples_logits_after_train: np.ndarray = None,
) -> tuple:
    assert y_train_rated.shape == y_train_rated_logits.shape
    train_pos_mask = y_train_rated > 0
    y_train_rated_pos_logits = y_train_rated_logits[train_pos_mask]
    assert y_val.shape == y_val_logits.shape
    val_pos_mask = y_val > 0
    y_val_pos_logits = y_val_logits[val_pos_mask]
    if y_negative_samples_logits_after_train is not None:
        y_negative_samples_train_logits = y_negative_samples_logits_after_train
    else:
        y_negative_samples_train_logits = y_negative_samples_logits
    assert y_negative_samples_train_logits.ndim == 1
    return y_train_rated_pos_logits, y_val_pos_logits, y_negative_samples_train_logits


def get_ranking_logits(
    i: int, y_negrated_ranking_logits: np.ndarray, y_negative_samples_logits: np.ndarray
) -> tuple:
    y_negrated_ranking_logits_i = y_negrated_ranking_logits[i]
    if y_negative_samples_logits.ndim == 2:
        y_negative_samples_logits_i = y_negative_samples_logits[i]
    else:
        y_negative_samples_logits_i = y_negative_samples_logits
    y_neg_ranking_all_logits_i = np.concatenate(
        (y_negrated_ranking_logits_i, y_negative_samples_logits_i)
    )
    return y_negrated_ranking_logits_i, y_negative_samples_logits_i, y_neg_ranking_all_logits_i


def get_ranking_ranks(
    y_pos_logit: float,
    y_negrated_ranking_logits: np.ndarray,
    y_negative_samples_logits: np.ndarray,
    y_neg_ranking_all_logits: np.ndarray,
) -> tuple:
    rank_pos = np.sum(y_negrated_ranking_logits >= y_pos_logit) + 1
    rank_pos_samples = np.sum(y_negative_samples_logits >= y_pos_logit) + 1
    rank_pos_all = np.sum(y_neg_ranking_all_logits >= y_pos_logit) + 1
    return rank_pos, rank_pos_samples, rank_pos_all


def get_ranking_softmax(all_logits: np.ndarray) -> dict:
    return {
        "05": softmax(all_logits / 0.5),
        "1": softmax(all_logits),
        "2": softmax(all_logits / 2.0),
    }


def get_scores_ranking_before_avging_split(
    y_pos_logits: np.ndarray,
    y_negrated_ranking_logits: np.ndarray,
    y_negative_samples_logits: np.ndarray,
) -> dict:
    scores_ranking_before_avging = {}
    for i, y_pos_logit in enumerate(y_pos_logits):
        y_negrated_ranking_logits_i, y_negative_samples_logits_i, y_neg_ranking_all_logits_i = (
            get_ranking_logits(i, y_negrated_ranking_logits, y_negative_samples_logits)
        )
        all_logits = np.concatenate((np.array([y_pos_logit]), y_neg_ranking_all_logits_i))
        rank_pos, rank_pos_samples, rank_pos_all = get_ranking_ranks(
            y_pos_logit=y_pos_logit,
            y_negrated_ranking_logits=y_negrated_ranking_logits_i,
            y_negative_samples_logits=y_negative_samples_logits_i,
            y_neg_ranking_all_logits=y_neg_ranking_all_logits_i,
        )
        softmax_dict = get_ranking_softmax(all_logits)
        for score in SCORES_BY_TYPE[Score_Type.RANKING]:
            score_ranking = get_score_ranking(
                score=score,
                rank_pos=rank_pos,
                rank_pos_samples=rank_pos_samples,
                rank_pos_all=rank_pos_all,
                softmax_dict=softmax_dict,
            )
            if score not in scores_ranking_before_avging:
                scores_ranking_before_avging[score] = []
            scores_ranking_before_avging[score].append(score_ranking)
    return scores_ranking_before_avging


def get_scores_ranking_before_avging(
    y_train_rated: np.ndarray,
    y_train_rated_logits: np.ndarray,
    y_val: np.ndarray,
    y_val_logits: np.ndarray,
    y_train_negrated_ranking_logits: np.ndarray,
    y_val_negrated_ranking_logits: np.ndarray,
    y_negative_samples_logits: np.ndarray,
    y_negative_samples_logits_after_train: np.ndarray = None,
) -> tuple:
    y_train_rated_pos_logits, y_val_pos_logits, y_negative_samples_train_logits = (
        get_scores_ranking_preprocessing(
            y_train_rated=y_train_rated,
            y_train_rated_logits=y_train_rated_logits,
            y_val=y_val,
            y_val_logits=y_val_logits,
            y_negative_samples_logits=y_negative_samples_logits,
            y_negative_samples_logits_after_train=y_negative_samples_logits_after_train,
        )
    )
    scores_ranking_before_avging_train = get_scores_ranking_before_avging_split(
        y_pos_logits=y_train_rated_pos_logits,
        y_negrated_ranking_logits=y_train_negrated_ranking_logits,
        y_negative_samples_logits=y_negative_samples_train_logits,
    )
    scores_ranking_before_avging_val = get_scores_ranking_before_avging_split(
        y_pos_logits=y_val_pos_logits,
        y_negrated_ranking_logits=y_val_negrated_ranking_logits,
        y_negative_samples_logits=y_negative_samples_logits,
    )
    assert scores_ranking_before_avging_train.keys() == scores_ranking_before_avging_val.keys()
    return scores_ranking_before_avging_train, scores_ranking_before_avging_val


def get_scores_ranking(
    user_scores: list,
    scores_to_indices_dict: dict,
    scores_ranking_before_avging_train: dict,
    scores_ranking_before_avging_val: dict,
) -> None:
    for score in SCORES_BY_TYPE[Score_Type.RANKING]:
        train_score = np.mean(scores_ranking_before_avging_train[score])
        val_score = np.mean(scores_ranking_before_avging_val[score])
        fill_user_scores_with_score(
            score, user_scores, scores_to_indices_dict, train_score, val_score
        )


def get_scores_ranking_bottom_1_pos(
    user_scores: list,
    scores_to_indices_dict: dict,
    scores_ranking_before_avging_train: dict,
    scores_ranking_before_avging_val: dict,
) -> None:
    for score in SCORES_BY_TYPE[Score_Type.RANKING_BOTTOM_1_POS]:
        lookup = SCORES_DICT[score]["lookup"]
        train_score = np.min(scores_ranking_before_avging_train[lookup])
        val_score = np.min(scores_ranking_before_avging_val[lookup])
        fill_user_scores_with_score(
            score, user_scores, scores_to_indices_dict, train_score, val_score
        )


def get_user_scores(
    scores_to_indices_dict: dict,
    val_data_dict: dict,
    user_outputs_dict: dict,
) -> tuple:
    user_scores = [None] * len(scores_to_indices_dict)

    get_scores_default(
        user_scores=user_scores,
        scores_to_indices_dict=scores_to_indices_dict,
        val_data_dict=val_data_dict,
        user_outputs_dict=user_outputs_dict,
    )
    get_scores_default_derivable(
        user_scores=user_scores,
        scores_to_indices_dict=scores_to_indices_dict,
    )
    get_scores_category(
        user_scores=user_scores,
        scores_to_indices_dict=scores_to_indices_dict,
        y_train_rated=val_data_dict["y_train_rated"],
        y_val=val_data_dict["y_val"],
        categories_dict=val_data_dict["categories_dict"],
    )
    scores_ranking_before_avging_train, scores_ranking_before_avging_val = (
        get_scores_ranking_before_avging(
            y_train_rated=val_data_dict["y_train_rated"],
            y_train_rated_logits=user_outputs_dict["y_train_rated_logits"],
            y_val=val_data_dict["y_val"],
            y_val_logits=user_outputs_dict["y_val_logits"],
            y_train_negrated_ranking_logits=user_outputs_dict["y_train_negrated_ranking_logits"],
            y_val_negrated_ranking_logits=user_outputs_dict["y_val_negrated_ranking_logits"],
            y_negative_samples_logits=user_outputs_dict["y_negative_samples_logits"],
            y_negative_samples_logits_after_train=user_outputs_dict.get(
                "y_negative_samples_logits_after_train", None
            ),
        )
    )
    get_scores_ranking(
        user_scores=user_scores,
        scores_to_indices_dict=scores_to_indices_dict,
        scores_ranking_before_avging_train=scores_ranking_before_avging_train,
        scores_ranking_before_avging_val=scores_ranking_before_avging_val
    )
    get_scores_ranking_bottom_1_pos(
        user_scores=user_scores,
        scores_to_indices_dict=scores_to_indices_dict,
        scores_ranking_before_avging_train=scores_ranking_before_avging_train,
        scores_ranking_before_avging_val=scores_ranking_before_avging_val
    )
    return tuple(user_scores)
