import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LinearRegression

from ..embeddings.embedding import Embedding
from .scores_definitions import (
    FIRST_LAST_PERCENT,
    SCORES_BY_TYPE,
    SCORES_DICT,
    Score,
    Score_Type,
    get_score_category,
    get_score_default,
    get_score_default_derivable,
    get_score_info,
    get_score_ranking,
    get_score_ranking_session,
)

FIRST_LAST_PERCENT = float(FIRST_LAST_PERCENT) / 100.0


def fill_user_predictions_dict(val_data_dict: dict) -> dict:
    return {
        "train_ids": val_data_dict["X_train_rated_papers_ids"],
        "train_labels": val_data_dict["y_train_rated"].tolist(),
        "train_time": val_data_dict["time_train_rated"].tolist(),
        "train_session_id": val_data_dict["session_id_train_rated"].tolist(),
        "val_ids": val_data_dict["X_val_papers_ids"],
        "val_labels": val_data_dict["y_val"].tolist(),
        "val_time": val_data_dict["time_val"].tolist(),
        "val_session_id": val_data_dict["session_id_val"].tolist(),
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
        "session_id_train_rated": train_ratings["session_id"].values,
        "X_val": X_val,
        "y_val": y_val,
        "X_val_papers_ids": X_val_papers_ids,
        "time_val": val_ratings["time"].values,
        "session_id_val": val_ratings["session_id"].values,
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
    user_info: dict,
    sessions_min_times: dict,
    save_users_predictions_bool: bool = False,
    user_scores: dict = None
) -> tuple:
    user_results, user_predictions = {}, {}
    for i, model in enumerate(user_models):
        if len(val_data_dict["y_val"]) <= 0:
            break
        if user_scores is not None:
            assert len(user_models) == 1
            user_outputs_dict = user_scores
        else:
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
            user_info=user_info,
            sessions_min_times=sessions_min_times,
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
    user_info: dict,
    sessions_min_times: dict,
    save_users_predictions_bool: bool = False,
    user_scores: dict = None
) -> tuple:
    user_results, user_predictions = {}, {}
    if user_scores is not None:
        user_outputs_dict = user_scores
    else:
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
        user_info=user_info,
        sessions_min_times=sessions_min_times,
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


def get_scores_ranking_convert_temporal(
    temporal_train_rated: np.ndarray, temporal_val: np.ndarray, score_type: Score_Type
) -> tuple:
    if score_type == Score_Type.RANKING_SESSION:
        return temporal_train_rated, temporal_val
    elif score_type == Score_Type.RANKING_TIME:
        ns_per_day = 24 * 60 * 60 * 1_000_000_000
        first_train_timestamp = temporal_train_rated[0]
        temporal_train_rated_days = (temporal_train_rated - first_train_timestamp).astype("int64")
        temporal_train_rated_days = temporal_train_rated_days / ns_per_day
        first_val_timestamp = temporal_val[0]
        temporal_val_days = (temporal_val - first_val_timestamp).astype("int64") / ns_per_day
        return temporal_train_rated_days, temporal_val_days
    else:
        raise ValueError(f"Unsupported score_type: {score_type}.")


def get_scores_ranking_temporal_slope(
    temporal_pos: np.ndarray, scores_ranking_before_avging: list
) -> float:
    y_scores = np.array(scores_ranking_before_avging)
    assert y_scores.shape == temporal_pos.shape
    assert len(temporal_pos) > 1
    model = LinearRegression()
    model.fit(temporal_pos.reshape(-1, 1), y_scores)
    return float(model.coef_[0])


def get_scores_ranking_temporal(
    user_scores: list,
    scores_to_indices_dict: dict,
    scores_ranking_before_avging_train: dict,
    scores_ranking_before_avging_val: dict,
    y_train_rated: np.ndarray,
    y_val: np.ndarray,
    temporal_train_rated: np.ndarray,
    temporal_val: np.ndarray,
    score_type: Score_Type,
) -> None:
    assert y_train_rated.shape == temporal_train_rated.shape
    assert y_val.shape == temporal_val.shape
    temporal_train_rated, temporal_val = get_scores_ranking_convert_temporal(
        temporal_train_rated, temporal_val, score_type
    )
    train_mask_pos = y_train_rated > 0
    val_mask_pos = y_val > 0
    temporal_train_rated_pos = temporal_train_rated[train_mask_pos]
    temporal_val_pos = temporal_val[val_mask_pos]
    assert np.all(np.diff(temporal_train_rated_pos) >= 0)
    assert np.all(np.diff(temporal_val_pos) >= 0)

    for score in SCORES_BY_TYPE[score_type]:
        lookup = SCORES_DICT[score]["lookup"]
        train_score = get_scores_ranking_temporal_slope(
            temporal_train_rated_pos, scores_ranking_before_avging_train[lookup]
        )
        val_score = get_scores_ranking_temporal_slope(
            temporal_val_pos, scores_ranking_before_avging_val[lookup]
        )
        fill_user_scores_with_score(
            score, user_scores, scores_to_indices_dict, train_score, val_score
        )


def get_relevant_idxs(val_data_dict: dict) -> dict:
    relevant_idxs = {}
    for split in ["train_rated", "val"]:
        relevant_idxs[split] = {}
        pos_mask = val_data_dict[f"y_{split}"] > 0
        pos_sessions_ids = val_data_dict[f"session_id_{split}"][pos_mask]
        distinct_sessions_ids = np.unique(pos_sessions_ids)
        for session_id in distinct_sessions_ids:
            session_mask = pos_sessions_ids == session_id
            relevant_idxs[split][session_id] = np.where(session_mask)[0].tolist()
    return relevant_idxs


def get_scores_ranking_per_session(
    scores_ranking_before_avging_train: dict,
    scores_ranking_before_avging_val: dict,
    relevant_idxs: dict,
) -> dict:
    scores_ranking_per_session = {}
    scores = SCORES_BY_TYPE[Score_Type.RANKING_SESSION]
    lookups = list(set([SCORES_DICT[score]["lookup"] for score in scores]))
    for split in ["train_rated", "val"]:
        scores_ranking_per_session[split] = {}
        relevant_idxs_split = relevant_idxs[split]
        distinct_sessions_ids = list(relevant_idxs_split.keys())
        if split == "train_rated":
            scores_ranking_before_avging = scores_ranking_before_avging_train
        else:
            scores_ranking_before_avging = scores_ranking_before_avging_val
        for lookup in lookups:
            scores_ranking_per_session[split][lookup] = {}
            values = scores_ranking_before_avging[lookup]
            for session_id in distinct_sessions_ids:
                relevant_values = np.array(values)[relevant_idxs_split[session_id]]
                scores_ranking_per_session[split][lookup][session_id] = relevant_values.tolist()
    return scores_ranking_per_session


def get_avgs_ranking_per_session(scores_ranking_per_session: dict) -> dict:
    avgs_ranking_per_session = {}
    for split in scores_ranking_per_session.keys():
        avgs_ranking_per_session[split] = {}
        for lookup in scores_ranking_per_session[split].keys():
            avgs_ranking_per_session[split][lookup] = {}
            for session_id, values in scores_ranking_per_session[split][lookup].items():
                avgs_ranking_per_session[split][lookup][session_id] = float(np.mean(values))
    return avgs_ranking_per_session


def get_first_last_sessions_idxs(
    scores_ranking_per_session: dict, sessions_min_times: dict
) -> tuple:
    first_key = list(scores_ranking_per_session.keys())[0]
    relevant_sessions_ids = np.array(list(scores_ranking_per_session[first_key].keys()))
    n_sessions = len(relevant_sessions_ids)

    n_sessions_to_include_for_sessions = max(1, int(n_sessions * FIRST_LAST_PERCENT))
    largest_session_idx_to_include_for_first_sessions = n_sessions_to_include_for_sessions - 1
    smallest_session_idx_to_include_for_last_sessions = (
        n_sessions - n_sessions_to_include_for_sessions
    )

    sessions_min_times = np.array([sessions_min_times[k] for k in relevant_sessions_ids])
    assert len(sessions_min_times) == len(relevant_sessions_ids)
    min_time, max_time = sessions_min_times.min(), sessions_min_times.max()
    total_time_span = pd.Timedelta(max_time - min_time)
    largest_time_to_include_for_first_times = min_time + total_time_span * FIRST_LAST_PERCENT
    smallest_time_to_include_for_last_times = max_time - total_time_span * FIRST_LAST_PERCENT
    
    sessions_ids_before_largest_time = relevant_sessions_ids[
        sessions_min_times <= largest_time_to_include_for_first_times
    ]
    assert len(sessions_ids_before_largest_time) > 0
    largest_session_idx_to_include_for_first_times = len(sessions_ids_before_largest_time) - 1
    sessions_ids_after_smallest_time = relevant_sessions_ids[
        sessions_min_times >= smallest_time_to_include_for_last_times
    ]
    assert len(sessions_ids_after_smallest_time) > 0
    smallest_session_idx_to_include_for_last_times = (
        n_sessions - len(sessions_ids_after_smallest_time)
    )
    return (
        largest_session_idx_to_include_for_first_sessions,
        smallest_session_idx_to_include_for_last_sessions,
        largest_session_idx_to_include_for_first_times,
        smallest_session_idx_to_include_for_last_times,
    )


def get_mid_time_session_idx(scores_ranking_per_session: dict, sessions_min_times: dict) -> int:
    first_key = list(scores_ranking_per_session.keys())[0]
    relevant_sessions_ids = np.array(list(scores_ranking_per_session[first_key].keys())).tolist()
    sessions_min_times = {k: v for k, v in sessions_min_times.items() if k in relevant_sessions_ids}
    times = np.array(list(sessions_min_times.values()))
    mid_time = pd.Timestamp((times.min().value + times.max().value) / 2)
    session_ids = np.array(list(sessions_min_times.keys()))
    sessions_ids_before_mid_time = session_ids[times <= mid_time]
    if len(sessions_ids_before_mid_time) == 0:
        return 0
    mid_time_session_id = sessions_ids_before_mid_time[-1]
    return relevant_sessions_ids.index(mid_time_session_id)


def get_scores_ranking_session(
    user_scores: list,
    scores_to_indices_dict: dict,
    scores_ranking_before_avging_train: dict,
    scores_ranking_before_avging_val: dict,
    relevant_idxs: dict,
    sessions_min_times: dict,
) -> None:
    scores_ranking_per_session = get_scores_ranking_per_session(
        scores_ranking_before_avging_train=scores_ranking_before_avging_train,
        scores_ranking_before_avging_val=scores_ranking_before_avging_val,
        relevant_idxs=relevant_idxs,
    )
    avgs_ranking_per_session = get_avgs_ranking_per_session(scores_ranking_per_session)

    (
        largest_session_idx_to_include_for_first_sessions_train_rated,
        smallest_session_idx_to_include_for_last_sessions_train_rated,
        largest_session_idx_to_include_for_first_times_train_rated,
        smallest_session_idx_to_include_for_last_times_train_rated,
    ) = get_first_last_sessions_idxs(scores_ranking_per_session["train_rated"], sessions_min_times)
    (
        largest_session_idx_to_include_for_first_sessions_val,
        smallest_session_idx_to_include_for_last_sessions_val,
        largest_session_idx_to_include_for_first_times_val,
        smallest_session_idx_to_include_for_last_times_val,
    ) = get_first_last_sessions_idxs(scores_ranking_per_session["val"], sessions_min_times)

    for score in SCORES_BY_TYPE[Score_Type.RANKING_SESSION]:
        lookup = SCORES_DICT[score]["lookup"]
        selection = SCORES_DICT[score]["selection"]
        train_score = get_score_ranking_session(
            scores_per_session=scores_ranking_per_session["train_rated"][lookup],
            avgs_per_session=avgs_ranking_per_session["train_rated"][lookup],
            selection=selection,
            largest_session_idx_to_include_for_first_sessions=largest_session_idx_to_include_for_first_sessions_train_rated,
            smallest_session_idx_to_include_for_last_sessions=smallest_session_idx_to_include_for_last_sessions_train_rated,
            largest_session_idx_to_include_for_first_times=largest_session_idx_to_include_for_first_times_train_rated,
            smallest_session_idx_to_include_for_last_times=smallest_session_idx_to_include_for_last_times_train_rated,
        )
        val_score = get_score_ranking_session(
            scores_per_session=scores_ranking_per_session["val"][lookup],
            avgs_per_session=avgs_ranking_per_session["val"][lookup],
            selection=selection,
            largest_session_idx_to_include_for_first_sessions=largest_session_idx_to_include_for_first_sessions_val,
            smallest_session_idx_to_include_for_last_sessions=smallest_session_idx_to_include_for_last_sessions_val,
            largest_session_idx_to_include_for_first_times=largest_session_idx_to_include_for_first_times_val,
            smallest_session_idx_to_include_for_last_times=smallest_session_idx_to_include_for_last_times_val,
        )
        fill_user_scores_with_score(
            score, user_scores, scores_to_indices_dict, train_score, val_score
        )


def get_pos_embeddings_per_session(val_data_dict: dict) -> dict:
    pos_embeddings_per_session = {}
    for split in ["train_rated", "val"]:
        pos_embeddings_per_session[split] = {}
        pos_mask = val_data_dict[f"y_{split}"] > 0
        pos_sessions_ids = val_data_dict[f"session_id_{split}"][pos_mask]
        distinct_sessions_ids = np.unique(pos_sessions_ids)
        embeddings = val_data_dict[f"X_{split}"][pos_mask]
        for session_id in distinct_sessions_ids:
            session_mask = pos_sessions_ids == session_id
            pos_embeddings_per_session[split][session_id] = embeddings[session_mask][:, :-100]
    return pos_embeddings_per_session


def get_scores_info(
    user_scores: list,
    scores_to_indices_dict: dict,
    val_data_dict: dict,
    user_info: dict,
) -> None:
    pos_embeddings_per_session = get_pos_embeddings_per_session(val_data_dict=val_data_dict)
    train_pos_mask = val_data_dict["y_train_rated"] > 0
    train_pos_embeddings_all_sessions = val_data_dict["X_train_rated"][train_pos_mask][:, :-100]
    for score in SCORES_BY_TYPE[Score_Type.INFO]:
        if "lookup_key" in SCORES_DICT[score]:
            key = SCORES_DICT[score]["lookup_key"]
            value = user_info[key]
            if isinstance(value, list):
                train_score = float(value[-1])
            else:
                train_score = float(value)
            val_score = train_score
        elif "calculator" in SCORES_DICT[score]:
            train_score = get_score_info(
                score=score,
                val_data_dict=val_data_dict,
                user_info=user_info,
                train_pos_embeddings_all_sessions=train_pos_embeddings_all_sessions,
                val_pos_embeddings_per_session=pos_embeddings_per_session["val"],
            )
            val_score = train_score
        else:
            train_score, val_score = None, None
        fill_user_scores_with_score(
            score, user_scores, scores_to_indices_dict, train_score, val_score
        )


def get_user_scores(
    scores_to_indices_dict: dict,
    val_data_dict: dict,
    user_outputs_dict: dict,
    user_info: dict,
    sessions_min_times: dict,
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
        scores_ranking_before_avging_val=scores_ranking_before_avging_val,
    )
    get_scores_ranking_bottom_1_pos(
        user_scores=user_scores,
        scores_to_indices_dict=scores_to_indices_dict,
        scores_ranking_before_avging_train=scores_ranking_before_avging_train,
        scores_ranking_before_avging_val=scores_ranking_before_avging_val,
    )
    relevant_idxs = get_relevant_idxs(val_data_dict=val_data_dict)
    get_scores_ranking_session(
        user_scores=user_scores,
        scores_to_indices_dict=scores_to_indices_dict,
        scores_ranking_before_avging_train=scores_ranking_before_avging_train,
        scores_ranking_before_avging_val=scores_ranking_before_avging_val,
        relevant_idxs=relevant_idxs,
        sessions_min_times=sessions_min_times,
    )
    get_scores_info(
        user_scores=user_scores,
        scores_to_indices_dict=scores_to_indices_dict,
        val_data_dict=val_data_dict,
        user_info=user_info,
    )
    return tuple(user_scores)
