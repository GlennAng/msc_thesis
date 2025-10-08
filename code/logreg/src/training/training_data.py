import json
import os
import random
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd

from ..embeddings.embedding import Embedding
from .users_ratings import N_NEGRATED_RANKING

FLOAT_PRECISION = 1e-10
LABEL_DTYPE = np.int64


def load_negrated_ranking_idxs_for_user_random(
    negrated_ranking: pd.DataFrame,
    negrated_ranking_idxs: np.ndarray,
    random_state: int,
    same_negrated_for_all_pos: bool,
) -> np.ndarray:
    neg_idxs = negrated_ranking.index.values.tolist()
    random.seed(random_state)
    if same_negrated_for_all_pos:
        negrated_ranking_idxs_sample = sorted(
            random.sample(neg_idxs, negrated_ranking_idxs.shape[1])
        )
        negrated_ranking_idxs = np.tile(
            negrated_ranking_idxs_sample, (negrated_ranking_idxs.shape[0], 1)
        )
    else:
        for i in range(negrated_ranking_idxs.shape[0]):
            negrated_ranking_idxs_sample = sorted(
                random.sample(neg_idxs, negrated_ranking_idxs.shape[1])
            )
            negrated_ranking_idxs[i] = negrated_ranking_idxs_sample
    return negrated_ranking_idxs


def load_negrated_ranking_idxs_for_user_timesort(
    pos_ratings: pd.DataFrame,
    negrated_ranking: pd.DataFrame,
    causal_mask: bool,
    negrated_ranking_idxs: np.ndarray,
) -> np.ndarray:
    assert len(pos_ratings) == negrated_ranking_idxs.shape[0]
    n_negrated = negrated_ranking_idxs.shape[1]

    pos_times = pos_ratings["time"].values
    pos_session_ids = pos_ratings["session_id"].values
    neg_times = negrated_ranking["time"].values
    neg_session_ids = negrated_ranking["session_id"].values
    neg_indices = negrated_ranking.index.values

    for i, (pos_time, pos_session_id) in enumerate(zip(pos_times, pos_session_ids)):
        if causal_mask:
            valid_mask = neg_session_ids >= pos_session_id
            assert valid_mask.any()
            valid_neg_times = neg_times[valid_mask]
            valid_neg_indices = neg_indices[valid_mask]
            time_diffs = np.abs(valid_neg_times - pos_time)
            closest_idxs = np.argsort(time_diffs)[:n_negrated]
            negrated_ranking_idxs[i] = valid_neg_indices[closest_idxs]
        else:
            time_diffs = np.abs(neg_times - pos_time)
            closest_idxs = np.argsort(time_diffs)[:n_negrated]
            negrated_ranking_idxs[i] = neg_indices[closest_idxs]
    return negrated_ranking_idxs


def load_negrated_ranking_idxs_for_user(
    ratings: pd.DataFrame,
    negrated_ranking: pd.DataFrame,
    timesort: bool,
    causal_mask: bool,
    random_state: int,
    same_negrated_for_all_pos: bool,
) -> np.ndarray:
    pos_ratings = ratings[ratings["rating"] > 0]
    n_pos, n_neg = len(pos_ratings), len(negrated_ranking)
    min_n_negrated = min(N_NEGRATED_RANKING, n_neg)
    negrated_ranking_idxs = np.zeros((n_pos, min_n_negrated), dtype=np.int64)
    if timesort:
        return load_negrated_ranking_idxs_for_user_timesort(
            pos_ratings=pos_ratings,
            negrated_ranking=negrated_ranking,
            causal_mask=causal_mask,
            negrated_ranking_idxs=negrated_ranking_idxs,
        )
    else:
        assert not causal_mask
        return load_negrated_ranking_idxs_for_user_random(
            negrated_ranking=negrated_ranking,
            negrated_ranking_idxs=negrated_ranking_idxs,
            random_state=random_state,
            same_negrated_for_all_pos=same_negrated_for_all_pos,
        )


def get_categories_ratios_for_validation() -> dict:
    categories_ratios = {
        "Computer Science": 0.15,
        "Physics": 0.15,
        "Mathematics": 0.1,
        "Biology": 0.1,
        "Medicine": 0.1,
        "Astronomy": 0.05,
        "Engineering": 0.05,
        "Chemistry": 0.05,
        "Economics": 0.05,
        "Psychology": 0.05,
        "Materials Science": 0.05,
        "Earth Science": 0.05,
        "Linguistics": 0.05,
    }
    return categories_ratios


def get_remaining_categories_to_fill(
    floored_categories: list,
    n_remaining: int,
    rng: np.random.RandomState,
) -> list:
    assert len(floored_categories) >= n_remaining
    remaining_categories_to_fill = rng.choice(
        floored_categories,
        size=n_remaining,
        replace=False,
    ).tolist()
    return remaining_categories_to_fill


def get_categories_counts(
    categories_ratios: dict,
    n_total: int,
    random_state: int = None,
    assert_no_flooring: bool = False,
) -> dict:
    exact_floats = {cat: n_total * ratio for cat, ratio in categories_ratios.items()}
    min_counts = {cat: floor(exact_float) for cat, exact_float in exact_floats.items()}
    categories_counts = min_counts.copy()
    floored_categories = [
        cat
        for cat in categories_ratios.keys()
        if (exact_floats[cat] - min_counts[cat]) > FLOAT_PRECISION
    ]
    n_total_min_counts = sum(min_counts.values())
    n_remaining = n_total - n_total_min_counts
    assert n_remaining >= 0
    if assert_no_flooring:
        assert n_remaining <= 0 and len(floored_categories) == 0
    if n_remaining > 0:
        assert random_state is not None
        rng = np.random.RandomState(random_state)
        remaining_categories_to_fill = get_remaining_categories_to_fill(
            floored_categories=floored_categories,
            n_remaining=n_remaining,
            rng=rng,
        )
        for cat in remaining_categories_to_fill:
            categories_counts[cat] += 1
    assert sum(categories_counts.values()) == n_total
    assert all(categories_counts[cat] >= min_counts[cat] for cat in categories_ratios.keys())
    assert all(categories_counts[cat] <= min_counts[cat] + 1 for cat in categories_ratios.keys())
    return categories_counts


def draw_categories_samples_ids(
    papers: pd.DataFrame,
    categories_counts: dict,
    random_state: int = 42,
    sort_samples: bool = False,
    papers_ids_to_exclude: list = None,
) -> tuple:
    papers = papers[papers["l1"].notna()].reset_index(drop=True)
    if papers_ids_to_exclude is not None:
        papers = papers[~papers["paper_id"].isin(papers_ids_to_exclude)].reset_index(drop=True)
    rng = np.random.RandomState(random_state)
    categories_samples_ids = {}
    categories_samples_ids_flattened = []
    for category, count in categories_counts.items():
        category_papers = papers[papers["l1"] == category]
        category_samples_ids = category_papers.sample(n=count, random_state=rng, replace=False)[
            "paper_id"
        ].tolist()
        if sort_samples:
            category_samples_ids = sorted(category_samples_ids)
            assert category_samples_ids == sorted(category_samples_ids)
        assert len(category_samples_ids) == count
        categories_samples_ids[category] = category_samples_ids
        categories_samples_ids_flattened.extend(category_samples_ids)
    categories_samples_ids_flattened = sorted(categories_samples_ids_flattened)
    return categories_samples_ids, categories_samples_ids_flattened


def get_categories_samples_ids(
    papers: pd.DataFrame,
    n_categories_samples: int,
    random_state: int,
    papers_ids_to_exclude: list = None,
) -> tuple:
    if n_categories_samples <= 0:
        return {}, []
    categories_ratios = get_categories_ratios_for_validation()
    categories_counts = get_categories_counts(
        categories_ratios=categories_ratios,
        n_total=(n_categories_samples * 2),
        random_state=None,
        assert_no_flooring=True,
    )
    categories_samples_ids, categories_samples_ids_flattened = draw_categories_samples_ids(
        papers=papers,
        categories_counts=categories_counts,
        random_state=random_state,
        sort_samples=False,
        papers_ids_to_exclude=papers_ids_to_exclude,
    )
    assert len(categories_samples_ids_flattened) == (n_categories_samples * 2)
    return categories_samples_ids, categories_samples_ids_flattened


def get_user_categories_ratios(
    categories_ratios: dict = None, categories_to_exclude: list = []
) -> dict:
    if type(categories_to_exclude) is pd.DataFrame:
        categories_to_exclude = categories_to_exclude["category"].tolist()
    if categories_ratios is None:
        categories_ratios = get_categories_ratios_for_validation()
    user_categories_ratios = {
        cat: ratio for cat, ratio in categories_ratios.items() if cat not in categories_to_exclude
    }
    total_ratio = sum(user_categories_ratios.values())
    user_categories_ratios = {
        cat: ratio / total_ratio for cat, ratio in user_categories_ratios.items()
    }
    total_sum = sum(user_categories_ratios.values())
    assert (
        abs(total_sum - 1.0) < FLOAT_PRECISION
    ), f"User Categories Ratios are {total_sum}, expected ~1.0"
    return user_categories_ratios


def select_user_categories_samples_ids(
    categories_samples_ids: dict,
    user_categories_counts: dict,
    papers_ids_to_exclude: list = None,
    sort_samples: bool = False,
) -> list:
    user_categories_samples_ids = []
    for category, count in user_categories_counts.items():
        category_samples_ids = categories_samples_ids[category]
        if papers_ids_to_exclude is not None:
            category_samples_ids = [
                pid for pid in category_samples_ids if pid not in papers_ids_to_exclude
            ]
        assert count <= len(category_samples_ids)
        user_categories_samples_ids.extend(category_samples_ids[:count])
    if sort_samples:
        user_categories_samples_ids = sorted(user_categories_samples_ids)
    assert len(user_categories_samples_ids) == sum(user_categories_counts.values())
    return user_categories_samples_ids


def get_user_categories_samples_ids(
    categories_samples_ids: dict,
    n_categories_samples: int,
    random_state: int,
    user_significant_categories: list = None,
    sort_samples: bool = False,
    user_categories_ratios: dict = None,
) -> list:
    if n_categories_samples <= 0:
        return []
    if user_categories_ratios is None:
        assert user_significant_categories is not None
        user_categories_ratios = get_user_categories_ratios(
            categories_to_exclude=user_significant_categories
        )
    user_categories_counts = get_categories_counts(
        categories_ratios=user_categories_ratios,
        n_total=n_categories_samples,
        random_state=random_state,
        assert_no_flooring=False,
    )
    user_categories_samples_ids = select_user_categories_samples_ids(
        categories_samples_ids=categories_samples_ids,
        user_categories_counts=user_categories_counts,
        papers_ids_to_exclude=None,
        sort_samples=sort_samples,
    )
    assert len(user_categories_samples_ids) == n_categories_samples
    return user_categories_samples_ids


def get_cache_papers_ids_categories_cache(
    papers: pd.DataFrame,
    n_cache: int,
    random_state: int,
    papers_ids_to_exclude: list = None,
    categories_ratios: dict = None,
) -> list:
    if categories_ratios is None:
        categories_ratios = get_categories_ratios_for_validation()
    categories_counts = get_categories_counts(
        categories_ratios=categories_ratios,
        n_total=n_cache,
        random_state=random_state,
        assert_no_flooring=True,
    )
    _, cache_papers_ids_flattened = draw_categories_samples_ids(
        papers=papers,
        categories_counts=categories_counts,
        random_state=random_state,
        sort_samples=True,
        papers_ids_to_exclude=papers_ids_to_exclude,
    )
    return cache_papers_ids_flattened


def get_val_negative_samples_ids(
    papers: pd.DataFrame,
    n_categories_samples: int,
    random_state: int,
    papers_ids_to_exclude: list,
) -> list:
    return get_categories_samples_ids(
        papers=papers,
        n_categories_samples=n_categories_samples,
        random_state=random_state,
        papers_ids_to_exclude=papers_ids_to_exclude,
    )[0]


def get_user_val_negative_samples(
    val_negative_samples_ids: list,
    n_negative_samples: int,
    random_state: int,
    user_categories_ratios: dict,
    embedding: Embedding = None,
) -> dict:
    val_negative_samples_ids = get_user_categories_samples_ids(
        categories_samples_ids=val_negative_samples_ids,
        n_categories_samples=n_negative_samples,
        random_state=random_state,
        sort_samples=True,
        user_categories_ratios=user_categories_ratios,
    )
    negative_samples_embeddings = None
    if embedding is not None:
        negative_samples_embeddings = embedding.matrix[embedding.get_idxs(val_negative_samples_ids)]
    return {
        "val_negative_samples_ids": val_negative_samples_ids,
        "val_negative_samples_embeddings": negative_samples_embeddings,
    }


def get_cache_papers_ids_old_cache(
    papers: pd.DataFrame,
    n_cache: int,
    random_state: int,
    papers_ids_to_exclude: list = None,
) -> list:
    papers_ids = papers[papers["in_cache"]]["paper_id"].tolist()
    if papers_ids_to_exclude is not None:
        papers_ids = [pid for pid in papers_ids if pid not in papers_ids_to_exclude]
    assert papers_ids == sorted(papers_ids) and len(papers_ids) == len(set(papers_ids))
    assert len(papers_ids) >= n_cache
    if len(papers_ids) > n_cache:
        rng = random.Random(random_state)
        papers_ids = sorted(rng.sample(papers_ids, n_cache))
    assert len(papers_ids) == n_cache
    assert papers_ids == sorted(papers_ids) and len(papers_ids) == len(set(papers_ids))
    return papers_ids


def get_cache_papers_ids_random_cache(
    papers: pd.DataFrame,
    n_cache: int,
    random_state: int,
    papers_ids_to_exclude: list = None,
) -> list:
    papers_ids = papers["paper_id"].tolist()
    if papers_ids_to_exclude is not None:
        papers_ids = [pid for pid in papers_ids if pid not in papers_ids_to_exclude]
    assert papers_ids == sorted(papers_ids) and len(papers_ids) == len(set(papers_ids))
    assert len(papers_ids) >= n_cache
    rng = random.Random(random_state)
    papers_ids = sorted(rng.sample(papers_ids, n_cache))
    assert len(papers_ids) == n_cache
    assert papers_ids == sorted(papers_ids) and len(papers_ids) == len(set(papers_ids))
    return papers_ids


def get_cache_papers_ids(
    cache_type: str,
    papers: pd.DataFrame,
    n_cache: int,
    random_state: int,
    papers_ids_to_exclude: list = None,
) -> list:
    if n_cache <= 0:
        return []
    if cache_type == "categories_cache":
        func = get_cache_papers_ids_categories_cache
    elif cache_type == "old_cache":
        func = get_cache_papers_ids_old_cache
    elif cache_type == "random_cache":
        func = get_cache_papers_ids_random_cache
    else:
        raise ValueError(f"Invalid cache_type: {cache_type}.")
    cache_papers_ids = func(
        papers=papers,
        n_cache=n_cache,
        random_state=random_state,
        papers_ids_to_exclude=papers_ids_to_exclude,
    )
    assert isinstance(cache_papers_ids, list)
    assert cache_papers_ids == sorted(cache_papers_ids)
    assert len(cache_papers_ids) == len(set(cache_papers_ids))
    return cache_papers_ids


def get_cache_papers_ids_full(
    papers: pd.DataFrame,
    cache_type: str,
    n_cache: int,
    random_state: int,
    n_categories_cache: int,
) -> tuple:
    cache_papers_ids = get_cache_papers_ids(
        cache_type=cache_type,
        papers=papers,
        n_cache=n_cache,
        random_state=random_state,
    )
    cache_papers_categories_ids = get_categories_samples_ids(
        papers=papers,
        n_categories_samples=n_categories_cache,
        random_state=random_state,
        papers_ids_to_exclude=cache_papers_ids,
    )[0]
    return cache_papers_categories_ids, cache_papers_ids


def get_user_cache_papers_ids(
    cache_type: str,
    cache_papers_ids: list,
    papers_ids_to_exclude: list = None,
) -> list:
    if cache_type in ["categories_cache", "old_cache", "random_cache"]:
        if papers_ids_to_exclude is not None:
            user_cache_papers_ids = [
                pid for pid in cache_papers_ids if pid not in papers_ids_to_exclude
            ]
        else:
            user_cache_papers_ids = cache_papers_ids
    else:
        raise ValueError(f"Invalid cache_type: {cache_type}.")
    assert user_cache_papers_ids == sorted(user_cache_papers_ids)
    assert len(user_cache_papers_ids) == len(set(user_cache_papers_ids))
    return user_cache_papers_ids


def get_user_cache_papers(
    cache_type: str,
    cache_papers_ids: list,
    cache_papers_categories_ids: list,
    n_categories_cache: int,
    random_state: int,
    papers_ids_to_exclude_from_cache: list,
    user_categories_ratios: dict,
    embedding: Embedding = None,
) -> dict:
    user_cache_papers_ids = get_user_cache_papers_ids(
        cache_type=cache_type,
        cache_papers_ids=cache_papers_ids,
    )
    user_cache_papers_categories_ids = get_user_categories_samples_ids(
        categories_samples_ids=cache_papers_categories_ids,
        n_categories_samples=n_categories_cache,
        random_state=random_state,
        sort_samples=False,
        user_categories_ratios=user_categories_ratios,
    )
    cache_papers_ids = sorted(
        list(set(user_cache_papers_categories_ids) | set(user_cache_papers_ids))
    )
    cache_papers_ids = [
        paper_id
        for paper_id in cache_papers_ids
        if paper_id not in papers_ids_to_exclude_from_cache
    ]
    assert user_cache_papers_ids == sorted(user_cache_papers_ids)
    assert len(user_cache_papers_ids) == len(set(user_cache_papers_ids))
    cache_n = len(cache_papers_ids)
    y_cache = np.zeros(cache_n, dtype=LABEL_DTYPE)
    cache_embedding_idxs = None
    if embedding is not None:
        cache_embedding_idxs = embedding.get_idxs(cache_papers_ids)
    return {
        "user_cache_papers_categories_ids": user_cache_papers_categories_ids,
        "cache_embedding_idxs": cache_embedding_idxs,
        "cache_n": cache_n,
        "y_cache": y_cache,
    }


def split_ratings(user_ratings: pd.DataFrame) -> tuple:
    if "split" not in user_ratings.columns:
        raise ValueError("User ratings DataFrame must contain 'split' column.")
    train_mask = user_ratings["split"] == "train"
    val_mask = user_ratings["split"] == "val"
    removed_mask = user_ratings["split"] == "removed"
    train_ratings = user_ratings.loc[train_mask]
    val_ratings = user_ratings.loc[val_mask]
    removed_ratings = user_ratings.loc[removed_mask]
    assert len(train_ratings) + len(val_ratings) + len(removed_ratings) == len(user_ratings)
    removed_ratings = removed_ratings[removed_ratings["rating"] == 0]
    removed_ratings = removed_ratings.iloc[:N_NEGRATED_RANKING]
    assert train_ratings["time"].is_monotonic_increasing
    assert val_ratings["time"].is_monotonic_increasing
    assert removed_ratings["time"].is_monotonic_increasing
    return train_ratings, val_ratings, removed_ratings


def split_negrated_ranking(
    train_ratings: pd.DataFrame, val_ratings: pd.DataFrame, removed_ratings: pd.DataFrame
) -> tuple:
    train_negrated_ranking = train_ratings[train_ratings["rating"] == 0]
    val_negrated_ranking = pd.concat([val_ratings[val_ratings["rating"] == 0], removed_ratings])
    return (
        train_negrated_ranking.reset_index(drop=True),
        val_negrated_ranking.reset_index(drop=True),
    )


def store_user_info_initial(user_ratings: pd.DataFrame, cache_n: int) -> dict:
    user_info = {"n_cache": cache_n, "n_base": 0, "n_zerorated": 0}
    user_info["n_posrated"] = len(user_ratings[user_ratings["rating"] == 1])
    user_info["n_negrated"] = len(user_ratings[user_ratings["rating"] == 0])
    assert user_info["n_posrated"] + user_info["n_negrated"] == len(user_ratings)
    user_info["n_sessions"] = user_ratings["session_id"].nunique()
    max_time, min_time = user_ratings["time"].max(), user_ratings["time"].min()
    user_info["time_range_days"] = (max_time - min_time).days
    return user_info


def update_user_info_split(
    user_info: dict, train_ratings: pd.DataFrame, val_ratings: pd.DataFrame
) -> None:
    user_info_split = {}
    n_train, n_val = len(train_ratings), len(val_ratings)
    user_info_split["train_rated_ratio"] = n_train / (n_train + n_val)
    splits = {"train": train_ratings, "val": val_ratings}
    for split_name, split_ratings in splits.items():
        n_posrated = len(split_ratings[split_ratings["rating"] == 1])
        user_info_split[f"n_posrated_{split_name}"] = n_posrated
        n_negrated = len(split_ratings[split_ratings["rating"] == 0])
        user_info_split[f"n_negrated_{split_name}"] = n_negrated
        assert n_posrated + n_negrated == len(split_ratings)
        user_info_split[f"n_sessions_{split_name}"] = split_ratings["session_id"].nunique()
        max_time, min_time = split_ratings["time"].max(), split_ratings["time"].min()
        user_info_split[f"time_range_days_{split_name}"] = (max_time - min_time).days
        pos_ratings = split_ratings[split_ratings["rating"] == 1]
        user_info_split[f"n_sessions_pos_{split_name}"] = pos_ratings["session_id"].nunique()
        pos_max_time, pos_min_time = pos_ratings["time"].max(), pos_ratings["time"].min()
        user_info_split[f"time_range_days_pos_{split_name}"] = (pos_max_time - pos_min_time).days
    for key, value in user_info_split.items():
        if key in user_info:
            assert isinstance(user_info[key], list)
            user_info[key].append(value)
        else:
            user_info[key] = [value]


def save_user_info(outputs_dir: Path, user_id: int, user_info: dict) -> None:
    for key, value in user_info.items():
        if isinstance(value, list):
            if len(value) == 1:
                user_info[key] = value[0]
            else:
                user_info[key] = sum(value) / len(value)
    folder = outputs_dir / "tmp" / f"user_{user_id}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    json.dump(user_info, open(folder / "user_info.json", "w"), indent=1)


def save_user_results(outputs_dir: Path, user_id: int, user_results_dict: dict) -> None:
    folder = outputs_dir / "tmp" / f"user_{user_id}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    try:
        json.dump(user_results_dict, open(folder / "user_results.json", "w"), indent=1)
    except:
        print(f"Error saving user {user_id} results.")
        raise


def save_user_predictions(outputs_dir: Path, user_id: int, user_predictions_dict: dict) -> None:
    folder = outputs_dir / "users_predictions" / f"user_{user_id}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    json.dump(user_predictions_dict, open(folder / "user_predictions.json", "w"), indent=1)


def save_user_coefs(outputs_dir: Path, user_id: int, user_coefs: np.ndarray) -> None:
    folder = outputs_dir / "tmp" / f"user_{user_id}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(folder / "user_coefs.npy", user_coefs)
