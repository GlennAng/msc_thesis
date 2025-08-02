import random
from math import floor

import numpy as np
import pandas as pd

FLOAT_PRECISION = 1e-10
LABEL_DTYPE = np.int64


def load_negrated_ranking_idxs_for_user_random(
    ratings: pd.DataFrame,
    pos_idxs: list,
    neg_idxs: list,
    negrated_ranking_idxs: np.ndarray,
    random_state: int,
    same_negrated_for_all_pos: bool,
) -> np.ndarray:
    assert len(pos_idxs) == negrated_ranking_idxs.shape[0]
    assert sorted(pos_idxs + neg_idxs) == list(range(len(ratings)))
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
    ratings: pd.DataFrame, pos_idxs: list, neg_idxs: list, negrated_ranking_idxs: np.ndarray
) -> np.ndarray:
    assert len(pos_idxs) == negrated_ranking_idxs.shape[0]
    assert sorted(pos_idxs + neg_idxs) == list(range(len(ratings)))
    n_negrated = negrated_ranking_idxs.shape[1]
    for i, pos_idx in enumerate(pos_idxs):
        pos_time = ratings.loc[pos_idx, "time"]
        time_diffs = np.abs(ratings.loc[neg_idxs, "time"] - pos_time)
        closest_time_diffs = np.argsort(time_diffs)[:n_negrated]
        closest_neg_idxs = sorted([neg_idxs[idx] for idx in closest_time_diffs])
        negrated_ranking_idxs[i] = closest_neg_idxs
    return negrated_ranking_idxs


def load_negrated_ranking_idxs_for_user(
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    random_neg: bool,
    random_state: int,
    same_negrated_for_all_pos: bool,
) -> tuple:
    train_pos_idxs = train_ratings[train_ratings["rating"] > 0].index.values.tolist()
    train_neg_idxs = train_ratings[train_ratings["rating"] <= 0].index.values.tolist()
    val_pos_idxs = val_ratings[val_ratings["rating"] > 0].index.values.tolist()
    val_neg_idxs = val_ratings[val_ratings["rating"] <= 0].index.values.tolist()
    n_pos_train, n_pos_val, n_neg_train, n_neg_val = (
        len(train_pos_idxs),
        len(val_pos_idxs),
        len(train_neg_idxs),
        len(val_neg_idxs),
    )
    min_n_negrated_train, min_n_negrated_val = min(4, n_neg_train), min(4, n_neg_val)
    train_negrated_ranking_idxs = np.zeros((n_pos_train, min_n_negrated_train), dtype=np.int64)
    val_negrated_ranking_idxs = np.zeros((n_pos_val, min_n_negrated_val), dtype=np.int64)
    if random_neg:
        train_negrated_ranking_idxs = load_negrated_ranking_idxs_for_user_random(
            train_ratings,
            train_pos_idxs,
            train_neg_idxs,
            train_negrated_ranking_idxs,
            random_state,
            same_negrated_for_all_pos,
        )
        val_negrated_ranking_idxs = load_negrated_ranking_idxs_for_user_random(
            val_ratings,
            val_pos_idxs,
            val_neg_idxs,
            val_negrated_ranking_idxs,
            random_state,
            same_negrated_for_all_pos,
        )
    else:
        train_negrated_ranking_idxs = load_negrated_ranking_idxs_for_user_timesort(
            train_ratings, train_pos_idxs, train_neg_idxs, train_negrated_ranking_idxs
        )
        val_negrated_ranking_idxs = load_negrated_ranking_idxs_for_user_timesort(
            val_ratings, val_pos_idxs, val_neg_idxs, val_negrated_ranking_idxs
        )
    return train_negrated_ranking_idxs, val_negrated_ranking_idxs


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


def count_to_str(count: int) -> str:
    return f"{count:,}"


def get_categories_distribution(
    papers_ids_to_categories: dict = None, print_results: bool = True
) -> tuple:
    unique_categories = set(papers_ids_to_categories.values())
    categories_counts = {category: 0 for category in unique_categories}
    n_total = 0
    for _, value in papers_ids_to_categories.items():
        categories_counts[value] += 1
        n_total += 1
    categories_counts = {category: count / n_total for category, count in categories_counts.items()}
    if print_results:
        categories_counts_copy = categories_counts.copy()
        categories_counts_copy["Total"] = 1.0
        categories_counts_copy = sorted(
            categories_counts_copy.items(), key=lambda x: x[1], reverse=True
        )
        for category, count in categories_counts_copy:
            print(f"{category}: {count:.2%} ({count_to_str(int(count * n_total))})")
        print("____________________________________________________________")
    return categories_counts, n_total


def get_categories_distribution_database(
    papers: pd.DataFrame, level: str = "l1", print_results: bool = True
) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    return get_categories_distribution(papers_ids_to_categories, print_results)


def get_categories_distribution_ratings(
    papers: pd.DataFrame, level: str = "l1", print_results: bool = True
) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    papers = papers[papers["in_ratings"]]
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    return get_categories_distribution(papers_ids_to_categories, print_results)


def get_categories_distribution_old_cache(
    papers: pd.DataFrame, level: str = "l1", print_results: bool = True
) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    cache_papers_ids = get_cache_papers_ids_old_cache(
        papers=papers,
        n_cache=5000,
        random_state=42,
        papers_ids_to_exclude=None,
    )
    papers = papers[papers["paper_id"].isin(cache_papers_ids)]
    if papers.empty:
        raise ValueError("No papers found in the cache.")
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    return get_categories_distribution(papers_ids_to_categories, print_results)


def get_categories_distribution_random_cache(
    papers: pd.DataFrame, level: str = "l1", print_results: bool = True
) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    cache_papers_ids = get_cache_papers_ids_random_cache(
        papers=papers,
        n_cache=5000,
        random_state=42,
        papers_ids_to_exclude=None,
    )
    papers = papers[papers["paper_id"].isin(cache_papers_ids)]
    if papers.empty:
        raise ValueError("No papers found in the cache.")
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    return get_categories_distribution(papers_ids_to_categories, print_results)


def get_categories_distribution_ratings_specific_users(
    papers: pd.DataFrame,
    users_ids: list,
    users_ratings: pd.DataFrame,
    level: str = "l1",
    print_results: bool = True,
) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    users_ratings = users_ratings[users_ratings["user_id"].isin(users_ids)].reset_index(drop=True)
    users_ratings = users_ratings[users_ratings["rating"] == 1]
    papers_users_ratings_ids = papers[papers["paper_id"].isin(users_ratings["paper_id"])]
    papers = papers[papers["paper_id"].isin(papers_users_ratings_ids["paper_id"])]
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    return get_categories_distribution(papers_ids_to_categories, print_results)


def get_categories_distribution_relevant_papers(
    papers: pd.DataFrame,
    level: str = "l1",
    print_results: bool = True,
) -> tuple:
    from ....src.load_files import load_relevant_papers_ids

    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    relevant_papers_ids = load_relevant_papers_ids()
    papers = papers[papers["paper_id"].isin(relevant_papers_ids)]
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    return get_categories_distribution(papers_ids_to_categories, print_results)


if __name__ == "__main__":

    from ....src.load_files import (
        load_papers,
        load_session_based_users_ids,
        load_users_ratings,
        load_users_significant_categories,
    )

    users_ids = load_session_based_users_ids()
    users_ids_non_cs = load_session_based_users_ids(
        select_non_cs_users_only=True,
    )
    users_ids_cs = sorted(list(set(users_ids) - set(users_ids_non_cs)))
    papers = load_papers()
    users_ratings = load_users_ratings(relevant_users_ids=users_ids)
    users_significant_categories = load_users_significant_categories(relevant_users_ids=users_ids)

    print("Categories Distribution for Database:")
    get_categories_distribution_database(papers=papers, level="l1", print_results=True)
    print("\nCategories Distribution for Ratings:")
    get_categories_distribution_ratings(papers=papers, level="l1", print_results=True)
    print("\nCategories Distribution for CS Users:")
    get_categories_distribution_ratings_specific_users(
        papers=papers, users_ids=users_ids_cs, users_ratings=users_ratings
    )
    print("\nCategories Distribution for Non-CS Users:")
    get_categories_distribution_ratings_specific_users(
        papers=papers, users_ids=users_ids_non_cs, users_ratings=users_ratings
    )
    print("\nCategories Distribution for Old Cache:")
    get_categories_distribution_old_cache(papers=papers, level="l1", print_results=True)
    print("\nCategories Distribution for Relevant Papers:")
    get_categories_distribution_relevant_papers(papers=papers, level="l1", print_results=True)
