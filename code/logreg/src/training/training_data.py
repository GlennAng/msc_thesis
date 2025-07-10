import random
from math import ceil, floor

import numpy as np
import pandas as pd

from ....src.load_files import load_users_significant_categories
from ..embeddings.embedding import Embedding

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


def load_global_cache(
    embedding: Embedding, papers: pd.DataFrame, max_cache: int = None, random_state: int = None
) -> tuple:
    global_cache_ids = get_global_cache_papers_ids(papers, max_cache, random_state)
    global_cache_idxs = embedding.get_idxs(global_cache_ids)
    global_cache_n = len(global_cache_idxs)
    y_global_cache = np.zeros(global_cache_n, dtype=LABEL_DTYPE)
    return global_cache_ids, global_cache_idxs, global_cache_n, y_global_cache


def get_global_cache_papers_ids(
    papers: pd.DataFrame, max_cache: int = None, random_state: int = None
) -> list:
    cache = papers[papers["in_cache"]]["paper_id"].tolist()
    n_cache = len(cache)
    max_cache = n_cache if max_cache is None else min(max_cache, n_cache)
    if n_cache < max_cache:
        raise ValueError(
            f"Required cache size ({max_cache}) is greater than the number of valid cache papers ({n_cache})."
        )
    elif n_cache > max_cache:
        cache = sorted(cache)
        rng = random.Random(random_state)
        cache = rng.sample(cache, max_cache)
    return sorted(cache)


def load_filtered_cache_for_user(
    embedding: Embedding,
    cache_papers_ids: list,
    rated_ids: list,
    max_cache: int = None,
    random_state: int = None,
) -> tuple:
    user_filtered_cache_ids = get_filtered_cache_papers_ids_for_user(
        cache_papers_ids, rated_ids, max_cache, random_state
    )
    user_filtered_cache_idxs = embedding.get_idxs(user_filtered_cache_ids)
    user_filtered_cache_n = len(user_filtered_cache_idxs)
    y_user_filtered_cache = np.zeros(user_filtered_cache_n, dtype=LABEL_DTYPE)
    return (
        user_filtered_cache_ids,
        user_filtered_cache_idxs,
        user_filtered_cache_n,
        y_user_filtered_cache,
    )


def get_filtered_cache_papers_ids_for_user(
    cache_papers_ids: list, rated_ids: list, max_cache: int = None, random_state: int = None
) -> list:
    user_filtered_cache_ids = [
        paper_id for paper_id in cache_papers_ids if paper_id not in rated_ids
    ]
    n_user_filtered_cache = len(user_filtered_cache_ids)
    max_cache = (
        n_user_filtered_cache if max_cache is None else min(max_cache, n_user_filtered_cache)
    )
    if n_user_filtered_cache < max_cache:
        raise ValueError(
            f"Required cache size ({max_cache}) is greater than the number of valid cache papers ({n_user_filtered_cache})."
        )
    elif n_user_filtered_cache > max_cache:
        user_filtered_cache_ids = sorted(user_filtered_cache_ids)
        rng = random.Random(random_state)
        user_filtered_cache_ids = rng.sample(user_filtered_cache_ids, max_cache)
    return sorted(user_filtered_cache_ids)


def get_categories_ratios() -> dict:
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


def get_papers_mask(
    papers: pd.DataFrame,
    exclude_in_ratings: bool = True,
    exclude_in_cache: bool = True,
    exclude_papers: list = None,
) -> pd.Series:
    mask = pd.Series(True, index=papers.index)
    mask &= papers["l1"].notna()
    if exclude_in_ratings:
        mask &= ~papers["in_ratings"]
    if exclude_in_cache:
        mask &= ~papers["in_cache"]
    if exclude_papers is not None:
        mask &= ~papers["paper_id"].isin(exclude_papers)
    return mask


def get_negative_samples_ids_per_category(
    papers: pd.DataFrame,
    n_negative_samples: int,
    random_state: int,
    papers_to_exclude: list = None,
    categories_ratios: dict = None,
    scalar_factor: float = 1.0,
) -> tuple:
    mask = get_papers_mask(papers, exclude_papers=papers_to_exclude)
    potential_samples = papers[mask].reset_index(drop=True)
    if categories_ratios is None:
        categories_ratios = get_categories_ratios()
    categories = list(categories_ratios.keys())
    rng = random.Random(random_state)
    negative_samples_ids_per_category = {}
    for cat in categories:
        potential_samples_ids_category = potential_samples[potential_samples["l1"] == cat][
            "paper_id"
        ].tolist()
        n_negative_samples_category = ceil(
            scalar_factor * n_negative_samples * categories_ratios[cat]
        )
        if len(potential_samples_ids_category) < n_negative_samples_category:
            raise ValueError(
                f"Not enough potential papers in category '{cat}' to sample {n_negative_samples_category} negative samples."
            )
        negative_samples_ids_category = rng.sample(
            potential_samples_ids_category, n_negative_samples_category
        )
        negative_samples_ids_per_category[cat] = sorted(negative_samples_ids_category)
    negative_samples_ids_list = []
    for negative_samples_ids_category in negative_samples_ids_per_category.values():
        negative_samples_ids_list.extend(negative_samples_ids_category)
    negative_samples_ids_list = sorted(negative_samples_ids_list)
    return negative_samples_ids_per_category, negative_samples_ids_list


def fill_n_samples_per_category(
    negative_samples_ids_per_category: dict,
    n_samples_per_category: dict,
    user_categories_ratios: dict,
    n_negative_samples: int,
    rng: random.Random,
) -> dict:
    n_samples_total = sum(n_samples_per_category.values())
    while n_samples_total < n_negative_samples:
        category = rng.choice(list(user_categories_ratios.keys()))
        if n_samples_per_category[category] < len(negative_samples_ids_per_category[category]):
            n_samples_per_category[category] += 1
            n_samples_total += 1
    return n_samples_per_category


def get_negative_samples_ids(
    negative_samples_ids_per_category: dict,
    users_significant_categories: pd.DataFrame,
    n_negative_samples: int,
    random_state: int,
    user_specific: bool = True,
    categories_ratios: dict = None,
) -> np.ndarray:
    if categories_ratios is None:
        categories_ratios = get_categories_ratios()
    users_ids = users_significant_categories["user_id"].unique().tolist()
    assert users_ids == sorted(users_ids), "Users IDs must be sorted."
    n_users = len(users_ids)
    negative_samples_ids = np.zeros(shape=(n_users, n_negative_samples), dtype=np.int64)
    rng = random.Random(random_state)
    for category, samples in negative_samples_ids_per_category.items():
        rng.shuffle(samples)
    for i, user_id in enumerate(users_ids):
        if user_specific:
            significant_categories_for_user = users_significant_categories[
                users_significant_categories["user_id"] == user_id
            ]["category"]
            if isinstance(significant_categories_for_user, str):
                significant_categories_for_user = [significant_categories_for_user]
            else:
                significant_categories_for_user = significant_categories_for_user.tolist()
            user_categories_ratios = {
                cat: ratio
                for cat, ratio in categories_ratios.items()
                if cat not in significant_categories_for_user
            }
            total_ratio = sum(user_categories_ratios.values())
            user_categories_ratios = {
                cat: ratio / total_ratio for cat, ratio in user_categories_ratios.items()
            }
        else:
            user_categories_ratios = categories_ratios
        n_samples_per_category = {
            cat: floor(n_negative_samples * ratio) for cat, ratio in user_categories_ratios.items()
        }
        n_samples_per_category = fill_n_samples_per_category(
            negative_samples_ids_per_category=negative_samples_ids_per_category,
            n_samples_per_category=n_samples_per_category,
            user_categories_ratios=user_categories_ratios,
            n_negative_samples=n_negative_samples,
            rng=rng,
        )
        user_negative_samples_ids = []
        for category, n_samples_category in n_samples_per_category.items():
            if n_samples_category <= 0:
                continue
            user_negative_samples_ids.extend(
                negative_samples_ids_per_category[category][:n_samples_category]
            )
        negative_samples_ids[i] = sorted(user_negative_samples_ids)
    return negative_samples_ids


def get_val_cache_attached_negative_samples_ids(
    users_ratings: pd.DataFrame,
    papers: pd.DataFrame,
    n_val_negative_samples: int,
    ranking_random_state: int,
    n_cache_attached: int,
    cache_random_state: int,
    cache_attached_user_specific: bool = True,
    return_all_papers_ids: bool = False,
) -> tuple:
    users_ids = users_ratings["user_id"].unique().tolist()
    n_users = len(users_ids)
    users_significant_categories = load_users_significant_categories(
        relevant_users_ids=users_ids,
    )
    val_negative_samples_ids_per_category, val_negative_samples_ids_list = (
        get_negative_samples_ids_per_category(
            papers=papers,
            n_negative_samples=n_val_negative_samples,
            random_state=ranking_random_state,
            papers_to_exclude=None,
            categories_ratios=None,
            scalar_factor=2.0,
        )
    )
    val_negative_samples_ids = get_negative_samples_ids(
        negative_samples_ids_per_category=val_negative_samples_ids_per_category,
        users_significant_categories=users_significant_categories,
        n_negative_samples=n_val_negative_samples,
        random_state=ranking_random_state,
        user_specific=True,
    )
    assert val_negative_samples_ids.shape == (n_users, n_val_negative_samples)
    cache_attached_papers_ids = None
    if n_cache_attached > 0:
        cache_attached_papers_ids_per_category, cache_attached_papers_ids_list = (
            get_negative_samples_ids_per_category(
                papers=papers,
                n_negative_samples=n_cache_attached,
                random_state=cache_random_state,
                papers_to_exclude=val_negative_samples_ids_list,
                categories_ratios=None,
                scalar_factor=(2.0 if cache_attached_user_specific else 1.0),
            )
        )
        cache_attached_papers_ids = get_negative_samples_ids(
            negative_samples_ids_per_category=cache_attached_papers_ids_per_category,
            users_significant_categories=users_significant_categories,
            n_negative_samples=n_cache_attached,
            random_state=cache_random_state,
            user_specific=cache_attached_user_specific,
        )
        assert cache_attached_papers_ids.shape == (n_users, n_cache_attached)
    all_papers_ids = None
    if return_all_papers_ids:
        all_papers_ids = val_negative_samples_ids_list
        if cache_attached_papers_ids is not None:
            all_papers_ids.extend(cache_attached_papers_ids_list)
        all_papers_ids = sorted(all_papers_ids)
    return val_negative_samples_ids, cache_attached_papers_ids, all_papers_ids


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


def get_categories_distribution_dataset(
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


def get_categories_distribution_cache(
    papers: pd.DataFrame, level: str = "l1", print_results: bool = True
) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    cache_papers_ids = get_global_cache_papers_ids(papers, max_cache=5000, random_state=42)
    papers = papers[papers["paper_id"].isin(cache_papers_ids)]
    if papers.empty:
        raise ValueError("No papers found in the cache.")
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    return get_categories_distribution(papers_ids_to_categories, print_results)


def get_categories_distribution_val_negative_samples(
    papers: pd.DataFrame,
    users_ratings: pd.DataFrame,
    level: str = "l1",
    print_results: bool = True,
) -> dict:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    users_ids = users_ratings["user_id"].unique().tolist()
    assert users_ids == sorted(users_ids), "Users IDs must be sorted."
    val_negative_samples_ids, _ = get_val_cache_attached_negative_samples_ids(
        users_ratings=users_ratings,
        papers=papers,
        n_val_negative_samples=100,
        ranking_random_state=42,
        n_cache_attached=0,
        cache_random_state=42,
        cache_attached_user_specific=False,
    )
    categories_distributions_dict = {}
    for i, user_id in enumerate(users_ids):
        user_val_negative_samples_ids = val_negative_samples_ids[i].tolist()
        assert user_val_negative_samples_ids == sorted(
            user_val_negative_samples_ids
        ), "User negative samples IDs must be sorted."
        user_papers = papers[papers["paper_id"].isin(user_val_negative_samples_ids)]
        if user_papers.empty:
            raise ValueError(
                f"No papers found for user {user_id} in the validation negative samples."
            )
        user_papers_ids_to_categories = user_papers.set_index("paper_id")[level].to_dict()
        categories_distributions_dict[user_id] = get_categories_distribution(
            user_papers_ids_to_categories, print_results=False
        )[0]
    return categories_distributions_dict


if __name__ == "__main__":
    from ....src.load_files import (
        load_finetuning_users,
        load_papers,
        load_users_ratings,
    )

    papers = load_papers()
    get_categories_distribution_dataset(papers, level="l1", print_results=True)
    get_categories_distribution_ratings(papers, level="l1", print_results=True)
    get_categories_distribution_cache(papers, level="l1", print_results=True)
    test_users = load_finetuning_users(selection="test")
    users_ratings = load_users_ratings(relevant_users_ids=test_users)
    categories_distributions_dict = get_categories_distribution_val_negative_samples(
        papers=papers, users_ratings=users_ratings
    )
    for key, dist in categories_distributions_dict.items():
        assert sum(dist.values()) == 1.0, "Categories distribution must sum to 1.0."
        if "Computer Science" in dist:
            print(f"User {key} - Computer Science: {dist['Computer Science']:.2%}")
            print(dist)
