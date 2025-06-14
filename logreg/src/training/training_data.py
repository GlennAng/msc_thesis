import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[3]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_paths_to_sys()

import random
import numpy as np, pandas as pd
from enum import Enum, auto

from embedding import Embedding
LABEL_DTYPE = np.int64

def load_negrated_ranking_ids_for_user(negrated_ids: list, random_state: int) -> list:
    min_n_negrated = min(4, len(negrated_ids))
    negrated_ids = sorted(negrated_ids)
    random.seed(random_state)
    random.shuffle(negrated_ids)
    negrated_ids = negrated_ids[:min_n_negrated]
    return sorted(negrated_ids)
   
def load_training_data_for_user(embedding: Embedding, include_base: bool, include_zerorated: bool, include_cache: bool, train_rated_idxs: np.ndarray, y_train_rated: np.ndarray, 
                                base_idxs: np.ndarray, y_base: np.ndarray, zerorated_idxs: np.ndarray, y_zerorated: np.ndarray, cache_idxs: np.ndarray, y_cache: np.ndarray) -> tuple:
    X_idxs, y = train_rated_idxs, y_train_rated
    if include_base:
        X_idxs = np.concatenate((X_idxs, base_idxs))
        y = np.concatenate((y, y_base))
    if include_zerorated:
        X_idxs = np.concatenate((X_idxs, zerorated_idxs))
        y = np.concatenate((y, y_zerorated))
    if include_cache:
        X_idxs = np.concatenate((X_idxs, cache_idxs))
        y = np.concatenate((y, y_cache))
    X_train = embedding.matrix[X_idxs]
    return X_train, y

def load_global_cache(embedding: Embedding, papers: pd.DataFrame, max_cache: int = None, random_state: int = None) -> tuple:
    global_cache_ids = get_global_cache_papers_ids(papers, max_cache, random_state)
    global_cache_idxs = embedding.get_idxs(global_cache_ids)
    global_cache_n = len(global_cache_idxs)
    y_global_cache = np.zeros(global_cache_n, dtype = LABEL_DTYPE)
    return global_cache_ids, global_cache_idxs, global_cache_n, y_global_cache

def get_global_cache_papers_ids(papers: pd.DataFrame, max_cache: int = None, random_state: int = None) -> list:
    cache = papers[papers["in_cache"]]["paper_id"].tolist()
    n_cache = len(cache)
    max_cache = n_cache if max_cache is None else min(max_cache, n_cache)
    if n_cache < max_cache:
        raise ValueError(f"Required cache size ({max_cache}) is greater than the number of valid cache papers ({n_cache}).")
    elif n_cache > max_cache:
        cache = sorted(cache)
        rng = random.Random(random_state)
        cache = rng.sample(cache, max_cache)
    return sorted(cache)

def load_filtered_cache_for_user(embedding: Embedding, cache_papers_ids: list, rated_ids: list, max_cache: int = None, random_state: int = None) -> tuple:
    user_filtered_cache_ids = get_filtered_cache_papers_ids_for_user(cache_papers_ids, rated_ids, max_cache, random_state)
    user_filtered_cache_idxs = embedding.get_idxs(user_filtered_cache_ids)
    user_filtered_cache_n = len(user_filtered_cache_idxs)
    y_user_filtered_cache = np.zeros(user_filtered_cache_n, dtype = LABEL_DTYPE)
    return user_filtered_cache_ids, user_filtered_cache_idxs, user_filtered_cache_n, y_user_filtered_cache

def get_filtered_cache_papers_ids_for_user(cache_papers_ids: list, rated_ids: list, max_cache: int = None, random_state: int = None) -> list:
    user_filtered_cache_ids = [paper_id for paper_id in cache_papers_ids if paper_id not in rated_ids]
    n_user_filtered_cache = len(user_filtered_cache_ids)
    max_cache = n_user_filtered_cache if max_cache is None else min(max_cache, n_user_filtered_cache)
    if n_user_filtered_cache < max_cache:
        raise ValueError(f"Required cache size ({max_cache}) is greater than the number of valid cache papers ({n_user_filtered_cache}).")
    elif n_user_filtered_cache > max_cache:
        user_filtered_cache_ids = sorted(user_filtered_cache_ids)
        rng = random.Random(random_state)
        user_filtered_cache_ids = rng.sample(user_filtered_cache_ids, max_cache)
    return sorted(user_filtered_cache_ids)

def load_negative_samples_embeddings(embedding: Embedding, papers: pd.DataFrame, n_negative_samples: int, random_state: int, papers_to_exclude: list = None,
                                     exclude_in_ratings: bool = False, exclude_in_cache: bool = False) -> tuple:
    negative_samples_ids = get_negative_samples_ids(papers, n_negative_samples, random_state, papers_to_exclude = papers_to_exclude,
                                                    exclude_in_ratings = exclude_in_ratings, exclude_in_cache = exclude_in_cache)
    return negative_samples_ids, embedding.matrix[embedding.get_idxs(negative_samples_ids)]

def get_categories_ratios() -> dict:
    categories_ratios = {"Physics": 0.2, "Astronomy": 0.1, "Biology": 0.15, "Medicine": 0.1, "Chemistry": 0.1, 
                         "Economics": 0.05, "Psychology": 0.05, "Materials Science": 0.05, "Earth Science": 0.05, 
                         "Linguistics": 0.05, "Philosophy": 0.05, "Geography": 0.05}
    return categories_ratios

def get_negative_samples_ids(papers: pd.DataFrame, n_negative_samples: int, random_state: int, categories_ratios: dict = None, papers_to_exclude: set = None, 
                             exclude_in_ratings: bool = False, exclude_in_cache: bool = False) -> list:
    mask = pd.Series(True, index = papers.index)
    if papers_to_exclude is not None:
        mask &= ~papers["paper_id"].isin(papers_to_exclude)
    if exclude_in_ratings:
        mask &= ~papers["in_ratings"]
    if exclude_in_cache:
        mask &= ~papers["in_cache"]
    potential_papers = papers.loc[mask, ["paper_id", "l1"]]
    if categories_ratios is None:
        categories_ratios = get_categories_ratios()
    samples_per_category = {category: int(n_negative_samples * ratio) for category, ratio in categories_ratios.items()}
    negative_samples_ids = []
    rng = random.Random(random_state)
    for category in list(categories_ratios.keys()):
        n_samples_category = samples_per_category[category]
        if n_samples_category == 0:
            continue
        potential_papers_category = potential_papers[potential_papers["l1"] == category]["paper_id"].tolist()
        if papers_to_exclude is not None:
            potential_papers_category = [paper_id for paper_id in potential_papers_category if paper_id not in papers_to_exclude]
            assert potential_papers_category == sorted(potential_papers_category)
        if len(potential_papers_category) < n_samples_category:
            raise ValueError(f"Not enough potential papers in category '{category}' to sample {n_samples_category} negative samples.")
        negative_samples_ids += rng.sample(potential_papers_category, n_samples_category)
    return sorted(negative_samples_ids)

def count_to_str(count: int) -> str:
    return f"{count:,}"

def get_categories_distribution(papers_ids_to_categories : dict = None, print_results : bool = True) -> tuple:
    unique_categories = set(papers_ids_to_categories.values())
    categories_counts = {category: 0 for category in unique_categories}
    n_total = 0
    for key, value in papers_ids_to_categories.items():
        categories_counts[value] += 1
        n_total += 1
    categories_counts = {category: count / n_total for category, count in categories_counts.items()}
    if print_results:
        categories_counts_copy = categories_counts.copy()
        categories_counts_copy["Total"] = 1.0
        categories_counts_copy = sorted(categories_counts_copy.items(), key = lambda x: x[1], reverse = True)
        for category, count in categories_counts_copy:
            print(f"{category}: {count:.2%} ({count_to_str(int(count * n_total))})")
        print("____________________________________________________________")
    return categories_counts, n_total

def get_categories_distribution_dataset(papers: pd.DataFrame, level: str = "l1", print_results: bool = True) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    return get_categories_distribution(papers_ids_to_categories, print_results)

def get_categories_distribution_ratings(papers: pd.DataFrame, level: str = "l1", print_results: bool = True) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    papers = papers[papers["in_ratings"]]
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    query = '''SELECT DISTINCT paper_id FROM users_ratings'''
    return get_categories_distribution(papers_ids_to_categories, print_results)

def get_categories_distribution_cache(papers: pd.DataFrame, level: str = "l1", print_results: bool = True) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    cache_papers_ids = get_global_cache_papers_ids(papers, max_cache = 5000, random_state = 42)
    papers = papers[papers["paper_id"].isin(cache_papers_ids)]
    if papers.empty:
        raise ValueError("No papers found in the cache.")
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    return get_categories_distribution(papers_ids_to_categories, print_results)

def get_categories_distribution_negative_samples(papers: pd.DataFrame, level: str = "l1", print_results: bool = True) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    negative_samples_ids = get_negative_samples_ids(papers, n_negative_samples = 100, random_state = 42, exclude_in_ratings = True, exclude_in_cache = True)
    papers_ids_to_categories = {paper_id: papers.loc[papers["paper_id"] == paper_id, level].values[0] for paper_id in negative_samples_ids}
    return get_categories_distribution(papers_ids_to_categories, print_results)

def get_categories_distribution_cache_attached(papers: pd.DataFrame, level: str = "l1", print_results: bool = True) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    negative_samples_ids = get_negative_samples_ids(papers, n_negative_samples = 100, random_state = 42, exclude_in_ratings = True, exclude_in_cache = True)
    cache_attached_ids = get_negative_samples_ids(papers, n_negative_samples = 5000, random_state = 42, papers_to_exclude = negative_samples_ids)
    papers_ids_to_categories = {paper_id: papers.loc[papers["paper_id"] == paper_id, level].values[0] for paper_id in cache_attached_ids}
    return get_categories_distribution(papers_ids_to_categories, print_results)

if __name__ == "__main__":
    from load_files import load_papers
    papers = load_papers(ProjectPaths.data_db_backup_date_papers_path())
    get_categories_distribution_dataset(papers, level = "l1", print_results = True)
    get_categories_distribution_ratings(papers, level = "l1", print_results = True)
    get_categories_distribution_cache(papers, level = "l1", print_results = True)
    get_categories_distribution_negative_samples(papers, level = "l1", print_results = True)
    get_categories_distribution_cache_attached(papers, level = "l1", print_results = True)