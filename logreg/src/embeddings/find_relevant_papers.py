import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[3]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_paths_to_sys()

import pickle, random
import pandas as pd
from tqdm import tqdm

from load_files import load_papers, load_finetuning_users, load_users_ratings
from training_data import get_filtered_cache_papers_ids_for_user, get_negative_samples_ids

def find_relevant_papers(seeds: list = [1, 2, 25, 26, 42, 75, 76, 100, 101, 150, 151]) -> list:
    users_ids = load_finetuning_users(selection = "test")
    print(len(users_ids), "Users loaded.")
    users_ratings = load_users_ratings(relevant_users_ids = users_ids)
    assert len(users_ratings["user_id"].unique()) == len(users_ids), "Users IDs do not match."
    papers = load_papers(relevant_columns = ["paper_id", "in_ratings", "in_cache", "l1"])
    relevant_papers_ids = set(users_ratings["paper_id"].values)
    print(len(relevant_papers_ids), "Relevant papers loaded.")

    for random_state in tqdm(seeds):
        negative_samples_ids = get_negative_samples_ids(papers, n_negative_samples = 100, random_state = random_state, exclude_in_ratings = True, exclude_in_cache = True)
        relevant_papers_ids.update(negative_samples_ids)
        cache_attached_ids = get_negative_samples_ids(papers, n_negative_samples = 5000, random_state = random_state, papers_to_exclude = negative_samples_ids)
        relevant_papers_ids.update(cache_attached_ids)
    print(len(relevant_papers_ids), "Total relevant papers IDs collected after negative sampling.")

    cache_papers_ids = papers[papers["in_cache"]]["paper_id"].tolist()
    assert cache_papers_ids == sorted(cache_papers_ids)
    relevant_papers_ids.update(cache_papers_ids)

    relevant_papers_ids = sorted(list(relevant_papers_ids))
    print(len(relevant_papers_ids), "Final number of relevant papers IDs in List.")
    return relevant_papers_ids

if __name__ == "__main__":
    find_relevant_papers()