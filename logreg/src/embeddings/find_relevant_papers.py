import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[3]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_src_paths_to_sys()

import pickle, random
import pandas as pd
from tqdm import tqdm

DEFAULT_SEEDS = [1, 2, 25, 26, 42, 75, 76, 100, 101, 150, 151]
from get_users_ratings import get_temporal_750_users_ratings
from load_files import load_papers
from training_data import get_filtered_cache_papers_ids_for_user, get_negative_samples_ids

users_ratings, users_ids = get_temporal_750_users_ratings()
print(len(users_ids), "Users loaded.")
papers = load_papers(ProjectPaths.data_db_backup_date_papers_path(), relevant_columns = ["paper_id", "in_ratings", "in_cache", "l1"])
relevant_papers_ids = set(users_ratings["paper_id"].values)
print(len(relevant_papers_ids), "Relevant papers loaded.")

for random_state in tqdm(DEFAULT_SEEDS):
    negative_samples_ids = get_negative_samples_ids(papers, n_negative_samples = 100, random_state = random_state, exclude_in_ratings = True, exclude_in_cache = True)
    relevant_papers_ids.update(negative_samples_ids)
    cache_attached_ids = get_negative_samples_ids(papers, n_negative_samples = 5000, random_state = random_state, papers_to_exclude = negative_samples_ids)
    relevant_papers_ids.update(cache_attached_ids)
print(len(relevant_papers_ids), "Total relevant papers IDs collected after negative sampling.")

cache_papers_ids = papers[papers["in_cache"]]["paper_id"].tolist()
assert cache_papers_ids == sorted(cache_papers_ids)
for user_id in tqdm(users_ids):
    user_ratings = users_ratings[users_ratings["user_id"] == user_id]
    posrated_ids = user_ratings[user_ratings["rating"] == 1]["paper_id"].tolist()
    negrated_ids = user_ratings[user_ratings["rating"] == 0]["paper_id"].tolist()
    rated_ids = posrated_ids + negrated_ids
    for random_state in DEFAULT_SEEDS:
        user_filtered_cache_ids = get_filtered_cache_papers_ids_for_user(cache_papers_ids, rated_ids, max_cache = 5000, random_state = random_state)
        relevant_papers_ids.update(user_filtered_cache_ids)

relevant_papers_ids = sorted(list(relevant_papers_ids))
print(len(relevant_papers_ids), "Final number of relevant papers IDs in List.")

with open(ProjectPaths.logreg_embeddings_path() / "relevant_papers.pkl", 'wb') as f:
    pickle.dump(relevant_papers_ids, f)