from ..embeddings.embedding import Embedding
from ....src.project_paths import ProjectPaths
import pandas as pd
import numpy as np
from numpy.linalg import norm
import random
from pathlib import Path
from ..training.users_ratings import load_users_ratings_from_selection, UsersRatingsSelection
from ....finetuning.src.finetuning_compare_embeddings import compute_sims_same_set, compute_sims
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from ..training.training_data import get_val_negative_samples_ids, get_user_categories_ratios, get_user_val_negative_samples
from ....src.load_files import load_papers, load_users_significant_categories

import matplotlib.pyplot as plt

# FLAG: "mean" or "max"
AGGREGATION = "max"

embedding = Embedding(ProjectPaths.finetuning_data_path() / "checkpoints" / "cat_best" / "embeddings")

users_ratings = load_users_ratings_from_selection(
    users_ratings_selection=UsersRatingsSelection.MSC_EARLY_SPLIT, relevant_users_ids = "finetuning_test"
)
users_ids = users_ratings["user_id"].unique().tolist()
n_users = len(users_ids)


pos_diff_sims = []
neg_diff_sims = []
for user_id in tqdm(users_ids):
    user_ratings = users_ratings[users_ratings["user_id"] == user_id]
    pos_papers_ids = user_ratings[user_ratings["rating"] == 1]["paper_id"].tolist()
    neg_papers_ids = user_ratings[user_ratings["rating"] == 0]["paper_id"].tolist()
    pos_papers_embs = embedding.matrix[embedding.get_idxs(pos_papers_ids)]
    neg_papers_embs = embedding.matrix[embedding.get_idxs(neg_papers_ids)]
    pos_diff_sims.append(compute_sims_same_set(pos_papers_embs))
    neg_diff_sims.append(compute_sims(pos_papers_embs, neg_papers_embs))

pos_diff_sims = np.mean(pos_diff_sims)
neg_diff_sims = np.mean(neg_diff_sims)
print(f"Pos diff sims: {pos_diff_sims}")
print(f"Neg diff sims: {neg_diff_sims}")