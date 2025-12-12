from ..embeddings.embedding import Embedding
from ....src.project_paths import ProjectPaths

import numpy as np
from numpy.linalg import norm
import random

from ....src.load_files import load_papers, load_users_ratings, load_finetuning_users_ids

from ..training.users_ratings import load_users_ratings_from_selection, UsersRatingsSelection
from ....finetuning.src.finetuning_compare_embeddings import compute_sims_same_set, compute_sims

embedding = Embedding(ProjectPaths.logreg_embeddings_path() / "after_pca" / "gte_large_256")
# normalize embeddings row-wise
# remove last 100 columns
embedding.matrix = embedding.matrix / norm(embedding.matrix, axis=1, keepdims=True)
print(norm(embedding.matrix[0]), norm(embedding.matrix[1]))


papers = load_papers()
print(papers["l1"].value_counts())
first_cat = "Physics"
second_cat = "Linguistics"


first_cat_papers_ids = papers[papers["l1"] == first_cat]["paper_id"].tolist()
second_cat_papers_ids = papers[papers["l1"] == second_cat]["paper_id"].tolist()



random.seed(42)
np.random.seed(42)
val = 0.0
val_first_cat = 0.0
val_second_cat = 0.0
n_papers = embedding.matrix.shape[0]
N = 100000
for _ in range(N):
    first_cat_paper_ids = random.sample(first_cat_papers_ids, 2)
    second_cat_paper_ids = random.sample(second_cat_papers_ids, 2)
    first_cat_papers_embeddings = embedding.matrix[embedding.get_idxs(first_cat_paper_ids)]
    second_cat_papers_embeddings = embedding.matrix[embedding.get_idxs(second_cat_paper_ids)]
    val += np.dot(first_cat_papers_embeddings[0], second_cat_papers_embeddings[0])
    val_first_cat += np.dot(first_cat_papers_embeddings[0], first_cat_papers_embeddings[1])
    val_second_cat += np.dot(second_cat_papers_embeddings[0], second_cat_papers_embeddings[1])
print(f"Average dot product between {first_cat} and {second_cat} papers: {val / N}")
print(f"Average dot product between {first_cat} papers: {val_first_cat / N}")
print(f"Average dot product between {second_cat} papers: {val_second_cat / N}")

users_ratings = load_users_ratings_from_selection(
    users_ratings_selection=UsersRatingsSelection.SESSION_BASED_NO_FILTERING_POS
)
users_ids = users_ratings["user_id"].unique().tolist()
n_users = len(users_ids)
users_val = 0.0
users_val_pos_neg = 0.0
for user_id in users_ids:
    user_ratings = users_ratings[users_ratings["user_id"] == user_id]
    pos_rated_papers_ids = user_ratings[user_ratings["rating"] == 1]["paper_id"].tolist()
    neg_rated_papers_ids = user_ratings[user_ratings["rating"] == 0]["paper_id"].tolist()
    embeddings = embedding.matrix[embedding.get_idxs(pos_rated_papers_ids)]
    embeddings_neg = embedding.matrix[embedding.get_idxs(neg_rated_papers_ids)]
    users_val += compute_sims_same_set(embeddings)
    users_val_pos_neg += compute_sims(embeddings, embeddings_neg)
print(f"Average dot product over users' positive rated papers: {users_val / n_users}")
print(f"Average dot product between users' positive and negative rated papers: {users_val_pos_neg / n_users}")
