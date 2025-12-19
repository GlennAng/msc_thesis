from ..embeddings.embedding import Embedding
from ....src.project_paths import ProjectPaths


import pandas as pd
# print all rows
pd.set_option('display.max_rows', None)
import numpy as np
from numpy.linalg import norm
import random
from pathlib import Path

from ....src.load_files import load_papers, load_users_ratings, load_finetuning_users_ids

from ..training.users_ratings import load_users_ratings_from_selection, UsersRatingsSelection
from ....finetuning.src.finetuning_compare_embeddings import compute_sims_same_set, compute_sims

from tqdm import tqdm

embedding = Embedding(Path("code/finetuning/data/experiments") / "gte_large_256_2025-12-19-04-05" / "embeddings")
# normalize
embedding.matrix = embedding.matrix / norm(embedding.matrix, axis=1, keepdims=True)
papers_ids = np.array(list(embedding.papers_ids_to_idxs.keys())).tolist()

users_ratings = load_users_ratings_from_selection(
    users_ratings_selection=UsersRatingsSelection.MSC_EARLY_SPLIT, relevant_users_ids = "finetuning_test"
)
n_users = users_ratings["user_id"].nunique()
print(f"Loaded {len(users_ratings)} ratings from {n_users} users.")
papers = load_papers(relevant_papers_ids=papers_ids)

users_pos_similarities = []
users_pos_neg_similarities = []
users_ids = users_ratings["user_id"].unique().tolist()
for user_id in tqdm(users_ids):
    user_ratings = users_ratings[users_ratings["user_id"] == user_id]
    pos_rated_papers_ids = user_ratings[user_ratings["rating"] == 1]["paper_id"].tolist()
    neg_rated_papers_ids = user_ratings[user_ratings["rating"] == 0]["paper_id"].tolist()
    embeddings = embedding.matrix[embedding.get_idxs(pos_rated_papers_ids)]
    embeddings_neg = embedding.matrix[embedding.get_idxs(neg_rated_papers_ids)]
    users_pos_similarities.append(compute_sims_same_set(embeddings))
    users_pos_neg_similarities.append(compute_sims(embeddings, embeddings_neg))
print(f"Average dot product over users' positive rated papers: {np.mean(users_pos_similarities)}")
print(f"Average dot product between users' positive and negative rated papers: {np.mean(users_pos_neg_similarities)}")
print(f"Difference: {np.mean(users_pos_similarities) - np.mean(users_pos_neg_similarities)}")

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
"""


users_ratings_early_split = load_users_ratings_from_selection(
    users_ratings_selection=UsersRatingsSelection.MSC_EARLY_SPLIT
)
val_ratings = users_ratings_early_split[users_ratings_early_split["split"] == "val"]
pos_val_ratings = val_ratings[val_ratings["rating"] == 1]

N_NEGRATED = 1
pos_val_ratings_after = pos_val_ratings[pos_val_ratings["n_negrated_still_to_come"] >= N_NEGRATED]
print(f"Number of positive val ratings before filtering: {len(pos_val_ratings)}")
print(f"Number of positive val ratings after filtering: {len(pos_val_ratings_after)}")


# count n_pos per user
n_pos_per_user = pos_val_ratings_after["user_id"].value_counts()
print("Number of users with at least N positive val ratings:")
for n_pos in range(1, 11):
    n_users = (n_pos_per_user >= n_pos).sum()
    print(f"At least {n_pos} positive val ratings: {n_users} users")

print()

df = val_ratings.copy()

# Get session stats
session_stats = df.groupby(['user_id', 'session_id']).agg({
    'rating': lambda x: (x == 1).sum(),
    'paper_id': 'count'
}).rename(columns={'rating': 'n_upvotes', 'paper_id': 'n_total'})
session_stats['n_downvotes'] = session_stats['n_total'] - session_stats['n_upvotes']
session_stats = session_stats.reset_index()

# For each upvote, check if there's a downvote within k future sessions
all_upvotes = df[df['rating'] == 1][['user_id', 'session_id', 'paper_id']].copy()
total_upvotes = len(all_upvotes)

df = val_ratings.copy()

# Analysis 1: Pairwise Accuracy Coverage
# Count sessions with at least 1 upvote AND at least 1 downvote
session_stats = df.groupby(['user_id', 'session_id']).agg({
    'rating': lambda x: (x == 1).sum(),  # count upvotes
    'paper_id': 'count'  # total ratings
}).rename(columns={'rating': 'n_upvotes', 'paper_id': 'n_total'})

session_stats['n_downvotes'] = session_stats['n_total'] - session_stats['n_upvotes']
session_stats['has_upvote'] = session_stats['n_upvotes'] > 0
session_stats['has_downvote'] = session_stats['n_downvotes'] > 0
session_stats['is_mixed'] = session_stats['has_upvote'] & session_stats['has_downvote']

# Get upvotes from mixed sessions
upvotes_in_mixed = df.merge(
    session_stats[['is_mixed']], 
    left_on=['user_id', 'session_id'], 
    right_index=True
)
upvotes_in_mixed = upvotes_in_mixed[(upvotes_in_mixed['rating'] == 1) & (upvotes_in_mixed['is_mixed'])]

total_upvotes = (df['rating'] == 1).sum()
upvotes_in_mixed_count = len(upvotes_in_mixed)
pairwise_coverage = upvotes_in_mixed_count / total_upvotes * 100

print("=" * 60)
print("ANALYSIS 1: Pairwise Accuracy Coverage")
print("=" * 60)
print(f"Total upvotes: {total_upvotes:,}")
print(f"Upvotes in sessions with ≥1 downvote: {upvotes_in_mixed_count:,}")
print(f"Data loss: {100 - pairwise_coverage:.1f}%")
print(f"Coverage: {pairwise_coverage:.1f}%")
print()

full_users_ids = users_ratings_early_split["user_id"].unique().tolist()
# count n upvotes per user
n_upvotes_per_user = upvotes_in_mixed['user_id'].value_counts()
users_with_at_least_five_upvotes = (n_upvotes_per_user >= 3).sum()
print(f"Number of users with at least 5 upvotes in mixed sessions: {users_with_at_least_five_upvotes} out of {len(full_users_ids)} total users")
# get the user ids
user_ids_with_at_least_five_upvotes = n_upvotes_per_user[n_upvotes_per_user >= 3].index.tolist()
users_ids_with_fewer_than_five_upvotes = set(full_users_ids) - set(user_ids_with_at_least_five_upvotes)
print(f"Number of users with fewer than 5 upvotes in mixed sessions: {len(users_ids_with_fewer_than_five_upvotes)}")

# sorted
user_ids_with_at_least_five_upvotes.sort()






users_ratings_late_split = load_users_ratings_from_selection(
    users_ratings_selection=UsersRatingsSelection.MSC_LATE_SPLIT
)
users_ids = users_ratings_early_split["user_id"].unique().tolist()

val_ratings = users_ratings_early_split[users_ratings_early_split["split"] == "val"]
# for each user id and session id count how often rating is 0 and 1
session_stats = val_ratings.groupby(['user_id', 'session_id']).agg({
    'rating': lambda x: (x == 0).sum(),  # count upvotes
    'paper_id': 'count'  # total ratings
}).rename(columns={'rating': 'n_downvotes', 'paper_id': 'n_total'})
print(session_stats['n_downvotes'].value_counts().sort_index())

min_val_pos = [0, 1, 2, 3, 4, 5]
pos_val_ratings = users_ratings_early_split[users_ratings_early_split["rating"] == 1]
pos_val_ratings = pos_val_ratings[pos_val_ratings["split"] == "val"]
n_ratings_before = len(pos_val_ratings)
pos_val_ratings_filtered = pos_val_ratings[pos_val_ratings["n_negrated_still_to_come"] >= 1]
n_ratings_after = len(pos_val_ratings_filtered)

print(f"Number of positive val ratings before filtering: {n_ratings_before}")
print(f"Number of positive val ratings after filtering: {n_ratings_after}")




def print_valid_users()



df = users_ratings.copy()
# Analysis 1: Pairwise Accuracy Coverage
# Count sessions with at least 1 upvote AND at least 1 downvote
session_stats = df.groupby(['user_id', 'session_id']).agg({
    'rating': lambda x: (x == 1).sum(),  # count upvotes
    'paper_id': 'count'  # total ratings
}).rename(columns={'rating': 'n_upvotes', 'paper_id': 'n_total'})

session_stats['n_downvotes'] = session_stats['n_total'] - session_stats['n_upvotes']
session_stats['has_upvote'] = session_stats['n_upvotes'] > 0
session_stats['has_downvote'] = session_stats['n_downvotes'] > 0
session_stats['is_mixed'] = session_stats['has_upvote'] & session_stats['has_downvote']

# Get upvotes from mixed sessions
upvotes_in_mixed = df.merge(
    session_stats[['is_mixed']], 
    left_on=['user_id', 'session_id'], 
    right_index=True
)
upvotes_in_mixed = upvotes_in_mixed[(upvotes_in_mixed['rating'] == 1) & (upvotes_in_mixed['is_mixed'])]

total_upvotes = (df['rating'] == 1).sum()
upvotes_in_mixed_count = len(upvotes_in_mixed)
pairwise_coverage = upvotes_in_mixed_count / total_upvotes * 100

print("=" * 60)
print("ANALYSIS 1: Pairwise Accuracy Coverage")
print("=" * 60)
print(f"Total upvotes: {total_upvotes:,}")
print(f"Upvotes in sessions with ≥1 downvote: {upvotes_in_mixed_count:,}")
print(f"Data loss: {100 - pairwise_coverage:.1f}%")
print(f"Coverage: {pairwise_coverage:.1f}%")
print()

# Count total pairs for statistical power
total_pairs = (session_stats['n_upvotes'] * session_stats['n_downvotes']).sum()
print(f"Total (upvote, downvote) pairs: {total_pairs:,}")
print()

# Analysis 2: NDCG with 4 Explicits Coverage
upvotes_with_4plus_negs = df.merge(
    session_stats[['n_downvotes']], 
    left_on=['user_id', 'session_id'], 
    right_index=True
)
upvotes_with_4plus_negs = upvotes_with_4plus_negs[
    (upvotes_with_4plus_negs['rating'] == 1) & 
    (upvotes_with_4plus_negs['n_downvotes'] >= 4)
]

upvotes_4plus_count = len(upvotes_with_4plus_negs)
ndcg4_coverage = upvotes_4plus_count / total_upvotes * 100

print("=" * 60)
print("ANALYSIS 2: NDCG with 4 Explicits Coverage")
print("=" * 60)
print(f"Total upvotes: {total_upvotes:,}")
print(f"Upvotes in sessions with ≥4 downvotes: {upvotes_4plus_count:,}")
print(f"Data loss: {100 - ndcg4_coverage:.1f}%")
print(f"Coverage: {ndcg4_coverage:.1f}%")
print()

# Bonus: Distribution of downvotes per session (for mixed sessions)
print("=" * 60)
print("BONUS: Downvote Distribution in Mixed Sessions")
print("=" * 60)
mixed_sessions = session_stats[session_stats['is_mixed']]
downvote_dist = mixed_sessions['n_downvotes'].value_counts().sort_index()
print(downvote_dist.head(10))
print()

# Summary recommendation
print("=" * 60)
print("RECOMMENDATION")
print("=" * 60)
if pairwise_coverage > 70 and pairwise_coverage > ndcg4_coverage:
    print("✓ Use two-metric approach (NDCG@101 + Pairwise)")
    print(f"  Reason: Pairwise covers {pairwise_coverage:.1f}% vs NDCG-4 covers {ndcg4_coverage:.1f}%")
elif ndcg4_coverage > 70 and ndcg4_coverage > pairwise_coverage:
    print("✓ Use NDCG@105 (100 randoms + 4 explicits)")
    print(f"  Reason: NDCG-4 covers {ndcg4_coverage:.1f}% vs Pairwise covers {pairwise_coverage:.1f}%")
elif pairwise_coverage > 60:
    print("✓ Use two-metric approach (NDCG@101 + Pairwise)")
    print(f"  Reason: More interpretable, and {pairwise_coverage:.1f}% coverage is acceptable")
else:
    print("⚠ Consider 'up to 4 explicits' to minimize data loss")
    print(f"  Both approaches lose significant data")



dot_products_temporal_split_mean = []
dot_products_temporal_split_max = []
dot_products_temporal_previous_session_mean = []
dot_products_temporal_previous_session_max = []
dot_products_temporal_current_session_mean = []
dot_products_temporal_current_session_max = []

users_ids = users_ratings["user_id"].unique().tolist()
for user_id in tqdm(users_ids):
    user_ratings = users_ratings[users_ratings["user_id"] == user_id]
    user_ratings_pos = user_ratings[user_ratings["rating"] == 1]
    user_ratings_pos_val = user_ratings_pos[user_ratings_pos["split"] == "val"]
    user_ratings_pos_train = user_ratings_pos[user_ratings_pos["split"] == "train"]
    pos_val_papers_ids = user_ratings_pos_val["paper_id"].tolist()
    pos_train_papers_ids = user_ratings_pos_train["paper_id"].tolist()
    pos_train_embeddings = embedding.matrix[embedding.get_idxs(pos_train_papers_ids)]
    user_dot_products_temporal_split_mean = []
    user_dot_products_temporal_split_max = []
    user_dot_products_temporal_previous_session_mean = []
    user_dot_products_temporal_previous_session_max = []
    user_dot_products_temporal_current_session_mean = []
    user_dot_products_temporal_current_session_max = []
    pos_val_sessions = user_ratings_pos_val["session_id"].unique().tolist()
    for i, session_id in enumerate(pos_val_sessions):
        session_papers_ids = user_ratings_pos_val[user_ratings_pos_val["session_id"] == session_id]["paper_id"].tolist()
        pos_train_papers_ids_previous_sessions = user_ratings_pos[user_ratings_pos["session_id"] < session_id]["paper_id"].tolist()
        pos_train_embeddings_previous_sessions = embedding.matrix[embedding.get_idxs(pos_train_papers_ids_previous_sessions)]
        pos_train_papers_ids_current_session = user_ratings_pos[user_ratings_pos["session_id"] <= session_id]["paper_id"].tolist()
        pos_train_embeddings_current_session = embedding.matrix[embedding.get_idxs(pos_train_papers_ids_current_session)]
        session_embeddings = embedding.matrix[embedding.get_idxs(session_papers_ids)]

        dot_products_train = np.dot(session_embeddings, pos_train_embeddings.T)
        dot_products_previous_sessions = np.dot(session_embeddings, pos_train_embeddings_previous_sessions.T)
        dot_products_current_session = np.dot(session_embeddings, pos_train_embeddings_current_session.T)
        # get mean and max per session embedding
        for j in range(session_embeddings.shape[0]):
            if dot_products_train.shape[1] > 0:
                user_dot_products_temporal_split_mean.append(np.mean(dot_products_train[j]))
                user_dot_products_temporal_split_max.append(np.max(dot_products_train[j]))
            if dot_products_previous_sessions.shape[1] > 0:
                user_dot_products_temporal_previous_session_mean.append(np.mean(dot_products_previous_sessions[j]))
                user_dot_products_temporal_previous_session_max.append(np.max(dot_products_previous_sessions[j]))
            if dot_products_current_session.shape[1] > 0:
                # remove self-dot-product
                current_paper_id = session_papers_ids[j]
                mask = np.array([paper_id != current_paper_id for paper_id in pos_train_papers_ids_current_session])
                dot_products_without_self = dot_products_current_session[j][mask]
                if dot_products_without_self.size > 0:
                    user_dot_products_temporal_current_session_mean.append(np.mean(dot_products_without_self))
                    user_dot_products_temporal_current_session_max.append(np.max(dot_products_without_self))
    dot_products_temporal_split_mean.append(np.mean(user_dot_products_temporal_split_mean))
    dot_products_temporal_split_max.append(np.mean(user_dot_products_temporal_split_max))
    dot_products_temporal_previous_session_mean.append(np.mean(user_dot_products_temporal_previous_session_mean))
    dot_products_temporal_previous_session_max.append(np.mean(user_dot_products_temporal_previous_session_max))
    dot_products_temporal_current_session_mean.append(np.mean(user_dot_products_temporal_current_session_mean))
    dot_products_temporal_current_session_max.append(np.mean(user_dot_products_temporal_current_session_max))

print(f"Average dot product temporal split mean: {np.mean(dot_products_temporal_split_mean)}")
print(f"Average dot product temporal split max: {np.mean(dot_products_temporal_split_max)}")
print(f"Average dot product temporal previous session mean: {np.mean(dot_products_temporal_previous_session_mean)}")
print(f"Average dot product temporal previous session max: {np.mean(dot_products_temporal_previous_session_max)}")
print(f"Average dot product temporal current session mean: {np.mean(dot_products_temporal_current_session_mean)}")
print(f"Average dot product temporal current session max: {np.mean(dot_products_temporal_current_session_max)}")

import numpy as np
from sklearn.model_selection import train_test_split

# First, create random 80/20 split
users_ratings_random = users_ratings.copy()
random_splits = []

for user_id in users_ratings_random["user_id"].unique():
    user_mask = users_ratings_random["user_id"] == user_id
    user_data = users_ratings_random[user_mask]
    user_pos = user_data[user_data["rating"] == 1]
    
    if len(user_pos) > 0:
        # 80/20 split of positive ratings
        train_indices, val_indices = train_test_split(
            user_pos.index, 
            test_size=0.2, 
            random_state=42
        )
        
        for idx in user_data.index:
            if idx in train_indices:
                random_splits.append("train")
            elif idx in val_indices:
                random_splits.append("val")
            else:
                # Keep negative ratings in train
                random_splits.append("train")
    else:
        random_splits.extend(["train"] * len(user_data))

users_ratings_random["split"] = random_splits

# Now compute similarities for random split
dot_products_random_split_mean = []
dot_products_random_split_max = []

users_ids = users_ratings_random["user_id"].unique().tolist()
for user_id in tqdm(users_ids):
    user_ratings = users_ratings_random[users_ratings_random["user_id"] == user_id]
    user_ratings_pos = user_ratings[user_ratings["rating"] == 1]
    user_ratings_pos_val = user_ratings_pos[user_ratings_pos["split"] == "val"]
    user_ratings_pos_train = user_ratings_pos[user_ratings_pos["split"] == "train"]
    
    pos_val_papers_ids = user_ratings_pos_val["paper_id"].tolist()
    pos_train_papers_ids = user_ratings_pos_train["paper_id"].tolist()
    
    if len(pos_val_papers_ids) == 0 or len(pos_train_papers_ids) == 0:
        continue
    
    pos_train_embeddings = embedding.matrix[embedding.get_idxs(pos_train_papers_ids)]
    pos_val_embeddings = embedding.matrix[embedding.get_idxs(pos_val_papers_ids)]
    
    # Compute dot products
    dot_products = np.dot(pos_val_embeddings, pos_train_embeddings.T)
    
    # Get mean and max for each val paper
    user_dot_products_mean = []
    user_dot_products_max = []
    
    for j in range(dot_products.shape[0]):
        if dot_products.shape[1] > 0:
            user_dot_products_mean.append(np.mean(dot_products[j]))
            user_dot_products_max.append(np.max(dot_products[j]))
    
    if len(user_dot_products_mean) > 0:
        dot_products_random_split_mean.append(np.mean(user_dot_products_mean))
        dot_products_random_split_max.append(np.mean(user_dot_products_max))

print("\n=== RANDOM 80/20 SPLIT ===")
print(f"Average dot product random split mean: {np.mean(dot_products_random_split_mean)}")
print(f"Average dot product random split max: {np.mean(dot_products_random_split_max)}")

print("\n=== TEMPORAL SPLIT (for comparison) ===")
print(f"Average dot product temporal split mean: {np.mean(dot_products_temporal_split_mean)}")
print(f"Average dot product temporal split max: {np.mean(dot_products_temporal_split_max)}")
"""