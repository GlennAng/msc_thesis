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

embedding = Embedding(ProjectPaths.logreg_embeddings_path() / "after_pca" / "gte_large_256")

users_ratings = load_users_ratings_from_selection(
    users_ratings_selection=UsersRatingsSelection.MSC_EARLY_SPLIT, relevant_users_ids = "finetuning_test"
)
users_ids = users_ratings["user_id"].unique().tolist()
n_users = len(users_ids)

pos_diff_sims = {}
neg_diff_sims = {}
pos_users_per_session = {}
neg_users_per_session = {}

for user_id in tqdm(users_ids):
    user_ratings = users_ratings[users_ratings["user_id"] == user_id]
    val_ratings = user_ratings[user_ratings["split"] == "val"]
    sessions_ids = val_ratings["session_id"].unique().tolist()
    
    for session_id in sessions_ids:
        # Only include sessions up to 75
        if session_id > 75:
            continue
            
        session_val_ratings = val_ratings[val_ratings["session_id"] == session_id]
        pos_session_val_ratings = session_val_ratings[session_val_ratings["rating"] == 1]
        neg_session_val_ratings = session_val_ratings[session_val_ratings["rating"] == 0]
        
        pos_session_val_embeddings = embedding.matrix[embedding.get_idxs(pos_session_val_ratings["paper_id"].tolist())]
        neg_session_val_embeddings = embedding.matrix[embedding.get_idxs(neg_session_val_ratings["paper_id"].tolist())]

        train_ratings = user_ratings[user_ratings["session_id"] < session_id]
        pos_train_ratings = train_ratings[train_ratings["rating"] == 1]
        neg_train_ratings = train_ratings[train_ratings["rating"] == 0]
        pos_train_embeddings = embedding.matrix[embedding.get_idxs(pos_train_ratings["paper_id"].tolist())]
        neg_train_embeddings = embedding.matrix[embedding.get_idxs(neg_train_ratings["paper_id"].tolist())]

        # Skip if no training data
        if len(pos_train_embeddings) == 0 or len(neg_train_embeddings) == 0:
            continue

        # For positively-rated validation papers
        if len(pos_session_val_embeddings) > 0:
            sims_pos = compute_sims(pos_session_val_embeddings, pos_train_embeddings, agg=False)
            sims_neg = compute_sims(pos_session_val_embeddings, neg_train_embeddings, agg=False)
            
            if AGGREGATION == "max":
                agg_sim_pos = np.max(sims_pos, axis=1)
                agg_sim_neg = np.max(sims_neg, axis=1)
            else:  # mean
                agg_sim_pos = np.mean(sims_pos, axis=1)
                agg_sim_neg = np.mean(sims_neg, axis=1)
            
            sim_diff = agg_sim_pos - agg_sim_neg
            pos_diff_sims.setdefault(session_id, []).append(np.mean(sim_diff))
            pos_users_per_session.setdefault(session_id, set()).add(user_id)

        # For negatively-rated validation papers
        if len(neg_session_val_embeddings) > 0:
            sims_pos = compute_sims(neg_session_val_embeddings, pos_train_embeddings, agg=False)
            sims_neg = compute_sims(neg_session_val_embeddings, neg_train_embeddings, agg=False)
            
            if AGGREGATION == "max":
                agg_sim_pos = np.max(sims_pos, axis=1)
                agg_sim_neg = np.max(sims_neg, axis=1)
            else:  # mean
                agg_sim_pos = np.mean(sims_pos, axis=1)
                agg_sim_neg = np.mean(sims_neg, axis=1)
            
            sim_diff = agg_sim_pos - agg_sim_neg
            neg_diff_sims.setdefault(session_id, []).append(np.mean(sim_diff))
            neg_users_per_session.setdefault(session_id, set()).add(user_id)

# Average over users for each session
pos_diff_sims_avg = {sid: np.mean(scores) for sid, scores in pos_diff_sims.items()}
neg_diff_sims_avg = {sid: np.mean(scores) for sid, scores in neg_diff_sims.items()}

# Calculate percentage of users included for each session
pos_percentages = {sid: len(users) / n_users for sid, users in pos_users_per_session.items()}
neg_percentages = {sid: len(users) / n_users for sid, users in neg_users_per_session.items()}
print(pos_percentages)
print(neg_percentages)

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))

# Set grey background
ax.set_facecolor("#f0f0f0")

# Add horizontal line at y=0 FIRST (so it's behind the plot lines)
ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-', alpha=0.6, zorder=1)

# Plot both difference lines with variable thickness
sessions_pos = sorted(pos_diff_sims_avg.keys())
sessions_neg = sorted(neg_diff_sims_avg.keys())

# Variable thickness parameters
min_width = 2.0
max_width = 5.0

# Plot positive papers with variable thickness
for i in range(len(sessions_pos) - 1):
    thickness = min_width + (max_width - min_width) * pos_percentages[sessions_pos[i]]
    ax.plot(
        sessions_pos[i : i + 2],
        [pos_diff_sims_avg[sessions_pos[i]], pos_diff_sims_avg[sessions_pos[i + 1]]],
        color='blue',
        linewidth=thickness,
        label='Positive Evaluation Papers' if i == 0 else None,
        zorder=2,
    )

# Plot negative papers with variable thickness
for i in range(len(sessions_neg) - 1):
    thickness = min_width + (max_width - min_width) * neg_percentages[sessions_neg[i]]
    ax.plot(
        sessions_neg[i : i + 2],
        [neg_diff_sims_avg[sessions_neg[i]], neg_diff_sims_avg[sessions_neg[i + 1]]],
        color='orange',
        linewidth=thickness,
        label='Negative Evaluation Papers' if i == 0 else None,
        zorder=2,
    )

# Set y-axis limits and major ticks
ax.set_ylim(-0.2, 0.2)
ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
ax.set_yticklabels(['-0.2', '-0.1', '0.0', '0.1', '0.2'], fontsize=16)

# Set minor y-axis ticks for gridlines at 0.05 intervals (no labels)
ax.set_yticks([-0.15, -0.05, 0.05, 0.15], minor=True)

# Remove minor tick marks (but keep the grid lines)
ax.tick_params(axis='y', which='minor', length=0)

# Set x-axis limits and ticks
ax.set_xlim(0, 75)
ax.set_xticks([0, 15, 30, 45, 60, 75])
ax.tick_params(axis='x', labelsize=16)

# Set axis labels
ax.set_xlabel('Session ID', fontsize=18)
ax.set_ylabel('Similarity Margin', fontsize=18)

# Enable grid for both major and minor ticks
ax.grid(True, which='major', alpha=0.7, linewidth=0.5)
ax.grid(True, which='minor', axis='y', alpha=0.7, linewidth=0.5)

# Legend
ax.legend(loc='lower right', fontsize=14)

plt.savefig(f"temporal_similarity_difference_{AGGREGATION}.pdf", bbox_inches="tight")
plt.close()