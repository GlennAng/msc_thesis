import csv
import pandas as pd
import numpy as np
from tqdm import tqdm


from ..training.users_ratings import load_users_ratings_from_selection, UsersRatingsSelection
from ..embeddings.embedding import Embedding
from ....src.project_paths import ProjectPaths
from ....finetuning.src.finetuning_compare_embeddings import compute_sims, compute_sims_same_set

ratings = load_users_ratings_from_selection(
    users_ratings_selection=UsersRatingsSelection.MSC_EARLY_SPLIT
)
users_ids = ratings["user_id"].unique()
embedding = Embedding(ProjectPaths.logreg_embeddings_path() / "after_pca" / "gte_large_256")

first_n_papers, other_n_papers = [], []
first_pos_ratio, other_pos_ratio = [], []
first_similarity, other_similarity = [], []

first_n_papers, other_n_papers = [], []
first_pos_ratio, other_pos_ratio = [], []
first_pos_pos_similarity, other_pos_pos_similarity = [], []
first_neg_neg_similarity, other_neg_neg_similarity = [], []
first_pos_neg_similarity, other_pos_neg_similarity = [], []

for user_id in tqdm(users_ids):
    user_ratings = ratings[ratings["user_id"] == user_id]
    
    # First session (session_id == 0)
    first_session = user_ratings[user_ratings["session_id"] == 0]
    if len(first_session) > 0:
        first_n_papers.append(len(first_session))
        n_pos = (first_session["rating"] == 1).sum()
        n_rated = (first_session["rating"] != 2).sum()  # Exclude neutral if present
        if n_rated > 0:
            first_pos_ratio.append(n_pos / n_rated)
        
        first_pos = first_session[first_session["rating"] == 1]
        first_neg = first_session[first_session["rating"] == 0]
        
        # Pos-Pos similarity
        if len(first_pos) > 1:
            pos_embeddings = embedding.matrix[embedding.get_idxs(first_pos["paper_id"].tolist())]
            first_pos_pos_similarity.append(compute_sims_same_set(pos_embeddings))
        
        # Neg-Neg similarity
        if len(first_neg) > 1:
            neg_embeddings = embedding.matrix[embedding.get_idxs(first_neg["paper_id"].tolist())]
            first_neg_neg_similarity.append(compute_sims_same_set(neg_embeddings))
        
        # Pos-Neg similarity
        if len(first_pos) > 0 and len(first_neg) > 0:
            pos_embeddings = embedding.matrix[embedding.get_idxs(first_pos["paper_id"].tolist())]
            neg_embeddings = embedding.matrix[embedding.get_idxs(first_neg["paper_id"].tolist())]
            first_pos_neg_similarity.append(compute_sims(pos_embeddings, neg_embeddings))
    
    # Other sessions (session_id > 0)
    other_sessions = user_ratings[user_ratings["session_id"] > 0]
    if len(other_sessions) > 0:
        session_ids = other_sessions["session_id"].unique()
        
        session_n_papers = []
        session_pos_ratios = []
        session_pos_pos_similarities = []
        session_neg_neg_similarities = []
        session_pos_neg_similarities = []
        
        for session_id in session_ids:
            session_data = other_sessions[other_sessions["session_id"] == session_id]
            
            session_n_papers.append(len(session_data))
            
            n_pos = (session_data["rating"] == 1).sum()
            n_rated = (session_data["rating"] != 2).sum()
            if n_rated > 0:
                session_pos_ratios.append(n_pos / n_rated)
            
            session_pos = session_data[session_data["rating"] == 1]
            session_neg = session_data[session_data["rating"] == 0]
            
            # Pos-Pos similarity
            if len(session_pos) > 1:
                pos_embeddings = embedding.matrix[embedding.get_idxs(session_pos["paper_id"].tolist())]
                session_pos_pos_similarities.append(compute_sims_same_set(pos_embeddings))
            
            # Neg-Neg similarity
            if len(session_neg) > 1:
                neg_embeddings = embedding.matrix[embedding.get_idxs(session_neg["paper_id"].tolist())]
                session_neg_neg_similarities.append(compute_sims_same_set(neg_embeddings))
            
            # Pos-Neg similarity
            if len(session_pos) > 0 and len(session_neg) > 0:
                pos_embeddings = embedding.matrix[embedding.get_idxs(session_pos["paper_id"].tolist())]
                neg_embeddings = embedding.matrix[embedding.get_idxs(session_neg["paper_id"].tolist())]
                session_pos_neg_similarities.append(compute_sims(pos_embeddings, neg_embeddings))
        
        # Average across this user's other sessions
        if session_n_papers:
            other_n_papers.append(np.mean(session_n_papers))
        if session_pos_ratios:
            other_pos_ratio.append(np.mean(session_pos_ratios))
        if session_pos_pos_similarities:
            other_pos_pos_similarity.append(np.mean(session_pos_pos_similarities))
        if session_neg_neg_similarities:
            other_neg_neg_similarity.append(np.mean(session_neg_neg_similarities))
        if session_pos_neg_similarities:
            other_pos_neg_similarity.append(np.mean(session_pos_neg_similarities))

# Final averages
print(f"Number of papers: First={np.mean(first_n_papers):.1f}, Other={np.mean(other_n_papers):.1f}")
print(f"Positive ratio: First={np.mean(first_pos_ratio):.3f}, Other={np.mean(other_pos_ratio):.3f}")
print(f"Pos-Pos similarity: First={np.mean(first_pos_pos_similarity):.3f}, Other={np.mean(other_pos_pos_similarity):.3f}")
print(f"Neg-Neg similarity: First={np.mean(first_neg_neg_similarity):.3f}, Other={np.mean(other_neg_neg_similarity):.3f}")
print(f"Pos-Neg similarity: First={np.mean(first_pos_neg_similarity):.3f}, Other={np.mean(other_pos_neg_similarity):.3f}")