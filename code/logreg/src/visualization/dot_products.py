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


def get_random_split(user_ratings: pd.DataFrame) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split
    user_ratings = user_ratings.copy()
    train_idx, val_idx = train_test_split(
        user_ratings.index,
        test_size=0.2,
        random_state=42,
        stratify=user_ratings['rating']
    )
    user_ratings['split'] = np.where(
        user_ratings.index.isin(val_idx),
        'val',
        'train'
    )
    return user_ratings

embedding = Embedding(ProjectPaths.logreg_embeddings_path() / "tfidf" / "tfidf_10k")

users_ratings_temporal = load_users_ratings_from_selection(
    users_ratings_selection=UsersRatingsSelection.MSC_LATE_SPLIT, relevant_users_ids = "finetuning_test"
)
users_ids = users_ratings_temporal["user_id"].unique().tolist()

train_pos_pos_temporal, train_pos_pos_random = [], []
train_neg_neg_temporal, train_neg_neg_random = [], []
val_pos_pos_temporal, val_pos_pos_random = [], []
val_neg_neg_temporal, val_neg_neg_random = [], []

temporal_stats = {
    'val_pos_train_pos': {'avg': [], 'max': []},
    'val_pos_train_neg': {'avg': [], 'max': []},
    'val_pos_val_neg': {'avg': [], 'max': []},
    'val_neg_train_pos': {'avg': [], 'max': []},
    'val_neg_train_neg': {'avg': [], 'max': []},
}
random_stats = {
    'val_pos_train_pos': {'avg': [], 'max': []},
    'val_pos_train_neg': {'avg': [], 'max': []},
    'val_pos_val_neg': {'avg': [], 'max': []},
    'val_neg_train_pos': {'avg': [], 'max': []},
    'val_neg_train_neg': {'avg': [], 'max': []},
}
prequential_stats = {
    'val_pos_train_pos': {'avg': [], 'max': []},
    'val_pos_train_neg': {'avg': [], 'max': []},
    'val_pos_val_neg': {'avg': [], 'max': []},
    'val_neg_train_pos': {'avg': [], 'max': []},
    'val_neg_train_neg': {'avg': [], 'max': []},
}

papers = load_papers(relevant_columns=["paper_id", "in_cache", "in_ratings", "l1", "l2"])

val_negative_samples_ids = get_val_negative_samples_ids(
    papers=papers,
    n_categories_samples=100,
    random_state=42,
    papers_ids_to_exclude=papers[papers["in_ratings"]]["paper_id"].tolist(),
)
users_significant_categories = load_users_significant_categories(
    relevant_users_ids=users_ids,
)

val_negative_samples_sims = []
val_negative_votes_sims = []
pos_ratings_sims = []
for user_id in tqdm(users_ids):
    user_significant_categories = users_significant_categories[
        users_significant_categories["user_id"] == user_id
    ]["category"].tolist()
    user_categories_ratios = get_user_categories_ratios(
        categories_to_exclude=user_significant_categories
    )
    user_val_negative_samples = get_user_val_negative_samples(
        val_negative_samples_ids=val_negative_samples_ids,
        n_negative_samples=100,
        random_state=42,
        user_categories_ratios=user_categories_ratios,
        embedding=embedding,
    )
    val_negative_samples_embeddings = user_val_negative_samples[
        "val_negative_samples_embeddings"
    ]
    user_ratings_full = users_ratings_temporal[
        users_ratings_temporal["user_id"] == user_id
    ]
    user_ratings_pos = user_ratings_full[user_ratings_full["rating"] == 1]
    user_ratings_neg = user_ratings_full[user_ratings_full["rating"] == 0]
    embeddings_pos = embedding.matrix[embedding.get_idxs(user_ratings_pos["paper_id"].tolist())]
    embeddings_neg = embedding.matrix[embedding.get_idxs(user_ratings_neg["paper_id"].tolist())]
    sims_negative_samples = compute_sims(embeddings_pos, val_negative_samples_embeddings)
    sims_negative_votes = compute_sims(embeddings_pos, embeddings_neg)
    val_negative_samples_sims.append(sims_negative_samples)
    val_negative_votes_sims.append(sims_negative_votes)
    sims_pos = compute_sims_same_set(embeddings_pos)
    pos_ratings_sims.append(sims_pos)
print("Average similarity of positive ratings to negative samples:", np.mean(val_negative_samples_sims))
print("Average similarity of negative ratings to negative samples:", np.mean(val_negative_votes_sims))
print("Average similarity among positive ratings:", np.mean(pos_ratings_sims))


def get_sims(user_ratings: pd.DataFrame) -> dict:
    user_ratings_train = user_ratings[user_ratings["split"] == "train"]
    user_ratings_val = user_ratings[user_ratings["split"] == "val"]
    pos_train_papers_ids = user_ratings_train[user_ratings_train["rating"] == 1]["paper_id"].tolist()
    neg_train_papers_ids = user_ratings_train[user_ratings_train["rating"] == 0]["paper_id"].tolist()
    pos_val_papers_ids = user_ratings_val[user_ratings_val["rating"] == 1]["paper_id"].tolist()
    neg_val_papers_ids = user_ratings_val[user_ratings_val["rating"] == 0]["paper_id"].tolist()
    all_avg_sims = {
        'val_pos_train_pos': [],
        'val_pos_train_neg': [],
        'val_pos_val_neg': [],
        'val_neg_train_pos': [],
        'val_neg_train_neg': [],
    }
    all_max_sims = {
        'val_pos_train_pos': [],
        'val_pos_train_neg': [],
        'val_pos_val_neg': [],
        'val_neg_train_pos': [],
        'val_neg_train_neg': [],
    }
    if len(pos_train_papers_ids) > 0 and len(pos_val_papers_ids) > 0:
        pos_train_embeddings = embedding.matrix[embedding.get_idxs(pos_train_papers_ids)]
        pos_val_embeddings = embedding.matrix[embedding.get_idxs(pos_val_papers_ids)]
        cos_sims = cosine_similarity(pos_val_embeddings, pos_train_embeddings)
        all_avg_sims['val_pos_train_pos'] = np.mean(cos_sims, axis=1).tolist()
        all_max_sims['val_pos_train_pos'] = np.max(cos_sims, axis=1).tolist()
    if len(neg_train_papers_ids) > 0 and len(pos_val_papers_ids) > 0:
        neg_train_embeddings = embedding.matrix[embedding.get_idxs(neg_train_papers_ids)]
        pos_val_embeddings = embedding.matrix[embedding.get_idxs(pos_val_papers_ids)]
        cos_sims = cosine_similarity(pos_val_embeddings, neg_train_embeddings)
        all_avg_sims['val_pos_train_neg'] = np.mean(cos_sims, axis=1).tolist()
        all_max_sims['val_pos_train_neg'] = np.max(cos_sims, axis=1).tolist()
    if len(pos_train_papers_ids) > 0 and len(neg_val_papers_ids) > 0:
        pos_train_embeddings = embedding.matrix[embedding.get_idxs(pos_train_papers_ids)]
        neg_val_embeddings = embedding.matrix[embedding.get_idxs(neg_val_papers_ids)]
        cos_sims = cosine_similarity(neg_val_embeddings, pos_train_embeddings)
        all_avg_sims['val_neg_train_pos'] = np.mean(cos_sims, axis=1).tolist()
        all_max_sims['val_neg_train_pos'] = np.max(cos_sims, axis=1).tolist()
    if len(neg_train_papers_ids) > 0 and len(neg_val_papers_ids) > 0:
        neg_train_embeddings = embedding.matrix[embedding.get_idxs(neg_train_papers_ids)]
        neg_val_embeddings = embedding.matrix[embedding.get_idxs(neg_val_papers_ids)]
        cos_sims = cosine_similarity(neg_val_embeddings, neg_train_embeddings)
        all_avg_sims['val_neg_train_neg'] = np.mean(cos_sims, axis=1).tolist()
        all_max_sims['val_neg_train_neg'] = np.max(cos_sims, axis=1).tolist()
    for session_id in user_ratings_val['session_id'].unique():
        session_ratings = user_ratings_val[user_ratings_val['session_id'] == session_id]
        pos_session = session_ratings[session_ratings['rating'] == 1]
        neg_session = session_ratings[session_ratings['rating'] == 0]
        if len(pos_session) > 0 and len(neg_session) > 0:
            pos_embeddings = embedding.matrix[embedding.get_idxs(pos_session['paper_id'].tolist())]
            neg_embeddings = embedding.matrix[embedding.get_idxs(neg_session['paper_id'].tolist())]
            cos_sims = cosine_similarity(pos_embeddings, neg_embeddings)
            all_avg_sims['val_pos_val_neg'].extend(np.mean(cos_sims, axis=1).tolist())
            all_max_sims['val_pos_val_neg'].extend(np.max(cos_sims, axis=1).tolist())
    return {key: {'avg': all_avg_sims[key], 'max': all_max_sims[key]} for key in all_avg_sims.keys()}

def get_sims_prequential(user_ratings: pd.DataFrame) -> dict:
    user_ratings_val = user_ratings[user_ratings["split"] == "val"]
    all_avg_sims = {
        'val_pos_train_pos': [],
        'val_pos_train_neg': [],
        'val_pos_val_neg': [],
        'val_neg_train_pos': [],
        'val_neg_train_neg': [],
    }
    all_max_sims = {
        'val_pos_train_pos': [],
        'val_pos_train_neg': [],
        'val_pos_val_neg': [],
        'val_neg_train_pos': [],
        'val_neg_train_neg': [],
    }
    for idx, val_row in user_ratings_val.iterrows():
        val_session_id = val_row['session_id']
        prior_ratings = user_ratings[user_ratings['session_id'] < val_session_id]
        if len(prior_ratings) == 0:
            continue
        prior_pos = prior_ratings[prior_ratings['rating'] == 1]
        prior_neg = prior_ratings[prior_ratings['rating'] == 0]
        val_embedding = embedding.matrix[embedding.get_idxs([val_row['paper_id']])]
        if val_row['rating'] == 1:
            if len(prior_pos) > 0:
                prior_pos_embeddings = embedding.matrix[embedding.get_idxs(prior_pos['paper_id'].tolist())]
                cos_sims = cosine_similarity(val_embedding, prior_pos_embeddings)
                all_avg_sims['val_pos_train_pos'].append(np.mean(cos_sims))
                all_max_sims['val_pos_train_pos'].append(np.max(cos_sims))
            if len(prior_neg) > 0:
                prior_neg_embeddings = embedding.matrix[embedding.get_idxs(prior_neg['paper_id'].tolist())]
                cos_sims = cosine_similarity(val_embedding, prior_neg_embeddings)
                all_avg_sims['val_pos_train_neg'].append(np.mean(cos_sims))
                all_max_sims['val_pos_train_neg'].append(np.max(cos_sims))
        else:
            if len(prior_pos) > 0:
                prior_pos_embeddings = embedding.matrix[embedding.get_idxs(prior_pos['paper_id'].tolist())]
                cos_sims = cosine_similarity(val_embedding, prior_pos_embeddings)
                all_avg_sims['val_neg_train_pos'].append(np.mean(cos_sims))
                all_max_sims['val_neg_train_pos'].append(np.max(cos_sims))
            if len(prior_neg) > 0:
                prior_neg_embeddings = embedding.matrix[embedding.get_idxs(prior_neg['paper_id'].tolist())]
                cos_sims = cosine_similarity(val_embedding, prior_neg_embeddings)
                all_avg_sims['val_neg_train_neg'].append(np.mean(cos_sims))
                all_max_sims['val_neg_train_neg'].append(np.max(cos_sims))
    for session_id in user_ratings_val['session_id'].unique():
        session_ratings = user_ratings_val[user_ratings_val['session_id'] == session_id]
        pos_session = session_ratings[session_ratings['rating'] == 1]
        neg_session = session_ratings[session_ratings['rating'] == 0]
        if len(pos_session) > 0 and len(neg_session) > 0:
            pos_embeddings = embedding.matrix[embedding.get_idxs(pos_session['paper_id'].tolist())]
            neg_embeddings = embedding.matrix[embedding.get_idxs(neg_session['paper_id'].tolist())]
            cos_sims = cosine_similarity(pos_embeddings, neg_embeddings)
            all_avg_sims['val_pos_val_neg'].extend(np.mean(cos_sims, axis=1).tolist())
            all_max_sims['val_pos_val_neg'].extend(np.max(cos_sims, axis=1).tolist())
    return {key: {'avg': all_avg_sims[key], 'max': all_max_sims[key]} for key in all_avg_sims.keys()}

for user_id in tqdm(users_ids):
    user_ratings_temporal = users_ratings_temporal[users_ratings_temporal["user_id"] == user_id]
    user_ratings_random = get_random_split(user_ratings_temporal)
    val_temporal = user_ratings_temporal[user_ratings_temporal["split"] == "val"]
    val_random = user_ratings_random[user_ratings_random["split"] == "val"]
    train_temporal = user_ratings_temporal[user_ratings_temporal["split"] == "train"]
    train_random = user_ratings_random[user_ratings_random["split"] == "train"]
    val_temporal_pos = val_temporal[val_temporal["rating"] == 1]
    val_temporal_neg = val_temporal[val_temporal["rating"] == 0]
    train_temporal_pos = train_temporal[train_temporal["rating"] == 1]
    train_temporal_neg = train_temporal[train_temporal["rating"] == 0]
    val_random_pos = val_random[val_random["rating"] == 1]
    val_random_neg = val_random[val_random["rating"] == 0]
    train_random_pos = train_random[train_random["rating"] == 1]
    train_random_neg = train_random[train_random["rating"] == 0]
    val_temporal_pos_embeddings = embedding.matrix[embedding.get_idxs(val_temporal_pos["paper_id"].tolist())]
    val_temporal_neg_embeddings = embedding.matrix[embedding.get_idxs(val_temporal_neg["paper_id"].tolist())]
    train_temporal_pos_embeddings = embedding.matrix[embedding.get_idxs(train_temporal_pos["paper_id"].tolist())]
    train_temporal_neg_embeddings = embedding.matrix[embedding.get_idxs(train_temporal_neg["paper_id"].tolist())]
    val_random_pos_embeddings = embedding.matrix[embedding.get_idxs(val_random_pos["paper_id"].tolist())]
    val_random_neg_embeddings = embedding.matrix[embedding.get_idxs(val_random_neg["paper_id"].tolist())]
    train_random_pos_embeddings = embedding.matrix[embedding.get_idxs(train_random_pos["paper_id"].tolist())]
    train_random_neg_embeddings = embedding.matrix[embedding.get_idxs(train_random_neg["paper_id"].tolist())]
    train_pos_pos_temporal.append(compute_sims_same_set(train_temporal_pos_embeddings))
    train_neg_neg_temporal.append(compute_sims_same_set(train_temporal_neg_embeddings))
    val_pos_pos_temporal.append(compute_sims_same_set(val_temporal_pos_embeddings))
    val_neg_neg_temporal.append(compute_sims_same_set(val_temporal_neg_embeddings))
    train_pos_pos_random.append(compute_sims_same_set(train_random_pos_embeddings))
    train_neg_neg_random.append(compute_sims_same_set(train_random_neg_embeddings))
    val_pos_pos_random.append(compute_sims_same_set(val_random_pos_embeddings))
    val_neg_neg_random.append(compute_sims_same_set(val_random_neg_embeddings))
    temporal_sims = get_sims(user_ratings_temporal)
    random_sims = get_sims(user_ratings_random)
    prequential_sims = get_sims_prequential(user_ratings_temporal)
    for key in temporal_stats.keys():
        if len(temporal_sims[key]['avg']) > 0:
            temporal_stats[key]['avg'].extend(temporal_sims[key]['avg'])
            temporal_stats[key]['max'].extend(temporal_sims[key]['max'])
        if len(random_sims[key]['avg']) > 0:
            random_stats[key]['avg'].extend(random_sims[key]['avg'])
            random_stats[key]['max'].extend(random_sims[key]['max'])
        if len(prequential_sims[key]['avg']) > 0:
            prequential_stats[key]['avg'].extend(prequential_sims[key]['avg'])
            prequential_stats[key]['max'].extend(prequential_sims[key]['max'])

print("TEMPORAL SPLIT:")
for key in temporal_stats.keys():
    if len(temporal_stats[key]['avg']) > 0:
        avg_vals = [v for v in temporal_stats[key]['avg'] if not np.isinf(v)]
        max_vals = [v for v in temporal_stats[key]['max'] if not np.isinf(v)]
        if len(avg_vals) > 0:
            avg_mean = np.mean(avg_vals)
            max_mean = np.mean(max_vals)
            print(f"{key}: avg={avg_mean:.4f}, max={max_mean:.4f}")
        else:
            print(f"{key}: no data")
    else:
        print(f"{key}: no data")
print(f"train_pos_pos: {np.mean(np.array(train_pos_pos_temporal)):.4f}, train_neg_neg: {np.mean(np.array(train_neg_neg_temporal)):.4f}")
print(f"val_pos_pos: {np.mean(np.array(val_pos_pos_temporal)):.4f}, val_neg_neg: {np.mean(np.array(val_neg_neg_temporal)):.4f}")
print("\nRANDOM SPLIT:")
for key in random_stats.keys():
    if len(random_stats[key]['avg']) > 0:
        avg_vals = [v for v in random_stats[key]['avg'] if not np.isinf(v)]
        max_vals = [v for v in random_stats[key]['max'] if not np.isinf(v)]
        if len(avg_vals) > 0:
            avg_mean = np.mean(avg_vals)
            max_mean = np.mean(max_vals)
            print(f"{key}: avg={avg_mean:.4f}, max={max_mean:.4f}")
        else:
            print(f"{key}: no data")
    else:
        print(f"{key}: no data")
print(f"train_pos_pos: {np.mean(np.array(train_pos_pos_random)):.4f}, train_neg_neg: {np.mean(np.array(train_neg_neg_random)):.4f}")
print(f"val_pos_pos: {np.mean(np.array(val_pos_pos_random)):.4f}, val_neg_neg: {np.mean(np.array(val_neg_neg_random)):.4f}")
print("\nPREQUENTIAL SPLIT:")
for key in prequential_stats.keys():
    if len(prequential_stats[key]['avg']) > 0:
        avg_vals = [v for v in prequential_stats[key]['avg'] if not np.isinf(v)]
        max_vals = [v for v in prequential_stats[key]['max'] if not np.isinf(v)]
        if len(avg_vals) > 0:
            avg_mean = np.mean(avg_vals)
            max_mean = np.mean(max_vals)
            print(f"{key}: avg={avg_mean:.4f}, max={max_mean:.4f}")
        else:
            print(f"{key}: no data")
    else:
        print(f"{key}: no data")