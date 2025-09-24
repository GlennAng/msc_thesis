import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from ....src.load_files import load_papers, load_sequence_users_ids, load_users_ratings
from ....src.project_paths import ProjectPaths
from .embedding import Embedding


def get_sequence_users_ids_all():
    users_ids_dict = load_sequence_users_ids()
    users_ids = []
    for key in users_ids_dict:
        users_ids += users_ids_dict[key]
    users_ids = sorted(users_ids)
    return users_ids


def get_users_ratings_for_multi_interest_analysis(users_ids: list = None) -> pd.DataFrame:
    if users_ids is None:
        users_ids = get_sequence_users_ids_all()
    users_ratings = load_users_ratings(relevant_users_ids=users_ids)
    users_ratings = users_ratings[users_ratings["rating"] == 1]
    papers = load_papers(relevant_columns=["paper_id", "l1", "l2"])
    users_ratings = users_ratings.merge(papers, on="paper_id", how="left")
    return users_ratings


def check_sufficiently_many_l1l2_categories(
    user_ratings: pd.DataFrame, min_n_categories: int = 2, min_n_papers_per_category: int = 3
) -> bool:
    l1l2_counts = user_ratings.groupby(["l1", "l2"]).size()
    sufficient_categories = l1l2_counts[l1l2_counts >= min_n_papers_per_category]
    return len(sufficient_categories) >= min_n_categories


def compute_n_dbscan_clusters_user(
    user_ratings: pd.DataFrame,
    embedding: Embedding,
    eps: float = 0.4,
    min_n_samples_per_cluster: int = 3,
) -> int:
    papers_ids = user_ratings["paper_id"].tolist()
    papers_embeddings = embedding.matrix[embedding.get_idxs(papers_ids)]
    n_papers = len(papers_embeddings)
    assert n_papers >= (2 * min_n_samples_per_cluster)

    distance_matrix = cosine_distances(papers_embeddings)
    dbscan = DBSCAN(eps=eps, min_samples=min_n_samples_per_cluster, metric="precomputed")
    cluster_labels = dbscan.fit_predict(distance_matrix)
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    return n_clusters


def check_sufficiently_many_clusters(
    user_ratings: pd.DataFrame,
    embedding: Embedding,
    min_n_clusters: int = 2,
    eps: float = 0.4,
    min_n_samples_per_cluster: int = 3,
) -> bool:
    n_clusters = compute_n_dbscan_clusters_user(
        user_ratings,
        embedding=embedding,
        eps=eps,
        min_n_samples_per_cluster=min_n_samples_per_cluster,
    )
    return n_clusters >= min_n_clusters


def compute_pairwise_cosine_distances(user_ratings: pd.DataFrame, embedding: Embedding) -> float:
    papers_ids = user_ratings["paper_id"].tolist()
    papers_embeddings = embedding.matrix[embedding.get_idxs(papers_ids)]
    if len(papers_embeddings) < 2:
        return 0.0
    distances = cosine_distances(papers_embeddings)
    upper_tri_indices = np.triu_indices_from(distances, k=1)
    pairwise_distances = distances[upper_tri_indices]
    average_distance = np.mean(pairwise_distances)
    return average_distance


def find_users_ids_with_multiple_interests(
    n_max_users: int,
    users_ids: list = None,
    embedding: Embedding = None,
    min_n_l1l2_categories: int = 2,
    min_n_papers_per_l1l2_category: int = 3,
    min_n_clusters: int = 2,
    eps_clusters: float = 0.4,
    min_n_samples_per_cluster: int = 3,
) -> list:
    if users_ids is None:
        users_ids = get_sequence_users_ids_all()
    if embedding is None:
        embedding = Embedding(
            ProjectPaths.logreg_embeddings_path()
            / "after_pca"
            / "gte_large_256_categories_l2_unit_100"
        )
        embedding.matrix = embedding.matrix[:, :-100]

    users_ratings = get_users_ratings_for_multi_interest_analysis(users_ids=users_ids)
    pairwise_cosine_distances = []
    for user_id in users_ids:
        user_ratings = users_ratings[users_ratings["user_id"] == user_id]
        if not check_sufficiently_many_l1l2_categories(
            user_ratings,
            min_n_categories=min_n_l1l2_categories,
            min_n_papers_per_category=min_n_papers_per_l1l2_category,
        ):
            continue
        if not check_sufficiently_many_clusters(
            user_ratings,
            embedding=embedding,
            min_n_clusters=min_n_clusters,
            eps=eps_clusters,
            min_n_samples_per_cluster=min_n_samples_per_cluster,
        ):
            continue
        pairwise_cosine_distances_user = compute_pairwise_cosine_distances(
            user_ratings, embedding=embedding
        )
        pairwise_cosine_distances.append(
            {"user_id": user_id, "score": pairwise_cosine_distances_user}
        )
    n_users = min(n_max_users, len(pairwise_cosine_distances))
    pairwise_cosine_distances = pd.DataFrame(pairwise_cosine_distances)
    pairwise_cosine_distances.sort_values(by="score", ascending=False, inplace=True)
    return pairwise_cosine_distances.head(n_users)["user_id"].tolist()
