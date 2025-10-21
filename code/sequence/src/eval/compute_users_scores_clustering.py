from enum import Enum, auto

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.cluster import KMeans
from tqdm import tqdm

from ....logreg.src.embeddings.embedding import Embedding
from ....logreg.src.training.algorithm import Algorithm, get_model
from ....logreg.src.training.evaluation import split_negrated_ranking, split_ratings
from ....logreg.src.training.scores import load_user_data_dicts
from ....logreg.src.training.training_data import load_negrated_ranking_idxs_for_user
from ....logreg.src.training.users_ratings import N_NEGRATED_RANKING
from ....src.load_files import load_papers
from ..data.users_embeddings_data import get_users_val_sessions_ids
from .compute_users_embeddings import (
    compute_session_min_time,
    get_user_train_set,
    init_users_ratings,
)
from .compute_users_embeddings_logreg import (
    get_sample_weights,
    logreg_get_embed_function_params,
    logreg_transform_embed_function_params,
)

DT = np.float64
N_EVAL_NEGATIVE_SAMPLES = 100


class ClusteringApproach(Enum):
    NONE = auto()
    K_MEANS_FIXED_K = auto()
    K_MEAN_SELECTION_SILHOUETTE = auto()


def get_clustering_approach(approach_str: str) -> ClusteringApproach:
    approach_str = approach_str.lower()
    if approach_str == "none":
        return ClusteringApproach.NONE
    elif approach_str == "k_means_fixed_k":
        return ClusteringApproach.K_MEANS_FIXED_K
    elif approach_str == "k_means_selection_silhouette":
        return ClusteringApproach.K_MEAN_SELECTION_SILHOUETTE
    else:
        raise ValueError(f"Unknown clustering approach: {approach_str}")


def get_user_train_embeddings_and_ratings(
    user_ratings: pd.DataFrame,
    session_id: int,
    embedding: Embedding,
    hard_constraint_min_n_train_posrated: int,
    hard_constraint_max_n_train_rated: int = None,
    soft_constraint_max_n_train_sessions: int = None,
    soft_constraint_max_n_train_days: int = None,
    remove_negrated_from_history: bool = False,
) -> tuple:
    session_min_time = compute_session_min_time(user_ratings, session_id)
    user_train_set = get_user_train_set(
        user_ratings=user_ratings,
        session_id=session_id,
        hard_constraint_min_n_train_posrated=hard_constraint_min_n_train_posrated,
        hard_constraint_max_n_train_rated=hard_constraint_max_n_train_rated,
        soft_constraint_max_n_train_sessions=soft_constraint_max_n_train_sessions,
        soft_constraint_max_n_train_days=soft_constraint_max_n_train_days,
        remove_negrated_from_history=remove_negrated_from_history,
    )
    user_train_set_time_diffs = (session_min_time - user_train_set["time"]).dt.days.to_numpy()
    user_train_set_papers_ids = user_train_set["paper_id"].tolist()
    user_train_set_embeddings = embedding.matrix[embedding.get_idxs(user_train_set_papers_ids)]
    user_train_set_ratings = user_train_set["rating"].values
    return user_train_set_embeddings, user_train_set_ratings, user_train_set_time_diffs


def fill_user_scores(user_scores: dict) -> dict:
    def proba(logits):
        return expit(logits).astype(DT)

    def pred(logits):
        return (logits > 0).astype(np.int64)

    user_scores["y_train_rated_proba"] = proba(user_scores["y_train_rated_logits"])
    user_scores["y_train_rated_pred"] = pred(user_scores["y_train_rated_logits"])
    user_scores["y_val_proba"] = proba(user_scores["y_val_logits"])
    user_scores["y_val_pred"] = pred(user_scores["y_val_logits"])
    user_scores["y_negative_samples_proba"] = proba(user_scores["y_negative_samples_logits"])
    user_scores["y_negative_samples_pred"] = pred(user_scores["y_negative_samples_logits"])
    user_scores["y_negative_samples_proba_after_train"] = proba(
        user_scores["y_negative_samples_logits_after_train"]
    )
    user_scores["y_negative_samples_pred_after_train"] = pred(
        user_scores["y_negative_samples_logits_after_train"]
    )
    return user_scores


def train_logreg_single_cluster(
    train_set_embeddings: np.ndarray,
    train_set_ratings: np.ndarray,
    train_set_time_diffs: np.ndarray,
    X_cache: np.ndarray,
    random_state: int,
    eval_settings: dict,
) -> object:
    is_sparse = sparse.isspmatrix(train_set_embeddings) or sparse.isspmatrix(X_cache)
    if is_sparse:
        X_train = sparse.vstack([train_set_embeddings, X_cache])
    else:
        X_train = np.vstack([train_set_embeddings, X_cache])
    y_cache = np.zeros(X_cache.shape[0], dtype=np.int64)
    y_train = np.hstack([train_set_ratings, y_cache])
    sample_weights = get_sample_weights(
        train_set_ratings, train_set_time_diffs, X_cache.shape[0], eval_settings
    )
    logreg = get_model(
        algorithm=Algorithm.LOGREG,
        max_iter=eval_settings["logreg_max_iter"],
        clf_C=eval_settings["logreg_clf_C"],
        random_state=random_state,
        logreg_solver=eval_settings["logreg_solver"],
    )
    logreg.fit(X_train, y_train, sample_weight=sample_weights)
    return logreg


def train_models_clustering_none(
    train_set_embeddings: np.ndarray,
    train_set_ratings: np.ndarray,
    train_set_time_diffs: np.ndarray,
    X_cache: np.ndarray,
    random_state: int,
    eval_settings: dict,
) -> tuple:
    logreg = train_logreg_single_cluster(
        train_set_embeddings=train_set_embeddings,
        train_set_ratings=train_set_ratings,
        train_set_time_diffs=train_set_time_diffs,
        X_cache=X_cache,
        random_state=random_state,
        eval_settings=eval_settings,
    )
    return logreg, [], None, []


def train_models_clustering_k_means_fixed_k(
    train_set_embeddings: np.ndarray,
    train_set_ratings: np.ndarray,
    train_set_time_diffs: np.ndarray,
    X_cache: np.ndarray,
    random_state: int,
    eval_settings: dict,
) -> tuple:
    global_logreg = train_logreg_single_cluster(
        train_set_embeddings=train_set_embeddings,
        train_set_ratings=train_set_ratings,
        train_set_time_diffs=train_set_time_diffs,
        X_cache=X_cache,
        random_state=random_state,
        eval_settings=eval_settings,
    )
    pos_mask, neg_mask = train_set_ratings == 1, train_set_ratings == 0
    train_set_embeddings_pos = train_set_embeddings[pos_mask]
    train_set_embeddings_neg = train_set_embeddings[neg_mask]
    train_set_ratings_neg = np.zeros(
        train_set_embeddings_neg.shape[0], dtype=train_set_ratings.dtype
    )
    train_set_time_diffs_pos, train_set_time_diffs_neg = None, None
    if train_set_time_diffs is not None:
        train_set_time_diffs_pos = train_set_time_diffs[pos_mask]
        train_set_time_diffs_neg = train_set_time_diffs[neg_mask]

    n_clusters = eval_settings["clustering_k_means_n_clusters"]
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, algorithm="elkan")
    clusters_labels = kmeans.fit_predict(train_set_embeddings_pos)

    clusters_logregs, clusters_with_sufficient_size = [], []
    for cluster_label in range(n_clusters):
        cluster_idxs = np.where(clusters_labels == cluster_label)[0]
        n_cluster_idxs = len(cluster_idxs)
        if n_cluster_idxs < eval_settings["clustering_selection_min_cluster_size"]:
            continue
        train_set_embeddings_pos_cluster = train_set_embeddings_pos[cluster_idxs]
        cluster_embeddings = np.vstack([train_set_embeddings_pos_cluster, train_set_embeddings_neg])
        train_set_ratings_pos_cluster = np.ones(n_cluster_idxs, dtype=train_set_ratings.dtype)
        cluster_ratings = np.hstack([train_set_ratings_pos_cluster, train_set_ratings_neg])
        if train_set_time_diffs is not None:
            train_set_time_diffs_pos_cluster = train_set_time_diffs_pos[cluster_idxs]
            cluster_time_diffs = np.hstack(
                [train_set_time_diffs_pos_cluster, train_set_time_diffs_neg]
            )
        else:
            cluster_time_diffs = None
        is_sparse = sparse.isspmatrix(cluster_embeddings) or sparse.isspmatrix(X_cache)
        if is_sparse:
            X_cluster_train = sparse.vstack([cluster_embeddings, X_cache])
        else:
            X_cluster_train = np.vstack([cluster_embeddings, X_cache])
        y_cache = np.zeros(X_cache.shape[0], dtype=np.int64)
        y_cluster_train = np.hstack([cluster_ratings, y_cache])
        sample_weights = get_sample_weights(
            cluster_ratings, cluster_time_diffs, X_cache.shape[0], eval_settings
        )
        logreg = get_model(
            algorithm=Algorithm.LOGREG,
            max_iter=eval_settings["logreg_max_iter"],
            clf_C=eval_settings["logreg_clf_C"],
            random_state=random_state,
            logreg_solver=eval_settings["logreg_solver"],
        )
        logreg.fit(X_cluster_train, y_cluster_train, sample_weight=sample_weights)
        clusters_with_sufficient_size.append(cluster_label)
        clusters_logregs.append(logreg)
    return global_logreg, clusters_logregs, kmeans, clusters_with_sufficient_size


def train_models_clustering(
    train_set_embeddings: np.ndarray,
    train_set_ratings: np.ndarray,
    train_set_time_diffs: np.ndarray,
    X_cache: np.ndarray,
    random_state: int,
    eval_settings: dict,
) -> tuple:
    clustering_approach = eval_settings.get("clustering_approach", ClusteringApproach.NONE)
    if clustering_approach == ClusteringApproach.NONE:
        return train_models_clustering_none(
            train_set_embeddings=train_set_embeddings,
            train_set_ratings=train_set_ratings,
            train_set_time_diffs=train_set_time_diffs,
            X_cache=X_cache,
            random_state=random_state,
            eval_settings=eval_settings,
        )
    elif clustering_approach == ClusteringApproach.K_MEANS_FIXED_K:
        return train_models_clustering_k_means_fixed_k(
            train_set_embeddings=train_set_embeddings,
            train_set_ratings=train_set_ratings,
            train_set_time_diffs=train_set_time_diffs,
            X_cache=X_cache,
            random_state=random_state,
            eval_settings=eval_settings,
        )


def scoring_function_clustering(
    global_logreg: object,
    clusters_logregs: list,
    clustering_model: object,
    clusters_with_sufficient_size: list,
    embeddings_to_score: np.ndarray,
) -> np.ndarray:
    if len(clusters_logregs) == 0:
        return global_logreg.decision_function(embeddings_to_score)

    clusters_labels = clustering_model.predict(embeddings_to_score)
    logits = np.zeros(embeddings_to_score.shape[0], dtype=DT)
    assigned_mask = np.zeros(embeddings_to_score.shape[0], dtype=bool)
    for cluster_label, logreg in zip(clusters_with_sufficient_size, clusters_logregs):
        cluster_idxs = np.where(clusters_labels == cluster_label)[0]
        if len(cluster_idxs) == 0:
            continue
        cluster_embeddings = embeddings_to_score[cluster_idxs]
        cluster_logits = logreg.decision_function(cluster_embeddings)
        logits[cluster_idxs] = cluster_logits
        assigned_mask[cluster_idxs] = True
    unassigned_idxs = np.where(~assigned_mask)[0]
    if len(unassigned_idxs) > 0:
        unassigned_embeddings = embeddings_to_score[unassigned_idxs]
        unassigned_logits = global_logreg.decision_function(unassigned_embeddings)
        logits[unassigned_idxs] = unassigned_logits
    return logits


def get_negrated_ranking_idxs(
    train_ratings: pd.DataFrame,
    train_negrated_ranking: pd.DataFrame,
    val_ratings: pd.DataFrame,
    val_negrated_ranking: pd.DataFrame,
    random_state: int,
) -> tuple:
    train_negrated_ranking_idxs = load_negrated_ranking_idxs_for_user(
        ratings=train_ratings,
        negrated_ranking=train_negrated_ranking,
        timesort=True,
        causal_mask=False,
        random_state=random_state,
        same_negrated_for_all_pos=False,
    )
    if "old_session_id" in val_ratings.columns:
        val_ratings = val_ratings.copy()
        val_ratings["session_id"] = val_ratings["old_session_id"]
        val_negrated_ranking = val_negrated_ranking.copy()
        val_negrated_ranking["session_id"] = val_negrated_ranking["old_session_id"]
    val_negrated_ranking_idxs = load_negrated_ranking_idxs_for_user(
        ratings=val_ratings,
        negrated_ranking=val_negrated_ranking,
        timesort=True,
        causal_mask=True,
        random_state=random_state,
        same_negrated_for_all_pos=False,
    )
    return train_negrated_ranking_idxs, val_negrated_ranking_idxs


def get_train_rated_logits_dict(
    val_data_dict: dict,
    val_negative_samples_embeddings: np.ndarray,
    train_negrated_ranking_idxs: np.ndarray,
    X_cache: np.ndarray,
    random_state: int,
    eval_settings: dict,
) -> dict:
    eval_settings = eval_settings.copy()
    eval_settings["logreg_temporal_decay"] = "none"
    global_logreg, clusters_logregs, clustering_model, clusters_with_sufficient_size = (
        train_models_clustering(
            train_set_embeddings=val_data_dict["X_train_rated"],
            train_set_ratings=val_data_dict["y_train_rated"],
            train_set_time_diffs=None,
            X_cache=X_cache,
            random_state=random_state,
            eval_settings=eval_settings,
        )
    )
    y_train_rated_logits = scoring_function_clustering(
        global_logreg=global_logreg,
        clusters_logregs=clusters_logregs,
        clustering_model=clustering_model,
        clusters_with_sufficient_size=clusters_with_sufficient_size,
        embeddings_to_score=val_data_dict["X_train_rated"],
    )
    train_negrated_ranking_idxs = train_negrated_ranking_idxs.reshape(-1)
    y_train_negrated_ranking_logits = y_train_rated_logits[train_negrated_ranking_idxs].reshape(
        (-1, N_NEGRATED_RANKING)
    )
    y_negative_samples_logits_after_train = scoring_function_clustering(
        global_logreg=global_logreg,
        clusters_logregs=clusters_logregs,
        clustering_model=clustering_model,
        clusters_with_sufficient_size=clusters_with_sufficient_size,
        embeddings_to_score=val_negative_samples_embeddings,
    )
    return {
        "y_train_rated_logits": y_train_rated_logits,
        "y_train_negrated_ranking_logits": y_train_negrated_ranking_logits,
        "y_negative_samples_logits_after_train": y_negative_samples_logits_after_train,
    }


def get_y_val_logits_session(
    global_logreg: object,
    clusters_logregs: list,
    clustering_model: object,
    clusters_with_sufficient_size: list,
    session_embeddings: np.ndarray,
    val_counter_all: int,
    n_session_papers: int,
    y_val_logits: np.ndarray,
) -> np.ndarray:
    session_logits = scoring_function_clustering(
        global_logreg=global_logreg,
        clusters_logregs=clusters_logregs,
        clustering_model=clustering_model,
        clusters_with_sufficient_size=clusters_with_sufficient_size,
        embeddings_to_score=session_embeddings,
    )
    y_val_logits[val_counter_all : val_counter_all + n_session_papers] = session_logits
    return y_val_logits


def get_y_val_negrated_ranking_logits_session(
    global_logreg: object,
    clusters_logregs: list,
    clustering_model: object,
    clusters_with_sufficient_size: list,
    val_negrated_ranking_idxs: np.ndarray,
    val_negrated_embeddings: np.ndarray,
    val_counter_pos: int,
    n_session_papers_pos: int,
    y_val_negrated_ranking_logits: np.ndarray,
) -> np.ndarray:
    if n_session_papers_pos <= 0:
        return y_val_negrated_ranking_logits
    val_negrated_ranking_idxs_session = val_negrated_ranking_idxs[
        val_counter_pos : val_counter_pos + n_session_papers_pos
    ]
    val_negrated_ranking_idxs_session_flat = val_negrated_ranking_idxs_session.reshape(-1)

    val_negrated_embeddings_session = val_negrated_embeddings[
        val_negrated_ranking_idxs_session_flat
    ]
    val_negrated_ranking_logits_session = scoring_function_clustering(
        global_logreg=global_logreg,
        clusters_logregs=clusters_logregs,
        clustering_model=clustering_model,
        clusters_with_sufficient_size=clusters_with_sufficient_size,
        embeddings_to_score=val_negrated_embeddings_session,
    ).reshape((-1, N_NEGRATED_RANKING))
    y_val_negrated_ranking_logits[val_counter_pos : val_counter_pos + n_session_papers_pos] = (
        val_negrated_ranking_logits_session
    )
    return y_val_negrated_ranking_logits


def get_y_negative_samples_logits_session(
    global_logreg: object,
    clusters_logregs: list,
    clustering_model: object,
    clusters_with_sufficient_size: list,
    val_negative_samples_embeddings: np.ndarray,
    val_counter_pos: int,
    n_session_papers_pos: int,
    y_negative_samples_logits: np.ndarray,
) -> np.ndarray:
    if n_session_papers_pos <= 0:
        return y_negative_samples_logits
    y_negative_samples_logits_session = scoring_function_clustering(
        global_logreg=global_logreg,
        clusters_logregs=clusters_logregs,
        clustering_model=clustering_model,
        clusters_with_sufficient_size=clusters_with_sufficient_size,
        embeddings_to_score=val_negative_samples_embeddings,
    )
    y_negative_samples_logits_session = np.tile(
        y_negative_samples_logits_session, (n_session_papers_pos, 1)
    )
    y_negative_samples_logits[val_counter_pos : val_counter_pos + n_session_papers_pos] = (
        y_negative_samples_logits_session
    )
    return y_negative_samples_logits


def compute_users_scores_clustering(eval_settings: dict, random_state: int) -> dict:
    users_ratings = init_users_ratings(eval_settings)
    users_val_sessions_ids = get_users_val_sessions_ids(users_ratings)
    embedding = Embedding(eval_settings["papers_embedding_path"])

    papers = load_papers(relevant_columns=["paper_id", "in_cache", "in_ratings", "l1", "l2"])
    users_ratings = users_ratings.merge(papers[["paper_id", "l1", "l2"]], on="paper_id", how="left")

    users_ids = users_ratings["user_id"].unique().tolist()
    logreg_params = logreg_get_embed_function_params(
        users_ids=users_ids, random_state=random_state, eval_settings=eval_settings
    )

    users_scores = {}
    for user_id in tqdm(users_ids, desc="Computing users scores with clustering"):
        user_ratings = users_ratings[users_ratings["user_id"] == user_id]
        user_params = logreg_transform_embed_function_params(
            user_id=user_id,
            user_ratings=user_ratings,
            embedding=embedding,
            random_state=random_state,
            users_significant_categories=logreg_params["users_significant_categories"],
            val_negative_samples_ids=logreg_params["val_negative_samples_ids"],
            cache_papers_categories_ids=logreg_params["cache_papers_categories_ids"],
            cache_papers_ids=logreg_params["cache_papers_ids"],
            eval_settings=eval_settings,
            compute_val_negative_samples_embeddings=True,
            n_negative_samples=N_EVAL_NEGATIVE_SAMPLES,
        )
        user_sessions_ids = users_val_sessions_ids[user_id]
        train_ratings, val_ratings, removed_ratings = split_ratings(user_ratings)
        train_negrated_ranking, val_negrated_ranking = split_negrated_ranking(
            train_ratings, val_ratings, removed_ratings
        )
        _, val_data_dict = load_user_data_dicts(
            train_ratings=train_ratings,
            val_ratings=val_ratings,
            train_negrated_ranking=train_negrated_ranking,
            val_negrated_ranking=val_negrated_ranking,
            embedding=embedding,
            load_user_train_data_dict_bool=False,
        )
        train_negrated_ranking_idxs, val_negrated_ranking_idxs = get_negrated_ranking_idxs(
            train_ratings=train_ratings,
            train_negrated_ranking=train_negrated_ranking,
            val_ratings=val_ratings,
            val_negrated_ranking=val_negrated_ranking,
            random_state=random_state,
        )
        val_negative_samples_embeddings = user_params["val_negative_samples_embeddings"]
        train_rated_logits_dict = get_train_rated_logits_dict(
            val_data_dict=val_data_dict,
            val_negative_samples_embeddings=val_negative_samples_embeddings,
            train_negrated_ranking_idxs=train_negrated_ranking_idxs,
            X_cache=user_params["X_cache"],
            random_state=random_state,
            eval_settings=eval_settings,
        )

        val_negrated_papers_ids = val_negrated_ranking["paper_id"].tolist()
        val_negrated_embeddings = embedding.matrix[embedding.get_idxs(val_negrated_papers_ids)]
        n_val_pos = (val_data_dict["y_val"] > 0).sum()
        y_val_logits = np.zeros(val_data_dict["y_val"].shape[0], dtype=DT)
        y_val_negrated_ranking_logits = np.zeros((n_val_pos, N_NEGRATED_RANKING), dtype=DT)
        n_val_negative_samples = val_negative_samples_embeddings.shape[0]
        y_negative_samples_logits = np.zeros((n_val_pos, n_val_negative_samples), dtype=DT)
        val_counter_all, val_counter_pos = 0, 0

        for session_id in user_sessions_ids:
            user_train_set_embeddings, user_train_set_ratings, user_train_set_time_diffs = (
                get_user_train_embeddings_and_ratings(
                    user_ratings=user_ratings,
                    session_id=session_id,
                    embedding=embedding,
                    hard_constraint_min_n_train_posrated=eval_settings[
                        "histories_hard_constraint_min_n_train_posrated"
                    ],
                    hard_constraint_max_n_train_rated=eval_settings[
                        "histories_hard_constraint_max_n_train_rated"
                    ],
                    soft_constraint_max_n_train_sessions=eval_settings[
                        "histories_soft_constraint_max_n_train_sessions"
                    ],
                    soft_constraint_max_n_train_days=eval_settings[
                        "histories_soft_constraint_max_n_train_days"
                    ],
                    remove_negrated_from_history=eval_settings[
                        "histories_remove_negrated_from_history"
                    ],
                )
            )
            session_ratings = user_ratings[user_ratings["session_id"] == session_id]
            n_session_papers = session_ratings.shape[0]
            n_session_papers_pos = (session_ratings["rating"] > 0).sum()
            session_embeddings = val_data_dict["X_val"][
                val_counter_all : val_counter_all + n_session_papers
            ]
            global_logreg, clusters_logregs, clustering_model, clusters_with_sufficient_size = (
                train_models_clustering(
                    train_set_embeddings=user_train_set_embeddings,
                    train_set_ratings=user_train_set_ratings,
                    train_set_time_diffs=user_train_set_time_diffs,
                    X_cache=user_params["X_cache"],
                    random_state=random_state,
                    eval_settings=eval_settings,
                )
            )

            y_val_logits = get_y_val_logits_session(
                global_logreg=global_logreg,
                clusters_logregs=clusters_logregs,
                clustering_model=clustering_model,
                clusters_with_sufficient_size=clusters_with_sufficient_size,
                session_embeddings=session_embeddings,
                val_counter_all=val_counter_all,
                n_session_papers=n_session_papers,
                y_val_logits=y_val_logits,
            )
            y_val_negrated_ranking_logits = get_y_val_negrated_ranking_logits_session(
                global_logreg=global_logreg,
                clusters_logregs=clusters_logregs,
                clustering_model=clustering_model,
                clusters_with_sufficient_size=clusters_with_sufficient_size,
                val_negrated_ranking_idxs=val_negrated_ranking_idxs,
                val_negrated_embeddings=val_negrated_embeddings,
                val_counter_pos=val_counter_pos,
                n_session_papers_pos=n_session_papers_pos,
                y_val_negrated_ranking_logits=y_val_negrated_ranking_logits,
            )
            y_negative_samples_logits = get_y_negative_samples_logits_session(
                global_logreg=global_logreg,
                clusters_logregs=clusters_logregs,
                clustering_model=clustering_model,
                clusters_with_sufficient_size=clusters_with_sufficient_size,
                val_negative_samples_embeddings=val_negative_samples_embeddings,
                val_counter_pos=val_counter_pos,
                n_session_papers_pos=n_session_papers_pos,
                y_negative_samples_logits=y_negative_samples_logits,
            )
            val_counter_all += n_session_papers
            val_counter_pos += n_session_papers_pos

        user_scores = {
            **train_rated_logits_dict,
            "y_val_logits": y_val_logits,
            "y_val_negrated_ranking_logits": y_val_negrated_ranking_logits,
            "y_negative_samples_logits": y_negative_samples_logits,
            "y_val_logits_pos": y_val_logits[val_data_dict["y_val"] == 1],
        }
        user_scores = fill_user_scores(user_scores)
        users_scores[user_id] = user_scores
    return users_scores
