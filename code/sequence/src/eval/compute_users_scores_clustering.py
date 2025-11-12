from enum import Enum, auto

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
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
from .compute_users_scores_upper_bound import get_user_scores_upper_bound

DT = np.float64
N_EVAL_NEGATIVE_SAMPLES = 100


class ClusteringApproach(Enum):
    NONE = auto()
    INCREMENTAL_K_MEANS = auto()
    K_MEANS_FIXED_K = auto()
    K_MEAN_SELECTION_SILHOUETTE = auto()
    UPPER_BOUND = auto()
    VAL_SPLIT = auto()


def get_clustering_approach_from_arg(arg: str) -> ClusteringApproach:
    arg = arg.lower()
    if arg == "none":
        return ClusteringApproach.NONE
    elif arg == "incremental_k_means":
        return ClusteringApproach.INCREMENTAL_K_MEANS
    elif arg == "k_means_fixed_k":
        return ClusteringApproach.K_MEANS_FIXED_K
    elif arg == "k_means_selection_silhouette":
        return ClusteringApproach.K_MEAN_SELECTION_SILHOUETTE
    elif arg == "upper_bound":
        return ClusteringApproach.UPPER_BOUND
    elif arg == "val_split":
        return ClusteringApproach.VAL_SPLIT
    else:
        raise ValueError(f"Unknown clustering approach: {arg}")


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

    for logits_string in [
        "y_train_rated_logits",
        "y_val_logits",
        "y_negative_samples_logits",
        "y_negative_samples_logits_after_train",
    ]:
        logits = user_scores[logits_string]
        proba_string = logits_string.replace("logits", "proba")
        pred_string = logits_string.replace("logits", "pred")
        user_scores[proba_string] = proba(logits)
        user_scores[pred_string] = pred(logits)
    return user_scores


def train_logreg_single_cluster(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_rated: int,
    rated_time_diffs: np.ndarray,
    random_state: int,
    eval_settings: dict,
    is_cluster: bool = False,
    pos_cluster_in_idxs: np.ndarray = None,
    pos_cluster_out_idxs: np.ndarray = None,
    neg_cluster_in_idxs: np.ndarray = None,
    neg_cluster_out_idxs: np.ndarray = None,
) -> object:
    sample_weights = get_sample_weights(
        y_train=y_train,
        n_rated=n_rated,
        rated_time_diffs=rated_time_diffs,
        eval_settings=eval_settings,
        is_cluster=is_cluster,
        pos_cluster_in_idxs=pos_cluster_in_idxs,
        pos_cluster_out_idxs=pos_cluster_out_idxs,
        neg_cluster_in_idxs=neg_cluster_in_idxs,
        neg_cluster_out_idxs=neg_cluster_out_idxs,
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
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_rated: int,
    rated_time_diffs: np.ndarray,
    random_state: int,
    eval_settings: dict,
) -> tuple:
    n_pos = np.sum(y_train == 1)
    assert n_pos >= eval_settings["clustering_selection_min_cluster_size"]
    logreg = train_logreg_single_cluster(
        X_train=X_train,
        y_train=y_train,
        n_rated=n_rated,
        rated_time_diffs=rated_time_diffs,
        random_state=random_state,
        eval_settings=eval_settings,
    )
    return logreg, [], None, []


def train_models_clustering_k_means_fixed_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_rated: int,
    rated_time_diffs: np.ndarray,
    random_state: int,
    eval_settings: dict,
) -> tuple:
    global_logreg = train_models_clustering_none(
        X_train=X_train,
        y_train=y_train,
        n_rated=n_rated,
        rated_time_diffs=rated_time_diffs,
        random_state=random_state,
        eval_settings=eval_settings,
    )[0]

    n_clusters = eval_settings["clustering_k_means_n_clusters"]
    assert n_clusters >= 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, algorithm="elkan", n_init=10)
    X_rated, y_rated = X_train[:n_rated], y_train[:n_rated]
    pos_rated_mask, neg_rated_mask = y_rated == 1, y_rated == 0
    pos_clusters_labels = kmeans.fit_predict(X_rated[pos_rated_mask])
    neg_clusters_labels = kmeans.predict(X_rated[neg_rated_mask])
    pos_original_idxs = np.where(pos_rated_mask)[0]
    neg_original_idxs = np.where(neg_rated_mask)[0]
    clusters_logregs, clusters_with_sufficient_size = [], []
    for cluster_label in range(n_clusters):
        pos_cluster_in_idxs = pos_original_idxs[np.where(pos_clusters_labels == cluster_label)[0]]
        pos_n_cluster_idxs = len(pos_cluster_in_idxs)
        if pos_n_cluster_idxs < eval_settings["clustering_selection_min_cluster_size"]:
            continue
        pos_cluster_out_idxs = pos_original_idxs[np.where(pos_clusters_labels != cluster_label)[0]]
        assert len(pos_cluster_in_idxs) + len(pos_cluster_out_idxs) == len(pos_original_idxs)
        neg_cluster_in_idxs = neg_original_idxs[np.where(neg_clusters_labels == cluster_label)[0]]
        neg_cluster_out_idxs = neg_original_idxs[np.where(neg_clusters_labels != cluster_label)[0]]
        assert len(neg_cluster_in_idxs) + len(neg_cluster_out_idxs) == len(neg_original_idxs)
        sample_weights = get_sample_weights(
            y_train=y_train,
            n_rated=n_rated,
            rated_time_diffs=rated_time_diffs,
            eval_settings=eval_settings,
            is_cluster=True,
            pos_cluster_in_idxs=pos_cluster_in_idxs,
            pos_cluster_out_idxs=pos_cluster_out_idxs,
            neg_cluster_in_idxs=neg_cluster_in_idxs,
            neg_cluster_out_idxs=neg_cluster_out_idxs,
        )
        logreg = get_model(
            algorithm=Algorithm.LOGREG,
            max_iter=eval_settings["logreg_max_iter"],
            clf_C=eval_settings["logreg_clf_C"],
            random_state=random_state,
            logreg_solver=eval_settings["logreg_solver"],
        )
        logreg.fit(X_train, y_train, sample_weight=sample_weights)
        clusters_with_sufficient_size.append(cluster_label)
        clusters_logregs.append(logreg)
    return global_logreg, clusters_logregs, kmeans, clusters_with_sufficient_size


def check_single_cluster(
    clustering_approach: ClusteringApproach, eval_settings: dict, n_pos: int
) -> bool:
    if clustering_approach == ClusteringApproach.NONE:
        return True
    if clustering_approach == ClusteringApproach.K_MEANS_FIXED_K:
        if eval_settings.get("clustering_k_means_n_clusters", 1) == 1:
            return True
    min_n_posrated = eval_settings.get("clustering_min_n_posrated", None)
    if min_n_posrated is not None and n_pos < min_n_posrated:
        return True
    max_n_posrated = eval_settings.get("clustering_max_n_posrated", None)
    if max_n_posrated is not None and n_pos > max_n_posrated:
        return True
    return False


def get_X_y_train(
    train_set_embeddings: np.ndarray,
    train_set_ratings: np.ndarray,
    X_cache: np.ndarray,
) -> tuple:
    is_sparse = sparse.isspmatrix(train_set_embeddings) or sparse.isspmatrix(X_cache)
    if is_sparse:
        X_train = sparse.vstack([train_set_embeddings, X_cache])
    else:
        X_train = np.vstack([train_set_embeddings, X_cache])
    y_cache = np.zeros(X_cache.shape[0], dtype=np.int64)
    y_train = np.hstack([train_set_ratings, y_cache])
    return X_train, y_train


def train_models_clustering(
    train_set_embeddings: np.ndarray,
    train_set_ratings: np.ndarray,
    train_set_time_diffs: np.ndarray,
    X_cache: np.ndarray,
    random_state: int,
    eval_settings: dict,
) -> tuple:
    clustering_approach = eval_settings.get("clustering_approach", ClusteringApproach.NONE)
    n_pos, n_rated = np.sum(train_set_ratings == 1), train_set_ratings.shape[0]
    single_cluster = check_single_cluster(clustering_approach, eval_settings, n_pos)
    X_train, y_train = get_X_y_train(
        train_set_embeddings=train_set_embeddings,
        train_set_ratings=train_set_ratings,
        X_cache=X_cache,
    )
    if single_cluster:
        return train_models_clustering_none(
            X_train=X_train,
            y_train=y_train,
            n_rated=n_rated,
            rated_time_diffs=train_set_time_diffs,
            random_state=random_state,
            eval_settings=eval_settings,
        )
    if clustering_approach == ClusteringApproach.K_MEANS_FIXED_K:
        return train_models_clustering_k_means_fixed_k(
            X_train=X_train,
            y_train=y_train,
            n_rated=n_rated,
            rated_time_diffs=train_set_time_diffs,
            random_state=random_state,
            eval_settings=eval_settings,
        )
    elif clustering_approach == ClusteringApproach.UPPER_BOUND:
        clusters_dict = {}
        clusters_dict[1] = train_models_clustering_none(
            X_train=X_train,
            y_train=y_train,
            n_rated=n_rated,
            rated_time_diffs=train_set_time_diffs,
            random_state=random_state,
            eval_settings=eval_settings,
        )
        for n_clusters in [2, 3, 4, 5, 7, 10]:
            eval_settings_copy = eval_settings.copy()
            eval_settings_copy["clustering_k_means_n_clusters"] = n_clusters
            clusters_dict[n_clusters] = train_models_clustering_k_means_fixed_k(
                X_train=X_train,
                y_train=y_train,
                n_rated=n_rated,
                rated_time_diffs=train_set_time_diffs,
                random_state=random_state,
                eval_settings=eval_settings_copy,
            )
        return clusters_dict


def scoring_function_clustering(
    global_logreg: object,
    clusters_logregs: list,
    clustering_model: object,
    clusters_with_sufficient_size: list,
    embeddings_to_score: np.ndarray,
    embeddings_ratings: np.ndarray = None,
) -> tuple:
    if len(clusters_logregs) == 0:
        n_global_model_pos, n_global_model_neg = None, None
        if embeddings_ratings is not None:
            n_global_model_pos = np.sum(embeddings_ratings == 1)
            n_global_model_neg = np.sum(embeddings_ratings == 0)
        return (
            global_logreg.decision_function(embeddings_to_score),
            n_global_model_pos,
            n_global_model_neg,
        )

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
    n_global_model_pos = None
    n_global_model_neg = None
    if embeddings_ratings is not None:
        unassigned_mask = ~assigned_mask
        n_global_model_pos = np.sum(embeddings_ratings[unassigned_mask] == 1)
        n_global_model_neg = np.sum(embeddings_ratings[unassigned_mask] == 0)
    return logits, n_global_model_pos, n_global_model_neg


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


def get_train_rated_logits_dict_upper_bound(
    val_data_dict: dict,
    val_negative_samples_embeddings: np.ndarray,
    train_negrated_ranking_idxs: np.ndarray,
    X_cache: np.ndarray,
    random_state: int,
    eval_settings: dict,
) -> tuple:
    assert eval_settings["clustering_approach"] == ClusteringApproach.UPPER_BOUND
    clusters_dict = train_models_clustering(
        train_set_embeddings=val_data_dict["X_train_rated"],
        train_set_ratings=val_data_dict["y_train_rated"],
        train_set_time_diffs=None,
        X_cache=X_cache,
        random_state=random_state,
        eval_settings=eval_settings,
    )
    global_logreg = clusters_dict[1][0]
    train_rated_logits_dict = {
        "y_train_rated_logits": {},
        "y_train_negrated_ranking_logits": {},
        "y_negative_samples_logits_after_train": {},
    }
    for n_clusters, clusters in clusters_dict.items():
        _, clusters_logregs, clustering_model, clusters_with_sufficient_size = clusters
        (
            y_train_rated_logits,
            y_train_negrated_ranking_logits,
            y_negative_samples_logits_after_train,
        ) = get_train_rated_logits_dict_components_single_k(
            val_data_dict=val_data_dict,
            train_negrated_ranking_idxs=train_negrated_ranking_idxs,
            val_negative_samples_embeddings=val_negative_samples_embeddings,
            global_logreg=global_logreg,
            clusters_logregs=clusters_logregs,
            clustering_model=clustering_model,
            clusters_with_sufficient_size=clusters_with_sufficient_size,
        )
        train_rated_logits_dict["y_train_rated_logits"][n_clusters] = y_train_rated_logits
        train_rated_logits_dict["y_train_negrated_ranking_logits"][
            n_clusters
        ] = y_train_negrated_ranking_logits
        train_rated_logits_dict["y_negative_samples_logits_after_train"][
            n_clusters
        ] = y_negative_samples_logits_after_train
    return train_rated_logits_dict, global_logreg


def get_train_rated_logits_dict_components_single_k(
    val_data_dict: dict,
    train_negrated_ranking_idxs: np.ndarray,
    val_negative_samples_embeddings: np.ndarray,
    global_logreg: object,
    clusters_logregs: list,
    clustering_model: object,
    clusters_with_sufficient_size: list,
) -> tuple:
    y_train_rated_logits, _, _ = scoring_function_clustering(
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
    y_negative_samples_logits_after_train, _, _ = scoring_function_clustering(
        global_logreg=global_logreg,
        clusters_logregs=clusters_logregs,
        clustering_model=clustering_model,
        clusters_with_sufficient_size=clusters_with_sufficient_size,
        embeddings_to_score=val_negative_samples_embeddings,
    )
    return (
        y_train_rated_logits,
        y_train_negrated_ranking_logits,
        y_negative_samples_logits_after_train,
    )


def get_train_rated_logits_dict(
    val_data_dict: dict,
    val_negative_samples_embeddings: np.ndarray,
    train_negrated_ranking_idxs: np.ndarray,
    X_cache: np.ndarray,
    random_state: int,
    eval_settings: dict,
) -> tuple:
    eval_settings = eval_settings.copy()
    eval_settings["logreg_temporal_decay"] = "none"

    if eval_settings["clustering_approach"] == ClusteringApproach.UPPER_BOUND:
        return get_train_rated_logits_dict_upper_bound(
            val_data_dict=val_data_dict,
            val_negative_samples_embeddings=val_negative_samples_embeddings,
            train_negrated_ranking_idxs=train_negrated_ranking_idxs,
            X_cache=X_cache,
            random_state=random_state,
            eval_settings=eval_settings,
        )

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
    y_train_rated_logits, y_train_negrated_ranking_logits, y_negative_samples_logits_after_train = (
        get_train_rated_logits_dict_components_single_k(
            val_data_dict=val_data_dict,
            train_negrated_ranking_idxs=train_negrated_ranking_idxs,
            val_negative_samples_embeddings=val_negative_samples_embeddings,
            global_logreg=global_logreg,
            clusters_logregs=clusters_logregs,
            clustering_model=clustering_model,
            clusters_with_sufficient_size=clusters_with_sufficient_size,
        )
    )
    train_rated_logits_dict = {
        "y_train_rated_logits": y_train_rated_logits,
        "y_train_negrated_ranking_logits": y_train_negrated_ranking_logits,
        "y_negative_samples_logits_after_train": y_negative_samples_logits_after_train,
    }
    return train_rated_logits_dict, global_logreg


def get_y_val_logits_session(
    global_logreg: object,
    clusters_logregs: list,
    clustering_model: object,
    clusters_with_sufficient_size: list,
    session_embeddings: np.ndarray,
    val_counter_all: int,
    n_session_papers: int,
    y_val_logits: np.ndarray,
    embeddings_ratings: np.ndarray = None,
) -> tuple:
    session_logits, n_global_model_pos, n_global_model_neg = scoring_function_clustering(
        global_logreg=global_logreg,
        clusters_logregs=clusters_logregs,
        clustering_model=clustering_model,
        clusters_with_sufficient_size=clusters_with_sufficient_size,
        embeddings_to_score=session_embeddings,
        embeddings_ratings=embeddings_ratings,
    )
    y_val_logits[val_counter_all : val_counter_all + n_session_papers] = session_logits
    return y_val_logits, n_global_model_pos, n_global_model_neg


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
    val_negrated_ranking_logits_session, _, _ = scoring_function_clustering(
        global_logreg=global_logreg,
        clusters_logregs=clusters_logregs,
        clustering_model=clustering_model,
        clusters_with_sufficient_size=clusters_with_sufficient_size,
        embeddings_to_score=val_negrated_embeddings_session,
    )
    val_negrated_ranking_logits_session = val_negrated_ranking_logits_session.reshape(
        (-1, N_NEGRATED_RANKING)
    )
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
    y_negative_samples_logits_session, _, _ = scoring_function_clustering(
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


def fill_users_scores_models_properties(logreg_models: dict, user_scores: dict) -> dict:
    logreg_coefs_cat = [model.coef_[0, :] for model in logreg_models]
    logreg_coefs_no_cat = [coef[:256] for coef in logreg_coefs_cat]
    d = {}

    for i, logreg_coefs_list in enumerate([logreg_coefs_cat, logreg_coefs_no_cat]):
        for j, coef in enumerate(logreg_coefs_list):
            s = "cat" if i == 0 else "no_cat"

            d.setdefault(f"norm_{s}", []).append(np.linalg.norm(coef))
            d.setdefault(f"mean_abs_coef_{s}", []).append(np.mean(np.abs(coef)))
            d.setdefault(f"max_abs_coef_{s}", []).append(np.max(np.abs(coef)))
            d.setdefault(f"std_coef_{s}", []).append(np.std(coef))

            threshold = 1e-3
            d.setdefault(f"ratio_nonzero_{s}", []).append(np.mean(np.abs(coef) >= threshold))
            d.setdefault(f"num_nonzero_{s}", []).append(np.count_nonzero(coef))

            if j > 0:
                prev_coef = logreg_coefs_list[j - 1]
                d.setdefault(f"distance_to_prev_{s}", []).append(np.linalg.norm(coef - prev_coef))
                sim_prev = cosine_similarity(coef.reshape(1, -1), prev_coef.reshape(1, -1))[0, 0]
                d.setdefault(f"sim_to_prev_{s}", []).append(sim_prev)
                init_coef = logreg_coefs_list[0]
                d.setdefault(f"distance_to_init_{s}", []).append(np.linalg.norm(coef - init_coef))
                sim_init = cosine_similarity(coef.reshape(1, -1), init_coef.reshape(1, -1))[0, 0]
                d.setdefault(f"sim_to_init_{s}", []).append(sim_init)

                k = 10
                top_k_curr = set(np.argsort(np.abs(coef))[-k:])
                top_k_prev = set(np.argsort(np.abs(prev_coef))[-k:])
                top_k_init = set(np.argsort(np.abs(init_coef))[-k:])
                d.setdefault(f"top{k}_overlap_prev_{s}", []).append(
                    len(top_k_curr & top_k_prev) / k
                )
                d.setdefault(f"top{k}_overlap_init_{s}", []).append(
                    len(top_k_curr & top_k_init) / k
                )
    for key, values in d.items():
        user_scores[key] = np.array(values, dtype=DT)
    return user_scores


def get_y_logits_components_single_k(
    global_logreg: object,
    clusters_logregs: list,
    clustering_model: object,
    clusters_with_sufficient_size: list,
    session_embeddings: np.ndarray,
    val_counter_all: int,
    n_session_papers: int,
    y_val_logits: np.ndarray,
    val_negrated_ranking_idxs: np.ndarray,
    val_negrated_embeddings: np.ndarray,
    val_counter_pos: int,
    n_session_papers_pos: int,
    y_val_negrated_ranking_logits: np.ndarray,
    val_negative_samples_embeddings: np.ndarray,
    y_negative_samples_logits: np.ndarray,
    embeddings_ratings: np.ndarray,
) -> tuple:
    y_val_logits, n_global_model_pos, n_global_model_neg = get_y_val_logits_session(
        global_logreg=global_logreg,
        clusters_logregs=clusters_logregs,
        clustering_model=clustering_model,
        clusters_with_sufficient_size=clusters_with_sufficient_size,
        session_embeddings=session_embeddings,
        val_counter_all=val_counter_all,
        n_session_papers=n_session_papers,
        y_val_logits=y_val_logits,
        embeddings_ratings=embeddings_ratings,
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
    return (
        y_val_logits,
        y_val_negrated_ranking_logits,
        y_negative_samples_logits,
        n_global_model_pos,
        n_global_model_neg,
    )


def init_y_logits_components(
    eval_settings: dict, val_data_dict: dict, n_val_pos: int, n_val_negative_samples: int
) -> tuple:
    clustering_approach = eval_settings.get("clustering_approach", ClusteringApproach.NONE)
    if clustering_approach == ClusteringApproach.UPPER_BOUND:
        r = [1, 2, 3, 4, 5, 7, 10]
        y_val_logits = {k: np.zeros(val_data_dict["y_val"].shape[0], dtype=DT) for k in r}
        y_val_negrated_ranking_logits = {
            k: np.zeros((n_val_pos, N_NEGRATED_RANKING), dtype=DT) for k in r
        }
        y_negative_samples_logits = {
            k: np.zeros((n_val_pos, n_val_negative_samples), dtype=DT) for k in r
        }
        n_global_model_pos, n_global_model_neg = {k: [] for k in r}, {k: [] for k in r}
        n_clusters_with_sufficient_size = {k: [] for k in r}
    else:
        y_val_logits = np.zeros(val_data_dict["y_val"].shape[0], dtype=DT)
        y_val_negrated_ranking_logits = np.zeros((n_val_pos, N_NEGRATED_RANKING), dtype=DT)
        y_negative_samples_logits = np.zeros((n_val_pos, n_val_negative_samples), dtype=DT)
        n_global_model_pos, n_global_model_neg = [], []
        n_clusters_with_sufficient_size = []
    return (
        y_val_logits,
        y_val_negrated_ranking_logits,
        y_negative_samples_logits,
        n_global_model_pos,
        n_global_model_neg,
        n_clusters_with_sufficient_size,
    )


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
        train_rated_logits_dict, global_logreg = get_train_rated_logits_dict(
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
        n_val_negative_samples = val_negative_samples_embeddings.shape[0]
        (
            y_val_logits,
            y_val_negrated_ranking_logits,
            y_negative_samples_logits,
            n_global_model_pos,
            n_global_model_neg,
            n_clusters_with_sufficient_size,
        ) = init_y_logits_components(
            eval_settings=eval_settings,
            val_data_dict=val_data_dict,
            n_val_pos=n_val_pos,
            n_val_negative_samples=n_val_negative_samples,
        )
        val_counter_all, val_counter_pos = 0, 0

        logreg_models = [global_logreg]
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

            if eval_settings["clustering_approach"] == ClusteringApproach.UPPER_BOUND:
                clusters_dict = train_models_clustering(
                    train_set_embeddings=user_train_set_embeddings,
                    train_set_ratings=user_train_set_ratings,
                    train_set_time_diffs=user_train_set_time_diffs,
                    X_cache=user_params["X_cache"],
                    random_state=random_state,
                    eval_settings=eval_settings,
                )
                global_logreg = clusters_dict[1][0]
                for n_clusters, clusters in clusters_dict.items():
                    _, clusters_logregs, clustering_model, clusters_with_sufficient_size = clusters
                    y_logits = get_y_logits_components_single_k(
                        global_logreg=global_logreg,
                        clusters_logregs=clusters_logregs,
                        clustering_model=clustering_model,
                        clusters_with_sufficient_size=clusters_with_sufficient_size,
                        session_embeddings=session_embeddings,
                        val_counter_all=val_counter_all,
                        n_session_papers=n_session_papers,
                        y_val_logits=y_val_logits[n_clusters],
                        val_negrated_ranking_idxs=val_negrated_ranking_idxs,
                        val_negrated_embeddings=val_negrated_embeddings,
                        val_counter_pos=val_counter_pos,
                        n_session_papers_pos=n_session_papers_pos,
                        y_val_negrated_ranking_logits=y_val_negrated_ranking_logits[n_clusters],
                        val_negative_samples_embeddings=val_negative_samples_embeddings,
                        y_negative_samples_logits=y_negative_samples_logits[n_clusters],
                        embeddings_ratings=val_data_dict["y_val"][
                            val_counter_all : val_counter_all + n_session_papers
                        ],
                    )
                    y_val_logits[n_clusters] = y_logits[0]
                    y_val_negrated_ranking_logits[n_clusters] = y_logits[1]
                    y_negative_samples_logits[n_clusters] = y_logits[2]
                    n_global_model_pos[n_clusters].append(y_logits[3])
                    n_global_model_neg[n_clusters].append(y_logits[4])
                    n_clusters_with_sufficient_size_k = (
                        1 if n_clusters == 1 else len(clusters_with_sufficient_size)
                    )
                    n_clusters_with_sufficient_size[n_clusters].append(
                        n_clusters_with_sufficient_size_k
                    )
            else:
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
                y_logits = get_y_logits_components_single_k(
                    global_logreg=global_logreg,
                    clusters_logregs=clusters_logregs,
                    clustering_model=clustering_model,
                    clusters_with_sufficient_size=clusters_with_sufficient_size,
                    session_embeddings=session_embeddings,
                    val_counter_all=val_counter_all,
                    n_session_papers=n_session_papers,
                    y_val_logits=y_val_logits,
                    val_negrated_ranking_idxs=val_negrated_ranking_idxs,
                    val_negrated_embeddings=val_negrated_embeddings,
                    val_counter_pos=val_counter_pos,
                    n_session_papers_pos=n_session_papers_pos,
                    y_val_negrated_ranking_logits=y_val_negrated_ranking_logits,
                    val_negative_samples_embeddings=val_negative_samples_embeddings,
                    y_negative_samples_logits=y_negative_samples_logits,
                    embeddings_ratings=val_data_dict["y_val"][
                        val_counter_all : val_counter_all + n_session_papers
                    ],
                )
                y_val_logits = y_logits[0]
                y_val_negrated_ranking_logits = y_logits[1]
                y_negative_samples_logits = y_logits[2]
                n_global_model_pos.append(y_logits[3])
                n_global_model_neg.append(y_logits[4])
                n_clusters_with_sufficient_size.append(len(clusters_with_sufficient_size))

            logreg_models.append(global_logreg)
            val_counter_all += n_session_papers
            val_counter_pos += n_session_papers_pos

        if eval_settings["clustering_approach"] == ClusteringApproach.UPPER_BOUND:
            user_scores = get_user_scores_upper_bound(
                user_ratings=user_ratings,
                user_sessions_ids=user_sessions_ids,
                train_rated_logits_dict=train_rated_logits_dict,
                y_val_logits=y_val_logits,
                y_val_negrated_ranking_logits=y_val_negrated_ranking_logits,
                y_negative_samples_logits=y_negative_samples_logits,
                eval_settings=eval_settings,
                n_global_model_pos=n_global_model_pos,
                n_global_model_neg=n_global_model_neg,
                n_clusters_with_sufficient_size=n_clusters_with_sufficient_size,
            )
        else:
            user_scores = {
                **train_rated_logits_dict,
                "y_val_logits": y_val_logits,
                "y_val_negrated_ranking_logits": y_val_negrated_ranking_logits,
                "y_negative_samples_logits": y_negative_samples_logits,
                "n_global_model_pos": np.array(n_global_model_pos, dtype=np.int64),
                "n_global_model_neg": np.array(n_global_model_neg, dtype=np.int64),
                "n_clusters_with_sufficient_size": np.array(
                    n_clusters_with_sufficient_size, dtype=np.int64
                ),
            }
        user_scores["y_val_logits_pos"] = user_scores["y_val_logits"][val_data_dict["y_val"] == 1]
        user_scores["y_val_logits_neg"] = user_scores["y_val_logits"][val_data_dict["y_val"] == 0]
        user_scores = fill_user_scores(user_scores)
        user_scores = fill_users_scores_models_properties(
            logreg_models=logreg_models, user_scores=user_scores
        )
        users_scores[user_id] = user_scores
    return users_scores
