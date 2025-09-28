import numpy as np
import pandas as pd
from tqdm import tqdm

from ....logreg.src.embeddings.embedding import Embedding
from ....logreg.src.training.evaluation import split_negrated_ranking, split_ratings
from ....logreg.src.training.training_data import (
    get_user_categories_ratios,
    get_user_val_negative_samples,
    get_val_negative_samples_ids,
    load_negrated_ranking_idxs_for_user,
)
from ....logreg.src.training.users_ratings import N_NEGRATED_RANKING
from ....src.load_files import load_papers, load_users_significant_categories
from ..data.users_embeddings_data import get_users_val_sessions_ids
from .compute_users_embeddings import get_user_train_set, init_users_ratings, parse_args
from .compute_users_embeddings_utils import EmbedFunction, get_embed_function_from_arg

N_EVAL_NEGATIVE_SAMPLES = 100


def scoring_function_max_pos_pooling() -> None:
    pass


def scoring_function_mean_pos_pooling() -> None:
    pass


def get_negrated_ranking_idxs(user_ratings: pd.DataFrame, random_state: int) -> tuple:
    train_ratings, val_ratings, removed_ratings = split_ratings(user_ratings)
    train_negrated_ranking, val_negrated_ranking = split_negrated_ranking(
        train_ratings, val_ratings, removed_ratings
    )
    train_negrated_ranking_idxs = load_negrated_ranking_idxs_for_user(
        ratings=train_ratings,
        negrated_ranking=train_negrated_ranking,
        timesort=True,
        causal_mask=False,
        random_state=random_state,
        same_negrated_for_all_pos=False,
    )
    train_negrated_papers_ids = train_negrated_ranking["paper_id"].tolist()
    val_negrated_ranking_idxs = load_negrated_ranking_idxs_for_user(
        ratings=val_ratings,
        negrated_ranking=val_negrated_ranking,
        timesort=True,
        causal_mask=True,
        random_state=random_state,
        same_negrated_for_all_pos=False,
    )
    val_negrated_papers_ids = val_negrated_ranking["paper_id"].tolist()
    return (
        train_negrated_ranking_idxs,
        train_negrated_papers_ids,
        val_negrated_ranking_idxs,
        val_negrated_papers_ids,
    )


def get_val_negative_samples_embeddings(
    user_significant_categories: pd.DataFrame,
    val_negative_samples_ids: dict,
    random_state: int,
    embedding: Embedding,
) -> np.ndarray:
    user_categories_ratios = get_user_categories_ratios(
        categories_to_exclude=user_significant_categories
    )
    val_negative_samples_embeddings = get_user_val_negative_samples(
        val_negative_samples_ids=val_negative_samples_ids,
        n_negative_samples=N_EVAL_NEGATIVE_SAMPLES,
        random_state=random_state,
        user_categories_ratios=user_categories_ratios,
        embedding=embedding,
    )["val_negative_samples_embeddings"]
    return val_negative_samples_embeddings


def get_user_counts(user_ratings: pd.DataFrame) -> tuple[int, int, int, int]:
    train_ratings = user_ratings[user_ratings["split"] == "train"]
    n_train = train_ratings.shape[0]
    n_train_pos = train_ratings[train_ratings["rating"] == 1].shape[0]
    val_ratings = user_ratings[user_ratings["split"] == "val"]
    n_val = val_ratings.shape[0]
    n_val_pos = val_ratings[val_ratings["rating"] == 1].shape[0]
    return n_train, n_train_pos, n_val, n_val_pos


def get_user_embeddings_dict(
    user_ratings: pd.DataFrame,
    user_significant_categories: pd.DataFrame,
    embedding: Embedding,
    val_negative_samples_ids: dict,
    random_state: int,
) -> dict:
    embeddings = {}
    rated_papers_ids = user_ratings["paper_id"].tolist()
    embeddings["rated_embeddings"] = embedding.matrix[embedding.get_idxs(rated_papers_ids)]
    embeddings["val_negative_samples_embeddings"] = get_val_negative_samples_embeddings(
        user_significant_categories=user_significant_categories,
        val_negative_samples_ids=val_negative_samples_ids,
        random_state=random_state,
        embedding=embedding,
    )





def compute_users_scores_general(
    users_ratings: pd.DataFrame,
    users_val_sessions_ids: dict,
    val_negative_samples_ids: dict,
    embedding: Embedding,
    scoring_function: callable,
    random_state: int,
    hard_constraint_min_n_train_posrated: int,
    hard_constraint_max_n_train_rated: int = None,
    soft_constraint_max_n_train_sessions: int = None,
    soft_constraint_max_n_train_days: int = None,
    remove_negrated_from_history: bool = False,
) -> dict:
    users_ids = users_ratings["user_id"].unique().tolist()
    assert users_ids == list(users_val_sessions_ids.keys())
    users_significant_categories = load_users_significant_categories(
        relevant_users_ids=users_ids,
    )
    for user_id in tqdm(users_ids):
        user_ratings = users_ratings[users_ratings["user_id"] == user_id]
        user_significant_categories = users_significant_categories[
            users_significant_categories["user_id"] == user_id
        ]
        user_sessions_ids = users_val_sessions_ids[user_id]
        (
            train_negrated_ranking_idxs,
            train_negrated_papers_ids,
            val_negrated_ranking_idxs,
            val_negrated_papers_ids,
        ) = get_negrated_ranking_idxs(user_ratings=user_ratings, random_state=random_state)
        val_negative_samples_embeddings = get_val_negative_samples_embeddings(
            user_id=user_id,
            users_significant_categories=users_significant_categories,
            val_negative_samples_ids=val_negative_samples_ids,
            random_state=random_state,
            embedding=embedding,
        )
        n_train, n_train_pos, n_val, n_val_pos = get_user_counts(user_ratings)
        break

        y_train_rated_logits = np.zeros(n_train, dtype=np.float64)
        y_train_negrated_ranking_logits = np.zeros(
            (n_train_pos, N_NEGRATED_RANKING), dtype=np.float64
        )
        y_val_logits = np.zeros(n_val, dtype=np.float64)
        y_val_negrated_ranking_logits = np.zeros((n_val_pos, N_NEGRATED_RANKING), dtype=np.float64)
        y_negative_samples_logits = np.zeros((n_val_pos, N_EVAL_NEGATIVE_SAMPLES), dtype=np.float64)
        y_negative_samples_logits_after_train = np.zeros(N_EVAL_NEGATIVE_SAMPLES, dtype=np.float64)

        val_counter_all, val_counter_pos = 0, 0
        for i, session_id in enumerate(user_sessions_ids):
            user_train_set = get_user_train_set(
                user_ratings=user_ratings,
                session_id=session_id,
                hard_constraint_min_n_train_posrated=hard_constraint_min_n_train_posrated,
                hard_constraint_max_n_train_rated=hard_constraint_max_n_train_rated,
                soft_constraint_max_n_train_sessions=soft_constraint_max_n_train_sessions,
                soft_constraint_max_n_train_days=soft_constraint_max_n_train_days,
                remove_negrated_from_history=remove_negrated_from_history,
            )
            user_train_set_papers_ids = user_train_set["paper_id"].tolist()
            user_train_set_embeddings = embedding.matrix[
                embedding.get_idxs(user_train_set_papers_ids)
            ]
            session_ratings = user_ratings[user_ratings["session_id"] == session_id]
            session_pos_mask = session_ratings["rating"] == 1
            session_papers_ids = session_ratings["paper_id"].tolist()
            session_embeddings = embedding.matrix[embedding.get_idxs(session_papers_ids)]
            n_session = session_ratings.shape[0]
            session_scores = scoring_function()
            y_val_logits[val_counter_all : val_counter_all + n_session] = session_scores
            n_session_pos = session_pos_mask.sum()

    return {}


def compute_users_scores(eval_settings: dict, random_state: int = None) -> dict:
    users_ratings = init_users_ratings(eval_settings)
    users_val_sessions_ids = get_users_val_sessions_ids(users_ratings)
    embedding = Embedding(eval_settings["papers_embedding_path"])
    embed_function = get_embed_function_from_arg(eval_settings["embed_function"])

    papers = load_papers(relevant_columns=["paper_id", "in_cache", "in_ratings", "l1", "l2"])
    users_ratings = users_ratings.merge(papers[["paper_id", "l1", "l2"]], on="paper_id", how="left")
    val_negative_samples_ids = get_val_negative_samples_ids(
        papers=papers,
        n_categories_samples=N_EVAL_NEGATIVE_SAMPLES,
        random_state=random_state,
        papers_ids_to_exclude=papers[papers["in_ratings"]]["paper_id"].tolist(),
    )

    if embed_function == EmbedFunction.MAX_POS_POOLING_SCORES:
        scoring_function = scoring_function_max_pos_pooling
    elif embed_function == EmbedFunction.MEAN_POS_POOLING_SCORES:
        scoring_function = scoring_function_mean_pos_pooling
    else:
        raise ValueError(f"Unsupported embed_function: {embed_function}")
    return compute_users_scores_general(
        users_ratings=users_ratings,
        users_val_sessions_ids=users_val_sessions_ids,
        val_negative_samples_ids=val_negative_samples_ids,
        embedding=embedding,
        scoring_function=scoring_function,
        random_state=random_state,
        hard_constraint_min_n_train_posrated=eval_settings[
            "histories_hard_constraint_min_n_train_posrated"
        ],
        hard_constraint_max_n_train_rated=eval_settings.get(
            "histories_hard_constraint_max_n_train_rated", None
        ),
        soft_constraint_max_n_train_sessions=eval_settings.get(
            "histories_soft_constraint_max_n_train_sessions", None
        ),
        soft_constraint_max_n_train_days=eval_settings.get(
            "histories_soft_constraint_max_n_train_days", None
        ),
        remove_negrated_from_history=True,
    )


if __name__ == "__main__":
    eval_settings, random_state = parse_args()
    users_scores = compute_users_scores(eval_settings, random_state)
    print("success")
