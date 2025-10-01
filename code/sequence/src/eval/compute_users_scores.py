import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from tqdm import tqdm

from ....logreg.src.embeddings.embedding import Embedding
from ....logreg.src.training.evaluation import split_negrated_ranking, split_ratings
from ....logreg.src.training.scores import load_user_data_dicts
from ....logreg.src.training.training_data import (
    get_user_categories_ratios,
    get_user_val_negative_samples,
    get_val_negative_samples_ids,
    load_negrated_ranking_idxs_for_user,
)
from ....logreg.src.training.users_ratings import N_NEGRATED_RANKING
from ....src.load_files import load_papers, load_users_significant_categories
from ..data.users_embeddings_data import get_users_val_sessions_ids
from .compute_users_embeddings import (
    get_user_train_set,
    init_users_ratings,
    parse_args,
)
from .compute_users_embeddings_utils import EmbedFunction, get_embed_function_from_arg

N_EVAL_NEGATIVE_SAMPLES = 100
DT = np.float64


def scoring_function_max_pos_pooling(
    train_set_embeddings: np.ndarray, train_set_ratings: np.ndarray, embeddings_to_score: np.ndarray
) -> np.ndarray:
    train_set_embeddings = train_set_embeddings[train_set_ratings > 0]
    if train_set_embeddings.shape[0] == 0:
        return np.zeros(embeddings_to_score.shape[0], dtype=DT)
    scores = embeddings_to_score @ train_set_embeddings.T
    scores = np.max(scores, axis=1)
    assert scores.shape[0] == embeddings_to_score.shape[0]
    return scores.astype(DT)


def scoring_function_mean_pos_pooling(
    train_set_embeddings: np.ndarray, train_set_ratings: np.ndarray, embeddings_to_score: np.ndarray
) -> np.ndarray:
    train_set_embeddings = train_set_embeddings[train_set_ratings > 0]
    if train_set_embeddings.shape[0] == 0:
        return np.zeros(embeddings_to_score.shape[0], dtype=DT)
    mean_train_embedding = np.mean(train_set_embeddings, axis=0)
    scores = embeddings_to_score @ mean_train_embedding
    assert scores.shape[0] == embeddings_to_score.shape[0]
    return scores.astype(DT)


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


def get_train_rated_logits_dict(
    train_rated_embeddings: np.ndarray,
    train_rated_ratings: np.ndarray,
    val_negative_samples_embeddings: np.ndarray,
    train_negrated_ranking_idxs: np.ndarray,
    scoring_function: callable,
) -> dict:
    y_train_rated_logits = scoring_function(
        train_set_embeddings=train_rated_embeddings,
        train_set_ratings=train_rated_ratings,
        embeddings_to_score=train_rated_embeddings,
    )
    train_negrated_ranking_idxs = train_negrated_ranking_idxs.reshape(-1)
    y_train_negrated_ranking_logits = y_train_rated_logits[train_negrated_ranking_idxs].reshape(
        (-1, N_NEGRATED_RANKING)
    )
    y_negative_samples_logits_after_train = scoring_function(
        train_set_embeddings=train_rated_embeddings,
        train_set_ratings=train_rated_ratings,
        embeddings_to_score=val_negative_samples_embeddings,
    )
    return {
        "y_train_rated_logits": y_train_rated_logits,
        "y_train_negrated_ranking_logits": y_train_negrated_ranking_logits,
        "y_negative_samples_logits_after_train": y_negative_samples_logits_after_train,
    }


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
    user_train_set_embeddings = embedding.matrix[embedding.get_idxs(user_train_set_papers_ids)]
    user_train_set_ratings = user_train_set["rating"].values
    return user_train_set_embeddings, user_train_set_ratings


def get_y_val_logits_session(
    session_embeddings: np.ndarray,
    user_train_set_embeddings: np.ndarray,
    user_train_set_ratings: np.ndarray,
    scoring_function: callable,
    val_counter_all: int,
    n_session_papers: int,
    y_val_logits: np.ndarray,
) -> np.ndarray:
    session_logits = scoring_function(
        train_set_embeddings=user_train_set_embeddings,
        train_set_ratings=user_train_set_ratings,
        embeddings_to_score=session_embeddings,
    )
    y_val_logits[val_counter_all : val_counter_all + n_session_papers] = session_logits
    return y_val_logits


def get_y_val_negrated_ranking_logits_session(
    val_negrated_embeddings: np.ndarray,
    val_negrated_ranking_idxs: np.ndarray,
    user_train_set_embeddings: np.ndarray,
    user_train_set_ratings: np.ndarray,
    scoring_function: callable,
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
    val_negrated_ranking_logits_session = scoring_function(
        train_set_embeddings=user_train_set_embeddings,
        train_set_ratings=user_train_set_ratings,
        embeddings_to_score=val_negrated_embeddings_session,
    ).reshape((-1, N_NEGRATED_RANKING))
    y_val_negrated_ranking_logits[val_counter_pos : val_counter_pos + n_session_papers_pos] = (
        val_negrated_ranking_logits_session
    )
    return y_val_negrated_ranking_logits


def get_y_negative_samples_logits_session(
    val_negative_samples_embeddings: np.ndarray,
    user_train_set_embeddings: np.ndarray,
    user_train_set_ratings: np.ndarray,
    scoring_function: callable,
    val_counter_pos: int,
    n_session_papers_pos: int,
    y_negative_samples_logits: np.ndarray,
) -> np.ndarray:
    if n_session_papers_pos <= 0:
        return y_negative_samples_logits
    y_negative_samples_logits_session = scoring_function(
        train_set_embeddings=user_train_set_embeddings,
        train_set_ratings=user_train_set_ratings,
        embeddings_to_score=val_negative_samples_embeddings,
    )
    y_negative_samples_logits_session = np.tile(
        y_negative_samples_logits_session, (n_session_papers_pos, 1)
    )
    y_negative_samples_logits[val_counter_pos : val_counter_pos + n_session_papers_pos] = (
        y_negative_samples_logits_session
    )
    return y_negative_samples_logits


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
    users_scores = {}
    users_ids = users_ratings["user_id"].unique().tolist()
    assert users_ids == list(users_val_sessions_ids.keys())
    users_significant_categories = load_users_significant_categories(
        relevant_users_ids=users_ids,
    )
    for user_id in tqdm(users_ids):
        user_ratings = users_ratings[users_ratings["user_id"] == user_id]
        user_sessions_ids = users_val_sessions_ids[user_id]
        user_significant_categories = users_significant_categories[
            users_significant_categories["user_id"] == user_id
        ]["category"].tolist()
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
        val_negative_samples_embeddings = get_val_negative_samples_embeddings(
            user_significant_categories=user_significant_categories,
            val_negative_samples_ids=val_negative_samples_ids,
            random_state=random_state,
            embedding=embedding,
        )
        train_rated_logits_dict = get_train_rated_logits_dict(
            train_rated_embeddings=val_data_dict["X_train_rated"],
            train_rated_ratings=val_data_dict["y_train_rated"],
            val_negative_samples_embeddings=val_negative_samples_embeddings,
            train_negrated_ranking_idxs=train_negrated_ranking_idxs,
            scoring_function=scoring_function,
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
            user_train_set_embeddings, user_train_set_ratings = (
                get_user_train_embeddings_and_ratings(
                    user_ratings=user_ratings,
                    session_id=session_id,
                    embedding=embedding,
                    hard_constraint_min_n_train_posrated=hard_constraint_min_n_train_posrated,
                    hard_constraint_max_n_train_rated=hard_constraint_max_n_train_rated,
                    soft_constraint_max_n_train_sessions=soft_constraint_max_n_train_sessions,
                    soft_constraint_max_n_train_days=soft_constraint_max_n_train_days,
                    remove_negrated_from_history=remove_negrated_from_history,
                )
            )
            session_ratings = user_ratings[user_ratings["session_id"] == session_id]
            n_session_papers = session_ratings.shape[0]
            n_session_papers_pos = (session_ratings["rating"] > 0).sum()
            session_embeddings = val_data_dict["X_val"][
                val_counter_all : val_counter_all + n_session_papers
            ]
            y_val_logits = get_y_val_logits_session(
                session_embeddings=session_embeddings,
                user_train_set_embeddings=user_train_set_embeddings,
                user_train_set_ratings=user_train_set_ratings,
                scoring_function=scoring_function,
                val_counter_all=val_counter_all,
                n_session_papers=n_session_papers,
                y_val_logits=y_val_logits,
            )
            y_val_negrated_ranking_logits = get_y_val_negrated_ranking_logits_session(
                val_negrated_embeddings=val_negrated_embeddings,
                val_negrated_ranking_idxs=val_negrated_ranking_idxs,
                user_train_set_embeddings=user_train_set_embeddings,
                user_train_set_ratings=user_train_set_ratings,
                scoring_function=scoring_function,
                val_counter_pos=val_counter_pos,
                n_session_papers_pos=n_session_papers_pos,
                y_val_negrated_ranking_logits=y_val_negrated_ranking_logits,
            )
            y_negative_samples_logits = get_y_negative_samples_logits_session(
                val_negative_samples_embeddings=val_negative_samples_embeddings,
                user_train_set_embeddings=user_train_set_embeddings,
                user_train_set_ratings=user_train_set_ratings,
                scoring_function=scoring_function,
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
        }
        user_scores = fill_user_scores(user_scores)
        users_scores[user_id] = user_scores
    return users_scores


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


def save_users_scores(users_scores: dict, users_scores_folder: Path) -> None:
    users_scores_folder.mkdir(parents=True, exist_ok=True)
    with open(users_scores_folder / "users_scores.pkl", "wb") as f:
        pickle.dump(users_scores, f)
    print(f"Users scores saved to {users_scores_folder}.")


if __name__ == "__main__":
    eval_settings, random_state = parse_args()
    users_scores = compute_users_scores(eval_settings, random_state)
    eval_data_folder = Path(eval_settings["eval_data_folder"]).resolve()
    users_embeddings_folder = eval_data_folder / "users_embeddings" / f"s_{random_state}"
    save_users_scores(users_scores, users_embeddings_folder)
