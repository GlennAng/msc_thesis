import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ...logreg.src.training.algorithm import Evaluation
from ...logreg.src.training.get_users_ratings import (
    MIN_N_NEGRATED_TRAIN,
    MIN_N_NEGRATED_VAL_USERS_SELECTION,
    MIN_N_POSRATED_TRAIN,
    MIN_N_POSRATED_VAL,
    TRAIN_SIZE,
    USERS_SELECTIONS,
    get_users_ratings,
)
from .sequence_data import (
    get_embedding_path,
    get_users_val_sessions_ids,
)

VALID_EMBED_FUNCTIONS = ["mean_pos", "logreg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute users embeddings")

    parser.add_argument("--embed_function", type=str, choices=VALID_EMBED_FUNCTIONS, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--single_val_session", action="store_true", default=False)

    parser.add_argument(
        "--users_selection", type=str, choices=USERS_SELECTIONS, default="session_based"
    )
    parser.add_argument("--embedding_path", type=str, required=False, default=None)
    parser.add_argument("--train_size", type=float, required=False, default=TRAIN_SIZE)
    parser.add_argument(
        "--min_n_posrated_train", type=int, required=False, default=MIN_N_POSRATED_TRAIN
    )
    parser.add_argument(
        "--min_n_negrated_train", type=int, required=False, default=MIN_N_NEGRATED_TRAIN
    )
    parser.add_argument(
        "--min_n_posrated_val", type=int, required=False, default=MIN_N_POSRATED_VAL
    )
    parser.add_argument(
        "--min_n_negrated_val", type=int, required=False, default=MIN_N_NEGRATED_VAL_USERS_SELECTION
    )
    parser.add_argument(
        "--not_filter_for_negrated_ranking",
        dest="filter_for_negrated_ranking",
        action="store_false",
    )

    parser.add_argument("--random_state", type=int, required=False, default=42)
    parser.add_argument("--n_last", type=int, required=False, default=None)

    args_dict = vars(parser.parse_args())
    args_dict["output_folder"] = Path(args_dict["output_folder"]).resolve()
    if args_dict["embedding_path"] is None:
        args_dict["embedding_path"] = get_embedding_path(args_dict["users_selection"])
    else:
        args_dict["embedding_path"] = Path(args_dict["embedding_path"]).resolve()
    return args_dict


def init_users_ratings(args_dict: dict) -> tuple:
    users_ratings = get_users_ratings(
        users_selection=args_dict["users_selection"],
        evaluation=Evaluation.SESSION_BASED,
        train_size=args_dict["train_size"],
        min_n_posrated_train=args_dict["min_n_posrated_train"],
        min_n_negrated_train=args_dict["min_n_negrated_train"],
        min_n_posrated_val=args_dict["min_n_posrated_val"],
        min_n_negrated_val=args_dict["min_n_negrated_val"],
        filter_for_negrated_ranking=args_dict["filter_for_negrated_ranking"],
    )
    users_ids = users_ratings["user_id"].unique().tolist()
    if args_dict["single_val_session"]:
        users_ratings = users_ratings.copy()
        val_mask = users_ratings["split"] == "val"
        min_session_ids = users_ratings[val_mask].groupby("user_id")["session_id"].min()
        assert len(min_session_ids) == len(users_ids)
        for user_id, min_session_id in min_session_ids.items():
            user_val_mask = (users_ratings["user_id"] == user_id) & (
                users_ratings["split"] == "val"
            )
            users_ratings.loc[user_val_mask, "session_id"] = min_session_id
    return users_ratings


def compute_users_embeddings(args_dict: dict) -> dict:
    users_ratings = init_users_ratings(args_dict)
    users_val_sessions_ids = get_users_val_sessions_ids(users_ratings)
    

    return {}
    


if __name__ == "__main__":
    args_dict = parse_args()
    users_embeddings = compute_users_embeddings(args_dict)
    


"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ...logreg.src.embeddings.embedding import Embedding
from ...logreg.src.training.evaluation import (
    get_cache_papers_ids_full,
    get_user_cache_papers,
    get_user_val_negative_samples,
    get_val_negative_samples_ids,
)
from ...logreg.src.training.get_users_ratings import (
    USERS_SELECTIONS,
    sequence_load_users_ratings,
)
from ...logreg.src.training.training_data import get_user_categories_ratios
from ...src.load_files import load_papers, load_users_significant_categories
from .sequence_data import (
    get_embedding_path,
    get_users_val_sessions_ids,
    save_users_embeddings_dict,
)

VALID_EMBED_FUNCTIONS = ["mean_pos", "logreg"]

CACHE_TYPE = "categories_cache"
N_CACHE = 5000
N_CATEGORIES_CACHE = 0
N_NEGATIVE_SAMPLES = 100




def compute_mean_pos_user_embedding(
    user_train_set_embeddings: np.ndarray, user_train_set_ratings: np.ndarray, user_id: int, n_last: int = None
) -> np.ndarray:
    assert user_train_set_embeddings.shape[0] == user_train_set_ratings.shape[0]
    pos_ratings = user_train_set_embeddings[user_train_set_ratings > 0]
    if pos_ratings.shape[0] == 0:
        return np.zeros(user_train_set_embeddings.shape[1])
    if n_last is not None:
        pos_ratings = pos_ratings[-n_last:]
    return pos_ratings.mean(axis=0)


def compute_logreg_user_embedding(
    user_train_set_embeddings: np.ndarray,
    user_train_set_ratings: np.ndarray,
    embedding: Embedding,
    user_id: int,
    users_ratings: pd.DataFrame,
    users_significant_categories: pd.DataFrame,
    val_negative_samples_ids: pd.DataFrame,
    cache_papers_ids: pd.DataFrame,
    cache_papers_categories_ids: pd.DataFrame,
    random_state: int,
) -> np.ndarray:
    user_ratings = users_ratings[users_ratings["user_id"] == user_id]
    user_significant_categories = users_significant_categories[
        users_significant_categories["user_id"] == user_id
    ]
    user_categories_ratios = get_user_categories_ratios(
        categories_to_exclude=user_significant_categories
    )
    user_val_negative_samples_ids = get_user_val_negative_samples(
        val_negative_samples_ids=val_negative_samples_ids,
        n_negative_samples=100,
        random_state=random_state,
        user_categories_ratios=user_categories_ratios,
        embedding=None,
    )["val_negative_samples_ids"]
    papers_ids_to_exclude_from_cache = (
        user_ratings["paper_id"].tolist() + user_val_negative_samples_ids
    )
    user_cache_papers = get_user_cache_papers(
        cache_type=CACHE_TYPE,
        cache_papers_ids=cache_papers_ids,
        cache_papers_categories_ids=cache_papers_categories_ids,
        n_categories_cache=N_CATEGORIES_CACHE,
        random_state=random_state,
        papers_ids_to_exclude_from_cache=papers_ids_to_exclude_from_cache,
        user_categories_ratios=user_categories_ratios,
        embedding=embedding,
    )

    return user_categories_ratios


def compute_users_embeddings_general(
    users_ratings: pd.DataFrame,
    users_val_sessions_ids: dict,
    embedding: Embedding,
    embed_function: callable,
    embed_function_params: dict = {},
) -> dict:
    users_embeddings = {}
    users_ids = users_ratings["user_id"].unique().tolist()
    assert users_ids == list(users_val_sessions_ids.keys())
    for user_id in users_ids:
        user_ratings = users_ratings[users_ratings["user_id"] == user_id]
        user_sessions_ids = users_val_sessions_ids[user_id]
        user_embeddings = np.zeros((len(user_sessions_ids), embedding.matrix.shape[1]))
        for i, session_id in enumerate(user_sessions_ids):
            user_train_set = user_ratings[user_ratings["session_id"] < session_id].reset_index(
                drop=True
            )
            user_train_set_papers_ids = user_train_set["paper_id"].tolist()
            user_train_set_ratings = user_train_set["rating"].to_numpy(dtype=np.int64)
            user_train_set_embeddings = embedding.matrix[
                embedding.get_idxs(user_train_set_papers_ids)
            ]
            user_embeddings[i] = embed_function(
                user_train_set_embeddings, user_train_set_ratings, user_id, **embed_function_params
            )
        users_embeddings[user_id] = {
            "sessions_ids": user_sessions_ids,
            "sessions_embeddings": user_embeddings,
        }
    return users_embeddings


def compute_users_embeddings_logreg(
    users_ratings: pd.DataFrame,
    users_val_sessions_ids: dict,
    embedding: Embedding,
    single_val_session: bool,
    random_state: int,
) -> dict:
    if single_val_session:
        users_ratings = users_ratings.copy()
        users_ids = users_ratings["user_id"].unique().tolist()
        val_mask = users_ratings["split"] == "val"
        min_session_ids = users_ratings[val_mask].groupby("user_id")["session_id"].min()
        assert len(min_session_ids) == len(users_ids)
        for user_id, min_session_id in min_session_ids.items():
            user_val_mask = (users_ratings["user_id"] == user_id) & (
                users_ratings["split"] == "val"
            )
            users_ratings.loc[user_val_mask, "session_id"] = min_session_id
        users_val_sessions_ids = get_users_val_sessions_ids(users_ratings=users_ratings)

    users_significant_categories = load_users_significant_categories(
        relevant_users_ids=users_ratings["user_id"].unique().tolist()
    )
    papers = load_papers(relevant_columns=["paper_id", "in_ratings", "l1"])
    val_negative_samples_ids = get_val_negative_samples_ids(
        papers=papers,
        n_categories_samples=100,
        random_state=random_state,
        papers_ids_to_exclude=papers[papers["in_ratings"]]["paper_id"].tolist(),
    )
    cache_papers_categories_ids, cache_papers_ids = get_cache_papers_ids_full(
        papers=papers,
        cache_type=CACHE_TYPE,
        n_cache=5000,
        random_state=random_state,
        n_categories_cache=0,
    )
    return compute_users_embeddings_general(
        users_ratings=users_ratings,
        users_val_sessions_ids=users_val_sessions_ids,
        embedding=embedding,
        embed_function=compute_logreg_user_embedding,
        embed_function_params={

    )


def compute_users_embeddings(args_dict: dict) -> dict:
    users_ratings = sequence_load_users_ratings(args_dict["users_selection"])
    users_val_sessions_ids = get_users_val_sessions_ids(users_ratings)
    embedding = Embedding(args_dict["embedding_path"])
    embed_function = args_dict["embed_function"]

    users_embeddings = None
    if embed_function == "mean_pos":
        users_embeddings = compute_users_embeddings_general(
            users_ratings=users_ratings,
            users_val_sessions_ids=users_val_sessions_ids,
            embedding=embedding,
            embed_function=compute_mean_pos_user_embedding,
            embed_function_params={"n_last": args_dict["n_last"]},
        )
    elif embed_function in ["logreg", "logreg_single_val_session"]:
        single_val_session = embed_function == "logreg_single_val_session"
        users_embeddings = compute_users_embeddings_logreg(
            users_ratings=users_ratings,
            users_val_sessions_ids=users_val_sessions_ids,
            embedding=embedding,
            single_val_session=single_val_session,
        )
    print(
        f"Computed users embeddings for function {args_dict['embed_function']}, "
        f"users selection {args_dict['users_selection']}."
    )
    return users_embeddings


if __name__ == "__main__":
    args_dict = parse_args()
    users_embeddings = compute_users_embeddings(args_dict)
    save_users_embeddings_dict(
        users_embeddings=users_embeddings,
        users_selection=args_dict["users_selection"],
        save_path=args_dict["output_path"],
        embedding_path=args_dict["embedding_path"],
    )

    
"""
