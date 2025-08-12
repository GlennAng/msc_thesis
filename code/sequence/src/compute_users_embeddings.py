import numpy as np
import pandas as pd

from ...logreg.src.embeddings.embedding import Embedding
from ...logreg.src.training.evaluation import get_val_negative_samples_ids, get_cache_papers_ids_full
from ...src.project_paths import ProjectPaths
from .sequence_get_users_ratings import sequence_get_users_ratings

pd.set_option("display.max_rows", None)


def compute_mean_pos_user_embedding(
    user_train_set_embeddings: np.ndarray, user_train_set_ratings: np.ndarray, n_last: int = None
) -> np.ndarray:
    assert user_train_set_embeddings.shape[0] == user_train_set_ratings.shape[0]
    pos_ratings = user_train_set_embeddings[user_train_set_ratings > 0]
    if pos_ratings.shape[0] == 0:
        return np.zeros(user_train_set_embeddings.shape[1])
    if n_last is not None:
        pos_ratings = pos_ratings[-n_last:]
    return pos_ratings.mean(axis=0)


def compute_logreg_user_embedding(
    user_train_set_embeddings: np.ndarray, user_train_set_ratings: np.ndarray,
) -> np.ndarray:
    pass

def logreg_get_cache_papers_ids_full() -> tuple:
    return get_cache_papers_ids_full(
        


    )


if __name__ == "__main__":

    embedding = Embedding(
        ProjectPaths.logreg_embeddings_path()
        / "after_pca"
        / "gte_large_256_session_based_categories_l2_unit_100"
    )

    users_ratings, users_ids, users_negrated_ranking = sequence_get_users_ratings(
        selection="session_based"
    )
    users_ratings_pos_val = users_ratings[
        (users_ratings["rating"] > 0) & (users_ratings["split"] == "val")
    ].reset_index(drop=True)
    for user_id in users_ids:
        user_ratings = users_ratings[users_ratings["user_id"] == user_id].reset_index(drop=True)
        user_ratings_pos_val = users_ratings_pos_val[
            users_ratings_pos_val["user_id"] == user_id
        ].reset_index(drop=True)
        user_ratings_pos_val = user_ratings_pos_val.drop(
            columns=["n_negrated_still_to_come", "rating", "split"]
        )
        user_sessions_ids_with_at_least_one_pos_val = (
            user_ratings_pos_val["session_id"].unique().tolist()
        )
        user_trained_embeddings = np.zeros(
            shape=(len(user_sessions_ids_with_at_least_one_pos_val), embedding.matrix.shape[1])
        )
        for i, session_id in enumerate(user_sessions_ids_with_at_least_one_pos_val):
            user_train_set = user_ratings[user_ratings["session_id"] < session_id].reset_index(
                drop=True
            )
            user_train_set_papers_ids = user_train_set["paper_id"].unique().tolist()
            user_train_set_ratings = user_train_set["rating"].to_numpy(dtype=np.int64)
            user_train_set_embeddings = embedding.matrix[
                embedding.get_idxs(user_train_set_papers_ids)
            ]
            user_trained_embedding = compute_mean_pos_user_embedding(
                user_train_set_embeddings, user_train_set_ratings, n_last=5
            )
            user_trained_embeddings[i] = user_trained_embedding
