import numpy as np

def compute_mean_pos_pooling_user_embedding(
    user_train_set_embeddings: np.ndarray,
    user_train_set_ratings: np.ndarray,
    user_train_set_time_diffs: np.ndarray = None,
) -> np.ndarray:
    assert user_train_set_embeddings.shape[0] == user_train_set_ratings.shape[0]
    assert np.all(user_train_set_ratings >= 0)
    if user_train_set_embeddings.shape[0] == 0:
        return np.zeros(user_train_set_embeddings.shape[1])
    mean_pos_pooling_user_embedding = user_train_set_embeddings.mean(axis=0)
    return mean_pos_pooling_user_embedding