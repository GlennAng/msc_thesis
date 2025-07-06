import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ....src.project_paths import ProjectPaths
from ..embeddings.embedding import Embedding
from ..training.algorithm import Evaluation
from ..training.get_users_ratings import get_users_ratings

np.random.seed(42)
N = 100

embedding = Embedding(
    ProjectPaths.logreg_embeddings_path() / "after_pca" / "gte_large_256_categories_l2_unit_100"
)
users_ratings, users_ids = get_users_ratings(
    "finetuning_val",
    Evaluation.SESSION_BASED,
    1.0,
    min_n_posrated_train=16,
    min_n_posrated_val=4,
    min_n_negrated_train=16,
    min_n_negrated_val=4,
    max_users=500,
)


def compute_sims(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> float:
    if embeddings_a.shape[0] == 0 or embeddings_b.shape[0] == 0:
        return 0.0
    sims = cosine_similarity(embeddings_a, embeddings_b)
    return np.mean(sims)


users_pos_first_10_train_sim_to_val = np.zeros(len(users_ids))
users_pos_random_10_train_sim_to_val = np.zeros(len(users_ids))
users_pos_last_10_train_sim_to_val = np.zeros(len(users_ids))
users_random_pos_sim_to_random_4_neg = np.zeros(len(users_ids))
users_random_pos_sim_to_closest_4_neg = np.zeros(len(users_ids))

for i, user_id in tqdm(enumerate(users_ids), total=len(users_ids)):
    user_pos_random_10_train_sim_to_val = np.zeros(N)
    user_random_pos_sim_to_random_4_neg = np.zeros(N)
    user_random_pos_sim_to_closest_4_neg = np.zeros(N)

    user_ratings = users_ratings[users_ratings["user_id"] == user_id]
    user_ratings_pos = user_ratings[user_ratings["rating"] > 0]
    user_ratings_neg = user_ratings[user_ratings["rating"] <= 0]
    user_ratings_pos_train = user_ratings_pos[user_ratings_pos["split"] == "train"]
    user_ratings_pos_val = user_ratings_pos[user_ratings_pos["split"] == "val"]

    user_pos_val_ids = user_ratings_pos_val["paper_id"].values
    user_pos_val_embeddings = embedding.matrix[embedding.get_idxs(user_pos_val_ids)]

    user_pos_first_10_train_ids = user_ratings_pos_train["paper_id"].values[:10]
    user_pos_first_10_train_embeddings = embedding.matrix[
        embedding.get_idxs(user_pos_first_10_train_ids)
    ]
    users_pos_first_10_train_sim_to_val[i] = compute_sims(
        user_pos_first_10_train_embeddings, user_pos_val_embeddings
    )

    user_pos_last_10_train_ids = user_ratings_pos_train["paper_id"].values[-10:]
    user_pos_last_10_train_embeddings = embedding.matrix[
        embedding.get_idxs(user_pos_last_10_train_ids)
    ]
    users_pos_last_10_train_sim_to_val[i] = compute_sims(
        user_pos_last_10_train_embeddings, user_pos_val_embeddings
    )

    for j in range(N):
        user_pos_random_10_train_ids = np.random.choice(
            user_ratings_pos_train["paper_id"].values, size=10, replace=False
        )
        user_pos_random_10_train_embeddings = embedding.matrix[
            embedding.get_idxs(user_pos_random_10_train_ids)
        ]
        user_pos_random_10_train_sim_to_val[j] = compute_sims(
            user_pos_random_10_train_embeddings, user_pos_val_embeddings
        )

        random_pos_idx = np.random.choice(user_ratings_pos.index)
        random_pos_id = user_ratings_pos.loc[random_pos_idx, "paper_id"]
        random_pos_time = user_ratings_pos.loc[random_pos_idx, "time"]
        random_pos_embedding = embedding.matrix[embedding.get_idxs([random_pos_id])]

        random_4_neg_ids = np.random.choice(
            user_ratings_neg["paper_id"].values, size=4, replace=False
        )
        random_4_neg_embeddings = embedding.matrix[embedding.get_idxs(random_4_neg_ids)]
        user_random_pos_sim_to_random_4_neg[j] = compute_sims(
            random_pos_embedding, random_4_neg_embeddings
        )

        neg_ratings_copy = user_ratings_neg.copy()
        neg_ratings_copy["time_diff"] = np.abs(neg_ratings_copy["time"] - random_pos_time)
        closest_4_neg = neg_ratings_copy.nsmallest(4, "time_diff")
        closest_4_neg_ids = closest_4_neg["paper_id"].values
        closest_4_neg_embeddings = embedding.matrix[embedding.get_idxs(closest_4_neg_ids)]
        user_random_pos_sim_to_closest_4_neg[j] = compute_sims(
            random_pos_embedding, closest_4_neg_embeddings
        )

    users_pos_random_10_train_sim_to_val[i] = np.mean(user_pos_random_10_train_sim_to_val)
    users_random_pos_sim_to_random_4_neg[i] = np.mean(user_random_pos_sim_to_random_4_neg)
    users_random_pos_sim_to_closest_4_neg[i] = np.mean(user_random_pos_sim_to_closest_4_neg)


print(
    f"Similarity of the first 10 Positives in Train to Val: Mean: {np.mean(users_pos_first_10_train_sim_to_val):.4f}, Std: {np.std(users_pos_first_10_train_sim_to_val):.4f}"
)
print(
    f"Similarity of Random 10 Positives in Train to Val: Mean: {np.mean(users_pos_random_10_train_sim_to_val):.4f}, Std: {np.std(users_pos_random_10_train_sim_to_val):.4f}"
)
print(
    f"Similarity of the last 10 Positives in Train to Val: Mean: {np.mean(users_pos_last_10_train_sim_to_val):.4f}, Std: {np.std(users_pos_last_10_train_sim_to_val):.4f}"
)
print(
    f"Similarity of Random Positive to Random 4 Negatives: Mean: {np.mean(users_random_pos_sim_to_random_4_neg):.4f}, Std: {np.std(users_random_pos_sim_to_random_4_neg):.4f}"
)
print(
    f"Similarity of Random Positive to Closest 4 Negatives: Mean: {np.mean(users_random_pos_sim_to_closest_4_neg):.4f}, Std: {np.std(users_random_pos_sim_to_closest_4_neg):.4f}"
)
