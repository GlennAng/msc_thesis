import json
import pandas as pd
import sys
from pathlib import Path

from ....finetuning.src.finetuning_compare_embeddings import compute_sims, compute_sims_same_set
from ....logreg.src.embeddings.embedding import Embedding
from ....logreg.src.training.users_ratings import load_users_ratings_from_selection, UsersRatingsSelection



embedding = Embedding("code/logreg/embeddings/after_pca/gte_large_256")
users_ratings = load_users_ratings_from_selection(UsersRatingsSelection.SESSION_BASED_FILTERING_OLD)
users_ratings = users_ratings[users_ratings["rating"] == 1]



# get df with number of papers per user
users_papers_count = users_ratings.groupby("user_id")["paper_id"].count().reset_index()
N_MIN_PAPERS = 0
N_MAX_PAPERS = None
users_ids = users_papers_count[users_papers_count["paper_id"] >= N_MIN_PAPERS]["user_id"].tolist()
if N_MAX_PAPERS is not None:
    users_ids = users_papers_count[users_papers_count["paper_id"] <= N_MAX_PAPERS]["user_id"].tolist()
users_ratings = users_ratings[users_ratings["user_id"].isin(users_ids)]
print(f"Number of users: {len(users_ids)}")
users_ratings_train = users_ratings[users_ratings["split"] == "train"]
users_ratings_val = users_ratings[users_ratings["split"] == "val"]
users_n_pos_train = users_ratings_train.groupby("user_id")["paper_id"].count().reset_index()
users_ids = users_n_pos_train[users_n_pos_train["paper_id"] < 80000]["user_id"].tolist()
print(f"Number of users with at least 40 positive training ratings: {len(users_ids)}")
users_ratings = users_ratings[users_ratings["user_id"].isin(users_ids)]
users_ratings_train = users_ratings_train[users_ratings_train["user_id"].isin(users_ids)]
users_ratings_val = users_ratings_val[users_ratings_val["user_id"].isin(users_ids)]


users_sims = {}


for user_id in users_ids:
    user_ratings_train = users_ratings_train[users_ratings_train["user_id"] == user_id]
    user_ratings_val = users_ratings_val[users_ratings_val["user_id"] == user_id]
    pos_embeds_train = embedding.matrix[embedding.get_idxs(user_ratings_train["paper_id"].tolist())]
    pos_embeds_val = embedding.matrix[embedding.get_idxs(user_ratings_val["paper_id"].tolist())]
    users_sims[user_id] = compute_sims(pos_embeds_train, pos_embeds_val)

if __name__ == "__main__":
    folder = Path(sys.argv[1])

    valid_path = None
    dfs = {}
    for dir in folder.iterdir():
        if not dir.is_dir():
            continue
        eval_file = dir / "eval_settings.json"
        if not eval_file.exists():
            continue
        with open(eval_file, "r") as f:
            eval_settings = json.load(f)
        clustering_approach = eval_settings["clustering_approach"]
        df_path = dir / "outputs" / "users_results.csv"
        if not df_path.exists():
            continue
        valid_path = dir / "outputs"  # assuming all have the same valid_path
        df = pd.read_csv(dir / "outputs" / "users_results.csv")
        if clustering_approach == "none":
            k = 1
        elif clustering_approach == "k_means_fixed_k":
            k = eval_settings["clustering_k_means_n_clusters"]
        else:
            raise ValueError(f"Unknown clustering approach: {clustering_approach}")
        df = df[["user_id", "val_ndcg_all"]]
        df = df[df["user_id"].isin(users_ids)]
        dfs[k] = df

    keys = sorted(list(dfs.keys()))
    result_df = None
    for k in keys:
        df = dfs[k].rename(columns={"val_ndcg_all": f"val_ndcg_all_{k}"})
        if result_df is None:
            result_df = df
        else:
            result_df = result_df.merge(df, on="user_id", how="outer")
    users_info = pd.read_csv(valid_path / "users_info.csv")
    cols = ["n_sessions_pos_val", "n_posrated_val", "time_range_days_pos_val"]
    users_info = users_info[["user_id"] + cols]
    result_df = result_df.merge(users_info, on="user_id", how="left")
    
    # Find which k has the highest NDCG for each user
    ndcg_cols = [f"val_ndcg_all_{k}" for k in sorted(dfs.keys())]
    result_df['best_k'] = result_df[ndcg_cols].idxmax(axis=1).str.extract(r'(\d+)$')[0].astype(int)

    result_df["users_sims"] = result_df["user_id"].map(users_sims)


    # Aggregate the last 3 columns AND all NDCG scores by best_k
    agg_cols = ['n_sessions_pos_val', 'n_posrated_val', 'time_range_days_pos_val', "users_sims"]
    print("Summary by best k:")
    summary = result_df.groupby('best_k').agg({
        'user_id': 'count',
        'val_ndcg_all_1': 'mean',
        'val_ndcg_all_2': 'mean',
        'val_ndcg_all_3': 'mean',
        'val_ndcg_all_4': 'mean',
        'val_ndcg_all_5': 'mean',
        'val_ndcg_all_7': 'mean',
        'val_ndcg_all_10': 'mean',

        'n_sessions_pos_val': 'mean',
        'n_posrated_val': 'mean',
        'time_range_days_pos_val': 'mean',
        'users_sims': 'mean'
    }).rename(columns={'user_id': 'n_users'})

    summary.columns = ['n_users', 'avg_ndcg_1', 'avg_ndcg_2', 'avg_ndcg_3', 'avg_ndcg_4', 'avg_ndcg_5', 'avg_ndcg_7', 'avg_ndcg_10',
                    'avg_sessions', 'avg_posrated', 'avg_time_range_days', 'avg_users_sims']
    print(summary)

    # print ndcg for k=1 over all users
    overall_ndcg_k1 = result_df['val_ndcg_all_1'].mean()
    # print ndcg for best k over all users
    overall_ndcg_best_k = result_df.apply(lambda row: row[f'val_ndcg_all_{int(row["best_k"])}'], axis=1).mean()
    print(f"Overall NDCG for k=1: {overall_ndcg_k1:.4f}")
    print(f"Overall NDCG for best k: {overall_ndcg_best_k:.4f}")
