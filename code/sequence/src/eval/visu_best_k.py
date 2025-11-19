import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from ....finetuning.src.finetuning_compare_embeddings import compute_sims, compute_sims_same_set
from ....logreg.src.embeddings.embedding import Embedding
from ....logreg.src.training.users_ratings import load_users_ratings_from_selection, UsersRatingsSelection



embedding = Embedding("code/logreg/embeddings/after_pca/gte_large_256")
users_ratings = load_users_ratings_from_selection(UsersRatingsSelection.SESSION_BASED_FILTERING)
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
users_ratings = users_ratings[users_ratings["user_id"].isin(users_ids)]
users_ratings_train = users_ratings_train[users_ratings_train["user_id"].isin(users_ids)]
users_ratings_val = users_ratings_val[users_ratings_val["user_id"].isin(users_ids)]



def compute_users_sims(users_ratings: pd.DataFrame, embedding: Embedding) -> dict:
    users_sims = {}
    users_ids = users_ratings["user_id"].unique().tolist()
    for user_id in users_ids:
        sims_scores = []
        user_ratings = users_ratings[users_ratings["user_id"] == user_id]
        val_sessions_ids = user_ratings[user_ratings["split"] == "val"]["session_id"].unique().tolist()
        for session_id in val_sessions_ids:
            session_ratings = user_ratings[user_ratings["session_id"] <= session_id]
            pos_embeds = embedding.matrix[embedding.get_idxs(session_ratings["paper_id"].tolist())]
            sim = compute_sims_same_set(pos_embeds)
            sims_scores.append(sim)
        users_sims[user_id] = np.mean(sims_scores) if len(sims_scores) > 0 else None
    return users_sims

users_sims = compute_users_sims(users_ratings, embedding)

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
        # Get all three metrics
        df = df[["user_id", "val_ndcg_all", "val_recall", "val_specificity", "val_mrr_all"]]
        df = df[df["user_id"].isin(users_ids)]
        dfs[k] = df

    keys = sorted(list(dfs.keys()))
    result_df = None
    for k in keys:
        df = dfs[k].rename(columns={
            "val_ndcg_all": f"val_ndcg_all_{k}",
            "val_recall": f"val_recall_all_{k}",
            "val_specificity": f"val_specificity_all_{k}",
            "val_mrr_all": f"val_mrr_all_{k}"
        })
        if result_df is None:
            result_df = df
        else:
            result_df = result_df.merge(df, on="user_id", how="outer")
    users_info = pd.read_csv(valid_path / "users_info.csv")
    cols = ["n_sessions_pos_val", "n_posrated_val", "time_range_days_pos_val"]
    users_info = users_info[["user_id"] + cols]
    result_df = result_df.merge(users_info, on="user_id", how="left")
    
    # Find which k has the highest NDCG for each user (keep grouping by NDCG)
    ndcg_cols = [f"val_ndcg_all_{k}" for k in sorted(dfs.keys())]
    result_df['best_k'] = result_df[ndcg_cols].idxmax(axis=1).str.extract(r'(\d+)$')[0].astype(int)

    result_df["users_sims"] = result_df["user_id"].map(users_sims)


    # Aggregate for all three metrics
    agg_dict = {
        'user_id': 'count',
        'n_sessions_pos_val': 'mean',
        'n_posrated_val': 'mean',
        'time_range_days_pos_val': 'mean',
        'users_sims': 'mean'
    }
    
    # Add all k values for all three metrics
    for k in sorted(dfs.keys()):
        agg_dict[f'val_ndcg_all_{k}'] = 'mean'
        agg_dict[f'val_recall_all_{k}'] = 'mean'
        agg_dict[f'val_specificity_all_{k}'] = 'mean'
        agg_dict[f'val_mrr_all_{k}'] = 'mean'
    
    print("\n" + "="*80)
    print("SUMMARY BY BEST K (based on NDCG)")
    print("="*80)
    
    summary = result_df.groupby('best_k').agg(agg_dict)
    summary = summary.rename(columns={'user_id': 'n_users'})
    
    # Rename columns for clarity
    col_renames = {'n_users': 'n_users'}
    for k in sorted(dfs.keys()):
        col_renames[f'val_ndcg_all_{k}'] = f'avg_ndcg_{k}'
        col_renames[f'val_recall_all_{k}'] = f'avg_recall_{k}'
        col_renames[f'val_specificity_all_{k}'] = f'avg_spec_{k}'
        col_renames[f'val_mrr_all_{k}'] = f'avg_mrr_{k}'
    col_renames.update({
        'n_sessions_pos_val': 'avg_sessions',
        'n_posrated_val': 'avg_posrated',
        'time_range_days_pos_val': 'avg_time_range_days',
        'users_sims': 'avg_users_sims'
    })
    summary = summary.rename(columns=col_renames)
    
    print("\nNDCG Summary:")
    print(summary[['n_users'] + [f'avg_ndcg_{k}' for k in sorted(dfs.keys())] + 
                  ['avg_sessions', 'avg_posrated', 'avg_time_range_days', 'avg_users_sims']])
    
    print("\nRecall Summary:")
    print(summary[['n_users'] + [f'avg_recall_{k}' for k in sorted(dfs.keys())] + 
                  ['avg_sessions', 'avg_posrated', 'avg_time_range_days', 'avg_users_sims']])
    
    print("\nSpecificity Summary:")
    print(summary[['n_users'] + [f'avg_spec_{k}' for k in sorted(dfs.keys())] + 
                  ['avg_sessions', 'avg_posrated', 'avg_time_range_days', 'avg_users_sims']])

    # Overall metrics per k and per line
    print("\n" + "="*80)
    print("OVERALL METRICS (all users)")
    print("="*80)
    
    for k in sorted(dfs.keys()):
        mean_ndcg = result_df[f'val_ndcg_all_{k}'].mean()
        mean_recall = result_df[f'val_recall_all_{k}'].mean()
        mean_spec = result_df[f'val_specificity_all_{k}'].mean()
        mean_mrr = result_df[f'val_mrr_all_{k}'].mean()
        print(f"k={k:2d} | NDCG: {mean_ndcg:.4f} | Recall: {mean_recall:.4f} | Specificity: {mean_spec:.4f} | MRR: {mean_mrr:.4f}")
    
    # Best k (based on NDCG grouping)
    overall_ndcg_best_k = result_df.apply(lambda row: row[f'val_ndcg_all_{int(row["best_k"])}'], axis=1).mean()
    overall_recall_best_k = result_df.apply(lambda row: row[f'val_recall_all_{int(row["best_k"])}'], axis=1).mean()
    overall_spec_best_k = result_df.apply(lambda row: row[f'val_specificity_all_{int(row["best_k"])}'], axis=1).mean()
    overall_mrr_best_k = result_df.apply(lambda row: row[f'val_mrr_all_{int(row["best_k"])}'], axis=1).mean()
    print(f"Best k (by NDCG) | NDCG: {overall_ndcg_best_k:.4f} | Recall: {overall_recall_best_k:.4f} | Specificity: {overall_spec_best_k:.4f} | MRR: {overall_mrr_best_k:.4f}")