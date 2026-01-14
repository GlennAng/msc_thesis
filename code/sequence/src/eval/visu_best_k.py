import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from ....finetuning.src.finetuning_compare_embeddings import compute_sims, compute_sims_same_set
from ....logreg.src.embeddings.embedding import Embedding
from ....logreg.src.training.users_ratings import load_users_ratings_from_selection, UsersRatingsSelection


K_VALUES = [1, 2, 3, 5, 7, 10, 15]



embedding = Embedding("code/logreg/embeddings/after_pca/gte_large_256")

users_ratings = load_users_ratings_from_selection(
    users_ratings_selection=UsersRatingsSelection.MSC_EARLY_SPLIT, relevant_users_ids="finetuning_test")
users_ratings = users_ratings[users_ratings["rating"] == 1]
users_sims = {}
users_ids = users_ratings["user_id"].unique().tolist()
for user_id in users_ids:
    users_sims[user_id] = compute_sims_same_set(
        embedding.matrix[embedding.get_idxs(
            users_ratings[(users_ratings["user_id"] == user_id)]["paper_id"].tolist()
        )]
    )


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
        # Get all metrics
        if k in K_VALUES:
            df = df[["user_id", "val_msc_auc", "val_recall", "val_specificity", "val_ndcg_samples"]]
            dfs[k] = df

    keys = sorted(list(dfs.keys()))
    result_df = None
    for k in keys:
        df = dfs[k].rename(columns={
            "val_msc_auc": f"val_msc_auc_{k}",
            "val_recall": f"val_recall_{k}",
            "val_specificity": f"val_specificity_{k}",
            "val_ndcg_samples": f"val_ndcg_samples_{k}",
        })
        if result_df is None:
            result_df = df
        else:
            result_df = result_df.merge(df, on="user_id", how="outer")
    users_info = pd.read_csv(valid_path / "users_info.csv")
    cols = ["n_sessions_pos_val", "n_posrated_val", "time_range_days_pos_val"]
    users_info = users_info[["user_id"] + cols]
    result_df = result_df.merge(users_info, on="user_id", how="left")
    
    # Find which k values have the highest AUC for each user (can be multiple if tied)
    auc_cols = [f"val_msc_auc_{k}" for k in sorted(dfs.keys())]
    
    # Mark users who have at least one non-NaN AUC
    result_df['has_valid_auc'] = result_df[auc_cols].notna().any(axis=1)
    result_df["users_sims"] = result_df["user_id"].map(users_sims)
    
    # Filter for valid users (those with at least one non-NaN AUC)
    valid_users_df = result_df[result_df['has_valid_auc']].copy()
    
    # Find all k values that tie for the maximum AUC for each user
    def get_best_ks(row):
        values = row[auc_cols]
        max_val = values.max()
        if pd.isna(max_val):
            return []
        # Find all k values that equal the max (within floating point tolerance)
        best_ks = []
        for col in auc_cols:
            if not pd.isna(row[col]) and np.isclose(row[col], max_val):
                k = int(col.split('_')[-1])
                best_ks.append(k)
        return best_ks
    
    valid_users_df['best_ks'] = valid_users_df.apply(get_best_ks, axis=1)
    
    # For oracle selection: pick the first best_k (or could pick randomly from ties)
    valid_users_df['oracle_best_k'] = valid_users_df['best_ks'].apply(lambda x: x[0] if len(x) > 0 else None)
    
    # Extract oracle metrics: the metrics at the best k for each user
    def get_oracle_metrics(row):
        k = row['oracle_best_k']
        if pd.isna(k):
            return pd.Series({
                'oracle_auc': np.nan,
                'oracle_recall': np.nan,
                'oracle_specificity': np.nan,
                'oracle_ndcg': np.nan
            })
        k = int(k)
        return pd.Series({
            'oracle_auc': row[f'val_msc_auc_{k}'],
            'oracle_recall': row[f'val_recall_{k}'],
            'oracle_specificity': row[f'val_specificity_{k}'],
            'oracle_ndcg': row[f'val_ndcg_samples_{k}']
        })
    
    oracle_metrics = valid_users_df.apply(get_oracle_metrics, axis=1)
    valid_users_df = pd.concat([valid_users_df, oracle_metrics], axis=1)
    
    # Expand dataframe so each user appears once per best_k
    expanded_rows = []
    for idx, row in valid_users_df.iterrows():
        for k in row['best_ks']:
            new_row = row.copy()
            new_row['best_k'] = k
            expanded_rows.append(new_row)
    
    expanded_df = pd.DataFrame(expanded_rows)

    # Aggregate for all metrics
    agg_dict = {
        'user_id': 'count',
        'n_sessions_pos_val': 'mean',
        'n_posrated_val': 'mean',
        'time_range_days_pos_val': 'mean',
        'users_sims': 'mean'
    }
    
    # Add all k values for all metrics
    for k in sorted(dfs.keys()):
        agg_dict[f'val_msc_auc_{k}'] = 'mean'
        agg_dict[f'val_recall_{k}'] = 'mean'
        agg_dict[f'val_specificity_{k}'] = 'mean'
        agg_dict[f'val_ndcg_samples_{k}'] = 'mean'
    
    print("\n" + "="*80)
    print("SUMMARY BY BEST K (based on AUC) - Users counted in multiple groups if tied")
    print("="*80)
    print(f"Total unique users: {len(result_df)}, Users with valid AUC: {len(valid_users_df)}")
    print(f"Users excluded (all NaN AUC): {len(result_df) - len(valid_users_df)}")
    
    # Count users with ties
    users_with_ties = sum(valid_users_df['best_ks'].apply(len) > 1)
    print(f"Users with ties (counted in multiple groups): {users_with_ties}")
    print(f"Total user-group assignments: {len(expanded_df)}")
    
    summary = expanded_df.groupby('best_k').agg(agg_dict)
    summary = summary.rename(columns={'user_id': 'n_users'})
    
    # Rename columns for clarity
    col_renames = {'n_users': 'n_users'}
    for k in sorted(dfs.keys()):
        col_renames[f'val_msc_auc_{k}'] = f'avg_msc_auc_{k}'
        col_renames[f'val_recall_{k}'] = f'avg_recall_{k}'
        col_renames[f'val_specificity_{k}'] = f'avg_specificity_{k}'
        col_renames[f'val_ndcg_samples_{k}'] = f'avg_ndcg_{k}'
    col_renames.update({
        'n_sessions_pos_val': 'avg_sessions',
        'n_posrated_val': 'avg_posrated',
        'time_range_days_pos_val': 'avg_time_range_days',
        'users_sims': 'avg_users_sims'
    })
    summary = summary.rename(columns=col_renames)
    
    print("\nAUC Summary:")
    print(summary[['n_users'] + [f'avg_msc_auc_{k}' for k in sorted(dfs.keys())] + 
                  ['avg_sessions', 'avg_posrated', 'avg_time_range_days', 'avg_users_sims']])
    
    print("\nRecall Summary:")
    print(summary[['n_users'] + [f'avg_recall_{k}' for k in sorted(dfs.keys())] + 
                  ['avg_sessions', 'avg_posrated', 'avg_time_range_days', 'avg_users_sims']])
    
    # Overall metrics per k (using valid users only)
    print("\n" + "="*80)
    print("OVERALL METRICS (users with valid AUC)")
    print("="*80)
    
    for k in sorted(dfs.keys()):
        mean_auc = valid_users_df[f'val_msc_auc_{k}'].mean()
        mean_recall = valid_users_df[f'val_recall_{k}'].mean()
        mean_specificity = valid_users_df[f'val_specificity_{k}'].mean()
        mean_ndcg = valid_users_df[f'val_ndcg_samples_{k}'].mean()
        print(f"k={k:2d} | AUC: {mean_auc:.4f} | Recall: {mean_recall:.4f} | Specificity: {mean_specificity:.4f} | NDCG: {mean_ndcg:.4f}")
    
    # ORACLE SELECTION: What if we could pick the best k for each user?
    print("\n" + "="*80)
    print("ORACLE SELECTION (using best k per user - knowledge we don't have in real world)")
    print("="*80)
    print(f"Average oracle AUC:         {valid_users_df['oracle_auc'].mean():.4f}")
    print(f"Average oracle Recall:      {valid_users_df['oracle_recall'].mean():.4f}")
    print(f"Average oracle Specificity: {valid_users_df['oracle_specificity'].mean():.4f}")
    print(f"Average oracle NDCG:        {valid_users_df['oracle_ndcg'].mean():.4f}")
    
    # Compare to fixed k strategies
    print("\nComparison to fixed k strategies:")
    for k in sorted(dfs.keys()):
        mean_auc = valid_users_df[f'val_msc_auc_{k}'].mean()
        delta = valid_users_df['oracle_auc'].mean() - mean_auc
        print(f"k={k:2d}: AUC={mean_auc:.4f}, Oracle gain: +{delta:.4f}")
    
    # BREAKDOWN BY NUMBER OF SESSIONS
    print("\n" + "="*80)
    print("BREAKDOWN BY NUMBER OF SESSIONS (75 users with most sessions)")
    print("="*80)
    
    # Sort by number of sessions
    valid_users_by_sessions = valid_users_df.sort_values('n_sessions_pos_val', ascending=False)
    top_75_sessions = valid_users_by_sessions.head(75)
    
    print("\nTop 75 users by number of sessions:")
    print(f"  Avg sessions:        {top_75_sessions['n_sessions_pos_val'].mean():.2f}")
    print(f"  Mean cosine sim:     {top_75_sessions['users_sims'].mean():.4f}")
    print(f"  Oracle AUC:          {top_75_sessions['oracle_auc'].mean():.4f}")
    print(f"  Oracle Recall:       {top_75_sessions['oracle_recall'].mean():.4f}")
    print(f"  Oracle Specificity:  {top_75_sessions['oracle_specificity'].mean():.4f}")
    print(f"  Oracle NDCG:         {top_75_sessions['oracle_ndcg'].mean():.4f}")
    print(f"  Avg posrated:        {top_75_sessions['n_posrated_val'].mean():.2f}")
    print(f"  Avg time range:      {top_75_sessions['time_range_days_pos_val'].mean():.2f}")
    
    # BREAKDOWN BY COSINE SIMILARITY
    print("\n" + "="*80)
    print("BREAKDOWN BY COSINE SIMILARITY (75 highest, 75 lowest)")
    print("="*80)
    
    # Sort by cosine similarity
    valid_users_sorted = valid_users_df.sort_values('users_sims', ascending=False)
    
    top_75_sim = valid_users_sorted.head(75)
    bottom_75_sim = valid_users_sorted.tail(75)
    
    print("\nTop 75 users by cosine similarity:")
    print(f"  Mean cosine sim:     {top_75_sim['users_sims'].mean():.4f}")
    print(f"  Oracle AUC:          {top_75_sim['oracle_auc'].mean():.4f}")
    print(f"  Oracle Recall:       {top_75_sim['oracle_recall'].mean():.4f}")
    print(f"  Oracle Specificity:  {top_75_sim['oracle_specificity'].mean():.4f}")
    print(f"  Oracle NDCG:         {top_75_sim['oracle_ndcg'].mean():.4f}")
    print(f"  Avg sessions:        {top_75_sim['n_sessions_pos_val'].mean():.2f}")
    print(f"  Avg posrated:        {top_75_sim['n_posrated_val'].mean():.2f}")
    print(f"  Avg time range:      {top_75_sim['time_range_days_pos_val'].mean():.2f}")
    
    print("\nBottom 75 users by cosine similarity:")
    print(f"  Mean cosine sim:     {bottom_75_sim['users_sims'].mean():.4f}")
    print(f"  Oracle AUC:          {bottom_75_sim['oracle_auc'].mean():.4f}")
    print(f"  Oracle Recall:       {bottom_75_sim['oracle_recall'].mean():.4f}")
    print(f"  Oracle Specificity:  {bottom_75_sim['oracle_specificity'].mean():.4f}")
    print(f"  Oracle NDCG:         {bottom_75_sim['oracle_ndcg'].mean():.4f}")
    print(f"  Avg sessions:        {bottom_75_sim['n_sessions_pos_val'].mean():.2f}")
    print(f"  Avg posrated:        {bottom_75_sim['n_posrated_val'].mean():.2f}")
    print(f"  Avg time range:      {bottom_75_sim['time_range_days_pos_val'].mean():.2f}")
    
    # BREAKDOWN BY PERFORMANCE (75 worst scoring users)
    print("\n" + "="*80)
    print("BREAKDOWN BY PERFORMANCE (75 worst scoring users by oracle AUC)")
    print("="*80)
    
    valid_users_by_perf = valid_users_df.sort_values('oracle_auc', ascending=True)
    worst_75_perf = valid_users_by_perf.head(75)
    
    print("\nWorst 75 users by oracle AUC:")
    print(f"  Mean cosine sim:     {worst_75_perf['users_sims'].mean():.4f}")
    print(f"  Oracle AUC:          {worst_75_perf['oracle_auc'].mean():.4f}")
    print(f"  Oracle Recall:       {worst_75_perf['oracle_recall'].mean():.4f}")
    print(f"  Oracle Specificity:  {worst_75_perf['oracle_specificity'].mean():.4f}")
    print(f"  Oracle NDCG:         {worst_75_perf['oracle_ndcg'].mean():.4f}")
    print(f"  Avg sessions:        {worst_75_perf['n_sessions_pos_val'].mean():.2f}")
    print(f"  Avg posrated:        {worst_75_perf['n_posrated_val'].mean():.2f}")
    print(f"  Avg time range:      {worst_75_perf['time_range_days_pos_val'].mean():.2f}")
    
    # Optional: Report recall metrics for ALL users (including those with NaN AUC)
    print("\n" + "="*80)
    print("RECALL METRICS (all users, including those with NaN AUC)")
    print("="*80)
    
    for k in sorted(dfs.keys()):
        mean_recall_all = result_df[f'val_recall_{k}'].mean()
        print(f"k={k:2d} | Recall: {mean_recall_all:.4f}")