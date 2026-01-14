import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ....src.load_files import load_users_significant_categories
from ..training.scores_definitions import SCORES_DICT, Score
from .visualization_tools import load_outputs_files
from .visualize_globally import Global_Visualizer


def bootstrap_ci(scores, n_bootstrap=10000, confidence=95, seed=42):
    """
    Calculate bootstrap confidence interval for mean of scores.
    
    Args:
        scores: array-like of scores
        n_bootstrap: number of bootstrap samples
        confidence: confidence level (default 95 for 95% CI)
        seed: random seed for reproducibility
    
    Returns:
        dict with 'mean', 'ci_lower', 'ci_upper'
    """
    np.random.seed(seed)
    scores = np.array(scores)
    n = len(scores)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        resampled = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.nanmean(resampled))
    
    bootstrap_means = np.array(bootstrap_means)
    observed_mean = np.nanmean(scores)
    
    alpha = (100 - confidence) / 2
    ci_lower = np.percentile(bootstrap_means, alpha)
    ci_upper = np.percentile(bootstrap_means, 100 - alpha)
    
    return {
        'mean': observed_mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


if __name__ == "__main__":
    np.random.seed(42)  # Set global seed as well
    
    path = "code/sequence/data/exponential_decay/outputs"
    config, users_info, hyp_combis, results_before_avging_over_folds = (
        load_outputs_files(path)
    )
    gv = Global_Visualizer(
        config=config,
        users_info=users_info,
        hyperparameters_combinations=hyp_combis,
        results_before_averaging_over_folds=results_before_avging_over_folds,
        folder=None,
    )
    
    scores = ["val_msc_auc", "val_ndcg_samples", "val_recall", "val_specificity"]
    groups = ["CosH", "CosL", "HSessPV", "Tail"]
    results = gv.results_after_averaging_over_folds
    
    for group in groups:
        group_ids = gv.users_groups_dict[group]["users_ids"]
        if not isinstance(group_ids, list):
            group_ids = group_ids.tolist()
        print(f"\n{group} (n={len(group_ids)})")
        
        for score in scores:
            # Filter by user_id column
            group_scores = results[results['user_id'].isin(group_ids)][score].values
            
            # Calculate bootstrap CI
            boot_result = bootstrap_ci(group_scores, seed=42)
            
            # Print: score_name: mean [lower, upper]
            print(f"  {score}: {boot_result['mean']:.4f} [{boot_result['ci_lower']:.4f}, {boot_result['ci_upper']:.4f}]")