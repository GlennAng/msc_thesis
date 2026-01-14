import argparse
from pathlib import Path

import pandas as pd

from ....src.load_files import load_users_significant_categories
from ..training.scores_definitions import SCORES_DICT, Score
from .visualization_tools import load_outputs_files
from .visualize_globally import Global_Visualizer

N_COMPARE_USERS = 35
SCORES_NAMES = [score.name for score in Score]


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Compare user performance")
    parser.add_argument("--old_path", type=str, required=True)
    parser.add_argument("--new_path", type=str, required=True)
    parser.add_argument("--score", type=str, default="AUROC_CLASSIFICATION", choices=SCORES_NAMES)
    parser.add_argument("--second_score", type=str, default=None, choices=SCORES_NAMES + [None])
    args = vars(parser.parse_args())
    args["old_path"] = Path(args["old_path"]).resolve()
    assert args["old_path"].exists()
    args["new_path"] = Path(args["new_path"]).resolve()
    assert args["new_path"].exists()
    args["score"] = Score[args["score"]]
    if args["second_score"] is not None:
        args["second_score"] = Score[args["second_score"]]
    return args


def print_user_info(user_id: int, users_infos: pd.DataFrame, sig_categories: pd.DataFrame) -> str:
    user_info = users_infos[users_infos["user_id"] == user_id].iloc[0]
    user_sig_cat = sig_categories[sig_categories["user_id"] == user_id].iloc[0]
    user_s = "  "
    user_s += f"{user_sig_cat['category']}"
    user_s += f"  N_POS_TRAIN: {int(user_info['n_posrated_train'])}"
    user_s += f"  N_POS_VAL_SESSIONS: {int(user_info['n_sessions_pos_val'])}"
    user_s += f"  N_POS_VAL_DAYS: {int(user_info['time_range_days_pos_val'])}"
    return user_s


if __name__ == "__main__":
    args = parse_args()
    config_old, users_info_old, hyp_combis_old, results_before_avging_over_folds_old = (
        load_outputs_files(args["old_path"])
    )
    config_new, users_info_new, hyp_combis_new, results_before_avging_over_folds_new = (
        load_outputs_files(args["new_path"])
    )
    gv_old = Global_Visualizer(
        config=config_old,
        users_info=users_info_old,
        hyperparameters_combinations=hyp_combis_old,
        results_before_averaging_over_folds=results_before_avging_over_folds_old,
        score=args["score"],
        folder=None,
    )
    gv_new = Global_Visualizer(
        config=config_new,
        users_info=users_info_new,
        hyperparameters_combinations=hyp_combis_new,
        results_before_averaging_over_folds=results_before_avging_over_folds_new,
        score=args["score"],
        folder=None,
    )
    from ..training.users_ratings import load_users_ratings_from_selection, UsersRatingsSelection
    users_ratings = load_users_ratings_from_selection(
        users_ratings_selection=UsersRatingsSelection.MSC_LATE_SPLIT)
    
    import numpy as np
    from ..embeddings.embedding import Embedding
    from ....src.project_paths import ProjectPaths
    from ....finetuning.src.finetuning_compare_embeddings import compute_sims_same_set
    embedding = Embedding(ProjectPaths().logreg_embeddings_path() / "after_pca" / "gte_large_256")

    groups = ["CosH", "CosL", "HSessPV", "Tail"]
    cosine_sims, sessions_counts = {}, {}
    for group in groups:
        users_ids = gv_new.users_groups_dict[group]["users_ids"]
        users_ratings_group = users_ratings[users_ratings["user_id"].isin(users_ids)]
        users_cosine_sims, users_sessions_counts = [], []
        for user_id in users_ids:
            user_ratings = users_ratings_group[users_ratings_group["user_id"] == user_id]
            users_sessions_counts.append(user_ratings["session_id"].nunique())
            user_ratings_pos = user_ratings[user_ratings["rating"] == 1]
            papers_ids = user_ratings_pos["paper_id"].tolist()
            embeds = embedding.matrix[embedding.get_idxs(papers_ids)]
            sims = compute_sims_same_set(embeds)
            users_cosine_sims.append(sims)
        cosine_sims[group] = np.nanmean(users_cosine_sims)
        sessions_counts[group] = np.nanmean(users_sessions_counts)
        print(f"Group {group}: Mean cosine sim: {cosine_sims[group]:.4f}, Mean #sessions: {sessions_counts[group]:.4f}")

    tail_users_old = gv_old.users_groups_dict["Tail"]["users_ids"]
    tail_users_new = gv_new.users_groups_dict["Tail"]["users_ids"]
    common_tail_users = set(tail_users_old).intersection(set(tail_users_new))
    print(f"Number of common Tail users: {len(common_tail_users)}")


    sig_categories = load_users_significant_categories(relevant_users_ids=gv_old.users_ids)
    sig_categories = sig_categories[sig_categories["rank"] == 1]

    correlation = gv_old.results_after_averaging_over_folds[["user_id", f"val_{args['score'].name.lower()}"]].merge(
        gv_new.results_after_averaging_over_folds[["user_id", f"val_{args['score'].name.lower()}"]],
        on="user_id",
        suffixes=("_old", "_new"),
    )
    corr_value = correlation[f"val_{args['score'].name.lower()}_old"].corr(
        correlation[f"val_{args['score'].name.lower()}_new"]
    )
    print(f"Correlation of {args['score'].name} between old and new: {corr_value:.4f}\n")

    score_col = f"val_{args['score'].name.lower()}"
    bottom_1_pos = "val_confidence_bottom_1_pos"

    if SCORES_DICT[args["score"]]["increase_better"]:
        improvement_values = (
            gv_new.results_after_averaging_over_folds[score_col]
            - gv_old.results_after_averaging_over_folds[score_col]
        )
    else:
        improvement_values = (
            gv_old.results_after_averaging_over_folds[score_col]
            - gv_new.results_after_averaging_over_folds[score_col]
        )

    user_results_improvements = pd.DataFrame(
        {
            "user_id": gv_new.results_after_averaging_over_folds["user_id"],
            "improvement": improvement_values,
        }
    )
    if args["second_score"] is not None:
        second_score_col = f"val_{args['second_score'].name.lower()}"
        if SCORES_DICT[args["second_score"]]["increase_better"]:
            second_improvement_values = (
                gv_new.results_after_averaging_over_folds[second_score_col]
                - gv_old.results_after_averaging_over_folds[second_score_col]
            )
        else:
            second_improvement_values = (
                gv_old.results_after_averaging_over_folds[second_score_col]
                - gv_new.results_after_averaging_over_folds[second_score_col]
            )
        user_results_improvements["second_improvement"] = second_improvement_values
    user_results_improvements["confidence_bottom_1_pos"] = gv_old.results_after_averaging_over_folds[bottom_1_pos]
    best_improvements = user_results_improvements.nlargest(N_COMPARE_USERS + 30, "improvement")

    print(f"Best {N_COMPARE_USERS} user improvements for {args['score'].name}:")
    for _, row in best_improvements.iterrows():
        user_id, improvement = int(row['user_id']), row['improvement']
        user_info_s = print_user_info(user_id, users_info_new, sig_categories)
        s = f"User {user_id}: {improvement:.4f} {user_info_s}"
        if args["second_score"] is not None:
            second_improvement = row['second_improvement']
            s += f" | {args['second_score'].name} improvement: {second_improvement:.4f}"
        print(s)

    print(f"\nWorst {N_COMPARE_USERS} user degradations for {args['score'].name}:")
    worst_degradations = user_results_improvements.nsmallest(N_COMPARE_USERS, 'improvement')
    for _, row in worst_degradations.iterrows():
        user_id, degradation = int(row['user_id']), row['improvement']
        user_info_s = print_user_info(user_id, users_info_new, sig_categories)
        s = f"User {user_id}: {degradation:.4f} {user_info_s}"
        if args["second_score"] is not None:
            second_degradation = row['second_improvement']
            s += f" | {args['second_score'].name} improvement: {second_degradation:.4f}"
        print(s)
