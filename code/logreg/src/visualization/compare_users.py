import numpy as np

from ....logreg.src.training.algorithm import SCORES_DICT, Score
from ....logreg.src.visualization.visualization_tools import (
    PRINT_SCORES,
    load_outputs_files,
)
from ....logreg.src.visualization.visualize_globally import Global_Visualizer
from ....src.project_paths import ProjectPaths

(
    config_og,
    users_info_og,
    hyperparameters_combinations_og,
    results_before_averaging_over_folds_og,
) = load_outputs_files(ProjectPaths.logreg_outputs_path() / "example_config_temporal")
(
    config_new,
    users_info_new,
    hyperparameters_combinations_new,
    results_before_averaging_over_folds_new,
) = load_outputs_files(
    ProjectPaths.finetuning_data_checkpoints_path()
    / "best_batch_125"
    / "embeddings"
    / "outputs"
    / "no_overlap_session_based"
    / "no_overlap_session_based_s75"
)

gv_og = Global_Visualizer(
    config=config_og,
    users_info=users_info_og,
    hyperparameters_combinations=hyperparameters_combinations_og,
    results_before_averaging_over_folds=results_before_averaging_over_folds_og,
    score=Score.NDCG_ALL,
    folder=None,
)
gv_new = Global_Visualizer(
    config=config_new,
    users_info=users_info_new,
    hyperparameters_combinations=hyperparameters_combinations_new,
    results_before_averaging_over_folds=results_before_averaging_over_folds_new,
    score=Score.NDCG_ALL,
    folder=None,
)
val_columns = [f"val_{score.name.lower()}_mean" for score in SCORES_DICT.keys()]
scores_og = {}
scores_new = {}
scores_improvements = {}
for score in SCORES_DICT.keys():
    og_score = gv_og.results_after_averaging_over_users[f"val_{score.name.lower()}_mean"].mean()
    new_score = gv_new.results_after_averaging_over_users[f"val_{score.name.lower()}_mean"].mean()
    scores_og[score] = og_score
    scores_new[score] = new_score
    if SCORES_DICT[score]["increase_better"]:
        scores_improvements[score] = new_score - og_score
    else:
        scores_improvements[score] = og_score - new_score


n_scores_improved = sum(1 for score in list(Score) if scores_improvements[score] > 0)
n_scores_same = sum(1 for score in list(Score) if scores_improvements[score] == 0)
n_scores_worsened = sum(1 for score in list(Score) if scores_improvements[score] < 0)
print(
    f"Scores Total: {len(SCORES_DICT)}, Improved: {n_scores_improved}, Same: {n_scores_same}, Worsened: {n_scores_worsened}"
)
for score in PRINT_SCORES:
    improv = scores_improvements[score] >= 0
    if SCORES_DICT[score]["increase_better"]:
        improv_sign = "+" if improv else "-"
    else:
        improv_sign = "-" if improv else "+"
    improv_string = "(BETTER)" if improv else "(WORSE)"

    print(
        f"{score.name + ':':<30} {scores_og[score]:>7.4f} (OG) "
        f"{scores_new[score]:>7.4f} (NEW) "
        f"{improv_sign}{np.abs(scores_improvements[score]):.4f} {improv_string}"
    )
print("---------------------")
print("5 Best Changes:")
sorted_scores = sorted(list(SCORES_DICT.keys()), key=lambda x: scores_improvements[x], reverse=True)
for score in sorted_scores[:5]:
    improv = scores_improvements[score] >= 0
    if SCORES_DICT[score]["increase_better"]:
        improv_sign = "+" if improv else "-"
    else:
        improv_sign = "-" if improv else "+"
    improv_string = "(BETTER)" if improv else "(WORSE)"

    print(
        f"{score.name + ':':<30} {scores_og[score]:>7.4f} (OG) "
        f"{scores_new[score]:>7.4f} (NEW) "
        f"{improv_sign}{np.abs(scores_improvements[score]):.4f} {improv_string}"
    )
print("---------------------")
print("10 Worst Changes:")
for i in range(len(sorted_scores) - 1, len(sorted_scores) - 11, -1):
    score = sorted_scores[i]
    improv = scores_improvements[score] >= 0
    if SCORES_DICT[score]["increase_better"]:
        improv_sign = "+" if improv else "-"
    else:
        improv_sign = "-" if improv else "+"
    improv_string = "(BETTER)" if improv else "(WORSE)"

    print(
        f"{score.name + ':':<30} {scores_og[score]:>7.4f} (OG) "
        f"{scores_new[score]:>7.4f} (NEW) "
        f"{improv_sign}{np.abs(scores_improvements[score]):.4f} {improv_string}"
    )
print("---------------------")


user_results_og = gv_og.results_after_averaging_over_folds[f"val_{gv_og.score.name.lower()}"]
user_results_new = gv_new.results_after_averaging_over_folds[f"val_{gv_new.score.name.lower()}"]
assert (gv_og.results_after_averaging_over_folds["user_id"] == gv_new.results_after_averaging_over_folds["user_id"]).all(), (
    "User IDs do not match between original and new results."
)
inc_better = SCORES_DICT[gv_og.score]["increase_better"]
n_users = len(user_results_og)

if inc_better:
    user_results_improvements = user_results_new - user_results_og
else:
    user_results_improvements = user_results_og - user_results_new

n_users_improved = sum(1 for i in range(n_users) if user_results_improvements[i] > 0)
n_users_same = sum(1 for i in range(n_users) if user_results_improvements[i] == 0)
n_users_worsened = sum(1 for i in range(n_users) if user_results_improvements[i] < 0)
users_ids = gv_og.results_after_averaging_over_folds["user_id"].tolist()  # Convert to list for easier indexing

print(
    f"Users Total: {n_users}, Improved: {n_users_improved}, Same: {n_users_same}, Worsened: {n_users_worsened}, "
    f"Average Change: {np.mean(user_results_improvements):.4f}"
)

user_improvement_pairs = [(user_results_improvements[i], i) for i in range(len(user_results_improvements))]
print("5 Users with Best Changes:")
top_5_pairs = sorted(user_improvement_pairs, reverse=True)[:5]
for improvement, user_index in top_5_pairs:
    user_id = users_ids[user_index]  # Get the actual user ID
    improv = improvement >= 0
    if inc_better:
        improv_sign = "+" if improv else "-"
    else:
        improv_sign = "-" if improv else "+"
    improv_string = "(BETTER)" if improv else "(WORSE)"
    
    print(
        f"User {user_id:<3}: {user_results_og[user_index]:>7.4f} (OG) "
        f"{user_results_new[user_index]:>7.4f} (NEW) "
        f"{improv_sign}{np.abs(improvement):.4f} {improv_string}"
    )
    recall_c = (
        gv_new.results_after_averaging_over_folds["val_recall"][user_index]
        - gv_og.results_after_averaging_over_folds["val_recall"][user_index]
    )
    spec_c = (
        gv_new.results_after_averaging_over_folds["val_specificity"][user_index]
        - gv_og.results_after_averaging_over_folds["val_specificity"][user_index]
    )
    ndcg_c = (
        gv_new.results_after_averaging_over_folds["val_ndcg"][user_index]
        - gv_og.results_after_averaging_over_folds["val_ndcg"][user_index]
    )
    ndcg_s_c = (
        gv_new.results_after_averaging_over_folds["val_ndcg_samples"][user_index]
        - gv_og.results_after_averaging_over_folds["val_ndcg_samples"][user_index]
    )
    print(
        f"Recall Change: {recall_c:.4f}, Specificity Change: {spec_c:.4f}, "
        f"nDCG Change: {ndcg_c:.4f}, nDCG Samples Change: {ndcg_s_c:.4f}\n"
    )


print("---------------------")
print("10 Users with Worst Changes:")

# Get the 10 worst improvements (lowest values)
worst_10_pairs = sorted(user_improvement_pairs, reverse=False)[:10]  # Sort ascending to get worst first

for improvement, user_index in worst_10_pairs:
    user_id = users_ids[user_index]  # Get the actual user ID
    improv = improvement >= 0
    if inc_better:
        improv_sign = "+" if improv else "-"
    else:
        improv_sign = "-" if improv else "+"
    improv_string = "(BETTER)" if improv else "(WORSE)"

    print(
        f"User {user_id:<3}: {user_results_og[user_index]:>7.4f} (OG) "
        f"{user_results_new[user_index]:>7.4f} (NEW) "
        f"{improv_sign}{np.abs(improvement):.4f} {improv_string}"
    )
    
    recall_c = (
        gv_new.results_after_averaging_over_folds["val_recall"][user_index]
        - gv_og.results_after_averaging_over_folds["val_recall"][user_index]
    )
    spec_c = (
        gv_new.results_after_averaging_over_folds["val_specificity"][user_index]
        - gv_og.results_after_averaging_over_folds["val_specificity"][user_index]
    )
    ndcg_c = (
        gv_new.results_after_averaging_over_folds["val_ndcg"][user_index]
        - gv_og.results_after_averaging_over_folds["val_ndcg"][user_index]
    )
    ndcg_s_c = (
        gv_new.results_after_averaging_over_folds["val_ndcg_samples"][user_index]
        - gv_og.results_after_averaging_over_folds["val_ndcg_samples"][user_index]
    )
    print(
        f"Recall Change: {recall_c:.4f}, Specificity Change: {spec_c:.4f}, "
        f"nDCG Change: {ndcg_c:.4f}, nDCG Samples Change: {ndcg_s_c:.4f}\n"
    )
print("---------------------")


# Specific users analysis
# 10830 Biology 0.61, Medicine 0.25
# 9079 Psychology: 0.42, Computer Science 0.19
# User 7260, Biology: 0.85, Psychology: 0.07
# User 4871, Biology: 0.39, Physics: 0.27
# User 5254, Physics: 0.52, Biology: 0.21
users_ids = [4871, 5254, 7260, 9079, 10830]
for user_id in users_ids:
    user_index = users_ids.index(user_id)
    og_score = user_results_og[user_index]
    new_score = user_results_new[user_index]
    improvement = user_results_improvements[user_index]
    
    improv = improvement >= 0
    if inc_better:
        improv_sign = "+" if improv else "-"
    else:
        improv_sign = "-" if improv else "+"
    improv_string = "(BETTER)" if improv else "(WORSE)"
    
    print(
        f"User {user_id:<5}: {og_score:>7.4f} (OG) "
        f"{new_score:>7.4f} (NEW) "
        f"{improv_sign}{np.abs(improvement):.4f} {improv_string}"
    )
    
    recall_c = (
        gv_new.results_after_averaging_over_folds["val_recall"][user_index]
        - gv_og.results_after_averaging_over_folds["val_recall"][user_index]
    )
    spec_c = (
        gv_new.results_after_averaging_over_folds["val_specificity"][user_index]
        - gv_og.results_after_averaging_over_folds["val_specificity"][user_index]
    )
    ndcg_c = (
        gv_new.results_after_averaging_over_folds["val_ndcg"][user_index]
        - gv_og.results_after_averaging_over_folds["val_ndcg"][user_index]
    )
    ndcg_s_c = (
        gv_new.results_after_averaging_over_folds["val_ndcg_samples"][user_index]
        - gv_og.results_after_averaging_over_folds["val_ndcg_samples"][user_index]
    )
    
    print(
        f"Recall Change: {recall_c:.4f}, Specificity Change: {spec_c:.4f}, "
        f"nDCG Change: {ndcg_c:.4f}, nDCG Samples Change: {ndcg_s_c:.4f}\n"
    )
