import sys

import numpy as np
import torch

from ...logreg.src.visualization.visualization_tools import format_number
from .finetuning_data import FinetuningDataset
from .finetuning_model import FinetuningModel, load_finetuning_model
from .finetuning_preprocessing import (
    load_finetuning_dataset,
    load_finetuning_papers_tokenized,
    load_negative_samples_matrix_val,
    load_val_users_embeddings_idxs,
)

FINETUNING_CLASSIFICATION_METRICS = ["bcel", "recall", "specificity", "balacc"]
FINETUNING_RANKING_METRICS = ["ndcg", "mrr", "hr@1", "infonce"]
FINETUNING_INFO_NCE_TEMPERATURE = 1.0


def get_user_scores(dataset: FinetuningDataset, user_idx: int, users_scores: torch.Tensor) -> tuple:
    user_starting_idx, user_ending_idx = (
        dataset.users_pos_starting_idxs[user_idx],
        dataset.users_pos_starting_idxs[user_idx] + dataset.users_counts[user_idx],
    )
    user_ratings = dataset.rating_tensor[user_starting_idx:user_ending_idx].to(torch.float32)
    user_scores = users_scores[user_starting_idx:user_ending_idx]
    user_scores_pos = user_scores[: dataset.users_pos_counts[user_idx]]
    return user_ratings, user_scores, user_scores_pos


def get_user_scores_explicit_negatives(
    dataset: FinetuningDataset, user_idx: int, user_scores: torch.Tensor
) -> torch.Tensor:
    user_pos_starting_idx = dataset.users_pos_starting_idxs[user_idx]
    user_pos_count, user_neg_count = (
        dataset.users_pos_counts[user_idx],
        dataset.users_neg_counts[user_idx],
    )
    user_neg_starting_idx = user_pos_starting_idx + user_pos_count
    user_pos_times = dataset.time_tensor[user_pos_starting_idx:user_neg_starting_idx]
    user_neg_times = dataset.time_tensor[
        user_neg_starting_idx : (user_neg_starting_idx + user_neg_count)
    ]
    user_neg_scores = user_scores[user_pos_count:]
    n_neg = min(4, user_neg_count)
    user_scores_explicit_negatives = torch.zeros(size=(user_pos_count, n_neg), dtype=torch.float32)
    for i, user_pos_time in enumerate(user_pos_times):
        time_diffs = torch.abs(user_neg_times - user_pos_time)
        closest_neg_indices = torch.topk(time_diffs, k=n_neg, largest=False).indices
        closest_neg_indices = closest_neg_indices.sort().values
        closest_neg_scores = user_neg_scores[closest_neg_indices]
        user_scores_explicit_negatives[i] = closest_neg_scores
    return user_scores_explicit_negatives


def compute_user_classification_metrics(
    user_ratings: torch.Tensor, user_scores: torch.Tensor
) -> torch.tensor:
    bcel = torch.nn.BCEWithLogitsLoss()(user_scores, user_ratings)
    recall = torch.sum(user_scores[user_ratings == 1] > 0) / torch.sum(user_ratings == 1)
    specificity = torch.sum(user_scores[user_ratings == 0] < 0) / torch.sum(user_ratings == 0)
    balanced_accuracy = (recall + specificity) / 2
    return torch.tensor([bcel, recall, specificity, balanced_accuracy], dtype=torch.float32)


def compute_user_ranking_metrics_single(pos_rank: int) -> torch.Tensor:
    ndcg = 1 / np.log2(pos_rank + 1)
    mrr = 1 / pos_rank
    hr_at_1 = 1.0 if pos_rank == 1 else 0.0
    results = [ndcg, mrr, hr_at_1]
    if len(results) < len(FINETUNING_RANKING_METRICS):
        results += [0.0] * (len(FINETUNING_RANKING_METRICS) - len(results))
    return torch.tensor(results, dtype=torch.float32)


def compute_user_ranking_metrics(
    user_scores_pos: torch.Tensor,
    user_scores_explicit_negatives: torch.Tensor,
    user_scores_negative_samples: torch.Tensor,
) -> tuple:
    assert user_scores_explicit_negatives.shape[0] == user_scores_pos.shape[0]
    user_ranking_metrics_explicit_negatives = torch.zeros(
        size=(len(user_scores_pos), len(FINETUNING_RANKING_METRICS)),
        dtype=torch.float32,
    )
    user_ranking_metrics_negative_samples = torch.zeros(
        size=(len(user_scores_pos), len(FINETUNING_RANKING_METRICS)),
        dtype=torch.float32,
    )
    user_ranking_metrics_all = torch.zeros(
        size=(len(user_scores_pos), len(FINETUNING_RANKING_METRICS)),
        dtype=torch.float32,
    )

    info_nce_tensor_explicit_negatives = (
        user_scores_explicit_negatives / FINETUNING_INFO_NCE_TEMPERATURE
    )
    info_nce_tensor_negative_samples = (
        user_scores_negative_samples / FINETUNING_INFO_NCE_TEMPERATURE
    )

    for i, pos_score in enumerate(user_scores_pos):
        pos_score_unsqueezed = pos_score.unsqueeze(0)
        pos_rank_explicit_negatives = (
            torch.sum(user_scores_explicit_negatives[i] >= pos_score).item() + 1
        )
        pos_rank_negative_samples = torch.sum(user_scores_negative_samples >= pos_score).item() + 1
        pos_rank_all = (
            torch.sum(
                torch.cat((user_scores_explicit_negatives[i], user_scores_negative_samples))
                >= pos_score
            ).item()
            + 1
        )

        user_ranking_metrics_explicit_negatives[i] = compute_user_ranking_metrics_single(
            pos_rank_explicit_negatives
        )
        user_ranking_metrics_explicit_negatives[i, -1] = -torch.log_softmax(
            torch.cat((pos_score_unsqueezed, info_nce_tensor_explicit_negatives[i])) + 1e-10,
            dim=0,
        )[0].item()
        user_ranking_metrics_negative_samples[i] = compute_user_ranking_metrics_single(
            pos_rank_negative_samples
        )
        user_ranking_metrics_negative_samples[i, -1] = -torch.log_softmax(
            torch.cat((pos_score_unsqueezed, info_nce_tensor_negative_samples)) + 1e-10,
            dim=0,
        )[0].item()
        user_ranking_metrics_all[i] = compute_user_ranking_metrics_single(pos_rank_all)
        info_nce_tensor_all = torch.cat(
            (info_nce_tensor_explicit_negatives[i], info_nce_tensor_negative_samples)
        )
        user_ranking_metrics_all[i, -1] = -torch.log_softmax(
            torch.cat((pos_score_unsqueezed, info_nce_tensor_all)) + 1e-10, dim=0
        )[0].item()
    user_ranking_metrics_explicit_negatives = torch.mean(
        user_ranking_metrics_explicit_negatives, dim=0
    )
    user_ranking_metrics_negative_samples = torch.mean(user_ranking_metrics_negative_samples, dim=0)
    user_ranking_metrics_all = torch.mean(user_ranking_metrics_all, dim=0)
    return (
        user_ranking_metrics_explicit_negatives,
        user_ranking_metrics_negative_samples,
        user_ranking_metrics_all,
    )


def get_metric_strings() -> dict:
    return {
        "bcel": "BCEL",
        "recall": "Recall",
        "specificity": "Specificity",
        "balacc": "Balanced Accuracy",
        "ndcg": "nDCG",
        "mrr": "MRR",
        "hr@1": "HR@1",
        "infonce": "InfoNCE",
        "worst_10_ndcg": "Worst 10 nDCG",
        "worst_3_ndcg": "Worst 3 nDCG",
        "worst_ndcg": "Worst nDCG",
        "worst_10_ndcg_diff": "Worst 10 nDCG Diff",
        "worst_3_ndcg_diff": "Worst 3 nDCG Diff",
        "worst_ndcg_diff": "Worst nDCG Diff",
        "best_10_ndcg_diff": "Best 10 nDCG Diff",
        "best_3_ndcg_diff": "Best 3 nDCG Diff",
        "best_ndcg_diff": "Best nDCG Diff",
    }


def print_metrics(scores_dict: dict, metrics: list) -> str:
    metric_strings = get_metric_strings()
    metrics_string = ""
    for i, metric in enumerate(metrics):
        metric_string = metric.split("_")[1]
        if i > 0:
            metrics_string += ", "
        metrics_string += f"{metric_strings[metric_string]}: {format_number(scores_dict[metric])}"
    return metrics_string


def print_validation(scores_dict: dict) -> str:
    validation_str = ""
    validation_str += "\nClassification:   " + print_metrics(
        scores_dict, [f"val_{metric}" for metric in FINETUNING_CLASSIFICATION_METRICS]
    )
    validation_str += "\nRanking (Explicit Negatives):   " + print_metrics(
        scores_dict,
        [f"val_{metric}_explicit_negatives" for metric in FINETUNING_RANKING_METRICS],
    )
    validation_str += "\nRanking (Negative Samples): " + print_metrics(
        scores_dict,
        [f"val_{metric}_negative_samples" for metric in FINETUNING_RANKING_METRICS],
    )
    validation_str += "\nRanking (All):   " + print_metrics(
        scores_dict, [f"val_{metric}_all" for metric in FINETUNING_RANKING_METRICS]
    )
    validation_str += "\nRanking (All) No CS:   " + print_metrics(
        scores_dict, [f"val_{metric}_all_no_cs" for metric in FINETUNING_RANKING_METRICS]
    )
    metric_strings = get_metric_strings()
    validation_str += "\nWorst nDCG:   "
    for i, ndcg_str in enumerate(["worst_10_ndcg", "worst_3_ndcg", "worst_ndcg"]):
        score = scores_dict[ndcg_str]
        if i > 0:
            validation_str += ", "
        validation_str += f"{metric_strings[ndcg_str]}: {format_number(score)}"
    diff_strings_worst = ["worst_10_ndcg_diff", "worst_3_ndcg_diff", "worst_ndcg_diff"]
    if all(diff_string in scores_dict for diff_string in diff_strings_worst):
        validation_str += "\nWorst nDCG Diff:   "
        for i, ndcg_diff_str in enumerate(diff_strings_worst):
            score = scores_dict[ndcg_diff_str]
            if i > 0:
                validation_str += ", "
            validation_str += f"{metric_strings[ndcg_diff_str]}: {format_number(score)}"
    diff_strings_best = ["best_10_ndcg_diff", "best_3_ndcg_diff", "best_ndcg_diff"]
    if all(diff_string in scores_dict for diff_string in diff_strings_best):
        validation_str += "\nBest nDCG Diff:   "
        for i, ndcg_diff_str in enumerate(diff_strings_best):
            score = scores_dict[ndcg_diff_str]
            if i > 0:
                validation_str += ", "
            validation_str += f"{metric_strings[ndcg_diff_str]}: {format_number(score)}"

    return validation_str


def run_validation(
    finetuning_model: FinetuningModel,
    val_dataset: FinetuningDataset,
    val_negative_samples: dict,
    val_negative_samples_matrix: torch.Tensor,
    print_results: bool = True,
    original_ndcg_scores: torch.Tensor = None,
) -> tuple:
    scores_dict = {}
    assert (
        val_dataset.user_idx_tensor.unique().tolist()
        == finetuning_model.val_users_embeddings_idxs.tolist()
    )
    training_mode = finetuning_model.training
    finetuning_model.eval()
    val_negative_samples_scores = finetuning_model.compute_val_negative_samples_scores(
        val_negative_samples["input_ids"],
        val_negative_samples["attention_mask"],
        val_negative_samples["l1"],
        val_negative_samples["l2"],
    )
    val_negative_samples_scores = torch.gather(
        val_negative_samples_scores, dim=1, index=val_negative_samples_matrix
    )
    val_users_scores = finetuning_model.compute_val_dataset_scores(
        val_dataset.input_ids_tensor,
        val_dataset.attention_mask_tensor,
        val_dataset.category_l1_tensor,
        val_dataset.category_l2_tensor,
        val_dataset.user_idx_tensor,
    )
    val_classification_metrics = torch.zeros(
        size=(val_dataset.n_users, len(FINETUNING_CLASSIFICATION_METRICS)),
        dtype=torch.float32,
    )
    val_ranking_metrics_explicit_negatives = torch.zeros(
        size=(val_dataset.n_users, len(FINETUNING_RANKING_METRICS)), dtype=torch.float32
    )
    val_ranking_metrics_negative_samples = torch.zeros(
        size=(val_dataset.n_users, len(FINETUNING_RANKING_METRICS)), dtype=torch.float32
    )
    val_ranking_metrics_all = torch.zeros(
        size=(val_dataset.n_users, len(FINETUNING_RANKING_METRICS)), dtype=torch.float32
    )
    val_ranking_metrics_all_no_cs = torch.zeros(
        size=(val_dataset.n_users, len(FINETUNING_RANKING_METRICS)), dtype=torch.float32
    )

    for i in range(val_dataset.n_users):
        val_user_ratings, val_user_scores, val_user_scores_pos = get_user_scores(
            val_dataset, i, val_users_scores
        )
        val_classification_metrics[i] = compute_user_classification_metrics(
            val_user_ratings, val_user_scores
        )
        val_user_scores_explicit_negatives = get_user_scores_explicit_negatives(
            val_dataset, i, val_user_scores
        )
        val_user_scores_negative_samples = val_negative_samples_scores[i]
        val_user_ranking_metrics = compute_user_ranking_metrics(
            val_user_scores_pos,
            val_user_scores_explicit_negatives,
            val_user_scores_negative_samples,
        )
        val_ranking_metrics_explicit_negatives[i] = val_user_ranking_metrics[0]
        val_ranking_metrics_negative_samples[i] = val_user_ranking_metrics[1]
        val_ranking_metrics_all[i] = val_user_ranking_metrics[2]
    ndcg_scores = val_ranking_metrics_all[:, 0]
    ndgc_scores_sorted = ndcg_scores.sort(descending=False).values
    scores_dict["worst_10_ndcg"] = ndgc_scores_sorted[:10].mean().item()
    scores_dict["worst_3_ndcg"] = ndgc_scores_sorted[:3].mean().item()
    scores_dict["worst_ndcg"] = ndgc_scores_sorted[0].item()
    if original_ndcg_scores is not None:
        ndcg_scores_diff = ndcg_scores - original_ndcg_scores
        ndcg_scores_diff_sorted = ndcg_scores_diff.sort(descending=False).values
        scores_dict["worst_10_ndcg_diff"] = ndcg_scores_diff_sorted[:10].mean().item()
        scores_dict["worst_3_ndcg_diff"] = ndcg_scores_diff_sorted[:3].mean().item()
        scores_dict["worst_ndcg_diff"] = ndcg_scores_diff_sorted[0].item()
        scores_dict["best_10_ndcg_diff"] = ndcg_scores_diff_sorted[-10:].mean().item()
        scores_dict["best_3_ndcg_diff"] = ndcg_scores_diff_sorted[-3:].mean().item()
        scores_dict["best_ndcg_diff"] = ndcg_scores_diff_sorted[-1].item()

    val_classification_metrics = torch.mean(val_classification_metrics, dim=0)
    val_ranking_metrics_explicit_negatives = torch.mean(
        val_ranking_metrics_explicit_negatives, dim=0
    )
    val_ranking_metrics_negative_samples = torch.mean(val_ranking_metrics_negative_samples, dim=0)
    val_ranking_metrics_all_no_cs = torch.mean(
        val_ranking_metrics_all[val_dataset.non_cs_users_selection], dim=0
    )
    val_ranking_metrics_all = torch.mean(val_ranking_metrics_all, dim=0)

    for i, metric in enumerate(FINETUNING_CLASSIFICATION_METRICS):
        scores_dict[f"val_{metric}"] = val_classification_metrics[i].item()
    for i, metric in enumerate(FINETUNING_RANKING_METRICS):
        scores_dict[f"val_{metric}_explicit_negatives"] = val_ranking_metrics_explicit_negatives[
            i
        ].item()
        scores_dict[f"val_{metric}_negative_samples"] = val_ranking_metrics_negative_samples[
            i
        ].item()
        scores_dict[f"val_{metric}_all"] = val_ranking_metrics_all[i].item()
        scores_dict[f"val_{metric}_all_no_cs"] = val_ranking_metrics_all_no_cs[i].item()
    validation_str = print_validation(scores_dict)
    if print_results:
        print(validation_str)
    if training_mode:
        finetuning_model.train()
    return scores_dict, validation_str, ndcg_scores


def test_validation(finetuning_model: FinetuningModel) -> None:
    dataset = FinetuningDataset(
        dataset=load_finetuning_dataset("val"),
        no_cs_users_selection="val",
    )
    negative_samples = load_finetuning_papers_tokenized("negative_samples_val")
    negative_samples_matrix = load_negative_samples_matrix_val()
    run_validation(
        finetuning_model=finetuning_model,
        val_dataset=dataset,
        val_negative_samples=negative_samples,
        val_negative_samples_matrix=negative_samples_matrix,
    )


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python finetuning_val.py <model_path>"
    model_path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_users_embeddings_idxs = load_val_users_embeddings_idxs()
    finetuning_model = load_finetuning_model(
        finetuning_model_path=model_path,
        device=device,
        mode="eval",
        val_users_embeddings_idxs=val_users_embeddings_idxs,
    )
    test_validation(finetuning_model)
