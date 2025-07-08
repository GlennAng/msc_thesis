import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ...logreg.src.visualization.visualization_tools import format_number
from ...scripts.create_example_configs import (
    create_example_config,
    create_example_config_temporal,
)
from ...src.load_files import load_finetuning_users
from ...src.project_paths import ProjectPaths
from .finetuning_data import FinetuningDataset, create_finetuning_dataset
from .finetuning_model import FinetuningModel, load_finetuning_model
from .finetuning_preprocessing import (
    TEST_RANDOM_STATES,
    VAL_RANDOM_STATE,
    load_finetuning_dataset,
    load_finetuning_papers,
    load_users_coefs_ids_to_idxs,
    load_val_users_embeddings_idxs,
)
from .finetuning_visualization import visualize_testing

FINETUNING_CLASSIFICATION_METRICS = ["bcel", "recall", "specificity", "balacc"]
FINETUNING_RANKING_METRICS = ["ndcg", "mrr", "hr@1", "infonce"]
FINETUNING_INFO_NCE_TEMPERATURE = 1.0


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description="Evaluation script.")
    parser.add_argument("--embeddings_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--no_overlap", action="store_false", dest="overlap")
    parser.add_argument("--not_perform_evaluation", action="store_false", dest="perform_evaluation")
    parser.add_argument("--allow_configs", action="store_true", default=False)
    parser.add_argument("--allow_outputs", action="store_true", default=False)
    parser.add_argument("--cross_validation", action="store_true", default=False)
    parser.add_argument("--session_based", action="store_true", default=False)
    args_dict = vars(parser.parse_args())

    if args_dict["embeddings_path"] is not None:
        args_dict["embeddings_path"] = Path(args_dict["embeddings_path"]).resolve()
        args_dict["configs_folder"] = args_dict["embeddings_path"] / "configs"
        args_dict["outputs_folder"] = args_dict["embeddings_path"] / "outputs"
        if args_dict["model_path"] is not None:
            raise ValueError(
                "Both --embeddings_path and --model_path are provided. Please provide only one."
            )
    else:
        if args_dict["model_path"] is not None:
            args_dict["model_path"] = Path(args_dict["model_path"]).resolve()
            args_dict["configs_folder"] = args_dict["model_path"].parent / "configs"
            args_dict["outputs_folder"] = args_dict["model_path"].parent / "outputs"
        else:
            raise ValueError("Either --embeddings_path or --model_path must be provided.")
    os.makedirs(args_dict["configs_folder"], exist_ok=True)
    os.makedirs(args_dict["outputs_folder"], exist_ok=True)
    if len(os.listdir(args_dict["configs_folder"])) and not args_dict["allow_configs"]:
        raise ValueError(
            f"Folder {args_dict['configs_folder']} is not empty. Please remove the files inside it."
        )
    if len(os.listdir(args_dict["outputs_folder"])) and not args_dict["allow_outputs"]:
        raise ValueError(
            f"Folder {args_dict['outputs_folder']} is not empty. Please remove the files inside it."
        )
    return args_dict


def save_users_coefs(
    finetuning_model: FinetuningModel,
    embeddings_folder: Path,
    users_embeddings_ids_to_idxs: dict,
) -> None:
    if not isinstance(embeddings_folder, Path):
        embeddings_folder = Path(embeddings_folder).resolve()
    users_coefs = finetuning_model.users_embeddings.weight.detach().cpu().numpy().astype(np.float64)
    np.save(embeddings_folder / "users_coefs.npy", users_coefs)
    with open(embeddings_folder / "users_coefs_ids_to_idxs.pkl", "wb") as f:
        pickle.dump(users_embeddings_ids_to_idxs, f)


def compute_test_embeddings(
    finetuning_model: FinetuningModel,
    test_papers: dict,
    embeddings_folder: Path,
    users_embeddings_ids_to_idxs: dict,
) -> None:
    training_mode = finetuning_model.training
    finetuning_model.eval()
    papers_ids_to_idxs = {
        paper_id.item(): idx for idx, paper_id in enumerate(test_papers["paper_id"])
    }
    embeddings = finetuning_model.compute_papers_embeddings(
        input_ids_tensor=test_papers["input_ids"],
        attention_mask_tensor=test_papers["attention_mask"],
        category_l1_tensor=test_papers["l1"],
        category_l2_tensor=test_papers["l2"],
    )
    np.save(f"{embeddings_folder / 'abs_X.npy'}", embeddings)
    with open(f"{embeddings_folder / 'abs_paper_ids_to_idx.pkl'}", "wb") as f:
        pickle.dump(papers_ids_to_idxs, f)
    save_users_coefs(finetuning_model, embeddings_folder, users_embeddings_ids_to_idxs)
    if training_mode:
        finetuning_model.train()


def save_finetuning_config(
    file_path: Path,
    embeddings_folder: Path,
    users_ids: list,
    evaluation: str,
    random_state: int,
    users_coefs_path: Path = None,
) -> None:
    if not isinstance(file_path, Path):
        file_path = Path(file_path).resolve()
    if not isinstance(embeddings_folder, Path):
        embeddings_folder = Path(embeddings_folder).resolve()
    if users_coefs_path is not None and not isinstance(users_coefs_path, Path):
        users_coefs_path = Path(users_coefs_path).resolve()
    example_config = (
        create_example_config()
        if evaluation == "cross_validation"
        else create_example_config_temporal()
    )
    example_config["embedding_folder"] = str(embeddings_folder)
    example_config["users_selection"] = users_ids
    example_config["model_random_state"] = random_state
    example_config["cache_random_state"] = random_state
    example_config["ranking_random_state"] = random_state
    if users_coefs_path is not None:
        example_config["users_coefs_path"] = str(users_coefs_path)
        example_config["load_users_coefs"] = True
    with open(file_path, "w") as config_file:
        json.dump(example_config, config_file, indent=3)


def run_testing_single(
    name: str,
    embeddings_folder: Path,
    configs_folder: Path,
    outputs_folder: Path,
    users_ids: list,
    evaluation: str,
    random_states: list,
    users_coefs_path: Path = None,
) -> None:
    for folder in [embeddings_folder, configs_folder, outputs_folder]:
        if not isinstance(folder, Path):
            folder = Path(folder).resolve()
    if users_coefs_path is not None and not isinstance(users_coefs_path, Path):
        users_coefs_path = Path(users_coefs_path).resolve()
    configs_names, configs_folder_single, outputs_folder_single = (
        [],
        configs_folder / name,
        outputs_folder / name,
    )
    os.makedirs(configs_folder_single, exist_ok=True)
    os.makedirs(outputs_folder_single, exist_ok=True)
    for random_state in random_states:
        config_name = f"{name}_s{random_state}"
        file_path = configs_folder_single / f"{config_name}.json"
        save_finetuning_config(
            file_path,
            embeddings_folder,
            users_ids,
            evaluation,
            random_state,
            users_coefs_path,
        )
        configs_names.append(config_name)

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "code.scripts.logreg_run",
                "--config_path",
                str(configs_folder_single),
                "--save_scores_tables",
            ]
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running logreg_run: {e}")
        raise

    for config_name in configs_names:
        src = ProjectPaths.logreg_outputs_path() / config_name
        dst = outputs_folder_single / config_name
        if not dst.exists():
            shutil.move(str(src), str(dst))


def run_testing(
    embeddings_folder: Path,
    configs_folder: Path,
    outputs_folder: Path,
    val_users_ids: list,
    test_users_no_overlap_ids: list,
    cross_validation: bool,
    session_based: bool,
    val_random_state: int = VAL_RANDOM_STATE,
    test_no_overlap_random_states: list = TEST_RANDOM_STATES,
) -> None:
    assert val_random_state not in test_no_overlap_random_states
    for folder in [embeddings_folder, configs_folder, outputs_folder]:
        if not isinstance(folder, Path):
            folder = Path(folder).resolve()
    if val_users_ids is not None:
        run_testing_single(
            "overlap",
            embeddings_folder,
            configs_folder,
            outputs_folder,
            val_users_ids,
            "session_based",
            [val_random_state],
        )
        if (embeddings_folder / "users_coefs.npy").exists():
            run_testing_single(
                "overlap_users_coefs",
                embeddings_folder,
                configs_folder,
                outputs_folder,
                val_users_ids,
                "session_based",
                [val_random_state],
                users_coefs_path=embeddings_folder,
            )
    if test_users_no_overlap_ids is not None:
        if session_based:
            run_testing_single(
                "no_overlap_session_based",
                embeddings_folder,
                configs_folder,
                outputs_folder,
                test_users_no_overlap_ids,
                "session_based",
                test_no_overlap_random_states,
            )
        if cross_validation:
            run_testing_single(
                "no_overlap_cross_validation",
                embeddings_folder,
                configs_folder,
                outputs_folder,
                test_users_no_overlap_ids,
                "cross_validation",
                test_no_overlap_random_states,
            )


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
        "ndcg": "NDCG",
        "mrr": "MRR",
        "hr@1": "HR@1",
        "infonce": "InfoNCE",
        "worst_10_ndcg": "Worst 10 NDCG",
        "worst_3_ndcg": "Worst 3 NDCG",
        "worst_ndcg": "Worst NDCG",
        "worst_10_ndcg_diff": "Worst 10 NDCG Diff",
        "worst_3_ndcg_diff": "Worst 3 NDCG Diff",
        "worst_ndcg_diff": "Worst NDCG Diff",
        "best_10_ndcg_diff": "Best 10 NDCG Diff",
        "best_3_ndcg_diff": "Best 3 NDCG Diff",
        "best_ndcg_diff": "Best NDCG Diff",
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
    val_users_scores = finetuning_model.compute_val_dataset_scores(
        val_dataset.input_ids_tensor,
        val_dataset.attention_mask_tensor,
        val_dataset.category_l1_tensor,
        val_dataset.category_l2_tensor,
        val_dataset.user_idx_tensor,
    )
    val_negative_samples_scores = finetuning_model.compute_val_negative_samples_scores(
        val_negative_samples["input_ids"],
        val_negative_samples["attention_mask"],
        val_negative_samples["l1"],
        val_negative_samples["l2"],
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
    validation_str = print_validation(scores_dict)
    if print_results:
        print(validation_str)
    if training_mode:
        finetuning_model.train()
    return scores_dict, validation_str, ndcg_scores


def test_validation(finetuning_model: FinetuningModel) -> None:
    val_dataset = create_finetuning_dataset(load_finetuning_dataset("val"))
    val_negative_samples = load_finetuning_papers("val_negative_samples")
    run_validation(finetuning_model, val_dataset, val_negative_samples, print_results=True)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation_config = parse_arguments()
    finetuning_users = load_finetuning_users()
    val_users_ids = finetuning_users["val"] if evaluation_config["overlap"] else None
    test_users_no_overlap_ids = finetuning_users["test"]

    if not evaluation_config["embeddings_path"]:
        users_embeddings_ids_to_idxs = load_users_coefs_ids_to_idxs()
        test_papers = load_finetuning_papers("test")
        val_users_embeddings_idxs = load_val_users_embeddings_idxs()
        finetuning_model = load_finetuning_model(
            evaluation_config["model_path"],
            device=device,
            mode="eval",
            n_unfreeze_layers=0,
            val_users_embeddings_idxs=val_users_embeddings_idxs,
        )
        evaluation_config["embeddings_path"] = evaluation_config["model_path"].parent / "embeddings"
        os.makedirs(evaluation_config["embeddings_path"], exist_ok=True)
        print("Computing test embeddings...")
        compute_test_embeddings(
            finetuning_model,
            test_papers,
            evaluation_config["embeddings_path"],
            users_embeddings_ids_to_idxs,
        )
    if evaluation_config["perform_evaluation"]:
        run_testing(
            evaluation_config["embeddings_path"],
            evaluation_config["configs_folder"],
            evaluation_config["outputs_folder"],
            val_users_ids,
            test_users_no_overlap_ids,
            evaluation_config["cross_validation"],
            evaluation_config["session_based"],
        )
    visualize_testing(evaluation_config["model_path"], evaluation_config["outputs_folder"])
