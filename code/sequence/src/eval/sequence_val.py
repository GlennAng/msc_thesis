import logging

import torch

from ....finetuning.src.finetuning_main import log_string
from ....finetuning.src.finetuning_val import (
    FINETUNING_RANKING_METRICS,
    compute_user_ranking_metrics,
    print_validation,
)
from ....logreg.src.training.users_ratings import N_NEGRATED_RANKING
from ....src.load_files import load_sequence_users_ids
from ....src.project_paths import ProjectPaths
from ..data.eval_data import load_eval_dataloader
from ..models.recommender import Recommender


def get_users_ids_sessions_ids_flattened(users_sessions_ids_to_idxs: dict) -> tuple:
    users_ids, sessions_ids = [], []
    for user_id in users_sessions_ids_to_idxs:
        sessions_ids_user = users_sessions_ids_to_idxs[user_id]
        users_ids.extend([user_id] * len(sessions_ids_user))
        sessions_ids.extend(sessions_ids_user)
    users_ids = torch.tensor(users_ids, dtype=torch.long)
    sessions_ids = torch.tensor(sessions_ids, dtype=torch.long)
    n_users_sessions = sum(len(sessions) for sessions in users_sessions_ids_to_idxs.values())
    assert len(users_ids) == len(sessions_ids) == n_users_sessions
    return users_ids, sessions_ids


def get_users_starting_indices(users_sessions_ids_to_idxs: dict) -> torch.Tensor:
    users_sessions_counts = torch.tensor(
        [len(sessions) for sessions in users_sessions_ids_to_idxs.values()], dtype=torch.long
    )
    users_starting_indices = torch.cat(
        [torch.tensor([0]), users_sessions_counts.cumsum(dim=0)[:-1]]
    )
    return users_starting_indices


def get_users_ending_indices(users_sessions_ids_to_idxs: dict) -> torch.Tensor:
    users_sessions_counts = torch.tensor(
        [len(sessions) for sessions in users_sessions_ids_to_idxs.values()], dtype=torch.long
    )
    users_ending_indices = users_sessions_counts.cumsum(dim=0)
    return users_ending_indices


def compute_scores_negative_samples_val(
    negative_samples_embeddings: torch.Tensor,
    negative_samples_matrix: torch.Tensor,
    users_embeddings: torch.Tensor,
    users_ending_indices: torch.Tensor,
) -> torch.Tensor:
    scores = []
    n_users, n_val_negative_samples = negative_samples_matrix.shape
    n_sessions_total = users_ending_indices[-1].item()
    users_starting_indices = torch.cat([torch.tensor([0]), users_ending_indices[:-1]])
    for i in range(n_users):
        user_starting_index = users_starting_indices[i]
        if i < n_users - 1:
            user_ending_index = users_starting_indices[i + 1]
        else:
            user_ending_index = n_sessions_total
        user_embeddings = users_embeddings[user_starting_index:user_ending_index]
        user_val_negative_samples_idxs = negative_samples_matrix[i]
        user_val_negative_samples_embeddings = negative_samples_embeddings[
            user_val_negative_samples_idxs
        ]
        scores.append(user_embeddings @ user_val_negative_samples_embeddings.T)
    scores = torch.cat(scores, dim=0)
    assert scores.shape == (n_sessions_total, n_val_negative_samples)
    return scores


def compute_scores_ranking_val(
    rated_papers_embeddings: torch.Tensor,
    users_embeddings: torch.Tensor,
    ranking_matrix: torch.Tensor,
    users_sessions_endings_indices: torch.Tensor,
) -> torch.Tensor:
    scores = []
    user_session_starting_index = 0
    for i in range(users_embeddings.shape[0]):
        user_session_embedding = users_embeddings[i, :]
        user_session_rated_papers_embeddings = rated_papers_embeddings[
            ranking_matrix[user_session_starting_index : users_sessions_endings_indices[i], :]
        ]
        result = user_session_rated_papers_embeddings @ user_session_embedding.unsqueeze(-1)
        result = result.squeeze(-1)
        scores.append(result)
        user_session_starting_index = users_sessions_endings_indices[i]
    scores = torch.cat(scores, dim=0)
    assert scores.shape == (ranking_matrix.shape[0], ranking_matrix.shape[1])
    return scores


def merge_scores(
    scores_negative_samples: torch.Tensor,
    scores_ranking: torch.Tensor,
    users_sessions_endings_indices: torch.Tensor,
) -> torch.Tensor:
    merged_scores = []
    user_session_starting_index = 0
    for i in range(users_sessions_endings_indices.shape[0]):
        user_session_ending_index = users_sessions_endings_indices[i]
        user_session_scores_ranking = scores_ranking[
            user_session_starting_index:user_session_ending_index, :
        ]
        user_session_scores_negative_samples = (
            scores_negative_samples[i, :]
            .unsqueeze(0)
            .repeat(user_session_scores_ranking.shape[0], 1)
        )
        user_session_merged_scores = torch.cat(
            [user_session_scores_ranking, user_session_scores_negative_samples], dim=1
        )
        merged_scores.append(user_session_merged_scores)
        user_session_starting_index = user_session_ending_index
    merged_scores = torch.cat(merged_scores, dim=0)
    assert merged_scores.shape == (
        scores_ranking.shape[0],
        scores_ranking.shape[1] + scores_negative_samples.shape[1],
    )
    return merged_scores


def extract_ranking_metrics_from_scores(
    merged_scores: torch.Tensor,
    users_endings_indices: torch.Tensor,
    users_sessions_endings_indices: torch.Tensor,
) -> dict:
    scores_dict = {}
    n_users = users_endings_indices.shape[0]
    val_ranking_metrics_explicit_negatives = torch.zeros(
        size=(n_users, len(FINETUNING_RANKING_METRICS)), dtype=torch.float32
    )
    val_ranking_metrics_negative_samples = torch.zeros(
        size=(n_users, len(FINETUNING_RANKING_METRICS)), dtype=torch.float32
    )
    val_ranking_metrics_all = torch.zeros(
        size=(n_users, len(FINETUNING_RANKING_METRICS)), dtype=torch.float32
    )

    for i in range(n_users):
        if i == 0:
            user_starting_index = 0
        else:
            user_starting_index = users_sessions_endings_indices[users_endings_indices[i - 1] - 1]
        user_ending_index = users_sessions_endings_indices[users_endings_indices[i] - 1]
        user_scores = merged_scores[user_starting_index:user_ending_index, :]
        user_scores_pos = user_scores[:, 0]
        user_scores_explicit_negatives = user_scores[:, 1 : (1 + N_NEGRATED_RANKING)]
        user_scores_negative_samples = user_scores[:, (1 + N_NEGRATED_RANKING) :]
        val_user_ranking_metrics = compute_user_ranking_metrics(
            user_scores_pos=user_scores_pos,
            user_scores_explicit_negatives=user_scores_explicit_negatives,
            user_scores_negative_samples=user_scores_negative_samples,
        )
        val_ranking_metrics_explicit_negatives[i] = val_user_ranking_metrics[0]
        val_ranking_metrics_negative_samples[i] = val_user_ranking_metrics[1]
        val_ranking_metrics_all[i] = val_user_ranking_metrics[2]

    non_cs_users_ids = load_sequence_users_ids(selection="val", select_non_cs_users_only=True)
    all_users_ids = load_sequence_users_ids(selection="val", select_non_cs_users_only=False)
    non_cs_users_idxs, cs_users_idxs = [], []
    for i, user_id in enumerate(all_users_ids):
        if user_id in non_cs_users_ids:
            non_cs_users_idxs.append(i)
        else:
            cs_users_idxs.append(i)
    non_cs_users_idxs = torch.tensor(non_cs_users_idxs, dtype=torch.long)
    cs_users_idxs = torch.tensor(cs_users_idxs, dtype=torch.long)
    val_ranking_metrics_all_non_cs = torch.mean(val_ranking_metrics_all[non_cs_users_idxs], dim=0)
    val_ranking_metrics_all_cs = torch.mean(val_ranking_metrics_all[cs_users_idxs], dim=0)

    val_ranking_metrics_explicit_negatives = torch.mean(
        val_ranking_metrics_explicit_negatives, dim=0
    )
    val_ranking_metrics_negative_samples = torch.mean(val_ranking_metrics_negative_samples, dim=0)
    val_ranking_metrics_all = torch.mean(val_ranking_metrics_all, dim=0)

    for i, metric in enumerate(FINETUNING_RANKING_METRICS):
        scores_dict[f"val_{metric}_explicit_negatives"] = val_ranking_metrics_explicit_negatives[
            i
        ].item()
        scores_dict[f"val_{metric}_negative_samples"] = val_ranking_metrics_negative_samples[
            i
        ].item()
        scores_dict[f"val_{metric}_all"] = val_ranking_metrics_all[i].item()
        scores_dict[f"val_{metric}_all_no_cs"] = val_ranking_metrics_all_non_cs[i].item()
        scores_dict[f"val_{metric}_all_cs"] = val_ranking_metrics_all_cs[i].item()
    return scores_dict


def run_validation(recommender: Recommender, val_data: dict, print_results: bool = True) -> tuple:
    train_mode = recommender.training
    recommender.eval()
    rated_papers_embeddings, rated_papers_ids_to_idxs = recommender.extract_papers_embeddings(
        papers_ids=val_data["rated_papers_ids"]
    )
    negative_samples_embeddings, _ = recommender.extract_papers_embeddings(
        papers_ids=val_data["negative_samples_ids"]
    )
    val_dataloader = load_eval_dataloader(
        dataset=val_data["dataset"],
        papers_embeddings=rated_papers_embeddings,
        papers_ids_to_idxs=rated_papers_ids_to_idxs,
    )
    users_embeddings, users_sessions_ids_to_idxs = recommender.compute_users_embeddings(
        dataloader=val_dataloader
    )
    users_ending_indices = get_users_ending_indices(users_sessions_ids_to_idxs)
    scores_negative_samples = compute_scores_negative_samples_val(
        negative_samples_embeddings=negative_samples_embeddings,
        negative_samples_matrix=val_data["negative_samples_matrix"],
        users_embeddings=users_embeddings,
        users_ending_indices=users_ending_indices,
    )
    scores_ranking = compute_scores_ranking_val(
        rated_papers_embeddings=rated_papers_embeddings,
        users_embeddings=users_embeddings,
        ranking_matrix=val_data["ranking_matrix"],
        users_sessions_endings_indices=val_data["users_sessions_endings_indices"],
    )
    scores = merge_scores(
        scores_negative_samples=scores_negative_samples,
        scores_ranking=scores_ranking,
        users_sessions_endings_indices=val_data["users_sessions_endings_indices"],
    )
    ranking_scores_dict = extract_ranking_metrics_from_scores(
        merged_scores=scores,
        users_endings_indices=users_ending_indices,
        users_sessions_endings_indices=val_data["users_sessions_endings_indices"],
    )
    if train_mode:
        recommender.train()
    val_string = print_validation(ranking_scores_dict)
    if print_results:
        print(val_string)
    return ranking_scores_dict, val_string


def process_validation(
    recommender: Recommender,
    args_dict: dict,
    val_data: dict,
    previous_best_score: float = None,
    early_stopping_counter: int = 0,
    logger: logging.Logger = None,
) -> tuple:
    ranking_scores_dict, val_string = run_validation(recommender, val_data, print_results=False)
    val_score_name = args_dict["val_score_name"]
    current_score = ranking_scores_dict[f"val_{val_score_name}"]
    if previous_best_score is None:
        is_improvement = True
        previous_best_score_string = previous_best_score
    else:
        if val_score_name.startswith("infonce"):
            is_improvement = current_score < previous_best_score
        else:
            is_improvement = current_score > previous_best_score
        previous_best_score_string = f"{previous_best_score:.4f}"
    if is_improvement:
        early_stopping_counter = 0
        best_score = current_score
        recommender.save_model(args_dict["outputs_folder"] / "model")
        val_string += f"\nNew Optimal Value of {val_score_name.upper()}: {current_score:.4f}"
    else:
        early_stopping_counter += 1
        best_score = previous_best_score
        val_string += f"\nNo Improvement of {val_score_name.upper()}: {current_score:.4f}."
    val_string += f" (Previous Best was {previous_best_score_string}).\n"
    if logger is not None:
        log_string(logger, val_string)
    return best_score, early_stopping_counter, current_score


if __name__ == "__main__":
    from ..data.eval_data import load_val_data
    from ..models.recommender import load_recommender_from_scratch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    users_encoder_type = "MeanPoolingUsersEncoder"
    embeddings_path = ProjectPaths.sequence_non_finetuned_embeddings_path()
    recommender = load_recommender_from_scratch(
        users_encoder_type=users_encoder_type,
        embeddings_path=embeddings_path,
        device=device,
    )
    val_data = load_val_data(histories_remove_negrated_from_history=True)
    run_validation(recommender=recommender, val_data=val_data)
