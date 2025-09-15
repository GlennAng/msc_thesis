import torch

from ....finetuning.src.finetuning_val import (
    FINETUNING_RANKING_METRICS,
    compute_user_ranking_metrics,
    print_validation,
)
from ....logreg.src.training.users_ratings import N_NEGRATED_RANKING
from ....src.load_files import load_sequence_users_ids
from ....src.project_paths import ProjectPaths
from ..data.sequence_dataset import load_val_data, load_val_dataloader
from ..models.recommender import Recommender, load_recommender


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
        if user_embeddings.shape[1] == user_val_negative_samples_embeddings.shape[1] + 1:
            user_embeddings = user_embeddings[:, :-1]
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
        if user_session_embedding.shape[0] == user_session_rated_papers_embeddings.shape[-1] + 1:
            user_session_embedding = user_session_embedding[:-1]
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
    print_results: bool = True,
) -> tuple:
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
    validation_str = print_validation(scores_dict)
    if print_results:
        print(validation_str)


def run_validation(recommender: Recommender, val_data: dict) -> None:
    recommender.eval()
    if recommender.use_papers_encoder:
        rated_papers_embeddings, rated_papers_ids_to_idxs = (
            recommender.papers_encoder.compute_papers_embeddings(
                dataloader=val_data["rated_papers_dataloader"]
            )
        )
        negative_samples_embeddings, _ = recommender.papers_encoder.compute_papers_embeddings(
            dataloader=val_data["negative_samples_dataloader"]
        )
        recommender.papers_encoder.to_device(torch.device("cpu"))
    else:
        rated_papers_embeddings, rated_papers_ids_to_idxs = recommender.extract_papers_embeddings(
            papers_ids=val_data["rated_papers_ids"]
        )
        negative_samples_embeddings, _ = recommender.extract_papers_embeddings(
            papers_ids=val_data["negative_samples_ids"]
        )
    val_dataloader = load_val_dataloader(
        dataset=val_data["dataset"],
        papers_embeddings=rated_papers_embeddings,
        papers_ids_to_idxs=rated_papers_ids_to_idxs,
    )
    users_embeddings, users_sessions_ids_to_idxs = (
        recommender.users_encoder.compute_users_embeddings(dataloader=val_dataloader)
    )
    recommender.users_encoder.to_device(torch.device("cpu"))
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
    extract_ranking_metrics_from_scores(
        merged_scores=scores,
        users_endings_indices=users_ending_indices,
        users_sessions_endings_indices=val_data["users_sessions_endings_indices"],
    )


def test_validation_load_recommender(device: torch.device, use_papers_encoder: bool) -> Recommender:
    users_encoder_dict = {"device": device}
    if use_papers_encoder:
        papers_encoder_dict = {
            "device": device,
            "path": ProjectPaths.sequence_data_model_state_dicts_papers_encoder_finetuned_path(),
        }
        recommender = load_recommender(
            users_encoder_dict=users_encoder_dict,
            papers_encoder_dict=papers_encoder_dict,
            use_papers_encoder=True,
        )
    else:
        path = ProjectPaths.sequence_data_model_state_dicts_papers_encoder_finetuned_path().parent
        papers_encoder_dict = {
            "device": device,
            "papers_embeddings_path": path / "all_embeddings",
        }
        recommender = load_recommender(
            users_encoder_dict=users_encoder_dict,
            papers_encoder_dict=papers_encoder_dict,
            use_papers_encoder=False,
        )
    return recommender


def test_validation(device: torch.device, use_papers_encoder: bool) -> None:
    recommender = test_validation_load_recommender(device, use_papers_encoder)
    val_data = load_val_data(
        use_papers_encoder=use_papers_encoder,
        remove_negrated_from_history=True,
    )
    run_validation(recommender=recommender, val_data=val_data)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for use_papers_encoder in [False, True]:
        test_validation(device=device, use_papers_encoder=use_papers_encoder)
