import torch
from torch.utils.data import DataLoader, Sampler

from .eval_papers import get_val_negative_samples_ids, get_val_rated_papers_ids
from .sequence_preprocessing import (
    load_negative_samples_matrix_val,
    load_ranking_matrix_val,
)
from .sessions_dataset import SessionsDataset, load_sessions_dataset_by_split


class EvalBatchSampler(Sampler):
    def __init__(self, dataset: SessionsDataset, max_hist_len_per_batch: int = 768) -> None:
        self.dataset = dataset
        self.max_hist_len_per_batch = max_hist_len_per_batch
        self.batches = self._create_batches()

    def _create_batches(self) -> list:
        batches = []
        current_batch = []
        hist_len_current_batch = 0
        for idx in range(len(self.dataset)):
            hist_len = self.dataset.histories_lengths[idx]
            assert hist_len <= self.max_hist_len_per_batch
            if hist_len_current_batch + hist_len > self.max_hist_len_per_batch:
                batches.append(current_batch)
                current_batch = [idx]
                hist_len_current_batch = hist_len
            else:
                current_batch.append(idx)
                hist_len_current_batch += hist_len
        if len(current_batch) > 0:
            batches.append(current_batch)
        return batches

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch


def eval_collate_fn(papers_embeddings: torch.Tensor, papers_ids_to_idxs: dict):
    def collate_fn(batch: dict) -> dict:
        users_ids, sessions_ids = [], []
        flattened_papers_ids_in_hist = []
        flattened_batch_idxs_in_hist = []
        flattened_papers_labels_in_hist = []
        flattened_papers_days_diffs_in_hist = []
        for sample_idx, item in enumerate(batch):
            users_ids.append(item["user_id"])
            sessions_ids.append(item["session_id"])
            history_papers_ids = item["history_dict"]["history_papers_ids"]
            history_labels = item["history_dict"]["history_labels"]
            flattened_papers_ids_in_hist.extend(history_papers_ids)
            flattened_batch_idxs_in_hist.extend([sample_idx] * len(history_papers_ids))
            flattened_papers_labels_in_hist.extend(history_labels)
            flattened_papers_days_diffs_in_hist.extend(item["history_dict"]["history_days_diffs"])
        embedding_indices = [
            papers_ids_to_idxs[paper_id] for paper_id in flattened_papers_ids_in_hist
        ]
        embedding_indices = torch.tensor(embedding_indices)
        x_hist = papers_embeddings[embedding_indices]
        return {
            "user_id": torch.tensor(users_ids, dtype=torch.int64),
            "session_id": torch.tensor(sessions_ids, dtype=torch.int64),
            "batch_hist": torch.tensor(flattened_batch_idxs_in_hist),
            "x_hist": x_hist,
            "y_hist": torch.tensor(flattened_papers_labels_in_hist, dtype=torch.int64),
            "days_diffs_hist": torch.tensor(flattened_papers_days_diffs_in_hist, dtype=torch.int64),
        }

    return collate_fn


def load_eval_dataloader(
    dataset: SessionsDataset,
    papers_embeddings: torch.Tensor,
    papers_ids_to_idxs: dict,
    max_hist_len_per_batch: int = None,
) -> DataLoader:
    if max_hist_len_per_batch is None:
        max_hist_len_per_batch = max(dataset.histories_lengths)
    sampler = EvalBatchSampler(dataset, max_hist_len_per_batch=max_hist_len_per_batch)
    collate_fn = eval_collate_fn(papers_embeddings, papers_ids_to_idxs)
    return DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=collate_fn)


def load_val_data(
    histories_hard_constraint_min_n_train_posrated: int = 0,
    histories_hard_constraint_max_n_train_rated: int = None,
    histories_soft_constraint_max_n_train_sessions: int = None,
    histories_soft_constraint_max_n_train_days: int = None,
    histories_remove_negrated_from_history: bool = False,
) -> dict:
    ranking_matrix, users_sessions_endings_indices = load_ranking_matrix_val()
    dataset = load_sessions_dataset_by_split(
        split="val",
        histories_hard_constraint_min_n_train_posrated=histories_hard_constraint_min_n_train_posrated,
        histories_hard_constraint_max_n_train_rated=histories_hard_constraint_max_n_train_rated,
        histories_soft_constraint_max_n_train_sessions=histories_soft_constraint_max_n_train_sessions,
        histories_soft_constraint_max_n_train_days=histories_soft_constraint_max_n_train_days,
        histories_remove_negrated_from_history=histories_remove_negrated_from_history,
    )
    val_data = {
        "rated_papers_ids": get_val_rated_papers_ids(),
        "negative_samples_ids": get_val_negative_samples_ids(),
        "ranking_matrix": ranking_matrix,
        "negative_samples_matrix": load_negative_samples_matrix_val(),
        "users_sessions_endings_indices": users_sessions_endings_indices,
        "dataset": dataset,
    }
    return val_data
