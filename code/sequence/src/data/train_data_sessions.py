import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .sessions_dataset import SessionsDataset, load_sessions_dataset_by_split


class TrainSessionsDataset(Dataset):
    def __init__(
        self,
        sessions_dataset: SessionsDataset,
        n_negrated_per_candidate: int,
        seed: int,
    ) -> None:
        self.sessions_dataset = sessions_dataset
        self.n_negrated_per_candidate = n_negrated_per_candidate
        self.rng = np.random.default_rng(seed)
        self._get_flattened_mapping()

    def _get_flattened_mapping(self):
        flattened_idxs_pairs = []
        for i in range(len(self.sessions_dataset)):
            n_candidates = self.sessions_dataset.candidates_lengths[i]
            for candidate_idx in range(n_candidates):
                flattened_idxs_pairs.append((i, candidate_idx))
        self.flattened_mapping = flattened_idxs_pairs

    def __len__(self) -> int:
        return len(self.flattened_mapping)

    def __getitem__(self, idx: int) -> dict:
        dataset_idx, candidate_idx = self.flattened_mapping[idx]
        dataset_item = self.sessions_dataset[dataset_idx]
        n_negrated = self.sessions_dataset.negrated_yet_to_come_lengths[dataset_idx]
        n_negrated_to_sample = min(self.n_negrated_per_candidate, n_negrated)
        negrated_idxs = self.rng.choice(n_negrated, size=n_negrated_to_sample, replace=False)
        return {
            "user_id": dataset_item["user_id"],
            "session_id": dataset_item["session_id"],
            "candidate_paper_id": dataset_item["candidates_dict"]["candidates_papers_ids"][
                candidate_idx
            ],
            "history_papers_ids": dataset_item["history_dict"]["history_papers_ids"],
            "history_labels": dataset_item["history_dict"]["history_labels"],
            "history_days_diffs": dataset_item["history_dict"]["history_days_diffs"],
            "negrated_papers_ids": dataset_item["negrated_yet_to_come_dict"][
                "negrated_yet_to_come_papers_ids"
            ][negrated_idxs],
        }


class TrainSessionsSampler(Sampler):
    def __init__(
        self,
        train_sessions_dataset: TrainSessionsDataset,
        n_candidates_per_batch: int,
        n_batches_total: int,
        seed: int,
    ) -> None:
        self.train_sessions_dataset = train_sessions_dataset
        self.n_candidates_per_batch = n_candidates_per_batch
        self.n_batches_total = n_batches_total
        self.seed = seed

    def __len__(self) -> int:
        return self.n_batches_total * self.n_candidates_per_batch

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        all_indices = []
        for _ in range(self.n_batches_total):
            batch_idxs = rng.choice(
                len(self.train_sessions_dataset),
                size=self.n_candidates_per_batch,
                replace=True,
            ).tolist()
            all_indices.extend(batch_idxs)
        for idx in all_indices:
            yield idx

    def run_test(self, batch: dict) -> bool:
        assert len(batch["user_id"]) == self.n_candidates_per_batch
        assert len(batch["session_id"]) == self.n_candidates_per_batch
        assert len(batch["x_candidates"]) == self.n_candidates_per_batch
        assert len(batch["x_hist"]) == len(batch["y_hist"])
        assert len(batch["x_hist"]) == len(batch["batch_hist"])
        assert len(batch["x_negrated"]) == len(batch["batch_negrated"])
        assert batch["x_candidates"].shape[1] == batch["x_hist"].shape[1]
        assert batch["x_candidates"].shape[1] == batch["x_negrated"].shape[1]
        return True


def train_data_sessions_collate_fn(
    batch: list[dict], papers_embeddings: torch.Tensor, papers_ids_to_idxs: dict
) -> dict:
    users_ids, sessions_ids, candidates_papers_ids = [], [], []
    histories_papers_ids, histories_labels, histories_batch_idxs = [], [], []
    histories_days_diffs = []
    negrated_papers_ids, negrated_batch_idxs = [], []

    for i, item in enumerate(batch):
        users_ids.append(item["user_id"])
        sessions_ids.append(item["session_id"])
        candidates_papers_ids.append(item["candidate_paper_id"])
        n_history = len(item["history_papers_ids"])
        histories_papers_ids.extend(item["history_papers_ids"])
        histories_labels.extend(item["history_labels"])
        histories_batch_idxs.extend([i] * n_history)
        histories_days_diffs.extend(item["history_days_diffs"])
        n_negrated = len(item["negrated_papers_ids"])
        negrated_papers_ids.extend(item["negrated_papers_ids"])
        negrated_batch_idxs.extend([i] * n_negrated)

    candidates_papers_idxs = [papers_ids_to_idxs[paper_id] for paper_id in candidates_papers_ids]
    candidates_embeddings = papers_embeddings[candidates_papers_idxs]
    histories_papers_idxs = [papers_ids_to_idxs[paper_id] for paper_id in histories_papers_ids]
    histories_embeddings = papers_embeddings[histories_papers_idxs]
    negrated_papers_idxs = [papers_ids_to_idxs[paper_id] for paper_id in negrated_papers_ids]
    negrated_embeddings = papers_embeddings[negrated_papers_idxs]

    return {
        "user_id": torch.tensor(users_ids, dtype=torch.long),
        "session_id": torch.tensor(sessions_ids, dtype=torch.long),
        "x_candidates": candidates_embeddings,
        "x_hist": histories_embeddings,
        "y_hist": torch.tensor(histories_labels, dtype=torch.long),
        "days_diffs_hist": torch.tensor(histories_days_diffs, dtype=torch.long),
        "batch_hist": torch.tensor(histories_batch_idxs, dtype=torch.long),
        "x_negrated": negrated_embeddings,
        "batch_negrated": torch.tensor(negrated_batch_idxs, dtype=torch.long),
    }


def get_train_sessions_dataloader(
    papers_embeddings: torch.Tensor,
    papers_ids_to_idxs: dict,
    n_candidates_per_batch: int,
    n_negrated_per_candidate: int,
    n_batches_total: int,
    seed: int,
    histories_hard_constraint_min_n_train_posrated: int = 0,
    histories_hard_constraint_max_n_train_rated: int = None,
    histories_soft_constraint_max_n_train_sessions: int = None,
    histories_soft_constraint_max_n_train_days: int = None,
    histories_remove_negrated_from_history: bool = False,
    negrated_hard_constraint_max_n_ratings: int = 15,
    negrated_hard_constraint_max_n_sessions: int = None,
    negrated_hard_constraint_max_n_days: int = None,
) -> DataLoader:
    negrated_causal_mask = not histories_remove_negrated_from_history
    sessions_dataset = load_sessions_dataset_by_split(
        split="train",
        histories_hard_constraint_min_n_train_posrated=histories_hard_constraint_min_n_train_posrated,
        histories_hard_constraint_max_n_train_rated=histories_hard_constraint_max_n_train_rated,
        histories_soft_constraint_max_n_train_sessions=histories_soft_constraint_max_n_train_sessions,
        histories_soft_constraint_max_n_train_days=histories_soft_constraint_max_n_train_days,
        histories_remove_negrated_from_history=histories_remove_negrated_from_history,
        negrated_hard_constraint_max_n_ratings=negrated_hard_constraint_max_n_ratings,
        negrated_hard_constraint_max_n_sessions=negrated_hard_constraint_max_n_sessions,
        negrated_hard_constraint_max_n_days=negrated_hard_constraint_max_n_days,
        negrated_causal_mask=negrated_causal_mask,
    )
    train_sessions_dataset = TrainSessionsDataset(
        sessions_dataset=sessions_dataset,
        n_negrated_per_candidate=n_negrated_per_candidate,
        seed=seed,
    )
    train_sessions_sampler = TrainSessionsSampler(
        train_sessions_dataset=train_sessions_dataset,
        n_candidates_per_batch=n_candidates_per_batch,
        n_batches_total=n_batches_total,
        seed=seed,
    )

    def collate_fn(batch):
        return train_data_sessions_collate_fn(
            batch=batch,
            papers_embeddings=papers_embeddings,
            papers_ids_to_idxs=papers_ids_to_idxs,
        )

    train_sessions_dataloader = DataLoader(
        dataset=train_sessions_dataset,
        sampler=train_sessions_sampler,
        batch_size=n_candidates_per_batch,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    first_batch = next(iter(train_sessions_dataloader))
    train_sessions_sampler.run_test(first_batch)
    return train_sessions_dataloader
