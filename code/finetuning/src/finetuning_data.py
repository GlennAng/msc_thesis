import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
from transformers import AutoTokenizer

from ...logreg.src.training.training_data import get_categories_ratios
from ...src.load_files import load_papers, load_papers_texts
from ...src.project_paths import ProjectPaths
from .finetuning_preprocessing import (
    load_categories_to_idxs,
    load_finetuning_dataset,
    load_finetuning_papers,
    load_negative_samples_ids_for_seeds,
    load_val_users_embeddings_idxs,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def round_number(number: float, decimal_places: int = 4) -> float:
    return round(number, decimal_places)


def print_train_ratings_batch(batch: dict) -> str:
    s = ""
    user_idx_list, paper_id_list, rating_list = (
        batch["user_idx"][:24].tolist(),
        batch["paper_id"][:24].tolist(),
        batch["rating"][:24].tolist(),
    )
    s += f"User IDXs: {user_idx_list}.\n"
    s += f"Paper IDs: {paper_id_list}.\n"
    s += f"Ratings: {rating_list}.\n"
    n_positive_samples, n_total_samples = sum(batch["rating"].tolist()), len(batch["rating"])
    s += f"Number of Samples: {n_total_samples} (positive: {n_positive_samples}, "
    s += f"negative: {n_total_samples - n_positive_samples})."
    return s


class FinetuningDataset(Dataset):
    def __init__(
        self,
        user_idx_tensor: torch.Tensor,
        paper_id_tensor: torch.Tensor,
        rating_tensor: torch.Tensor,
        input_ids_tensor: torch.Tensor,
        attention_mask_tensor: torch.Tensor,
        category_l1_tensor: torch.Tensor,
        category_l2_tensor: torch.Tensor,
        time_tensor: torch.Tensor,
    ) -> None:
        n_samples = len(user_idx_tensor)
        assert (
            n_samples
            == len(paper_id_tensor)
            == len(rating_tensor)
            == len(input_ids_tensor)
            == len(attention_mask_tensor)
        )
        assert n_samples == len(category_l1_tensor) == len(category_l2_tensor) == len(time_tensor)
        self.user_idx_tensor, self.paper_id_tensor, self.rating_tensor = (
            user_idx_tensor,
            paper_id_tensor,
            rating_tensor,
        )
        self.input_ids_tensor, self.attention_mask_tensor = (
            input_ids_tensor,
            attention_mask_tensor,
        )
        self.category_l1_tensor, self.category_l2_tensor, self.time_tensor = (
            category_l1_tensor,
            category_l2_tensor,
            time_tensor,
        )
        self.n_users = len(user_idx_tensor.unique())
        assert self.user_idx_tensor.unique().tolist() == sorted(
            self.user_idx_tensor.unique().tolist()
        )
        self.get_users_data()

    def __len__(self) -> int:
        return len(self.user_idx_tensor)

    def __getitem__(self, idx: int) -> dict:
        return {
            "user_idx": self.user_idx_tensor[idx],
            "paper_id": self.paper_id_tensor[idx],
            "rating": self.rating_tensor[idx],
            "input_ids": self.input_ids_tensor[idx],
            "attention_mask": self.attention_mask_tensor[idx],
            "category_l1": self.category_l1_tensor[idx],
            "category_l2": self.category_l2_tensor[idx],
            "time": self.time_tensor[idx],
        }

    def get_users_data(self) -> None:
        unique_values, idxs = torch.unique(self.user_idx_tensor, return_inverse=True)
        assert unique_values.tolist() == sorted(unique_values.tolist())
        assert len(unique_values) == self.n_users
        users_counts = torch.bincount(idxs)
        assert users_counts.sum() == self.__len__()
        assert len(users_counts) == self.n_users
        users_pos_starting_idxs = torch.cat(
            (torch.tensor([0]), torch.cumsum(users_counts, dim=0)[:-1])
        )
        pos_tensor, neg_tensor = (
            idxs[self.rating_tensor == 1],
            idxs[self.rating_tensor == 0],
        )
        users_pos_counts, users_neg_counts = torch.bincount(pos_tensor), torch.bincount(neg_tensor)
        assert len(users_pos_counts) == len(users_counts)
        assert len(users_neg_counts) == len(users_counts)
        users_neg_starting_idxs = users_pos_starting_idxs + users_pos_counts
        self.users_counts, self.users_pos_counts, self.users_neg_counts = (
            users_counts,
            users_pos_counts,
            users_neg_counts,
        )
        self.users_pos_starting_idxs, self.users_neg_starting_idxs = (
            users_pos_starting_idxs,
            users_neg_starting_idxs,
        )
        self.users_counts_ids_to_idxs = {
            user_id.item(): user_idx
            for user_id, user_idx in zip(self.user_idx_tensor.unique(), range(self.n_users))
        }


def create_finetuning_dataset(dataset: dict) -> FinetuningDataset:
    user_idx_tensor, paper_id_tensor, rating_tensor = (
        dataset["user_idx"],
        dataset["paper_id"],
        dataset["rating"],
    )
    input_ids_tensor, attention_mask_tensor = (
        dataset["input_ids"],
        dataset["attention_mask"],
    )
    category_l1_tensor, category_l2_tensor, time_tensor = (
        dataset["category_l1"],
        dataset["category_l2"],
        dataset["time"],
    )
    return FinetuningDataset(
        user_idx_tensor=user_idx_tensor,
        paper_id_tensor=paper_id_tensor,
        rating_tensor=rating_tensor,
        input_ids_tensor=input_ids_tensor,
        attention_mask_tensor=attention_mask_tensor,
        category_l1_tensor=category_l1_tensor,
        category_l2_tensor=category_l2_tensor,
        time_tensor=time_tensor,
    )


class TrainDatasetBatchSampler(BatchSampler):
    def __init__(self, dataset: FinetuningDataset, args_dict: dict) -> None:
        self.dataset = dataset
        self.read_args_dict(args_dict)
        self.users_probas = self.get_users_probas()
        assert abs(self.users_probas.sum() - 1) < 1e-5

    def read_args_dict(self, args_dict: dict) -> None:
        self.batch_size, self.users_sampling_strategy, self.n_samples_per_user = (
            args_dict["batch_size"],
            args_dict["users_sampling_strategy"],
            args_dict["n_samples_per_user"],
        )
        self.n_min_positive_samples_per_user, self.n_max_positive_samples_per_user = (
            args_dict["n_min_positive_samples_per_user"],
            args_dict["n_max_positive_samples_per_user"],
        )
        self.n_min_negative_samples_per_user, self.n_max_negative_samples_per_user = (
            args_dict["n_min_negative_samples_per_user"],
            args_dict["n_max_negative_samples_per_user"],
        )
        self.n_samples_from_most_recent_positive_votes = args_dict[
            "n_samples_from_most_recent_positive_votes"
        ]
        self.n_samples_from_closest_negative_votes = args_dict[
            "n_samples_from_closest_negative_votes"
        ]
        self.n_batches_total, self.seed, self.n_users_per_batch = (
            args_dict["n_batches_total"],
            args_dict["seed"],
            self.batch_size // self.n_samples_per_user,
        )

    def get_users_probas(self) -> torch.Tensor:
        if self.users_sampling_strategy == "uniform":
            users_probas = torch.ones(self.dataset.n_users) / self.dataset.n_users
        elif self.users_sampling_strategy in [
            "proportional",
            "square_root_proportional",
            "cube_root_proportional",
        ]:
            if self.users_sampling_strategy == "proportional":
                users_counts = self.dataset.users_counts
            elif self.users_sampling_strategy == "square_root_proportional":
                users_counts = torch.sqrt(self.dataset.users_counts)
            elif self.users_sampling_strategy == "cube_root_proportional":
                users_counts = torch.pow(self.dataset.users_counts, 1 / 3)
            users_probas = users_counts / torch.sum(users_counts)
        else:
            raise ValueError(f"Unknown users sampling strategy: {self.users_sampling_strategy}")
        return users_probas

    def __len__(self) -> int:
        return self.n_batches_total

    def __iter__(self):
        self.rng = np.random.RandomState(self.seed)
        for _ in range(len(self)):
            selected_users_idxs = self.rng.choice(
                np.arange(self.dataset.n_users),
                size=self.n_users_per_batch,
                p=self.users_probas.numpy(),
                replace=False,
            )
            batch_idxs = []
            for user_idx in selected_users_idxs:
                user_idxs = self._sample_user_idxs_pick_from_most_recent_pos_and_closest_neg(
                    user_idx
                )
                batch_idxs.extend(user_idxs)
            self.rng.shuffle(batch_idxs)
            yield batch_idxs

    def _sample_user_idxs(self, user_idx: int) -> list:
        user_pos_starting_index, user_neg_starting_index = (
            self.dataset.users_pos_starting_idxs[user_idx],
            self.dataset.users_neg_starting_idxs[user_idx],
        )
        user_pos_count, user_neg_count = (
            self.dataset.users_pos_counts[user_idx],
            self.dataset.users_neg_counts[user_idx],
        )
        user_pos_indices = np.arange(
            user_pos_starting_index, user_pos_starting_index + user_pos_count
        )
        user_neg_indices = np.arange(
            user_neg_starting_index, user_neg_starting_index + user_neg_count
        )

        if self.n_samples_per_user > user_pos_count + user_neg_count:
            raise ValueError(
                f"n_samples_per_user ({self.n_samples_per_user}) must be "
                f"<= the number of samples for user {user_idx} ({user_pos_count + user_neg_count})."
            )
        elif self.n_min_positive_samples_per_user > user_pos_count:
            raise ValueError(
                f"n_min_positive_samples_per_user ({self.n_min_positive_samples_per_user}) must be "
                f"<= the number of positive samples for user {user_idx} ({user_pos_count})."
            )
        elif self.n_min_negative_samples_per_user > user_neg_count:
            raise ValueError(
                f"n_min_negative_samples_per_user ({self.n_min_negative_samples_per_user}) must be "
                f"<= the number of negative samples for user {user_idx} ({user_neg_count})."
            )

        self.rng.shuffle(user_pos_indices)
        self.rng.shuffle(user_neg_indices)
        selected_pos_indices = user_pos_indices[: self.n_min_positive_samples_per_user].tolist()
        selected_neg_indices = user_neg_indices[: self.n_min_negative_samples_per_user].tolist()
        remaining_pos_indices = user_pos_indices[self.n_min_positive_samples_per_user :].tolist()
        remaining_neg_indices = user_neg_indices[self.n_min_negative_samples_per_user :].tolist()
        n_remaining_to_select = self.n_samples_per_user - (
            self.n_min_positive_samples_per_user + self.n_min_negative_samples_per_user
        )

        if n_remaining_to_select == 0:
            return selected_pos_indices + selected_neg_indices
        n_pos_still_allowed = (
            self.n_max_positive_samples_per_user - self.n_min_positive_samples_per_user
        )
        if n_pos_still_allowed == 0:
            if n_remaining_to_select > len(remaining_neg_indices):
                raise ValueError(
                    f"Not enough negative samples for user {user_idx}. "
                    f"Required: {n_remaining_to_select}, available: {len(remaining_neg_indices)}."
                )
            return (
                selected_pos_indices
                + selected_neg_indices
                + remaining_neg_indices[:n_remaining_to_select]
            )
        n_neg_still_allowed = (
            self.n_max_negative_samples_per_user - self.n_min_negative_samples_per_user
        )
        if n_neg_still_allowed == 0:
            if n_remaining_to_select > len(remaining_pos_indices):
                raise ValueError(
                    f"Not enough positive samples for user {user_idx}. "
                    f"Required: {n_remaining_to_select}, available: {len(remaining_pos_indices)}."
                )
            return (
                selected_pos_indices
                + selected_neg_indices
                + remaining_pos_indices[:n_remaining_to_select]
            )

        joint_remaining = [(idx, 1) for idx in remaining_pos_indices] + [
            (idx, 0) for idx in remaining_neg_indices
        ]
        self.rng.shuffle(joint_remaining)
        for idx, rating in joint_remaining:
            if n_remaining_to_select <= 0:
                break
            if rating == 1 and n_pos_still_allowed > 0:
                selected_pos_indices.append(idx)
                n_remaining_to_select -= 1
                n_pos_still_allowed -= 1
            elif rating == 0 and n_neg_still_allowed > 0:
                selected_neg_indices.append(idx)
                n_remaining_to_select -= 1
                n_neg_still_allowed -= 1
        if n_remaining_to_select > 0:
            raise ValueError(
                f"Not enough samples for user {user_idx}. Required {n_remaining_to_select} more."
            )
        return selected_pos_indices + selected_neg_indices

    def _sample_user_idxs_pick_from_most_recent_pos_and_closest_neg(self, user_idx: int) -> list:
        assert self.n_min_negative_samples_per_user == self.n_max_negative_samples_per_user == 4
        assert self.n_samples_per_user == self.n_min_negative_samples_per_user + 1
        user_pos_starting_index, user_neg_starting_index = (
            self.dataset.users_pos_starting_idxs[user_idx],
            self.dataset.users_neg_starting_idxs[user_idx],
        )
        user_pos_count, user_neg_count = (
            self.dataset.users_pos_counts[user_idx],
            self.dataset.users_neg_counts[user_idx],
        )
        user_pos_indices = np.arange(
            user_pos_starting_index, user_pos_starting_index + user_pos_count
        )
        user_neg_indices = np.arange(
            user_neg_starting_index, user_neg_starting_index + user_neg_count
        )

        user_pos_indices = user_pos_indices[-self.n_samples_from_most_recent_positive_votes :]
        self.rng.shuffle(user_pos_indices)
        selected_pos_indices = user_pos_indices[:1].tolist()
        pos_time = self.dataset.time_tensor[selected_pos_indices[0]].item()
        user_neg_indices = user_neg_indices.tolist()
        user_neg_indices.sort(key=lambda idx: abs(self.dataset.time_tensor[idx].item() - pos_time))
        selected_neg_indices = user_neg_indices[: self.n_samples_from_closest_negative_votes]
        self.rng.shuffle(selected_neg_indices)
        selected_neg_indices = selected_neg_indices[:4]
        return selected_pos_indices + selected_neg_indices

    def run_test(self, batch: dict) -> bool:
        user_idx_tensor, paper_id_tensor, rating_tensor = (
            batch["user_idx"],
            batch["paper_id"],
            batch["rating"],
        )
        input_ids_tensor, attention_mask_tensor = (
            batch["input_ids"],
            batch["attention_mask"],
        )
        category_l1_tensor, category_l2_tensor, time_tensor = (
            batch["category_l1"],
            batch["category_l2"],
            batch["time"],
        )
        len_batch = len(user_idx_tensor)
        assert (
            len_batch
            == len(paper_id_tensor)
            == len(rating_tensor)
            == len(input_ids_tensor)
            == len(attention_mask_tensor)
        )
        assert len_batch == len(category_l1_tensor) == len(category_l2_tensor) == len(time_tensor)
        assert len_batch == self.batch_size
        assert len(user_idx_tensor.unique()) == self.n_users_per_batch
        for user_idx in user_idx_tensor.unique():
            n_pos = rating_tensor[user_idx_tensor == user_idx].sum().item()
            n_total = len(user_idx_tensor[user_idx_tensor == user_idx])
            n_neg = n_total - n_pos
            assert n_total == self.n_samples_per_user
            assert (
                n_pos >= self.n_min_positive_samples_per_user
                and n_pos <= self.n_max_positive_samples_per_user
            )
            assert (
                n_neg >= self.n_min_negative_samples_per_user
                and n_neg <= self.n_max_negative_samples_per_user
            )
        return True


def get_train_dataset_dataloader(args_dict: dict) -> DataLoader:
    train_dataset = create_finetuning_dataset(load_finetuning_dataset(dataset_type="train"))
    train_dataset_batch_sampler = TrainDatasetBatchSampler(train_dataset, args_dict)
    train_dataset_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_dataset_batch_sampler,
        num_workers=4,
        pin_memory=True,
    )
    first_batch = next(iter(train_dataset_dataloader))
    train_dataset_batch_sampler.run_test(first_batch)
    return train_dataset_dataloader


class TrainNegativeSamplesDataset(Dataset):
    def __init__(
        self,
        paper_id_tensor: torch.Tensor,
        category_l1_tensor: torch.Tensor,
        category_l2_tensor: torch.Tensor,
    ) -> None:
        assert paper_id_tensor.tolist() == sorted(paper_id_tensor.tolist())
        assert len(paper_id_tensor) == len(category_l1_tensor) == len(category_l2_tensor)
        self.paper_id_tensor, self.category_l1_tensor, self.category_l2_tensor = (
            paper_id_tensor,
            category_l1_tensor,
            category_l2_tensor,
        )
        categories_l1 = self.category_l1_tensor.unique().tolist()
        self.tensor_idxs_per_category_l1 = {
            category_l1: torch.where(self.category_l1_tensor == category_l1)[0]
            for category_l1 in categories_l1
        }
        n_papers_per_category_l1 = {
            category_l1: len(tensor_idxs)
            for category_l1, tensor_idxs in self.tensor_idxs_per_category_l1.items()
        }
        assert len(self.paper_id_tensor) == sum(n_papers_per_category_l1.values())

    def __len__(self) -> int:
        return len(self.paper_id_tensor)

    def __getitem__(self, idx: int) -> dict:
        return {
            "paper_id": self.paper_id_tensor[idx],
            "category_l1": self.category_l1_tensor[idx],
            "category_l2": self.category_l2_tensor[idx],
        }


class TrainNegativeSamplesBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: TrainNegativeSamplesDataset,
        n_papers_per_category_l1: dict,
        n_batches_total: int,
        seed: int,
    ) -> None:
        self.tensor_idxs_per_category_l1 = dataset.tensor_idxs_per_category_l1
        self.n_papers_per_category_l1 = n_papers_per_category_l1
        self.n_batches_total, self.seed = n_batches_total, seed

    def __len__(self) -> int:
        return self.n_batches_total

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        for _ in range(len(self)):
            batch_idxs = []
            for category_idx, n_papers in self.n_papers_per_category_l1.items():
                tensor_idxs = self.tensor_idxs_per_category_l1[category_idx]
                if len(tensor_idxs) < n_papers:
                    raise ValueError(
                        f"Not enough papers for category {category_idx}. "
                        f"Required: {n_papers}, available: {len(tensor_idxs)}."
                    )
                sampled_tensor_idxs = rng.choice(tensor_idxs, size=n_papers, replace=False)
                batch_idxs.extend(sampled_tensor_idxs.tolist())
            rng.shuffle(batch_idxs)
            yield batch_idxs

    def run_test(self, batch: dict) -> bool:
        paper_id_tensor, category_l1_tensor, category_l2_tensor = (
            batch["paper_id"],
            batch["category_l1"],
            batch["category_l2"],
        )
        input_ids_tensor, attention_mask_tensor = (
            batch["input_ids"],
            batch["attention_mask"],
        )
        len_batch = len(paper_id_tensor)
        assert (
            len_batch
            == len(category_l1_tensor)
            == len(category_l2_tensor)
            == len(input_ids_tensor)
            == len(attention_mask_tensor)
        )
        assert len_batch == sum(self.n_papers_per_category_l1.values())
        assert len(paper_id_tensor.unique()) == len_batch
        for category_idx, n_papers in self.n_papers_per_category_l1.items():
            assert (category_l1_tensor == category_idx).sum().item() == n_papers
        return True


class TrainNegativeSamplesBatchProcesser:
    def __init__(self, papers_texts: pd.DataFrame, tokenizer: AutoTokenizer):
        self.papers_texts = papers_texts
        self.tokenizer = tokenizer

    def collate_fn(self, batch: dict) -> dict:
        paper_id_tensor = torch.stack([b["paper_id"] for b in batch])
        category_l1_tensor = torch.stack([b["category_l1"] for b in batch])
        category_l2_tensor = torch.stack([b["category_l2"] for b in batch])
        assert len(paper_id_tensor) == len(category_l1_tensor) == len(category_l2_tensor)

        batch_papers = self.papers_texts.loc[
            self.papers_texts["paper_id"].isin(paper_id_tensor.tolist())
        ]
        batch_papers = batch_papers[["paper_id", "title", "abstract"]].values.tolist()
        batch_papers_ids, batch_papers_titles, batch_papers_abstracts = zip(*batch_papers)
        batch_papers_ids, batch_papers_titles, batch_papers_abstracts = (
            list(batch_papers_ids),
            list(batch_papers_titles),
            list(batch_papers_abstracts),
        )
        batch_papers_titles_abstracts = [
            f"{title} {self.tokenizer.sep_token} {abstract}"
            for title, abstract in zip(batch_papers_titles, batch_papers_abstracts)
        ]
        batch_tokenized_papers = self.tokenizer(
            text=batch_papers_titles_abstracts,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "paper_id": paper_id_tensor,
            "category_l1": category_l1_tensor,
            "category_l2": category_l2_tensor,
            "input_ids": batch_tokenized_papers["input_ids"],
            "attention_mask": batch_tokenized_papers["attention_mask"],
        }


def get_train_negative_samples_dataloader(
    n_train_negative_samples: int,
    n_batches_total: int,
    seed: int,
    categories_ratios: dict = None,
    categories_to_idxs_l1: dict = None,
    categories_to_idxs_l2: dict = None,
) -> DataLoader:
    if categories_to_idxs_l1 is None:
        categories_to_idxs_l1 = load_categories_to_idxs("l1")
    if categories_to_idxs_l2 is None:
        categories_to_idxs_l2 = load_categories_to_idxs("l2")
    if categories_ratios is None:
        categories_ratios = get_categories_ratios()
    assert set(categories_ratios.keys()) <= set(categories_to_idxs_l1.keys())
    tokenizer = AutoTokenizer.from_pretrained(ProjectPaths.finetuning_data_model_hf())

    n_papers_per_category_l1, non_zero_categories = {}, []
    for category, idx in categories_to_idxs_l1.items():
        if category in categories_ratios:
            ratio = categories_ratios[category]
            if ratio > 0:
                n_papers_per_category_l1[idx] = round(ratio * n_train_negative_samples)
                non_zero_categories.append(category)
    assert n_train_negative_samples == sum(n_papers_per_category_l1.values())

    papers = load_papers()
    negative_samples_ids_for_seeds = load_negative_samples_ids_for_seeds()
    all_negative_samples_ids = [
        item for sublist in negative_samples_ids_for_seeds.values() for item in sublist
    ]
    all_negative_samples_ids = sorted(list(set(all_negative_samples_ids)))
    mask = ~papers["in_ratings"] & ~papers["in_cache"]
    if all_negative_samples_ids:
        mask &= ~papers["paper_id"].isin(all_negative_samples_ids)
    if non_zero_categories:
        mask &= papers["l1"].isin(non_zero_categories)
    potential_papers = papers.loc[mask, ["paper_id", "l1", "l2"]]
    paper_id_tensor = potential_papers["paper_id"].values.tolist()
    papers_texts = load_papers_texts(
        relevant_papers_ids=paper_id_tensor,
        relevant_columns=["paper_id", "title", "abstract"],
    )
    paper_id_tensor = torch.tensor(paper_id_tensor)
    paper_id_tensor = torch.tensor(potential_papers["paper_id"].values, dtype=torch.int64)
    category_l1_tensor = torch.tensor(
        potential_papers["l1"].map(categories_to_idxs_l1).values, dtype=torch.int64
    )
    category_l2_tensor = torch.tensor(
        potential_papers["l2"].map(categories_to_idxs_l2).values, dtype=torch.int64
    )
    train_negative_samples_dataset = TrainNegativeSamplesDataset(
        paper_id_tensor, category_l1_tensor, category_l2_tensor
    )
    train_negative_samples_batch_sampler = TrainNegativeSamplesBatchSampler(
        train_negative_samples_dataset, n_papers_per_category_l1, n_batches_total, seed
    )
    train_negative_samples_dataloader = DataLoader(
        train_negative_samples_dataset,
        batch_sampler=train_negative_samples_batch_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=TrainNegativeSamplesBatchProcesser(papers_texts, tokenizer).collate_fn,
    )
    first_batch = next(iter(train_negative_samples_dataloader))
    train_negative_samples_batch_sampler.run_test(first_batch)
    return train_negative_samples_dataloader


def load_val_data() -> dict:
    val_data = {}
    val_data["val_users_embeddings_idxs"] = load_val_users_embeddings_idxs()
    val_data["val_dataset"] = create_finetuning_dataset(load_finetuning_dataset("val"))
    val_data["val_negative_samples"] = load_finetuning_papers("val_negative_samples")
    return val_data
