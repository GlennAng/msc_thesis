import os

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

from ...logreg.src.training.training_data import (
    get_categories_ratios_for_validation,
)
from .finetuning_preprocessing import (
    load_categories_to_idxs,
    load_finetuning_dataset,
    load_finetuning_papers_tokenized,
    load_finetuning_users_ids,
    load_negative_samples_matrix_val,
    load_negative_samples_tokenized_train,
    load_users_coefs_ids_to_idxs,
    load_val_users_embeddings_idxs,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

N_SAMPLES_PRINT_TRAIN_RATINGS_BATCH = 24


def round_number(number: float, decimal_places: int = 4) -> float:
    return round(number, decimal_places)


def print_train_ratings_batch(batch: dict) -> str:
    s = ""
    user_idx_list = batch["user_idx"][:N_SAMPLES_PRINT_TRAIN_RATINGS_BATCH].tolist()
    paper_id_list = batch["paper_id"][:N_SAMPLES_PRINT_TRAIN_RATINGS_BATCH].tolist()
    rating_list = batch["rating"][:N_SAMPLES_PRINT_TRAIN_RATINGS_BATCH].tolist()
    s += f"User IDXs: {user_idx_list}.\n"
    s += f"Paper IDs: {paper_id_list}.\n"
    s += f"Ratings: {rating_list}.\n"
    n_positive_samples, n_total_samples = sum(batch["rating"].tolist()), len(batch["rating"])
    s += f"Number of Samples: {n_total_samples} (positive: {n_positive_samples}, "
    s += f"negative: {n_total_samples - n_positive_samples})."
    return s


def get_non_cs_users_sorted_indices(
    unique_users_idxs: torch.Tensor, non_cs_users_selection: str
) -> torch.Tensor:
    assert len(unique_users_idxs) == len(torch.unique(unique_users_idxs))
    assert torch.all(unique_users_idxs[:-1] <= unique_users_idxs[1:])
    non_cs_users_ids = load_finetuning_users_ids(
        selection=non_cs_users_selection,
        select_non_cs_users_only=True,
    )
    users_ids_to_idxs = load_users_coefs_ids_to_idxs()
    non_cs_users_idxs = [users_ids_to_idxs[user_id] for user_id in non_cs_users_ids]
    non_cs_users_idxs = torch.tensor(non_cs_users_idxs, dtype=torch.int64)
    assert torch.all(non_cs_users_idxs[:-1] <= non_cs_users_idxs[1:])
    assert torch.all(torch.isin(non_cs_users_idxs, unique_users_idxs))
    return torch.searchsorted(unique_users_idxs, non_cs_users_idxs)


class FinetuningDataset(Dataset):
    def __init__(
        self,
        dataset: dict,
        non_cs_users_selection: str = None,
    ) -> None:
        """
        dataset: Dictionary containing the following keys:
            - "user_idx"
            - "paper_id"
            - "rating"
            - "input_ids"
            - "attention_mask"
            - "category_l1"
            - "category_l2"
            - "time"
        """
        self._setup(dataset)
        self._assert_lengths()
        unique_users_idxs = self.user_idx_tensor.unique()
        assert unique_users_idxs.tolist() == sorted(unique_users_idxs.tolist())
        self.n_users = len(unique_users_idxs)

        if non_cs_users_selection is not None:
            self.non_cs_users_selection = get_non_cs_users_sorted_indices(
                unique_users_idxs, non_cs_users_selection
            )
        self.get_users_data()

    def _setup(self, dataset: dict) -> None:
        self.user_idx_tensor = dataset["user_idx"]
        self.paper_id_tensor = dataset["paper_id"]
        self.rating_tensor = dataset["rating"]
        self.input_ids_tensor = dataset["input_ids"]
        self.attention_mask_tensor = dataset["attention_mask"]
        self.category_l1_tensor = dataset["category_l1"]
        self.category_l2_tensor = dataset["category_l2"]
        self.time_tensor = dataset["time"]

    def _assert_lengths(self) -> None:
        n_samples = len(self.user_idx_tensor)
        tensor_lengths = [
            len(self.paper_id_tensor),
            len(self.rating_tensor),
            len(self.input_ids_tensor),
            len(self.attention_mask_tensor),
            len(self.category_l1_tensor),
            len(self.category_l2_tensor),
            len(self.time_tensor),
        ]
        assert all(length == n_samples for length in tensor_lengths)

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
        self.users_counts = users_counts
        self.users_pos_counts = users_pos_counts
        self.users_neg_counts = users_neg_counts
        self.users_pos_starting_idxs = users_pos_starting_idxs
        self.users_neg_starting_idxs = users_neg_starting_idxs
        self.users_counts_ids_to_idxs = {
            user_id.item(): user_idx
            for user_id, user_idx in zip(self.user_idx_tensor.unique(), range(self.n_users))
        }


class TrainDatasetBatchSampler(BatchSampler):
    def __init__(self, dataset: FinetuningDataset, args_dict: dict) -> None:
        self.dataset = dataset
        self.read_args_dict(args_dict)
        self.users_probas = self.get_users_probas()

    def read_args_dict(self, args_dict: dict) -> None:
        self.batch_size = args_dict["batch_size"]
        self.users_sampling_strategy = args_dict["users_sampling_strategy"]
        self.n_samples_per_user = args_dict["n_samples_per_user"]
        self.n_min_positive_samples_per_user = args_dict["n_min_positive_samples_per_user"]
        self.n_max_positive_samples_per_user = args_dict["n_max_positive_samples_per_user"]
        self.n_min_negative_samples_per_user = args_dict["n_min_negative_samples_per_user"]
        self.n_max_negative_samples_per_user = args_dict["n_max_negative_samples_per_user"]
        self.n_samples_from_most_recent_positive_votes = args_dict[
            "n_samples_from_most_recent_positive_votes"
        ]
        self.n_samples_from_closest_negative_votes = args_dict[
            "n_samples_from_closest_negative_votes"
        ]
        self.n_batches_total = args_dict["n_batches_total"]
        self.seed = args_dict["seed"]
        self.n_users_per_batch = self.batch_size // self.n_samples_per_user

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
        assert abs(users_probas.sum() - 1) < 1e-5
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
    train_dataset = FinetuningDataset(dataset=load_finetuning_dataset(dataset_type="train"))
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
    def __init__(self, train_negative_samples_tokenized: dict) -> None:
        self._stack_category_tensors(train_negative_samples_tokenized)

    def _stack_category_tensors(self, train_negative_samples_tokenized: dict) -> None:
        self.categories_starting_idxs, self.categories_ending_idxs = {}, {}
        current_idx = 0
        tensor_keys = ["paper_id", "input_ids", "attention_mask", "l1", "l2"]
        tensor_lists = {key: [] for key in tensor_keys}
        tensor_mapping = {
            "paper_id": "paper_id_tensor",
            "input_ids": "input_ids_tensor",
            "attention_mask": "attention_mask_tensor",
            "l1": "category_l1_tensor",
            "l2": "category_l2_tensor",
        }

        for category, tensors_dict in train_negative_samples_tokenized.items():
            self.categories_starting_idxs[category] = current_idx
            for key in tensor_keys:
                tensor_lists[key].extend(tensors_dict[key])
            current_idx += len(tensors_dict["paper_id"])
            self.categories_ending_idxs[category] = current_idx
        for key, attr_name in tensor_mapping.items():
            tensor_list = tensor_lists[key]
            setattr(self, attr_name, torch.stack(tensor_list) if tensor_list else torch.empty(0))
        assert len(self.paper_id_tensor) == sum(
            len(tensors_dict["paper_id"])
            for tensors_dict in train_negative_samples_tokenized.values()
        )

    def __len__(self) -> int:
        return len(self.paper_id_tensor)

    def __getitem__(self, idx: int) -> dict:
        return {
            "paper_id": self.paper_id_tensor[idx],
            "input_ids": self.input_ids_tensor[idx],
            "attention_mask": self.attention_mask_tensor[idx],
            "category_l1": self.category_l1_tensor[idx],
            "category_l2": self.category_l2_tensor[idx],
        }


class TrainNegativeSamplesBatchSampler(BatchSampler):
    def __init__(
        self,
        n_train_negative_samples: int,
        n_batches_total: int,
        seed: int,
        categories_ratios: dict,
        categories_starting_idxs: dict,
        categories_ending_idxs: dict,
    ) -> None:
        self.n_train_negative_samples = n_train_negative_samples
        self.n_batches_total = n_batches_total
        self.seed = seed
        self.categories = list(categories_ratios.keys())
        self.categories_p = list(categories_ratios.values())
        self.categories_starting_idxs = categories_starting_idxs
        self.categories_ending_idxs = categories_ending_idxs

    def __len__(self) -> int:
        return self.n_batches_total

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        for _ in range(len(self)):
            batch_idxs = []
            for _ in range(self.n_train_negative_samples):
                category = rng.choice(self.categories, p=self.categories_p)
                start_idx = self.categories_starting_idxs[category]
                end_idx = self.categories_ending_idxs[category]
                idx = rng.randint(start_idx, end_idx)
                batch_idxs.append(idx)
            rng.shuffle(batch_idxs)
            yield batch_idxs

    def run_test(self, batch: dict) -> bool:
        paper_id_tensor = batch["paper_id"]
        input_ids_tensor = batch["input_ids"]
        attention_mask_tensor = batch["attention_mask"]
        category_l1_tensor = batch["category_l1"]
        category_l2_tensor = batch["category_l2"]
        len_batch = self.n_train_negative_samples
        assert (
            len_batch
            == len(paper_id_tensor)
            == len(input_ids_tensor)
            == len(attention_mask_tensor)
            == len(category_l1_tensor)
            == len(category_l2_tensor)
        )
        return True


def convert_categories_to_idxs(categories_dict: dict, categories_to_idxs_l1: dict) -> dict:
    if set(categories_dict.keys()) <= set(categories_to_idxs_l1.keys()):
        return {
            categories_to_idxs_l1[category]: ratio for category, ratio in categories_dict.items()
        }
    return categories_dict


def get_train_negative_samples_dataloader(
    n_train_negative_samples: int,
    n_train_negative_samples_per_category_to_sample_from: int,
    n_batches_total: int,
    seed: int,
    categories_ratios: dict = None,
) -> tuple:
    categories_to_idxs_l1 = load_categories_to_idxs("l1")
    if categories_ratios is None:
        categories_ratios = get_categories_ratios_for_validation()
    categories_ratios = convert_categories_to_idxs(categories_ratios, categories_to_idxs_l1)
    train_negative_samples_tokenized = load_negative_samples_tokenized_train(
        attach_l1=True,
        attach_l2=True,
        shuffle_papers=True,
        random_state=seed,
    )
    for category, tensors_dict in train_negative_samples_tokenized.items():
        n_samples_before = len(tensors_dict["paper_id"])
        if n_samples_before > n_train_negative_samples_per_category_to_sample_from:
            train_negative_samples_tokenized[category] = {
                key: value[:n_train_negative_samples_per_category_to_sample_from]
                for key, value in tensors_dict.items()
            }
        n_samples_after = len(train_negative_samples_tokenized[category]["paper_id"])
        assert n_samples_after <= n_train_negative_samples_per_category_to_sample_from
    train_negative_samples_tokenized = convert_categories_to_idxs(
        train_negative_samples_tokenized, categories_to_idxs_l1
    )
    train_negative_samples_dataset = TrainNegativeSamplesDataset(
        train_negative_samples_tokenized=train_negative_samples_tokenized,
    )
    train_negative_samples_batch_sampler = TrainNegativeSamplesBatchSampler(
        n_train_negative_samples=n_train_negative_samples,
        n_batches_total=n_batches_total,
        seed=seed,
        categories_ratios=categories_ratios,
        categories_starting_idxs=train_negative_samples_dataset.categories_starting_idxs,
        categories_ending_idxs=train_negative_samples_dataset.categories_ending_idxs,
    )
    train_negative_samples_dataloader = DataLoader(
        train_negative_samples_dataset,
        batch_sampler=train_negative_samples_batch_sampler,
        num_workers=4,
        pin_memory=True,
    )
    first_batch = next(iter(train_negative_samples_dataloader))
    train_negative_samples_batch_sampler.run_test(first_batch)
    return train_negative_samples_dataloader


def load_val_data() -> dict:
    val_data = {}
    val_data["val_users_embeddings_idxs"] = load_val_users_embeddings_idxs()
    val_data["val_dataset"] = FinetuningDataset(
        dataset=load_finetuning_dataset("val"),
        non_cs_users_selection="val",
    )
    val_data["val_negative_samples"] = load_finetuning_papers_tokenized(
        papers_type="negative_samples_val"
    )
    val_data["val_negative_samples_matrix"] = load_negative_samples_matrix_val()
    return val_data
