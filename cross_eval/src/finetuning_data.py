from finetuning_preprocessing import *
from torch.utils.data import Dataset, Sampler
import torch

class FinetuningDataset(Dataset):
    def __init__(self, user_idx_tensor : torch.Tensor, paper_id_tensor : torch.Tensor, label_tensor : torch.Tensor, input_ids_tensor : torch.Tensor, attention_mask_tensor : torch.Tensor) -> None:
        assert len(user_idx_tensor) == len(paper_id_tensor) == len(label_tensor) == len(input_ids_tensor) == len(attention_mask_tensor)
        self.n_samples = len(user_idx_tensor)
        self.user_idx_tensor, self.paper_id_tensor, self.label_tensor = user_idx_tensor, paper_id_tensor, label_tensor
        self.input_ids_tensor, self.attention_mask_tensor = input_ids_tensor, attention_mask_tensor

        self.n_users = len(user_idx_tensor.unique())
        assert self.user_idx_tensor.unique().tolist() == sorted(self.user_idx_tensor.unique().tolist()), "User idxs must be sorted"

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx : int) -> dict:
        batch = {"user_idx" : self.user_idx_tensor[idx], "paper_id" : self.paper_id_tensor[idx], "label" : self.label_tensor[idx],
                 "input_ids" : self.input_ids_tensor[idx], "attention_mask" : self.attention_mask_tensor[idx]}
        return batch

    def get_users_counts(self) -> None:
        self.users_counts = torch.bincount(self.user_idx_tensor)
        assert self.users_counts.sum() == self.n_samples, "Users counts must sum to the number of samples"
        self.users_pos_starting_indices = torch.cat((torch.tensor([0]), torch.cumsum(self.users_counts, dim = 0)[:-1]))
        pos_tensor, neg_tensor = self.user_idx_tensor[self.label_tensor == 1], self.user_idx_tensor[self.label_tensor == 0]
        self.users_pos_counts, self.users_neg_counts = torch.bincount(pos_tensor), torch.bincount(neg_tensor)
        assert len(self.users_pos_counts) == len(self.users_counts), "Every user must have at least one positive sample"
        assert len(self.users_neg_counts) == len(self.users_counts), "Every user must have at least one negative sample"
        self.users_neg_starting_indices = self.users_pos_starting_indices + self.users_pos_counts

class FinetuningSampler(Sampler):
    def __init__(self, dataset : FinetuningDataset, batch_size : int, n_samples_per_user : int, users_sampling_strategy : str, class_balancing : bool, seed : int) -> None:
        super().__init__()
        self.dataset, self.batch_size, self.n_samples_per_user = dataset, batch_size, n_samples_per_user
        self.users_sampling_strategy, self.class_balancing = users_sampling_strategy, class_balancing

        self.n_users_per_batch = self.batch_size // self.n_samples_per_user
        self.n_batches_per_epoch = len(self.dataset) // self.batch_size
        self.n_samples_per_epoch = self.n_batches_per_epoch * self.batch_size

        self.dataset.get_users_counts()
        self.users_probas = self.get_users_probas()
        assert abs(self.users_probas.sum() - 1) < 1e-5, "Users sampling probabilities must sum to 1"

        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.n_samples_per_epoch

    def __iter__(self) -> int:
        epoch_seed = self.seed + self.epoch
        self.rng = np.random.RandomState(epoch_seed)
        for _ in range(self.n_batches_per_epoch):
            batch_indices = []
            selected_users_idxs = self.rng.choice(np.arange(self.dataset.n_users), size = self.n_users_per_batch, p = self.users_probas.numpy(), replace = False)
            for user_idx in selected_users_idxs:
                user_indices = self._sample_user_indices(user_idx)
                batch_indices.extend(user_indices)
            for idx in batch_indices:
                yield idx

    def _sample_user_indices(self, user_idx : int) -> list:
        user_pos_starting_index, user_neg_starting_index = self.dataset.users_pos_starting_indices[user_idx], self.dataset.users_neg_starting_indices[user_idx]
        user_pos_count, user_neg_count = self.dataset.users_pos_counts[user_idx], self.dataset.users_neg_counts[user_idx]
        user_pos_ending_index, user_neg_ending_index = user_pos_starting_index + user_pos_count, user_neg_starting_index + user_neg_count
        if self.class_balancing:
            assert self.n_samples_per_user % 2 == 0, "n_samples_per_user must be even for class balancing"
            n_samples_per_class = self.n_samples_per_user // 2
            pos_replace, neg_replace = n_samples_per_class > user_pos_count, n_samples_per_class > user_neg_count
            if pos_replace:
                print(f"Warning: User {user_idx} has fewer positive samples ({user_pos_count}) than requested ({n_samples_per_class}). Using replacement.")
            if neg_replace:
                print(f"Warning: User {user_idx} has fewer negative samples ({user_neg_count}) than requested ({n_samples_per_class}). Using replacement.")
            pos_range, neg_range = np.arange(user_pos_starting_index, user_pos_ending_index), np.arange(user_neg_starting_index, user_neg_ending_index)
            pos_idxs = self.rng.choice(pos_range, size = n_samples_per_class, replace = pos_replace).tolist()
            neg_idxs = self.rng.choice(neg_range, size = n_samples_per_class, replace = neg_replace).tolist()
            idxs = pos_idxs + neg_idxs
        else:
            all_range = np.arange(user_pos_starting_index, user_neg_ending_index)
            replace = self.n_samples_per_user > len(all_range)
            if replace:
                print(f"Warning: User {user_idx} has fewer samples ({len(all_range)}) than requested ({self.n_samples_per_user}). Using replacement.")
            idxs = self.rng.choice(all_range, size = self.n_samples_per_user, replace=replace).tolist()
        return idxs

    def get_users_probas(self) -> torch.tensor:
        if self.users_sampling_strategy == "uniform":
            users_probas = torch.ones(self.dataset.n_users) / self.dataset.n_users
        elif self.users_sampling_strategy in ["proportional", "square_root_proportional", "cube_root_proportional"]:
            if self.users_sampling_strategy == "proportional":
                users_counts = self.dataset.users_counts
            elif self.users_sampling_strategy == "square_root_proportional":
                users_counts = torch.sqrt(self.dataset.users_counts)
            elif self.users_sampling_strategy == "cube_root_proportional":
                users_counts = torch.pow(self.dataset.users_counts, 1/3)
            users_probas = users_counts / torch.sum(users_counts)
        else:
            raise ValueError(f"Unknown users sampling strategy: {self.users_sampling_strategy}")
        return users_probas

    def run_test(self, batch : dict) -> None:
        user_idx_tensor, paper_id_tensor, label_tensor = batch["user_idx"], batch["paper_id"], batch["label"]
        input_ids_tensor, attention_mask_tensor = batch["input_ids"], batch["attention_mask"]
        assert len(user_idx_tensor) == len(paper_id_tensor) == len(label_tensor) == len(input_ids_tensor) == len(attention_mask_tensor), "Batch tensors must have the same length"
        len_batch = len(user_idx_tensor)
        assert len_batch == self.batch_size, f"Batch size must be {self.batch_size}, but got {len_batch}"
        assert len(user_idx_tensor.unique()) == self.n_users_per_batch, f"Batch must contain {self.n_users_per_batch} unique users, but got {len(user_idx.unique())}"
        for user_idx in user_idx_tensor.unique():
            n_pos = label_tensor[user_idx_tensor == user_idx].sum().item()
            n_total = len(user_idx_tensor[user_idx_tensor == user_idx])
            n_neg = n_total - n_pos
            assert n_total == self.n_samples_per_user, f"User {user_idx} must have {self.n_samples_per_user} samples, but got {n_total}"
            if self.class_balancing:
                assert n_pos == n_neg, f"User {user_idx} must have the same number of positive and negative samples, but got {n_pos} positive and {n_neg} negative samples"

def create_dataset(dataset : dict, users_embeddings_ids_to_idxs : dict) -> FinetuningDataset:
    user_idx_tensor = torch.tensor([users_embeddings_ids_to_idxs[user_id.item()] for user_id in dataset["user_id"]], dtype = dataset["user_id"].dtype)
    return FinetuningDataset(user_idx_tensor, dataset["paper_id"], dataset["label"], dataset["input_ids"], dataset["attention_mask"])

def create_train_val_dataset(train_dataset : FinetuningDataset, val_dataset : FinetuningDataset) -> FinetuningDataset:
    val_users_idxs = val_dataset.user_idx_tensor.unique()
    mask = torch.isin(train_dataset.user_idx_tensor, val_users_idxs)
    idxs_where_label_is_1 = torch.where((train_dataset.label_tensor == 1) & mask)[0]
    user_idx_tensor = train_dataset.user_idx_tensor[idxs_where_label_is_1]
    paper_id_tensor = train_dataset.paper_id_tensor[idxs_where_label_is_1]
    label_tensor = train_dataset.label_tensor[idxs_where_label_is_1]
    input_ids_tensor = train_dataset.input_ids_tensor[idxs_where_label_is_1]
    attention_mask_tensor = train_dataset.attention_mask_tensor[idxs_where_label_is_1]
    return FinetuningDataset(user_idx_tensor, paper_id_tensor, label_tensor, input_ids_tensor, attention_mask_tensor)

def load_datasets_dict() -> dict:
    datasets_dict = {}
    datasets_dict["train_users_ids"], datasets_dict["val_users_ids"], datasets_dict["test_users_no_overlap_ids"] = load_finetuning_users_ids()
    datasets_dict["users_embeddings_ids_to_idxs"] = load_users_embeddings_ids_to_idxs()
    train_dataset, val_dataset = load_train_dataset(), load_val_dataset()
    datasets_dict["train_dataset"] = create_dataset(train_dataset, datasets_dict["users_embeddings_ids_to_idxs"])
    datasets_dict["val_dataset"] = create_dataset(val_dataset, datasets_dict["users_embeddings_ids_to_idxs"])
    datasets_dict["train_val_dataset"] = create_train_val_dataset(datasets_dict["train_dataset"], datasets_dict["val_dataset"])
    datasets_dict["val_users_embeddings_idxs"] = load_val_users_embeddings_idxs(datasets_dict["val_users_ids"], datasets_dict["users_embeddings_ids_to_idxs"])
    datasets_dict["negative_samples"] = load_negative_samples()
    datasets_dict["test_papers"] = load_test_papers()
    return datasets_dict