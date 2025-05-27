
from data_handling import get_categories_distribution, get_categories_ratios, get_titles_and_abstracts, get_papers_to_exclude
from finetuning_preprocessing import *
from torch.utils.data import Dataset, DataLoader, BatchSampler
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def round_number(number : float, decimal_places : int = 4) -> float:
    return round(number, decimal_places)

def print_train_ratings_batch(batch : dict) -> str:
    s = ""
    user_idx_list, paper_id_list, label_list = batch["user_idx"][:24].tolist(), batch["paper_id"][:24].tolist(), batch["label"][:24].tolist()
    s += f"User IDXs: {user_idx_list}.\n"
    s += f"Paper IDs: {paper_id_list}.\n"
    s += f"Labels: {label_list}.\n"
    n_positive_samples, n_total_samples = sum(batch["label"].tolist()), len(batch["label"])
    s += f"Number of Samples: {n_total_samples} (positive: {n_positive_samples}, negative: {n_total_samples - n_positive_samples})."
    return s

class FinetuningDataset(Dataset):
    def __init__(self, user_idx_tensor : torch.Tensor, paper_id_tensor : torch.Tensor, label_tensor : torch.Tensor, input_ids_tensor : torch.Tensor, attention_mask_tensor : torch.Tensor,
                       category_idx_tensor : torch.Tensor) -> None:
        assert len(user_idx_tensor) == len(paper_id_tensor) == len(label_tensor) == len(input_ids_tensor) == len(attention_mask_tensor) == len(category_idx_tensor)
        self.user_idx_tensor, self.paper_id_tensor, self.label_tensor = user_idx_tensor, paper_id_tensor, label_tensor
        self.input_ids_tensor, self.attention_mask_tensor, self.category_idx_tensor = input_ids_tensor, attention_mask_tensor, category_idx_tensor

        self.n_users = len(user_idx_tensor.unique())
        assert self.user_idx_tensor.unique().tolist() == sorted(self.user_idx_tensor.unique().tolist()), "User idxs must be sorted"
        self.get_users_data()

    def __len__(self) -> int:
        return len(self.user_idx_tensor)

    def __getitem__(self, idx : int) -> dict:
        return {"user_idx" : self.user_idx_tensor[idx], "paper_id" : self.paper_id_tensor[idx], "label" : self.label_tensor[idx],
                "input_ids" : self.input_ids_tensor[idx], "attention_mask" : self.attention_mask_tensor[idx], "category_idx": self.category_idx_tensor[idx]}

    def get_users_data(self) -> None:
        unique_values, idxs = torch.unique(self.user_idx_tensor, return_inverse = True)
        assert unique_values.tolist() == sorted(unique_values.tolist()), "User idxs must be sorted"
        assert len(unique_values) == self.n_users, "There must be n_users unique user idxs"
        users_counts = torch.bincount(idxs)
        assert users_counts.sum() == self.__len__(), "The sum of the users counts must be equal to the length of the dataset"
        assert len(users_counts) == self.n_users
        users_pos_starting_idxs = torch.cat((torch.tensor([0]), torch.cumsum(users_counts, dim = 0)[:-1]))
        pos_tensor, neg_tensor = idxs[self.label_tensor == 1], idxs[self.label_tensor == 0]
        users_pos_counts, users_neg_counts = torch.bincount(pos_tensor), torch.bincount(neg_tensor)
        assert len(users_pos_counts) == len(users_counts), "Every user must have at least one positive sample"
        assert len(users_neg_counts) == len(users_counts), "Every user must have at least one negative sample"
        users_neg_starting_idxs = users_pos_starting_idxs + users_pos_counts
        self.users_counts, self.users_pos_counts, self.users_neg_counts = users_counts, users_pos_counts, users_neg_counts
        self.users_pos_starting_idxs, self.users_neg_starting_idxs = users_pos_starting_idxs, users_neg_starting_idxs
        self.users_counts_ids_to_idxs = {user_id.item() : user_idx for user_id, user_idx in zip(self.user_idx_tensor.unique(), range(self.n_users))}

def create_dataset(dataset : dict) -> FinetuningDataset:
    user_idx_tensor, paper_id_tensor, label_tensor = dataset["user_idx"], dataset["paper_id"], dataset["label"]
    input_ids_tensor, attention_mask_tensor, category_idx_tensor = dataset["input_ids"], dataset["attention_mask"], dataset["category_idx"]
    return FinetuningDataset(user_idx_tensor = user_idx_tensor, paper_id_tensor = paper_id_tensor, label_tensor = label_tensor,
                             input_ids_tensor = input_ids_tensor, attention_mask_tensor = attention_mask_tensor, category_idx_tensor = category_idx_tensor)

class TrainRatingsBatchSampler(BatchSampler):
    def __init__(self, dataset : FinetuningDataset, args_dict : dict) -> None:
        self.dataset = dataset
        self.read_args_dict(args_dict)
        self.users_probas = self.get_users_probas()
        assert abs(self.users_probas.sum() - 1) < 1e-5, "Users sampling probabilities must sum to 1"

    def read_args_dict(self, args_dict : dict) -> None:
        self.batch_size, self.users_sampling_strategy, self.n_samples_per_user = args_dict["batch_size"], args_dict["users_sampling_strategy"], args_dict["n_samples_per_user"]
        self.n_min_positive_samples_per_user, self.n_max_positive_samples_per_user = args_dict["n_min_positive_samples_per_user"], args_dict["n_max_positive_samples_per_user"]
        self.n_min_negative_samples_per_user, self.n_max_negative_samples_per_user = args_dict["n_min_negative_samples_per_user"], args_dict["n_max_negative_samples_per_user"]
        self.n_batches_total, self.seed, self.n_users_per_batch = args_dict["n_batches_total"], args_dict["seed"], self.batch_size // self.n_samples_per_user

    def get_users_probas(self) -> torch.tensor:
        if self.users_sampling_strategy == "uniform":
            users_probas = torch.ones(self.dataset.n_users) / self.dataset.n_users
        elif self.users_sampling_strategy in ["proportional", "square_root_proportional", "cube_root_proportional"]:
            if users_sampling_strategy == "proportional":
                users_counts = self.dataset.users_counts
            elif self.users_sampling_strategy == "square_root_proportional":
                users_counts = torch.sqrt(self.dataset.users_counts)
            elif self.users_sampling_strategy == "cube_root_proportional":
                users_counts = torch.pow(self.dataset.users_counts, 1/3)
            users_probas = users_counts / torch.sum(users_counts)
        else:
            raise ValueError(f"Unknown users sampling strategy: {self.users_sampling_strategy}")
        return users_probas

    def __len__(self) -> int:
        return self.n_batches_total

    def __iter__(self) -> list:
        self.rng = np.random.RandomState(self.seed)
        for _ in range(len(self)):
            selected_users_idxs = self.rng.choice(np.arange(self.dataset.n_users), size = self.n_users_per_batch, p = self.users_probas.numpy(), replace = False)
            batch_idxs = []
            for user_idx in selected_users_idxs:
                user_idxs = self._sample_user_idxs(user_idx)
                batch_idxs.extend(user_idxs)
            self.rng.shuffle(batch_idxs)
            yield batch_idxs

    def _sample_user_idxs(self, user_idx : int) -> list:
        user_pos_starting_index, user_neg_starting_index = self.dataset.users_pos_starting_idxs[user_idx], self.dataset.users_neg_starting_idxs[user_idx]
        user_pos_count, user_neg_count = self.dataset.users_pos_counts[user_idx], self.dataset.users_neg_counts[user_idx]
        user_pos_indices = np.arange(user_pos_starting_index, user_pos_starting_index + user_pos_count)
        user_neg_indices = np.arange(user_neg_starting_index, user_neg_starting_index + user_neg_count)

        if self.n_samples_per_user > user_pos_count + user_neg_count:
            raise ValueError(f"n_samples_per_user ({self.n_samples_per_user}) must be <= the number of samples for user {user_idx} ({user_pos_count + user_neg_count}).")
        elif self.n_min_positive_samples_per_user > user_pos_count:
            raise ValueError(f"n_min_positive_samples_per_user ({self.n_min_positive_samples_per_user}) must be <= the number of positive samples for user {user_idx} ({user_pos_count}).")
        elif self.n_min_negative_samples_per_user > user_neg_count:
            raise ValueError(f"n_min_negative_samples_per_user ({self.n_min_negative_samples_per_user}) must be <= the number of negative samples for user {user_idx} ({user_neg_count}).")

        self.rng.shuffle(user_pos_indices)
        self.rng.shuffle(user_neg_indices)
        selected_pos_indices = user_pos_indices[:self.n_min_positive_samples_per_user].tolist()
        selected_neg_indices = user_neg_indices[:self.n_min_negative_samples_per_user].tolist()
        remaining_pos_indices = user_pos_indices[self.n_min_positive_samples_per_user:].tolist()
        remaining_neg_indices = user_neg_indices[self.n_min_negative_samples_per_user:].tolist()
        n_remaining_to_select = self.n_samples_per_user - (self.n_min_positive_samples_per_user + self.n_min_negative_samples_per_user)

        if n_remaining_to_select == 0:
            return selected_pos_indices + selected_neg_indices
        n_pos_still_allowed = self.n_max_positive_samples_per_user - self.n_min_positive_samples_per_user
        if n_pos_still_allowed == 0:
            if n_remaining_to_select > len(remaining_neg_indices):
                raise ValueError(f"Not enough negative samples for user {user_idx}. Required: {n_remaining_to_select}, available: {len(remaining_neg_indices)}.")
            return selected_pos_indices + selected_neg_indices + remaining_neg_indices[:n_remaining_to_select]
        n_neg_still_allowed = self.n_max_negative_samples_per_user - self.n_min_negative_samples_per_user
        if n_neg_still_allowed == 0:
            if n_remaining_to_select > len(remaining_pos_indices):
                raise ValueError(f"Not enough positive samples for user {user_idx}. Required: {n_remaining_to_select}, available: {len(remaining_pos_indices)}.")
            return selected_pos_indices + selected_neg_indices + remaining_pos_indices[:n_remaining_to_select]
        
        joint_remaining = [(idx, 1) for idx in remaining_pos_indices] + [(idx, 0) for idx in remaining_neg_indices]
        self.rng.shuffle(joint_remaining)
        for idx, label in joint_remaining:
            if n_remaining_to_select <= 0:
                break
            if label == 1 and n_pos_still_allowed > 0:
                selected_pos_indices.append(idx)
                n_remaining_to_select -= 1
                n_pos_still_allowed -= 1
            elif label == 0 and n_neg_still_allowed > 0:
                selected_neg_indices.append(idx)
                n_remaining_to_select -= 1
                n_neg_still_allowed -= 1
        if n_remaining_to_select > 0:
            raise ValueError(f"Not enough samples for user {user_idx}. Required {n_remaining_to_select} more.")
        return selected_pos_indices + selected_neg_indices

    def run_test(self, batch : dict) -> bool:
        user_idx_tensor, paper_id_tensor, label_tensor = batch["user_idx"], batch["paper_id"], batch["label"]
        input_ids_tensor, attention_mask_tensor, category_idx_tensor = batch["input_ids"], batch["attention_mask"], batch["category_idx"]
        assert len(user_idx_tensor) == len(paper_id_tensor) == len(label_tensor) == len(input_ids_tensor) == len(attention_mask_tensor) == len(category_idx_tensor)
        len_batch = len(user_idx_tensor)
        assert len_batch == self.batch_size, f"Batch size must be {self.batch_size}, but got {len_batch}"
        assert len(user_idx_tensor.unique()) == self.n_users_per_batch, f"Batch must contain {self.n_users_per_batch} unique users, but got {len(user_idx.unique())}"
        for user_idx in user_idx_tensor.unique():
            n_pos = label_tensor[user_idx_tensor == user_idx].sum().item()
            n_total = len(user_idx_tensor[user_idx_tensor == user_idx])
            n_neg = n_total - n_pos
            assert n_total == self.n_samples_per_user, f"User {user_idx} must have {self.n_samples_per_user} samples, but got {n_total}"
            assert n_pos >= self.n_min_positive_samples_per_user and n_pos <= self.n_max_positive_samples_per_user
            assert n_neg >= self.n_min_negative_samples_per_user and n_neg <= self.n_max_negative_samples_per_user
        return True

def get_train_ratings_dataloader(args_dict : dict, users_embeddings_ids_to_idxs : dict, papers_ids_to_categories_idxs : dict) -> DataLoader:
    train_ratings = create_dataset(load_dataset("train", users_embeddings_ids_to_idxs, papers_ids_to_categories_idxs))
    train_ratings_batch_sampler = TrainRatingsBatchSampler(train_ratings, args_dict)
    train_ratings_dataloader = DataLoader(train_ratings, batch_sampler = train_ratings_batch_sampler, num_workers = 4, pin_memory = True)
    first_batch = next(iter(train_ratings_dataloader))
    return train_ratings_dataloader
    
class TrainNegativeSamplesDataset(Dataset):
    def __init__(self, paper_id_tensor : torch.tensor, category_idx_tensor : torch.tensor) -> None:
        assert paper_id_tensor.tolist() == sorted(paper_id_tensor.tolist()), "Papers IDs must be sorted"
        assert len(paper_id_tensor) == len(category_idx_tensor)
        self.paper_id_tensor, self.category_idx_tensor = paper_id_tensor, category_idx_tensor
        categories_idxs = category_idx_tensor.unique().tolist()
        assert categories_idxs == sorted(categories_idxs), "Categories idxs must be sorted"
        self.tensor_idxs_per_category_idx = {category_idx : torch.where(category_idx_tensor == category_idx)[0] for category_idx in categories_idxs}
        n_papers_per_category_idx = {category_idx : len(tensor_idxs) for category_idx, tensor_idxs in self.tensor_idxs_per_category_idx.items()}
        assert len(self.paper_id_tensor) == sum(n_papers_per_category_idx.values())
    
    def __len__(self) -> int:
        return len(self.paper_id_tensor)

    def __getitem__(self, idx : int) -> dict:
        return {"paper_id" : self.paper_id_tensor[idx], "category_idx" : self.category_idx_tensor[idx]}

class TrainNegativeSamplesBatchSampler(BatchSampler):
    def __init__(self, dataset : TrainNegativeSamplesDataset, categories_idxs_counts : dict, n_batches_total : int, seed : int) -> None:
        self.tensor_idxs_per_category_idx, self.categories_idxs_counts = dataset.tensor_idxs_per_category_idx, categories_idxs_counts
        self.n_batches_total, self.seed = n_batches_total, seed

    def __len__(self) -> int:
        return self.n_batches_total

    def __iter__(self) -> list:
        rng = np.random.RandomState(self.seed)
        for _ in range(len(self)):
            batch_idxs = []
            for category_idx, n_papers in self.categories_idxs_counts.items():
                tensor_idxs = self.tensor_idxs_per_category_idx[category_idx]
                if len(tensor_idxs) < n_papers:
                    raise ValueError(f"Not enough papers for category {category_idx}. Required: {n_papers}, available: {len(tensor_idxs)}.")
                sampled_tensor_idxs = rng.choice(tensor_idxs, size = n_papers, replace = False)
                batch_idxs.extend(sampled_tensor_idxs.tolist())
            rng.shuffle(batch_idxs)
            yield batch_idxs

    def run_test(self, batch : dict) -> bool:
        paper_id_tensor, category_idx_tensor = batch["paper_id"], batch["category_idx"]
        input_ids_tensor, attention_mask_tensor = batch["input_ids"], batch["attention_mask"]
        assert len(paper_id_tensor) == len(category_idx_tensor) == len(input_ids_tensor) == len(attention_mask_tensor)
        len_batch = len(paper_id_tensor)
        assert len_batch == sum(self.categories_idxs_counts.values()), f"Batch size must be {sum(self.categories_idxs_counts.values())}, but got {len_batch}"
        assert len(paper_id_tensor.unique()) == len_batch, f"Batch must contain {len_batch} unique papers, but got {len(paper_id_tensor.unique())}"
        for category_idx, n_papers in self.categories_idxs_counts.items():
            assert (category_idx_tensor == category_idx).sum().item() == n_papers
        return True

class TrainNegativeSamplesBatchProcesser:
    def __init__(self, tokenizer : AutoTokenizer):
        self.tokenizer = tokenizer

    def collate_fn(self, batch : dict) -> dict:
        paper_id_tensor, category_idx_tensor = torch.stack([b["paper_id"] for b in batch]), torch.stack([b["category_idx"] for b in batch])
        assert len(paper_id_tensor) == len(category_idx_tensor)
        batch_papers = get_titles_and_abstracts(paper_id_tensor.tolist())
        batch_papers_ids, batch_papers_titles, batch_papers_abstracts = zip(*batch_papers)
        batch_papers_ids, batch_papers_titles, batch_papers_abstracts = list(batch_papers_ids), list(batch_papers_titles), list(batch_papers_abstracts)
        batch_papers_titles_abstracts = [f"{title} {self.tokenizer.sep_token} {abstract}" for title, abstract in zip(batch_papers_titles, batch_papers_abstracts)]
        batch_tokenized_papers = self.tokenizer(text = batch_papers_titles_abstracts, max_length = 512, padding = "max_length", truncation = True, return_tensors = "pt")
        return {"paper_id" : paper_id_tensor, "category_idx" : category_idx_tensor, "input_ids" : batch_tokenized_papers["input_ids"], 
                "attention_mask" : batch_tokenized_papers["attention_mask"]}

def get_train_negative_samples_dataloader(n_train_negative_samples : int, papers_ids_to_categories_idxs : dict, n_batches_total : int, seed : int) -> DataLoader:
    categories_to_idxs, categories_ratios = load_categories_to_idxs(), get_categories_ratios()
    assert set(categories_ratios.keys()) <= set(categories_to_idxs.keys())
    categories_idxs_counts = {}
    for category, idx in categories_to_idxs.items():
        if category in categories_ratios:
            ratio = categories_ratios[category]
            if ratio > 0:
                categories_idxs_counts[idx] = round(ratio * n_train_negative_samples)
    assert n_train_negative_samples == sum(categories_idxs_counts.values())
    relevant_categories_idxs, papers_to_exclude = list(categories_idxs_counts.keys()), get_papers_to_exclude()
    assert relevant_categories_idxs == sorted(relevant_categories_idxs)
    paper_id_tensor, category_idx_tensor = [], []
    for paper_id, category_idx in papers_ids_to_categories_idxs.items():
        if category_idx in relevant_categories_idxs and paper_id not in papers_to_exclude:
            paper_id_tensor.append(paper_id)
            category_idx_tensor.append(category_idx)
    paper_id_tensor, category_idx_tensor = torch.tensor(paper_id_tensor), torch.tensor(category_idx_tensor)
    train_negative_samples_dataset = TrainNegativeSamplesDataset(paper_id_tensor, category_idx_tensor)
    train_negative_samples_batch_sampler = TrainNegativeSamplesBatchSampler(train_negative_samples_dataset, categories_idxs_counts, n_batches_total, seed)
    tokenizer = AutoTokenizer.from_pretrained(GTE_LARGE_PATH)
    train_negative_samples_dataloader = DataLoader(train_negative_samples_dataset, batch_sampler = train_negative_samples_batch_sampler, num_workers = 4, pin_memory = True,
                                                    collate_fn = TrainNegativeSamplesBatchProcesser(tokenizer).collate_fn)
    return train_negative_samples_dataloader

def load_val_data(users_embeddings_ids_to_idxs : dict, papers_ids_to_categories_idxs : dict) -> dict:
    val_data = {}
    val_users_ids = load_finetuning_users_ids()[1]
    val_data["val_users_embeddings_idxs"] = load_val_users_embeddings_idxs(val_users_ids, users_embeddings_ids_to_idxs)
    val_data["val_ratings"] = create_dataset(load_dataset("val", users_embeddings_ids_to_idxs, papers_ids_to_categories_idxs))
    val_data["val_negative_samples"] = load_papers(path = "val_negative_samples", papers_ids_to_categories_idxs = papers_ids_to_categories_idxs)
    return val_data
