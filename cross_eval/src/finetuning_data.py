from finetuning_preprocessing import *
from torch.utils.data import Dataset, Sampler
import torch

class FinetuningDataset(Dataset):
    def __init__(self, user_idx_tensor : torch.Tensor, paper_id_tensor : torch.Tensor, label_tensor : torch.Tensor, input_ids_tensor : torch.Tensor, attention_mask_tensor : torch.Tensor) -> None:
        assert len(user_idx_tensor) == len(paper_id_tensor) == len(label_tensor) == len(input_ids_tensor) == len(attention_mask_tensor)
        self.n_samples = len(user_idx_tensor)
        self.user_idx_tensor, self.paper_id_tensor, self.label_tensor = user_idx_tensor, paper_id_tensor, label_tensor
        self.input_ids_tensor, self.attention_mask_tensor = input_ids_tensor, attention_mask_tensor

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx : int) -> dict:
        batch = {"user_idx" : self.user_idx_tensor[idx], "paper_id" : self.paper_id_tensor[idx], "label" : self.label_tensor[idx],
                 "input_ids" : self.input_ids_tensor[idx], "attention_mask" : self.attention_mask_tensor[idx]}
        return batch

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

def load_datasets() -> tuple:
    train_users_ids, val_users_ids, test_users_no_overlap_ids = load_finetuning_users_ids()
    train_dataset, val_dataset, users_embeddings_ids_to_idxs = load_train_dataset(), load_val_dataset(), load_users_embeddings_ids_to_idxs()
    train_dataset = create_dataset(train_dataset, users_embeddings_ids_to_idxs)
    val_dataset = create_dataset(val_dataset, users_embeddings_ids_to_idxs)
    train_val_dataset = create_train_val_dataset(train_dataset, val_dataset)
    val_users_embeddings_idxs = load_val_users_embeddings_idxs(val_users_ids, users_embeddings_ids_to_idxs)
    negative_samples = load_negative_samples()
    test_papers = load_test_papers()
    return train_dataset, val_dataset, train_val_dataset, val_users_embeddings_idxs, negative_samples, test_papers