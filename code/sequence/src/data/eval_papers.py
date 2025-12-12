import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler

from ..data.sequence_preprocessing import load_sequence_papers_tokenized


class EvalPaperDataset(Dataset):
    def __init__(self, papers_tokenized: dict):
        self.paper_id_tensor = papers_tokenized["paper_id"]
        self.input_ids_tensor = papers_tokenized["input_ids"]
        self.attention_mask_tensor = papers_tokenized["attention_mask"]
        self.l1_tensor = papers_tokenized["l1"]
        self.l2_tensor = papers_tokenized["l2"]

        self.length = self.paper_id_tensor.shape[0]
        assert all(
            tensor.shape[0] == self.length
            for tensor in [
                self.input_ids_tensor,
                self.attention_mask_tensor,
                self.l1_tensor,
                self.l2_tensor,
            ]
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        return {
            "paper_id": self.paper_id_tensor[idx],
            "input_ids": self.input_ids_tensor[idx],
            "attention_mask": self.attention_mask_tensor[idx],
            "l1": self.l1_tensor[idx],
            "l2": self.l2_tensor[idx],
        }


class EvalPaperBatchSampler(Sampler):
    def __init__(
        self,
        dataset: EvalPaperDataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.length = len(dataset)

    def __iter__(self):
        indices = list(range(self.length))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, self.length, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            yield batch_indices

    def __len__(self) -> int:
        if self.drop_last:
            return self.length // self.batch_size
        else:
            return (self.length + self.batch_size - 1) // self.batch_size


def get_eval_papers_dataloader(papers_type: str, batch_size: int = 768) -> DataLoader:
    papers_tokenized = load_sequence_papers_tokenized(papers_type=papers_type)
    dataset = EvalPaperDataset(papers_tokenized)
    sampler = EvalPaperBatchSampler(dataset, batch_size=batch_size)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)


def get_eval_papers_ids(papers_type: str) -> list:
    papers_tokenized = load_sequence_papers_tokenized(papers_type=papers_type)
    papers_ids = papers_tokenized["paper_id"].tolist()
    assert len(papers_ids) == len(set(papers_ids))
    assert papers_ids == sorted(papers_ids)
    return papers_ids


def get_val_rated_papers_ids() -> list:
    return get_eval_papers_ids(papers_type="eval_val_users")


def get_val_negative_samples_ids() -> list:
    return get_eval_papers_ids(papers_type="negative_samples_val")
