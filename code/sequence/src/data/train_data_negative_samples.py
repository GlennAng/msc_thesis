import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import default_collate

from ....finetuning.src.finetuning_preprocessing import (
    get_negative_samples_ids_per_category_dict_train,
    load_categories_to_idxs,
)
from ....logreg.src.training.training_data import get_categories_ratios_for_validation


class TrainNegativeSamplesDataset(Dataset):
    def __init__(
        self, train_negative_samples_ids_per_category: dict, categories_to_idxs: dict = None
    ) -> None:
        self._stack_category_tensors(train_negative_samples_ids_per_category, categories_to_idxs)

    def _stack_category_tensors(
        self, train_negative_samples_ids_per_category: dict, categories_to_idxs: dict
    ) -> None:
        self.categories_starting_idxs, self.categories_ending_idxs = {}, {}
        current_idx = 0
        if categories_to_idxs is None:
            categories_to_idxs = load_categories_to_idxs()
        paper_id_list, category_l1_list = [], []
        for category, papers_ids in train_negative_samples_ids_per_category.items():
            category_idx = categories_to_idxs[category]
            self.categories_starting_idxs[category_idx] = current_idx
            paper_id_list.extend(papers_ids)
            n_papers = len(papers_ids)
            category_l1_list.extend([category_idx] * n_papers)
            current_idx += n_papers
            self.categories_ending_idxs[category_idx] = current_idx
        self.paper_id_tensor = torch.tensor(paper_id_list, dtype=torch.long)
        self.category_l1_tensor = torch.tensor(category_l1_list, dtype=torch.long)
        assert len(self.paper_id_tensor) == len(self.category_l1_tensor)
        assert len(self.paper_id_tensor) == sum(
            len(v) for v in train_negative_samples_ids_per_category.values()
        )

    def __len__(self) -> int:
        return len(self.paper_id_tensor)

    def __getitem__(self, idx: int) -> dict:
        return {
            "paper_id": self.paper_id_tensor[idx],
            "category_l1": self.category_l1_tensor[idx],
        }


class TrainNegativeSamplesSampler(Sampler):
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
                category = rng.choice(self.categories, p=self.categories_p).item()
                start_idx = self.categories_starting_idxs[category]
                end_idx = self.categories_ending_idxs[category]
                idx = rng.randint(start_idx, end_idx)
                batch_idxs.append(idx)
            rng.shuffle(batch_idxs)
            yield batch_idxs

    def run_test(self, batch: dict, embedding_dim: int = None) -> bool:
        x_negative_samples = batch["x_negative_samples"]
        cs_mask = batch["cs_mask"]
        assert x_negative_samples.shape[0] == self.n_train_negative_samples
        assert cs_mask.shape[0] == self.n_train_negative_samples
        if embedding_dim is not None:
            assert x_negative_samples.shape[1] == embedding_dim
        return True


def train_negative_samples_collate_fn(
    batch: dict, papers_embeddings: torch.Tensor, papers_ids_to_idxs: dict, cs_idx: int
) -> dict:
    batched = default_collate(batch)
    papers_ids = batched["paper_id"].tolist()
    embedding_indices = [papers_ids_to_idxs[paper_id] for paper_id in papers_ids]
    x_negative_samples = papers_embeddings[embedding_indices]
    cs_mask = batched["category_l1"] == cs_idx
    return {
        "x_negative_samples": x_negative_samples,
        "cs_mask": cs_mask,
    }


def get_train_negative_samples_dataloader(
    papers_embeddings: torch.Tensor,
    papers_ids_to_idxs: dict,
    n_train_negative_samples: int,
    n_batches_total: int,
    seed: int,
    categories_ratios: dict = None,
) -> DataLoader:
    categories_to_idxs_l1 = load_categories_to_idxs("l1")
    cs_idx = categories_to_idxs_l1["Computer Science"]
    if categories_ratios is None:
        categories_ratios = get_categories_ratios_for_validation()
    categories_ratios = {categories_to_idxs_l1[k]: v for k, v in categories_ratios.items()}
    negative_samples_ids_per_category_dict_train = get_negative_samples_ids_per_category_dict_train(
        finetuning=False,
        n_train_negative_samples_per_category_max=None,
        selection_random_state=None,
    )
    train_negative_samples_dataset = TrainNegativeSamplesDataset(
        train_negative_samples_ids_per_category=negative_samples_ids_per_category_dict_train,
        categories_to_idxs=categories_to_idxs_l1,
    )
    train_negative_samples_sampler = TrainNegativeSamplesSampler(
        n_train_negative_samples=n_train_negative_samples,
        n_batches_total=n_batches_total,
        seed=seed,
        categories_ratios=categories_ratios,
        categories_starting_idxs=train_negative_samples_dataset.categories_starting_idxs,
        categories_ending_idxs=train_negative_samples_dataset.categories_ending_idxs,
    )

    def collate_fn(batch):
        return train_negative_samples_collate_fn(
            batch=batch,
            papers_embeddings=papers_embeddings,
            papers_ids_to_idxs=papers_ids_to_idxs,
            cs_idx=cs_idx,
        )

    dataloader = DataLoader(
        dataset=train_negative_samples_dataset,
        batch_sampler=train_negative_samples_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    first_batch = next(iter(dataloader))
    assert train_negative_samples_sampler.run_test(
        batch=first_batch, embedding_dim=papers_embeddings.shape[1]
    )
    return dataloader
