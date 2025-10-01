import json
import pickle
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_batch

from .gru_users_encoder import GRUUsersEncoder
from .mean_pooling_users_encoder import MeanPoolingUsersEncoder
from .nrms_users_encoder import NRMSUsersEncoder
from .users_encoder import (
    UsersEncoder,
    UsersEncoderType,
    get_users_encoder_type_from_arg,
)


class Recommender(nn.Module):
    def __init__(
        self,
        users_encoder: UsersEncoder,
        papers_embeddings: torch.Tensor,
        papers_ids_to_idxs: dict,
    ):
        super(Recommender, self).__init__()
        self.users_encoder = users_encoder
        self.papers_embeddings = papers_embeddings
        self.papers_ids_to_idxs = papers_ids_to_idxs

    def forward(self, batch: dict) -> tuple:
        users_embeddings = self.users_encoder(batch)
        negative_samples_dot_products = torch.matmul(
            users_embeddings, batch["x_negative_samples"].T
        )
        negrated_vector_agg, mask_negrated = self.get_negrated(batch)
        candidates_expanded = batch["x_candidates"].unsqueeze(1)
        stacked_items = torch.cat([candidates_expanded, negrated_vector_agg], dim=1)
        candidates_and_negrated_dots = torch.bmm(
            users_embeddings.unsqueeze(1), stacked_items.transpose(1, 2)
        ).squeeze(1)
        all_dot_products = torch.cat(
            [candidates_and_negrated_dots, negative_samples_dot_products], dim=1
        )
        return all_dot_products, mask_negrated

    def get_negrated(self, batch: dict) -> tuple:
        batch_size = batch["x_candidates"].size(0)
        if len(batch["batch_negrated"]) == 0:
            feature_dim = batch["x_candidates"].size(-1)
            negrated_vector_agg = torch.zeros(batch_size, 0, feature_dim, device=self.get_device())
            mask_negrated = torch.zeros(batch_size, 0, device=self.get_device(), dtype=torch.bool)
        else:
            negrated_vector_agg, mask_negrated = to_dense_batch(
                batch["x_negrated"], batch["batch_negrated"], batch_size=batch_size
            )
        return negrated_vector_agg, mask_negrated

    def save_model(self, path: Path) -> None:
        if not isinstance(path, Path):
            path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        self.users_encoder.save_model(path)

    def to_device(self, device: torch.device) -> None:
        self.users_encoder.to_device(device)
        assert self.users_encoder.get_device() == device

    def get_device(self) -> tuple:
        return self.users_encoder.get_device()

    def get_memory_footprint(self) -> dict:
        return self.users_encoder.get_memory_footprint()

    def compute_users_embeddings(self, dataloader: DataLoader) -> tuple:
        return self.users_encoder.compute_users_embeddings(dataloader)

    def extract_papers_embeddings(self, papers_ids: list) -> tuple:
        if isinstance(papers_ids, torch.Tensor):
            papers_ids = papers_ids.tolist()
        idxs = [self.papers_ids_to_idxs[pid] for pid in papers_ids]
        with torch.no_grad():
            papers_embeddings = self.papers_embeddings[idxs]
        papers_ids_to_idxs = {pid: idx for idx, pid in enumerate(papers_ids)}
        return papers_embeddings, papers_ids_to_idxs


def get_users_encoder_class(users_encoder_type: UsersEncoderType) -> UsersEncoder:
    if users_encoder_type == UsersEncoderType.MEAN_POS_POOLING:
        return MeanPoolingUsersEncoder
    elif users_encoder_type == UsersEncoderType.NRMS:
        return NRMSUsersEncoder
    elif users_encoder_type == UsersEncoderType.GRU:
        return GRUUsersEncoder
    else:
        raise ValueError(f"Unknown users_encoder_type: {users_encoder_type}")


def load_recommender_pretrained_users_encoder(
    recommender_model_path: Path, users_encoder_type: UsersEncoderType
) -> UsersEncoder:
    if not isinstance(recommender_model_path, Path):
        recommender_model_path = Path(recommender_model_path).resolve()
    config_path = recommender_model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    users_encoder_state_dict_path = recommender_model_path / "users_encoder.pt"
    if not users_encoder_state_dict_path.exists():
        raise FileNotFoundError(
            f"Users Encoder state dict not found at: {users_encoder_state_dict_path}"
        )

    with open(config_path, "r") as f:
        config = json.load(f)
    users_encoder_class = get_users_encoder_class(users_encoder_type)
    config = {k: v for k, v in config.items() if k != "users_encoder_type"}
    users_encoder = users_encoder_class(**config)
    users_encoder.load_state_dict(torch.load(users_encoder_state_dict_path, weights_only=True))
    return users_encoder


def load_recommender_pretrained_embeddings(embeddings_path: Path) -> tuple:
    if not isinstance(embeddings_path, Path):
        embeddings_path = Path(embeddings_path).resolve()
    papers_embeddings_path = embeddings_path / "abs_X.npy"
    if not papers_embeddings_path.exists():
        raise FileNotFoundError(f"Papers embeddings not found at: {papers_embeddings_path}")
    papers_ids_to_idxs_path = embeddings_path / "abs_paper_ids_to_idx.pkl"
    if not papers_ids_to_idxs_path.exists():
        raise FileNotFoundError(f"Papers ids to idxs not found at: {papers_ids_to_idxs_path}")

    papers_embeddings = torch.from_numpy(np.load(papers_embeddings_path)).requires_grad_(False)
    with open(papers_ids_to_idxs_path, "rb") as f:
        papers_ids_to_idxs = pickle.load(f)
    return papers_embeddings, papers_ids_to_idxs


def load_recommender_pretrained(
    recommender_model_path: Path,
    embeddings_path: Path,
    users_encoder_type_str: str,
    device: torch.device = None,
) -> Recommender:
    users_encoder_type = get_users_encoder_type_from_arg(users_encoder_type_str)
    users_encoder = load_recommender_pretrained_users_encoder(
        recommender_model_path, users_encoder_type
    )
    papers_embeddings, papers_ids_to_idxs = load_recommender_pretrained_embeddings(embeddings_path)
    recommender = Recommender(
        users_encoder=users_encoder,
        papers_embeddings=papers_embeddings,
        papers_ids_to_idxs=papers_ids_to_idxs,
    )
    if device is not None:
        recommender.to_device(device)
    return recommender


@contextmanager
def temporary_seed(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def load_recommender_from_scratch(
    users_encoder_type: UsersEncoderType,
    embeddings_path: Path,
    random_seed: int = None,
    device: torch.device = None,
    **users_encoder_kwargs,
) -> Recommender:
    if random_seed is not None:
        with temporary_seed(random_seed):
            users_encoder_class = get_users_encoder_class(users_encoder_type)
    else:
        users_encoder_class = get_users_encoder_class(users_encoder_type)
    users_encoder = users_encoder_class(**users_encoder_kwargs)
    papers_embeddings, papers_ids_to_idxs = load_recommender_pretrained_embeddings(embeddings_path)
    recommender = Recommender(
        users_encoder=users_encoder,
        papers_embeddings=papers_embeddings,
        papers_ids_to_idxs=papers_ids_to_idxs,
    )
    if device is not None:
        recommender.to_device(device)
    return recommender
