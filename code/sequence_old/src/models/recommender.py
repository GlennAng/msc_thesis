import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .papers_encoder import PapersEncoder, load_papers_encoder
from .users_encoder import UsersEncoder, load_users_encoder


class Recommender(nn.Module):
    def __init__(
        self,
        users_encoder: UsersEncoder,
        papers_encoder: PapersEncoder = None,
        papers_embeddings: torch.Tensor = None,
        papers_ids_to_idxs: dict = None,
        use_papers_encoder: bool = False,
    ):
        super(Recommender, self).__init__()
        self.users_encoder = users_encoder
        self.use_papers_encoder = use_papers_encoder
        if self.use_papers_encoder:
            assert papers_encoder is not None
            self.papers_encoder = papers_encoder
        else:
            assert papers_embeddings is not None
            assert papers_ids_to_idxs is not None
            self.papers_embeddings = papers_embeddings
            self.papers_ids_to_idxs = papers_ids_to_idxs

    def save_model(self, path: Path) -> None:
        if not isinstance(path, Path):
            path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        self.users_encoder.save_model(path / "users_encoder")
        if self.use_papers_encoder:
            self.papers_encoder.save_model(path / "papers_encoder")

    def to_device(self, device: torch.device) -> None:
        self.users_encoder.to(device)
        if self.use_papers_encoder:
            self.papers_encoder.to(device)
        users_encoder_device, papers_encoders_device = self.get_devices()
        assert users_encoder_device == device
        if self.use_papers_encoder:
            assert papers_encoders_device == device

    def get_devices(self) -> tuple:
        users_encoder_device = self.users_encoder.get_device()
        if self.use_papers_encoder:
            papers_encoder_device = self.papers_encoder.get_device()
        else:
            papers_encoder_device = None
        return users_encoder_device, papers_encoder_device

    def get_memory_footprint(self) -> dict:
        memory_info = {}
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        memory_info["total_parameters"] = total_params
        memory_info["trainable_parameters"] = trainable_params
        if total_params > 0:
            memory_info["trainable_percentage"] = trainable_params / total_params
        else:
            memory_info["trainable_percentage"] = 0.0
        if torch.cuda.is_available():
            memory_info["gpu_memory_allocated (GB)"] = torch.cuda.memory_allocated() / 1024**3
            memory_info["gpu_memory_reserved (GB)"] = torch.cuda.memory_reserved() / 1024**3
        return memory_info

    def extract_papers_embeddings(self, papers_ids: list) -> tuple:
        assert not self.use_papers_encoder
        if isinstance(papers_ids, torch.Tensor):
            papers_ids = papers_ids.tolist()
        idxs = [self.papers_ids_to_idxs[pid] for pid in papers_ids]
        with torch.no_grad():
            papers_embeddings = self.papers_embeddings[idxs]
        papers_ids_to_idxs = {pid: idx for idx, pid in enumerate(papers_ids)}
        return papers_embeddings, papers_ids_to_idxs


def load_recommender(
    users_encoder_dict: dict, papers_encoder_dict: dict, use_papers_encoder: bool = False
) -> Recommender:
    users_encoder = load_users_encoder(**users_encoder_dict)
    if use_papers_encoder:
        papers_encoder = load_papers_encoder(**papers_encoder_dict)
        recommender = Recommender(
            users_encoder=users_encoder, papers_encoder=papers_encoder, use_papers_encoder=True
        )
    else:
        assert "papers_embeddings_path" in papers_encoder_dict
        path = papers_encoder_dict["papers_embeddings_path"]
        if not isinstance(path, Path):
            path = Path(path).resolve()
        embed_path = path / "abs_X.npy"
        ids_to_idxs_path = path / "abs_paper_ids_to_idx.pkl"
        papers_embeddings = torch.from_numpy(np.load(embed_path)).requires_grad_(False)
        with open(ids_to_idxs_path, "rb") as f:
            papers_ids_to_idxs = pickle.load(f)
        recommender = Recommender(
            users_encoder=users_encoder,
            papers_embeddings=papers_embeddings,
            papers_ids_to_idxs=papers_ids_to_idxs,
            use_papers_encoder=False,
        )
    return recommender
