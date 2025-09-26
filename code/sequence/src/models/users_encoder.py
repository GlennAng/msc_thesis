import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ....logreg.src.embeddings.compute_embeddings import get_gpu_info
from ..data.users_embeddings_data import save_users_embeddings


def save_users_embeddings_as_pickle(
    path: Path, embeddings: torch.Tensor, users_sessions_ids_to_idxs: dict
) -> None:
    if not isinstance(path, Path):
        path = Path(path).resolve()
    users_dict = {
        uid: {"sessions_ids": list(users_sessions_ids_to_idxs[uid].keys())}
        for uid in users_sessions_ids_to_idxs
    }
    for uid in users_dict:
        users_dict[uid]["sessions_embeddings"] = (
            embeddings[list(users_sessions_ids_to_idxs[uid].values())].numpy().astype(np.float64)
        )
    save_users_embeddings(
        users_embeddings=users_dict,
        users_embeddings_folder=path,
        single_val_session=False,
    )


class UsersEncoder(nn.Module, ABC):
    def __init__(self):
        super(UsersEncoder, self).__init__()
        self.required_batch_keys = self._get_required_batch_keys()

    @abstractmethod
    def _get_required_batch_keys(self) -> list:
        pass

    @abstractmethod
    def _encode_user(self, batch: dict) -> torch.Tensor:
        pass

    @abstractmethod
    def get_config(self) -> dict:
        pass

    def to_device(self, device: torch.device) -> None:
        self.to(device)
        assert self.get_device() == device

    def get_device(self) -> torch.device:
        device = next(self.parameters()).device
        assert all(param.device == device for param in self.parameters())
        return device

    def forward(self, batch: dict) -> torch.Tensor:
        self._verify_batch(batch)
        return self._encode_user(batch)

    def _verify_batch(self, batch: dict) -> None:
        if len(self.required_batch_keys) == 0:
            return
        target_device = self.get_device()
        for key in self.required_batch_keys:
            if key not in batch:
                raise ValueError(f"Batch is missing required key: {key}")
            tensor = batch[key]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Batch key {key} must be a torch.Tensor")
            tensor_device = tensor.device
            if target_device != tensor_device:
                raise ValueError(
                    f"Batch key {key} is on device {tensor_device}, expected {target_device}"
                )

    def save_model(self, path: Path) -> None:
        if not isinstance(path, Path):
            path = Path(path).resolve()
        os.makedirs(path, exist_ok=True)

        users_encoder_path = path / "users_encoder.pt"
        torch.save(self.state_dict(), users_encoder_path)
        config_path = path / "config.json"
        config = self.get_config()
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Users Encoder saved at: {users_encoder_path}")
        print(f"Config saved at: {config_path}")

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

    def compute_users_embeddings(self, dataloader: DataLoader) -> tuple:
        device = self.get_device()
        in_training = self.training
        self.eval()
        all_embeddings = []
        users_sessions_ids_to_idxs = {}
        current_idx = 0
        pbar = None

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if not hasattr(self, "required_batch_keys"):
                    required_batch_keys = list(batch.keys())
                else:
                    required_batch_keys = self.required_batch_keys

                for key in required_batch_keys:
                    batch[key] = batch[key].to(device)

                with torch.inference_mode():
                    embeddings = self(batch)

                if batch_idx == 0:
                    print(f"First Batch: {get_gpu_info()}")
                    pbar = tqdm(total=len(dataloader), desc="Computing user embeddings")
                pbar.update(1)

                all_embeddings.append(embeddings.cpu())
                users_ids = batch["user_id"].cpu().tolist()
                sessions_ids = batch["session_id"].cpu().tolist()

                for user_id, session_id in zip(users_ids, sessions_ids):
                    if user_id not in users_sessions_ids_to_idxs:
                        users_sessions_ids_to_idxs[user_id] = {}
                    users_sessions_ids_to_idxs[user_id][session_id] = current_idx
                    current_idx += 1

        if pbar is not None:
            pbar.close()

        all_embeddings = torch.cat(all_embeddings, dim=0)
        if in_training:
            self.train()
        return all_embeddings, users_sessions_ids_to_idxs
