import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

from ....logreg.src.embeddings.compute_embeddings import get_gpu_info
from ..eval.users_embeddings_data import save_users_embeddings


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
        users_dict, args_dict={"output_folder": path, "single_val_session": False}
    )


class UsersEncoder(nn.Module):
    def __init__(self):
        super(UsersEncoder, self).__init__()
        self.required_batch_keys = self._get_required_batch_keys()

    def _get_required_batch_keys(self) -> list:
        return ["x_hist", "batch_hist"]

    def to_device(self, device: torch.device) -> None:
        self.device = device

    def get_device(self) -> torch.device:
        return self.device

    def forward(self, batch: dict) -> torch.Tensor:
        self._verify_batch(batch)
        hist_vector_agg, mask_hist = to_dense_batch(batch["x_hist"], batch["batch_hist"])
        user_embeddings = hist_vector_agg.sum(dim=1) / mask_hist.sum(dim=1, keepdim=True)
        batch_size = user_embeddings.size(0)
        zeros_column = torch.zeros(batch_size, 1, device=user_embeddings.device)
        user_embeddings = torch.cat([user_embeddings, zeros_column], dim=1)
        return user_embeddings

    def _verify_batch(self, batch: dict) -> None:
        if len(self.required_batch_keys) == 0:
            return
        device = self.get_device()
        for key in self.required_batch_keys:
            if key not in batch:
                raise ValueError(f"Batch is missing required key: {key}")
            tensor = batch[key]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Batch key {key} must be a torch.Tensor")
            if tensor.device != device:
                raise ValueError(f"Batch key {key} is on device {tensor.device}, expected {device}")

    def save_model(self, path: Path) -> None:
        if not isinstance(path, Path):
            path = Path(path).resolve()
        components_path = path / "components.pt"
        if os.path.exists(components_path):
            print(f"Users Encoder components already exist at: {components_path}. Skipping save.")
            return
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path / "components.pt")
        print(f"Users Encoder components saved at: {components_path}")

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
                with torch.autocast(device_type=device.type, dtype=torch.float16):
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


def load_users_encoder(path: Path = None, device: torch.device = None) -> UsersEncoder:
    users_encoder = UsersEncoder()
    if path is not None:
        if not isinstance(path, Path):
            path = Path(path).resolve()
        state_dict = torch.load(path / "components.pt", weights_only=True)
        users_encoder.load_state_dict(state_dict)
    if device is not None:
        users_encoder.to_device(device)
    return users_encoder
