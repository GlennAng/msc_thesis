import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[2]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_paths_to_sys()
ProjectPaths.add_finetuning_paths_to_sys()

import json, gc, os, pickle, time, torch
import numpy as np, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from compute_embeddings import get_gpu_info

class FinetuningModel(nn.Module):
    def __init__(self, transformer_model: AutoModel, projection: nn.Linear, users_embeddings: nn.Embedding, categories_embeddings_l1: nn.Embedding,
                 categories_embeddings_l2: nn.Embedding = none, val_users_embeddings_idxs: torch.Tensor = None, n_unfreeze_layers: int = 4) -> None:
        super().__init__()
        self.transformer_model = transformer_model
        self.transformer_model_name = ProjectPaths.finetuning_data_model_path().stem
        self.eval_batch_size = 512
        self.projection = projection
        self.users_embeddings = users_embeddings
        self.categories_embeddings_l1 = categories_embeddings_l1
        self.categories_embeddings_l2 = categories_embeddings_l2
        self.device = next(transformer_model.parameters()).device
        assert self.device == self.projection.weight.device == self.users_embeddings.weight.device == self.categories_embeddings_l1.weight.device
        if categories_embeddings_l2 is not None:
            assert self.device == self.categories_embeddings_l2.weight.device
        if val_users_embeddings_idxs is not None:
            self.val_users_embeddings_idxs = val_users_embeddings_idxs.to(self.device)
            assert self.val_users_embeddings_idxs.tolist() == sorted(self.val_users_embeddings_idxs.tolist())
        self.unfreeze_layers(n_unfreeze_layers)

    def unfreeze_layers(self, n_unfreeze_layers: int) -> None:
        for param in self.transformer_model.parameters():
            param.requires_grad = False
        transformer_layers = self.transformer_model.encoder.layer
        for i in range(len(transformer_layers) - n_unfreeze_layers, len(transformer_layers)):
            transformer_layers[i].requires_grad_(True)

    def count_transformer_layers(self) -> int:
        return len(self.transformer_model.encoder.layer)

    def count_transformer_parameters(self) -> tuple:
        n_params_total = 0
        n_params_unfreeze = 0
        for name, param in self.named_parameters():
            n_params_total += param.numel()
            if param.requires_grad:
                n_params_unfreeze += param.numel()
        return n_params_total, n_params_unfreeze




def load_transformer_model(transformer_path: str, device: torch.device) -> AutoModel:
    return AutoModel.from_pretrained(transformer_path, trust_remote_code = True, unpad_inputs = True, torch_dtype = "auto").to(device)

def save_transformer_model(transformer_model: AutoModel, model_path: Path) -> None:
    transformer_model.save_pretrained(model_path)

def get_transformer_embeddings_dim(transformer_model: AutoModel) -> int:
    return transformer_model.config.hidden_size

def load_projection(device: torch.device, transformer_embeddings_dim: int = 1024, projection_dim: int = 256, projection_state_dict: dict = None, 
                    dtype = torch.float32) -> nn.Linear:
    if projection_state_dict is not None:
        projection_dim, transformer_embeddings_dim = projection_state_dict["weight"].shape
    projection = nn.Linear(transformer_embeddings_dim, projection_dim)
    if projection_state_dict is not None:
        projection.load_state_dict(projection_state_dict)
    projection = projection.to(device, dtype = dtype)
    return projection

def load_embeddings(device: torch.device, num_embeddings: int = None, embeddings_dim: int = None, embeddings_state_dict: dict = None, 
                    dtype = torch.float32) -> nn.Embedding:
    if embeddings_state_dict is not None:
        num_embeddings, embeddings_dim = embeddings_state_dict["weight"].shape
    embeddings = nn.Embedding(num_embeddings, embeddings_dim)
    if embeddings_state_dict is not None:
        embeddings.load_state_dict(embeddings_state_dict)
    embeddings = embeddings.to(device, dtype = dtype)
    return embeddings

def load_categories_embeddings_l2(device: torch.device, num_embeddings: int, embeddings_dim: int, dtype = torch.float32) -> nn.Embedding:
    categories_embeddings_l2 = nn.Embedding(num_embeddings, embeddings_dim, dtype = dtype, device = device)
    nn.init.normal_(categories_embeddings_l2.weight, std = 0.01)
    return categories_embeddings_l2

def fix_gte_config(model_path: Path) -> None:
    config_path = model_path / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    if "auto_map" in config:
        for key in config["auto_map"]:
            if key == "AutoConfig" and not config["auto_map"][key].startswith("Alibaba-NLP"):
                config["auto_map"][key] = "Alibaba-NLP/new-impl--configuration.NewConfig"
            elif not config["auto_map"][key].startswith("Alibaba-NLP"):
                model_class = config["auto_map"][key].split(".")[-1]
                config["auto_map"][key] = f"Alibaba-NLP/new-impl--modeling.{model_class}"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent = 2)

def load_finetuning_model(finetuning_model_path: str, device: torch.device, val_users_embeddings_idxs: torch.tensor = None, n_unfreeze_layers: int = 4,
                          projection_state_dict: dict = None, users_embeddings_state_dict: dict = None, categories_embeddings_l1_state_dict: dict = None,
                          pretrained_projection: bool = True, pretrained_users_embeddings: bool = True, pretrained_categories_embeddings_l1: bool = True,
                          projection_dim: int = None, n_users: int = None, n_categories : int = None, categories_embeddings_l1_dim: int = None,
                          categories_embeddings_l2_state_dict: dict = None, n_categories_l2: int = None) -> FinetuningModel:
    finetuning_model_path = finetuning_model_path.rstrip("/") + "/"
    transformer_model_path = finetuning_model_path + "transformer_model"
    fix_gte_config(transformer_model_path)
    transformer_model = load_transformer_model(transformer_model_path, device)
    transformer_embeddings_dim = get_transformer_embeddings_dim(transformer_model)

    if projection_state_dict is None and pretrained_projection:
        projection_state_dict = torch.load(finetuning_model_path + "projection.pt", map_location = device, weights_only = True)
    projection = load_projection(device, transformer_embeddings_dim, projection_dim, projection_state_dict)
    if users_embeddings_state_dict is None and pretrained_users_embeddings:
        users_embeddings_state_dict = torch.load(finetuning_model_path + "users_embeddings.pt", map_location = device, weights_only = True)
    users_embeddings = load_embeddings(device, n_users, transformer_embeddings_dim + categories_embeddings_dim + 1, users_embeddings_state_dict)
    if categories_embeddings_state_dict is None and pretrained_categories_embeddings:
        categories_embeddings_state_dict = torch.load(finetuning_model_path + "categories_embeddings.pt", map_location = device, weights_only = True)
    categories_embeddings = load_embeddings(device, n_categories, categories_embeddings_dim, categories_embeddings_state_dict)
    if n_categories_l2 > 0:
        if categories_embeddings_l2_state_dict is None:
            if os.path.exists(finetuning_model_path + "categories_embeddings_l2.pt"):
                categories_embeddings_l2_state_dict = torch.load(finetuning_model_path + "categories_embeddings_l2.pt", map_location = device, weights_only = True)
                categories_embeddings_l2 = load_embeddings(device, n_categories_l2, categories_embeddings_dim, categories_embeddings_l2_state_dict)
            else:
                categories_embeddings_l2 = load_categories_embeddings_l2(device, n_categories_l2, categories_embeddings_dim, categories_embeddings_l2_state_dict)
        else:
            categories_embeddings_l2 = load_embeddings(device, n_categories_l2, categories_embeddings_dim, categories_embeddings_l2_state_dict)
    else:
        categories_embeddings_l2 = None
    return FinetuningModel(transformer_model, projection, users_embeddings, categories_embeddings, val_users_embeddings_idxs, categories_embeddings_l2, n_unfreeze_layers = n_unfreeze_layers)


    