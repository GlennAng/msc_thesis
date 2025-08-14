import gc
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from ...logreg.src.embeddings.compute_embeddings import get_gpu_info
from ...src.project_paths import ProjectPaths
from .finetuning_preprocessing import (
    load_categories_to_idxs,
    load_users_coefs_ids_to_idxs,
)

EVAL_BATCH_SIZE = 512
FINETUNING_DTYPE = torch.float32
CS_ID = 4


class FinetuningModel(nn.Module):
    def __init__(
        self,
        transformer_model: AutoModel,
        projection: nn.Linear,
        users_embeddings: nn.Embedding,
        categories_embeddings_l1: nn.Embedding,
        categories_embeddings_l2: nn.Embedding = None,
        val_users_embeddings_idxs: torch.Tensor = None,
        n_unfreeze_layers: int = 4,
        unfreeze_word_embeddings: bool = False,
        unfreeze_from_bottom: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_model = transformer_model
        self.projection = projection
        self.users_embeddings = users_embeddings
        self.categories_embeddings_l1 = categories_embeddings_l1
        self.categories_embeddings_l2 = categories_embeddings_l2

        self.device = next(transformer_model.parameters()).device
        self._validate_and_setup_components(val_users_embeddings_idxs)
        self._unfreeze_layers(n_unfreeze_layers, unfreeze_word_embeddings, unfreeze_from_bottom)

    def _validate_and_setup_components(self, val_users_embeddings_idxs: torch.Tensor) -> None:
        components = [
            self.projection.weight,
            self.users_embeddings.weight,
            self.categories_embeddings_l1.weight,
        ]
        if self.categories_embeddings_l2 is not None:
            components.append(self.categories_embeddings_l2.weight)
        if val_users_embeddings_idxs is not None:
            self.val_users_embeddings_idxs = val_users_embeddings_idxs.to(self.device)
            components.append(self.val_users_embeddings_idxs)
            val_users_embeddings_idxs_sorted = (
                self.val_users_embeddings_idxs[1:] - self.val_users_embeddings_idxs[:-1]
            )
            if not torch.all(val_users_embeddings_idxs_sorted >= 0):
                raise ValueError(
                    "Tensor val_users_embeddings_idxs must be sorted in ascending order."
                )
        all_same_device = all(component.device == self.device for component in components)
        if not all_same_device:
            raise ValueError("All components must be on the same device as the transformer model.")

    def _unfreeze_layers(
        self, n_unfreeze_layers: int, unfreeze_word_embeddings: bool, unfreeze_from_bottom: bool
    ) -> None:
        for param in self.transformer_model.parameters():
            param.requires_grad = False
        if unfreeze_word_embeddings:
            for param in self.transformer_model.embeddings.parameters():
                param.requires_grad = True
        transformer_layers = self.transformer_model.encoder.layer
        if unfreeze_from_bottom:
            for i in range(n_unfreeze_layers):
                transformer_layers[i].requires_grad_(True)
        else:
            for i in range(len(transformer_layers) - n_unfreeze_layers, len(transformer_layers)):
                transformer_layers[i].requires_grad_(True)

    def forward(
        self,
        eval_type: str,
        tensors_dict: dict,
        n_negative_samples_per_user: int = "all",
        categories_cosine_term: bool = False,
    ) -> torch.Tensor:
        """
        tensors_dict: Dictionary containing the following keys:
            - input_ids_tensor (required)
            - attention_mask_tensor (required)
            - category_l1_tensor (required)
            - category_l2_tensor (optional)
            - user_idx_tensor (optional)
            - sorted_unique_user_idx_tensor (optional)
            - batch_negatives_indices (optional)
        """
        valid_eval_types = ["train", "negative_samples", "val", "test"]
        if eval_type is None or eval_type not in valid_eval_types:
            raise ValueError(f"eval_type must be one of {valid_eval_types}, but got {eval_type}.")
        tensors_dict = self._check_and_move_tensors(tensors_dict)

        papers_embeddings = self._compute_papers_embeddings(
            input_ids_tensor=tensors_dict["input_ids_tensor"],
            attention_mask_tensor=tensors_dict["attention_mask_tensor"],
            category_l1_tensor=tensors_dict["category_l1_tensor"],
            category_l2_tensor=tensors_dict.get("category_l2_tensor", None),
        )
        if eval_type == "test":
            return papers_embeddings
        elif eval_type == "negative_samples":
            return self._compute_negative_samples_scores(
                negative_samples_embeddings=papers_embeddings,
                sorted_unique_users_idx_tensor=tensors_dict["sorted_unique_user_idx_tensor"],
                n_negative_samples_per_user=n_negative_samples_per_user,
                categories_cosine_term=categories_cosine_term,
                category_l1_tensor=tensors_dict["category_l1_tensor"],
            )
        else:
            return self._compute_train_val_scores(
                papers_embeddings=papers_embeddings,
                user_idx_tensor=tensors_dict["user_idx_tensor"],
                sorted_unique_users_idx_tensor=tensors_dict.get(
                    "sorted_unique_user_idx_tensor", None
                ),
                batch_negatives_indices=tensors_dict.get("batch_negatives_indices", None),
            )

    def _check_and_move_tensors(self, tensors_dict: dict) -> dict:
        device_checked = {}
        for key, tensor in tensors_dict.items():
            if tensor is not None and hasattr(tensor, "device"):
                if tensor.device != self.device:
                    device_checked[key] = tensor.to(self.device)
                else:
                    device_checked[key] = tensor
            else:
                device_checked[key] = tensor
        return device_checked

    def _compute_papers_embeddings(
        self,
        input_ids_tensor: torch.Tensor,
        attention_mask_tensor: torch.Tensor,
        category_l1_tensor: torch.Tensor,
        category_l2_tensor: torch.Tensor = None,
    ) -> torch.Tensor:
        papers_embeddings = self.transformer_model(
            input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
        ).last_hidden_state[:, 0, :]
        papers_embeddings = self.projection(papers_embeddings)
        papers_embeddings = F.normalize(papers_embeddings, p=2, dim=1)
        categories_embeddings = self.categories_embeddings_l1(category_l1_tensor)
        if self.categories_embeddings_l2 is not None and category_l2_tensor is not None:
            categories_embeddings_l2 = self.categories_embeddings_l2(category_l2_tensor)
            categories_embeddings = categories_embeddings + categories_embeddings_l2
        papers_embeddings = torch.cat((papers_embeddings, categories_embeddings), dim=1)
        return papers_embeddings

    def _compute_negative_samples_scores(
        self,
        negative_samples_embeddings: torch.Tensor,
        sorted_unique_users_idx_tensor: torch.Tensor,
        n_negative_samples_per_user: int = "all",
        categories_cosine_term: bool = False,
        category_l1_tensor: torch.Tensor = None,
    ) -> torch.Tensor:
        sorted_unique_users_embeddings = self.users_embeddings(sorted_unique_users_idx_tensor)
        if n_negative_samples_per_user == "all":
            dot_products = torch.matmul(
                sorted_unique_users_embeddings[:, :-1],
                negative_samples_embeddings.transpose(0, 1),
            )
        else:
            if len(negative_samples_embeddings) % n_negative_samples_per_user != 0:
                raise ValueError(
                    "Length of negative_samples_embeddings must be divisible by n_negative_samples_per_user."
                )
            n_users = len(sorted_unique_users_embeddings)
            negative_samples_batches = negative_samples_embeddings.view(
                n_users, n_negative_samples_per_user, -1
            )
            dot_products = torch.bmm(
                sorted_unique_users_embeddings[:, :-1].unsqueeze(1),
                negative_samples_batches.transpose(1, 2),
            ).squeeze(1)
        dot_products = dot_products + sorted_unique_users_embeddings[:, -1].unsqueeze(1)
        categories_dot_products = None
        if categories_cosine_term:
            categories_dot_products = self._compute_categories_repulsion_loss(
                negative_samples_embeddings, category_l1_tensor)
        return dot_products, categories_dot_products

    def _compute_categories_repulsion_loss(
        self,
        negative_samples_embeddings: torch.Tensor,
        category_l1_tensor: torch.Tensor,
        margin: float = 0.4
    ) -> torch.Tensor:
        non_cs_mask = category_l1_tensor != CS_ID
        if non_cs_mask.sum() <= 1:
            return torch.tensor(0.0, device=negative_samples_embeddings.device)
        non_cs_embeddings = negative_samples_embeddings[non_cs_mask]
        non_cs_categories = category_l1_tensor[non_cs_mask]
        category_mask = non_cs_categories.unsqueeze(0) != non_cs_categories.unsqueeze(1)
        diagonal_mask = ~torch.eye(len(non_cs_categories), dtype=torch.bool, device=category_mask.device)
        mask = category_mask & diagonal_mask
        normalized_embeddings = F.normalize(non_cs_embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        different_category_similarities = similarity_matrix[mask]
        if different_category_similarities.numel() > 0:
            penalties = torch.clamp(different_category_similarities - margin, min=0)
            return penalties.mean()
        else:
            return torch.tensor(0.0, device=negative_samples_embeddings.device)

    def _compute_train_val_scores(
        self,
        papers_embeddings: torch.Tensor,
        user_idx_tensor: torch.Tensor,
        sorted_unique_users_idx_tensor: torch.Tensor = None,
        batch_negatives_indices: torch.Tensor = None,
    ) -> tuple:
        users_embeddings = self.users_embeddings(user_idx_tensor)
        dot_products = (
            torch.sum(papers_embeddings * users_embeddings[:, :-1], dim=1) + users_embeddings[:, -1]
        )
        batch_negatives_scores = None
        if batch_negatives_indices is not None:
            batch_negatives_scores = self._compute_batch_negatives_scores(
                papers_embeddings=papers_embeddings,
                sorted_unique_users_idx_tensor=sorted_unique_users_idx_tensor,
                batch_negatives_indices=batch_negatives_indices,
            )
        return dot_products, batch_negatives_scores

    def _compute_batch_negatives_scores(
        self,
        papers_embeddings: torch.Tensor,
        sorted_unique_users_idx_tensor: torch.Tensor,
        batch_negatives_indices: torch.Tensor,
    ) -> torch.Tensor:
        if sorted_unique_users_idx_tensor is None:
            raise ValueError(
                "sorted_unique_users_idx_tensor must be provided when batch_negatives_indices is not None."
            )
        batch_negatives_papers = papers_embeddings[batch_negatives_indices]
        sorted_unique_users_embeddings = self.users_embeddings(sorted_unique_users_idx_tensor)
        batch_negatives_scores = torch.bmm(
            batch_negatives_papers,
            sorted_unique_users_embeddings[:, :-1].unsqueeze(-1),
        ).squeeze(-1)
        batch_negatives_scores = batch_negatives_scores + sorted_unique_users_embeddings[
            :, -1
        ].unsqueeze(1)
        return batch_negatives_scores

    def save_finetuning_model(self, model_path: Path) -> None:
        if not isinstance(model_path, Path):
            model_path = Path(model_path).resolve()
        os.makedirs(model_path, exist_ok=True)
        self.transformer_model.save_pretrained(model_path / "transformer_model")
        torch.save(self.projection.state_dict(), model_path / "projection.pt")
        torch.save(self.users_embeddings.state_dict(), model_path / "users_embeddings.pt")
        torch.save(
            self.categories_embeddings_l1.state_dict(),
            model_path / "categories_embeddings_l1.pt",
        )
        if self.categories_embeddings_l2 is not None:
            torch.save(
                self.categories_embeddings_l2.state_dict(),
                model_path / "categories_embeddings_l2.pt",
            )

    def count_transformer_layers(self) -> int:
        return len(self.transformer_model.encoder.layer)

    def count_transformer_parameters(self) -> tuple:
        n_params_total = 0
        n_params_unfreeze = 0
        for _, param in self.named_parameters():
            n_params_total += param.numel()
            if param.requires_grad:
                n_params_unfreeze += param.numel()
        return n_params_total, n_params_unfreeze

    def perform_inference(
        self,
        max_batch_size: int,
        input_ids_tensor: torch.Tensor,
        attention_mask_tensor: torch.Tensor,
        category_l1_tensor: torch.Tensor,
        category_l2_tensor: torch.Tensor = None,
        user_idx_tensor: torch.Tensor = None,
        eval_type: str = None,
    ) -> torch.Tensor:
        MAX_TRIES = 10
        results_total = []
        n_samples, n_samples_processed, batch_num = len(input_ids_tensor), 0, 1
        start_time = time.time()
        print("Performing Inference...")
        while n_samples_processed < n_samples:
            batch_start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            n_tries_so_far, batch_size, successfully_processed = (
                0,
                max_batch_size,
                False,
            )
            while n_tries_so_far < MAX_TRIES:
                try:
                    upper_bound = min(n_samples, n_samples_processed + batch_size)
                    batch_input_ids_tensor = input_ids_tensor[n_samples_processed:upper_bound]
                    batch_attention_mask_tensor = attention_mask_tensor[
                        n_samples_processed:upper_bound
                    ]
                    batch_category_l1_tensor = None
                    if category_l1_tensor is not None:
                        batch_category_l1_tensor = category_l1_tensor[
                            n_samples_processed:upper_bound
                        ]
                    batch_category_l2_tensor = None
                    if category_l2_tensor is not None:
                        batch_category_l2_tensor = category_l2_tensor[
                            n_samples_processed:upper_bound
                        ]
                    batch_user_idx_tensor = None
                    if user_idx_tensor is not None:
                        batch_user_idx_tensor = user_idx_tensor[n_samples_processed:upper_bound]
                    tensors_dict = {
                        "input_ids_tensor": batch_input_ids_tensor,
                        "attention_mask_tensor": batch_attention_mask_tensor,
                        "category_l1_tensor": batch_category_l1_tensor,
                        "category_l2_tensor": batch_category_l2_tensor,
                        "user_idx_tensor": batch_user_idx_tensor,
                    }
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        with torch.inference_mode():
                            results = self(
                                eval_type=eval_type,
                                tensors_dict=tensors_dict,
                            )
                            if isinstance(results, tuple) and len(results) == 2:
                                results = results[0]
                            results = results.cpu()
                            gpu_info = get_gpu_info()
                    results_total.append(results)
                    n_samples_processed = min(n_samples, n_samples_processed + batch_size)
                    if batch_num == 1 or batch_num % 10 == 0:
                        print(
                            f"Finished Batch {batch_num} in {time.time() - batch_start_time:.2f} Seconds: Samples {n_samples_processed} / {n_samples}. {gpu_info}"
                        )
                    batch_num += 1
                    successfully_processed = True
                except Exception as e:
                    print(f"Error in encoding Batch {batch_num}: \n{e}")
                    n_tries_so_far += 1
                    batch_size = batch_size // 2
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                if successfully_processed:
                    break
            if n_tries_so_far >= MAX_TRIES:
                os._exit(0)
        results_total = torch.cat(results_total, dim=0)
        print(
            f"Finished all Batches in {time.time() - start_time:.2f} seconds. Total Samples: {n_samples_processed}.\n"
        )
        return results_total

    def compute_papers_embeddings(
        self,
        input_ids_tensor: torch.Tensor,
        attention_mask_tensor: torch.Tensor,
        category_l1_tensor: torch.Tensor,
        category_l2_tensor: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.perform_inference(
            max_batch_size=EVAL_BATCH_SIZE,
            input_ids_tensor=input_ids_tensor,
            attention_mask_tensor=attention_mask_tensor,
            category_l1_tensor=category_l1_tensor,
            category_l2_tensor=category_l2_tensor,
            eval_type="test",
        )

    def compute_val_dataset_scores(
        self,
        input_ids_tensor: torch.Tensor,
        attention_mask_tensor: torch.Tensor,
        category_l1_tensor: torch.Tensor,
        category_l2_tensor: torch.Tensor = None,
        user_idx_tensor: torch.Tensor = None,
    ) -> torch.Tensor:
        assert user_idx_tensor is not None
        return self.perform_inference(
            max_batch_size=EVAL_BATCH_SIZE,
            input_ids_tensor=input_ids_tensor,
            attention_mask_tensor=attention_mask_tensor,
            category_l1_tensor=category_l1_tensor,
            category_l2_tensor=category_l2_tensor,
            user_idx_tensor=user_idx_tensor,
            eval_type="val",
        )

    def compute_val_negative_samples_scores(
        self,
        input_ids_tensor: torch.Tensor,
        attention_mask_tensor: torch.Tensor,
        category_l1_tensor: torch.Tensor,
        category_l2_tensor: torch.Tensor = None,
    ) -> torch.Tensor:
        tensors_dict = {
            "input_ids_tensor": input_ids_tensor,
            "attention_mask_tensor": attention_mask_tensor,
            "category_l1_tensor": category_l1_tensor,
            "category_l2_tensor": category_l2_tensor,
            "sorted_unique_user_idx_tensor": self.val_users_embeddings_idxs,
        }
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            with torch.inference_mode():
                return self(
                    eval_type="negative_samples",
                    tensors_dict=tensors_dict,
                )[0].cpu()

    def get_memory_footprint(self) -> dict:
        memory_info = {}
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        memory_info["total_parameters"] = total_params
        memory_info["trainable_parameters"] = trainable_params
        memory_info["trainable_percentage"] = trainable_params / total_params
        if torch.cuda.is_available():
            memory_info["gpu_memory_allocated (GB)"] = torch.cuda.memory_allocated() / 1024**3
            memory_info["gpu_memory_reserved (GB)"] = torch.cuda.memory_reserved() / 1024**3
        return memory_info


def load_transformer_model(transformer_path: str, device: torch.device) -> AutoModel:
    return AutoModel.from_pretrained(
        transformer_path, trust_remote_code=True, unpad_inputs=True, torch_dtype="auto"
    ).to(device)


def save_transformer_model(transformer_model: AutoModel, model_path: Path) -> None:
    transformer_model.save_pretrained(model_path)


def get_transformer_embeddings_dim(transformer_model: AutoModel) -> int:
    return transformer_model.config.hidden_size


def load_unfreeze_parameters(unfreeze_parameters_dict: dict, mode: str) -> tuple:
    if unfreeze_parameters_dict is None:
        unfreeze_parameters_dict = {}
    n_unfreeze_layers = unfreeze_parameters_dict.get("n_unfreeze_layers", None)
    unfreeze_word_embeddings = unfreeze_parameters_dict.get("unfreeze_word_embeddings", False)
    unfreeze_from_bottom = unfreeze_parameters_dict.get("unfreeze_from_bottom", False)
    if mode == "eval":
        if n_unfreeze_layers is None:
            n_unfreeze_layers = 0
        assert n_unfreeze_layers == 0
    return n_unfreeze_layers, unfreeze_word_embeddings, unfreeze_from_bottom


def load_projection(
    tensors_parameters_dict: dict,
    device: torch.device,
    projection_path: Path = None,
    transformer_embeddings_dim: int = None,
) -> nn.Linear:
    projection_pretrained = tensors_parameters_dict.get("projection_pretrained", True)
    projection_dim = tensors_parameters_dict.get("projection_dim", None)
    projection_state_dict = None

    if projection_pretrained:
        if not projection_path.exists():
            raise FileNotFoundError(f"Projection state dict not found at {projection_path}.")
        projection_state_dict = torch.load(
            projection_path,
            map_location=device,
            weights_only=True,
        )
        projection_dim, transformer_embeddings_dim = projection_state_dict["weight"].shape
    else:
        if transformer_embeddings_dim is None:
            raise ValueError(
                "transformer_embeddings_dim must be provided if pretrained_projection is False."
            )
        if projection_dim is None:
            raise ValueError("projection_dim must be provided if pretrained_projection is False.")

    projection = nn.Linear(
        in_features=transformer_embeddings_dim,
        out_features=projection_dim,
    )
    if projection_state_dict is not None:
        projection.load_state_dict(projection_state_dict)
    projection = projection.to(device, dtype=FINETUNING_DTYPE)
    return projection


def load_embeddings(
    embeddings_type: str,
    tensors_parameters_dict: dict,
    device: torch.device,
    embeddings_path: Path = None,
) -> nn.Embedding:
    valid_embeddings_types = [
        "users_embeddings",
        "categories_embeddings_l1",
        "categories_embeddings_l2",
    ]
    if embeddings_type not in valid_embeddings_types:
        raise ValueError(
            f"embeddings_type must be one of {valid_embeddings_types}, but got {embeddings_type}."
        )
    pretrained_embeddings = tensors_parameters_dict.get(f"{embeddings_type}_pretrained", True)
    embeddings_dim = tensors_parameters_dict.get(f"{embeddings_type}_dim", None)
    embeddings_state_dict = None

    if pretrained_embeddings:
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings state dict not found at {embeddings_path}.")
        embeddings_state_dict = torch.load(
            embeddings_path,
            map_location=device,
            weights_only=True,
        )
        num_embeddings, embeddings_dim = embeddings_state_dict["weight"].shape
    else:
        if embeddings_dim is None:
            raise ValueError("embeddings_dim must be provided if pretrained_embeddings is False.")
        if embeddings_type == "users_embeddings":
            num_embeddings = len(load_users_coefs_ids_to_idxs())
        elif embeddings_type == "categories_embeddings_l1":
            num_embeddings = len(load_categories_to_idxs("l1"))
        elif embeddings_type == "categories_embeddings_l2":
            num_embeddings = len(load_categories_to_idxs("l2"))

    embeddings = nn.Embedding(num_embeddings, embeddings_dim)
    if embeddings_state_dict is not None:
        embeddings.load_state_dict(embeddings_state_dict)
    embeddings = embeddings.to(device, dtype=FINETUNING_DTYPE)
    if embeddings_type == "categories_embeddings_l2" and embeddings_state_dict is None:
        nn.init.normal_(embeddings.weight, std=0.01)
    return embeddings


def fix_gte_config(model_path: Path) -> None:
    config_path = model_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    if "auto_map" in config:
        for key in config["auto_map"]:
            if key == "AutoConfig" and not config["auto_map"][key].startswith("Alibaba-NLP"):
                config["auto_map"][key] = "Alibaba-NLP/new-impl--configuration.NewConfig"
            elif not config["auto_map"][key].startswith("Alibaba-NLP"):
                model_class = config["auto_map"][key].split(".")[-1]
                config["auto_map"][key] = f"Alibaba-NLP/new-impl--modeling.{model_class}"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def load_finetuning_model(
    finetuning_model_path: Path,
    device: torch.device,
    mode: str,
    unfreeze_parameters_dict: dict = {},
    tensors_parameters_dict: dict = {},
    include_categories_embeddings_l2: bool = False,
    val_users_embeddings_idxs: torch.Tensor = None,
) -> FinetuningModel:
    valid_modes = ["train", "eval"]
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}, but got {mode}.")
    if not isinstance(finetuning_model_path, Path):
        finetuning_model_path = Path(finetuning_model_path).resolve()
    n_unfreeze_layers, unfreeze_word_embeddings, unfreeze_from_bottom = load_unfreeze_parameters(
        unfreeze_parameters_dict, mode
    )
    transformer_model_path = finetuning_model_path / "transformer_model"
    fix_gte_config(transformer_model_path)
    transformer_model = load_transformer_model(transformer_model_path, device)
    transformer_embeddings_dim = get_transformer_embeddings_dim(transformer_model)

    projection = load_projection(
        tensors_parameters_dict=tensors_parameters_dict,
        device=device,
        projection_path=finetuning_model_path / "projection.pt",
    )
    categories_embeddings_l1 = load_embeddings(
        embeddings_type="categories_embeddings_l1",
        tensors_parameters_dict=tensors_parameters_dict,
        device=device,
        embeddings_path=finetuning_model_path / "categories_embeddings_l1.pt",
    )
    categories_embeddings_l1_dim = categories_embeddings_l1.embedding_dim
    tensors_parameters_dict.update(
        {"users_embeddings_dim": transformer_embeddings_dim + categories_embeddings_l1_dim + 1}
    )
    users_embeddings = load_embeddings(
        embeddings_type="users_embeddings",
        tensors_parameters_dict=tensors_parameters_dict,
        device=device,
        embeddings_path=finetuning_model_path / "users_embeddings.pt",
    )
    categories_embeddings_l2 = None
    if include_categories_embeddings_l2:
        tensors_parameters_dict.update(
            {"categories_embeddings_l2_dim": categories_embeddings_l1_dim}
        )
        categories_embeddings_l2 = load_embeddings(
            embeddings_type="categories_embeddings_l2",
            tensors_parameters_dict=tensors_parameters_dict,
            device=device,
            embeddings_path=finetuning_model_path / "categories_embeddings_l2.pt",
        )

    return FinetuningModel(
        transformer_model=transformer_model,
        projection=projection,
        users_embeddings=users_embeddings,
        categories_embeddings_l1=categories_embeddings_l1,
        categories_embeddings_l2=categories_embeddings_l2,
        val_users_embeddings_idxs=val_users_embeddings_idxs,
        n_unfreeze_layers=n_unfreeze_layers,
        unfreeze_word_embeddings=unfreeze_word_embeddings,
        unfreeze_from_bottom=unfreeze_from_bottom,
    )
