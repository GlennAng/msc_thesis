import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from ....finetuning.src.finetuning_preprocessing import (
    load_categories_to_idxs,
    save_categories_embeddings_tensor,
    save_projection_tensor,
    save_transformer_model,
)
from ....logreg.src.embeddings.compute_embeddings import get_gpu_info
from ....src.project_paths import ProjectPaths


def save_papers_embeddings_as_numpy(
    path: Path, embeddings: torch.Tensor, papers_ids_to_idxs: dict
) -> None:
    if not isinstance(path, Path):
        path = Path(path).resolve()
    np.save(f"{path/ 'abs_X.npy'}", embeddings)
    with open(f"{path / 'abs_paper_ids_to_idx.pkl'}", "wb") as f:
        pickle.dump(papers_ids_to_idxs, f)


class PapersEncoder(nn.Module):
    def __init__(
        self,
        transformer_model: AutoModel,
        projection: nn.Linear,
        categories_embeddings_l1: nn.Embedding,
        l1_scale: float = 1.0,
        categories_embeddings_l2: nn.Embedding = None,
        l2_scale: float = 1.0,
        unfreeze_l1_scale: bool = False,
        unfreeze_l2_scale: bool = False,
        n_unfreeze_layers: int = 0,
        unfreeze_word_embeddings: bool = False,
        unfreeze_from_bottom: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_model = transformer_model
        self.projection = projection
        self.categories_embeddings_l1 = categories_embeddings_l1
        self.l1_scale = nn.Parameter(torch.tensor(l1_scale), requires_grad=unfreeze_l1_scale)

        if categories_embeddings_l2 is not None:
            self.uses_categories_embeddings_l2 = True
            self.categories_embeddings_l2 = categories_embeddings_l2
            assert l2_scale is not None and isinstance(l2_scale, float)
            self.l2_scale = nn.Parameter(torch.tensor(l2_scale), requires_grad=unfreeze_l2_scale)
        else:
            self.uses_categories_embeddings_l2 = False

        self._extract_dims()
        self._unfreeze_transformer_model_layers(
            n_unfreeze_layers=n_unfreeze_layers,
            unfreeze_word_embeddings=unfreeze_word_embeddings,
            unfreeze_from_bottom=unfreeze_from_bottom,
        )

    def _extract_dims(self) -> None:
        self.text_dim = self.projection.out_features
        self.categories_dim = self.categories_embeddings_l1.embedding_dim
        if self.uses_categories_embeddings_l2:
            assert self.categories_dim == self.categories_embeddings_l2.embedding_dim
        self.embedding_dim = self.text_dim + self.categories_dim

    def _unfreeze_transformer_model_layers(
        self,
        n_unfreeze_layers: int,
        unfreeze_word_embeddings: bool = False,
        unfreeze_from_bottom: bool = False,
    ) -> None:
        for param in self.transformer_model.parameters():
            param.requires_grad = False
        if unfreeze_word_embeddings:
            for param in self.transformer_model.embeddings.parameters():
                param.requires_grad = True
        if n_unfreeze_layers == 0:
            return
        transformer_layers = self.transformer_model.encoder.layer
        n_transformer_layers = len(transformer_layers)
        assert n_unfreeze_layers >= 0 and n_unfreeze_layers <= n_transformer_layers
        if unfreeze_from_bottom:
            for i in range(n_unfreeze_layers):
                for param in transformer_layers[i].parameters():
                    param.requires_grad = True
        else:
            for i in range(len(transformer_layers) - n_unfreeze_layers, len(transformer_layers)):
                for param in transformer_layers[i].parameters():
                    param.requires_grad = True

    def to_device(self, device: torch.device) -> None:
        self.to(device)
        assert self.get_device() == device

    def get_device(self) -> torch.device:
        device = next(self.transformer_model.parameters()).device
        components_list = [
            self.projection.weight,
            self.categories_embeddings_l1.weight,
            self.l1_scale,
        ]
        if self.uses_categories_embeddings_l2:
            components_list.extend([self.categories_embeddings_l2.weight, self.l2_scale])
        for component in components_list:
            assert component.device == device
        return device

    def forward(
        self,
        input_ids_tensor: torch.Tensor,
        attention_mask_tensor: torch.Tensor,
        category_l1_tensor: torch.Tensor,
        category_l2_tensor: torch.Tensor,
        normalize_embeddings: bool,
    ) -> torch.Tensor:
        self._verify_batch(
            input_ids_tensor=input_ids_tensor,
            attention_mask_tensor=attention_mask_tensor,
            category_l1_tensor=category_l1_tensor,
            category_l2_tensor=category_l2_tensor,
        )
        papers_embeddings = self.transformer_model(
            input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
        ).last_hidden_state[:, 0, :]
        papers_embeddings = self.projection(papers_embeddings)
        if normalize_embeddings:
            papers_embeddings = F.normalize(papers_embeddings, p=2, dim=1)
        categories_embeddings = self.l1_scale * self.categories_embeddings_l1(category_l1_tensor)
        if self.uses_categories_embeddings_l2:
            categories_embeddings_l2 = self.l2_scale * self.categories_embeddings_l2(
                category_l2_tensor
            )
            categories_embeddings = categories_embeddings + categories_embeddings_l2
        papers_embeddings = torch.cat((papers_embeddings, categories_embeddings), dim=1)
        return papers_embeddings

    def _verify_batch(
        self,
        input_ids_tensor: torch.Tensor,
        attention_mask_tensor: torch.Tensor,
        category_l1_tensor: torch.Tensor,
        category_l2_tensor: torch.Tensor,
    ) -> None:
        device = self.get_device()
        tensors_list = [
            input_ids_tensor,
            attention_mask_tensor,
            category_l1_tensor,
        ]
        if self.uses_categories_embeddings_l2:
            tensors_list.append(category_l2_tensor)
        for tensor in tensors_list:
            assert isinstance(tensor, torch.Tensor) and tensor.device == device

    def save_model(self, path: Path) -> None:
        if not isinstance(path, Path):
            path = Path(path).resolve()
        os.makedirs(path.parent, exist_ok=True)
        self.transformer_model.save_pretrained(path / "transformer_model")
        non_transformer_state = {
            k: v for k, v in self.state_dict().items() if not k.startswith("transformer_model.")
        }
        torch.save(non_transformer_state, path / "components.pt")

    def compute_papers_embeddings(self, dataloader: DataLoader) -> tuple:
        device = self.get_device()
        in_training = self.training
        self.eval()
        all_embeddings = []
        papers_ids_to_idxs = {}
        current_idx = 0
        pbar = None
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                category_l1 = batch["l1"].to(device)
                category_l2 = None
                if self.uses_categories_embeddings_l2:
                    category_l2 = batch["l2"].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    with torch.inference_mode():
                        embeddings = self(
                            input_ids_tensor=input_ids,
                            attention_mask_tensor=attention_mask,
                            category_l1_tensor=category_l1,
                            category_l2_tensor=category_l2,
                            normalize_embeddings=True,
                        )
                if batch_idx == 0:
                    print(f"First Batch: {get_gpu_info()}")
                    pbar = tqdm(total=len(dataloader), desc="Computing paper embeddings")
                pbar.update(1)
                all_embeddings.append(embeddings.cpu())
                for paper_id in batch["paper_id"]:
                    papers_ids_to_idxs[paper_id.item()] = current_idx
                    current_idx += 1
        if pbar is not None:
            pbar.close()
        all_embeddings = torch.cat(all_embeddings, dim=0)
        if in_training:
            self.train()
        return all_embeddings, papers_ids_to_idxs


def _verify_papers_encoder(papers_encoder: PapersEncoder, embeddings_path: Path) -> None:
    assert isinstance(papers_encoder, PapersEncoder)
    device = papers_encoder.get_device()
    assert embeddings_path.exists()

    N_PAPERS = 1000
    from ....finetuning.src.finetuning_preprocessing import (
        load_finetuning_papers_tokenized,
    )
    from ....logreg.src.embeddings.embedding import Embedding

    papers_tokenized = load_finetuning_papers_tokenized("eval_test_users")
    papers_ids = papers_tokenized["paper_id"][:N_PAPERS].tolist()
    embedding = Embedding(embeddings_path)
    papers_embeddings = embedding.matrix[embedding.get_idxs(papers_ids)]
    papers_embeddings = torch.tensor(papers_embeddings, device=device)

    input_ids_tensor = papers_tokenized["input_ids"][:N_PAPERS].to(device)
    attention_mask_tensor = papers_tokenized["attention_mask"][:N_PAPERS].to(device)
    category_l1_tensor = papers_tokenized["l1"][:N_PAPERS].to(device)
    category_l2_tensor = papers_tokenized["l2"][:N_PAPERS].to(device)

    papers_encoder.eval()
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        with torch.inference_mode():
            papers_encoded = papers_encoder(
                input_ids_tensor=input_ids_tensor,
                attention_mask_tensor=attention_mask_tensor,
                category_l1_tensor=category_l1_tensor,
                category_l2_tensor=category_l2_tensor,
                normalize_embeddings=True,
            )
    assert papers_encoded.shape == papers_embeddings.shape
    assert torch.allclose(papers_encoded, papers_embeddings, atol=1e-2)


def verify_papers_encoder(model_path: Path, embeddings_path: Path) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not embeddings_path.exists():
        print(f"Cannot verify Papers Encoder: embeddings path not found: {embeddings_path}")
        return
    papers_encoder = load_papers_encoder(
        path=model_path,
        overwrite_l1_scale=None,
        overwrite_use_l2_embeddings=None,
        overwrite_l2_scale=None,
        l2_init_seed=None,
        unfreeze_l1_scale=False,
        unfreeze_l2_scale=False,
        n_unfreeze_layers=0,
        unfreeze_word_embeddings=False,
        unfreeze_from_bottom=False,
        device=device,
    )
    _verify_papers_encoder(papers_encoder, embeddings_path)
    print(f"Papers Encoder verification passed for model at: {model_path}")


def load_categories_l2(
    components_dict: dict,
    overwrite_use_l2_embeddings: bool,
    overwrite_l2_scale: float,
    embedding_dim: int = None,
    seed: int = None,
) -> tuple:
    if overwrite_use_l2_embeddings is not None and not overwrite_use_l2_embeddings:
        return None, None

    categories_embeddings_l2, l2_scale = None, None
    l2_weight_key = "categories_embeddings_l2.weight"
    if l2_weight_key in components_dict:
        categories_embeddings_l2 = load_embeddings_from_weight(
            embedding_weight=components_dict[l2_weight_key]
        )
    else:
        if overwrite_use_l2_embeddings is not None and overwrite_use_l2_embeddings:
            assert embedding_dim is not None and isinstance(embedding_dim, int)
            assert seed is not None and isinstance(seed, int)
            categories_to_idxs = load_categories_to_idxs(level="l2")
            categories_embeddings_l2 = nn.Embedding(
                num_embeddings=len(categories_to_idxs), embedding_dim=embedding_dim
            )
            torch.manual_seed(seed)
            nn.init.normal_(categories_embeddings_l2.weight, std=0.01)
    if categories_embeddings_l2 is not None:
        if overwrite_l2_scale is not None:
            l2_scale = overwrite_l2_scale
        else:
            if "l2_scale" in components_dict:
                l2_scale = components_dict["l2_scale"].item()
            else:
                l2_scale = 1.0
    return categories_embeddings_l2, l2_scale


def load_categories_l2_initial(path: Path) -> tuple:
    if not isinstance(path, Path):
        path = Path(path).resolve()
    if path.exists():
        categories_embeddings_l2 = load_embeddings(path=path)
        l2_scale = 1.0
    else:
        categories_embeddings_l2, l2_scale = None, None
    return categories_embeddings_l2, l2_scale


def load_transformer_model(path: Path) -> AutoModel:
    if not isinstance(path, Path):
        path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Transformer model directory not found: {path}")
    transformer_model = AutoModel.from_pretrained(
        path, trust_remote_code=True, unpad_inputs=True, torch_dtype="auto"
    )
    return transformer_model


def load_projection_from_weight_bias(
    projection_weight: torch.Tensor, projection_bias: torch.Tensor
) -> nn.Linear:
    output_dim, input_dim = projection_weight.shape
    projection = nn.Linear(input_dim, output_dim, bias=True)
    projection.load_state_dict({"weight": projection_weight, "bias": projection_bias})
    return projection


def load_projection(path: Path) -> nn.Linear:
    if not isinstance(path, Path):
        path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Projection file not found: {path}")
    projection_dict = torch.load(path, weights_only=True)
    projection_weight, projection_bias = projection_dict["weight"], projection_dict["bias"]
    return load_projection_from_weight_bias(
        projection_weight=projection_weight, projection_bias=projection_bias
    )


def load_embeddings_from_weight(embedding_weight: torch.Tensor) -> nn.Embedding:
    num_embeddings, embedding_dim = embedding_weight.shape
    embedding = nn.Embedding(num_embeddings, embedding_dim)
    embedding.load_state_dict({"weight": embedding_weight})
    return embedding


def load_embeddings(path: Path) -> nn.Embedding:
    if not isinstance(path, Path):
        path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    embedding_dict = torch.load(path, weights_only=True)
    embedding_weight = embedding_dict["weight"]
    return load_embeddings_from_weight(embedding_weight=embedding_weight)


def save_papers_encoder_from_components(path: Path) -> None:
    if not isinstance(path, Path):
        path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Paper Encoder directory not found: {path}")
    components_path = path / "components.pt"
    if components_path.exists():
        print(f"Papers Encoder components already exist at: {components_path}")
        return

    transformer_model = load_transformer_model(path=(path / "transformer_model"))
    projection = load_projection(path=(path / "projection.pt"))
    categories_embeddings_l1 = load_embeddings(path=(path / "categories_embeddings_l1.pt"))
    l1_scale = 1.0
    categories_embeddings_l2, l2_scale = load_categories_l2_initial(
        path=(path / "categories_embeddings_l2.pt")
    )
    papers_encoder = PapersEncoder(
        transformer_model=transformer_model,
        projection=projection,
        categories_embeddings_l1=categories_embeddings_l1,
        l1_scale=l1_scale,
        categories_embeddings_l2=categories_embeddings_l2,
        l2_scale=l2_scale,
        unfreeze_l1_scale=False,
        unfreeze_l2_scale=False,
        n_unfreeze_layers=0,
        unfreeze_word_embeddings=False,
        unfreeze_from_bottom=False,
    )
    papers_encoder.save_model(path=path)
    print(f"Saved Papers Encoder components at: {components_path}")


def save_papers_encoder_components() -> None:
    path = ProjectPaths.sequence_data_model_state_dicts_papers_encoder_path()
    os.makedirs(path, exist_ok=True)
    save_transformer_model(model_path=(path / "transformer_model"))
    save_projection_tensor(projection_tensor_path=(path / "projection.pt"))
    save_categories_embeddings_tensor(
        categories_embeddings_l1_path=(path / "categories_embeddings_l1.pt"),
        categories_to_idxs_l1_path=(path / "categories_to_idxs_l1.pkl"),
        categories_to_idxs_l2_path=(path / "categories_to_idxs_l2.pkl"),
    )


def load_papers_encoder(
    path: Path = ProjectPaths.sequence_data_model_state_dicts_papers_encoder_path(),
    overwrite_l1_scale: float = None,
    overwrite_use_l2_embeddings: bool = None,
    overwrite_l2_scale: float = None,
    l2_init_seed: int = None,
    unfreeze_l1_scale: bool = False,
    unfreeze_l2_scale: bool = False,
    n_unfreeze_layers: int = 0,
    unfreeze_word_embeddings: bool = False,
    unfreeze_from_bottom: bool = False,
    device: torch.device = None,
) -> PapersEncoder:
    if not isinstance(path, Path):
        path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Paper Encoder directory not found: {path}")

    transformer_model = load_transformer_model(path=(path / "transformer_model"))
    components_path = path / "components.pt"
    if not components_path.exists():
        raise FileNotFoundError(f"Components file not found: {components_path}")
    components_dict = torch.load(components_path, weights_only=True)

    projection = load_projection_from_weight_bias(
        projection_weight=components_dict["projection.weight"],
        projection_bias=components_dict["projection.bias"],
    )
    categories_embeddings_l1 = load_embeddings_from_weight(
        embedding_weight=components_dict["categories_embeddings_l1.weight"]
    )
    l1_scale = components_dict["l1_scale"].item()
    l1_scale = overwrite_l1_scale if overwrite_l1_scale is not None else l1_scale
    categories_embeddings_l2, l2_scale = load_categories_l2(
        components_dict=components_dict,
        overwrite_use_l2_embeddings=overwrite_use_l2_embeddings,
        overwrite_l2_scale=overwrite_l2_scale,
        embedding_dim=categories_embeddings_l1.embedding_dim,
        seed=l2_init_seed,
    )
    papers_encoder = PapersEncoder(
        transformer_model=transformer_model,
        projection=projection,
        categories_embeddings_l1=categories_embeddings_l1,
        l1_scale=l1_scale,
        categories_embeddings_l2=categories_embeddings_l2,
        l2_scale=l2_scale,
        unfreeze_l1_scale=unfreeze_l1_scale,
        unfreeze_l2_scale=unfreeze_l2_scale,
        n_unfreeze_layers=n_unfreeze_layers,
        unfreeze_word_embeddings=unfreeze_word_embeddings,
        unfreeze_from_bottom=unfreeze_from_bottom,
    )
    if device is not None:
        papers_encoder.to_device(device)
    return papers_encoder


if __name__ == "__main__":
    save_papers_encoder_components()

    path = ProjectPaths.sequence_data_model_state_dicts_papers_encoder_path()
    save_papers_encoder_from_components(path=path)
    embeddings_path = ProjectPaths.finetuning_data_model_path() / "embeddings"
    if embeddings_path.exists():
        verify_papers_encoder(model_path=path, embeddings_path=embeddings_path)

    finetuning_path = ProjectPaths.sequence_data_model_state_dicts_papers_encoder_finetuned_path()
    if finetuning_path.exists():
        save_papers_encoder_from_components(path=finetuning_path)
        model_path = finetuning_path
        embeddings_path = finetuning_path.parent / "all_embeddings"
        verify_papers_encoder(model_path=model_path, embeddings_path=embeddings_path)
