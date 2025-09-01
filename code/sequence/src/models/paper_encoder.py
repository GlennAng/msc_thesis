import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from ....finetuning.src.finetuning_model import load_transformer_model
from ....finetuning.src.finetuning_preprocessing import load_categories_to_idxs
from ....src.project_paths import ProjectPaths


class PaperEncoder(nn.Module):
    def __init__(
        self,
        transformer_model: AutoModel,
        projection: nn.Linear,
        categories_embeddings_l1: nn.Embedding,
        categories_embeddings_l2: nn.Embedding,
        l1_scale: float = 1.0,
        l2_scale: float = 1.0,
        n_unfreeze_layers: int = 0,
        unfreeze_word_embeddings: bool = False,
        unfreeze_from_bottom: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_model = transformer_model
        self.device = next(transformer_model.parameters()).device
        self.projection = projection
        self.categories_embeddings_l1 = categories_embeddings_l1
        self.categories_embeddings_l2 = categories_embeddings_l2
        self.l1_scale = nn.Parameter(torch.tensor(l1_scale, device=self.device))
        self.l2_scale = nn.Parameter(torch.tensor(l2_scale, device=self.device))

        self._validate_and_setup_components()
        self._extract_dims()
        self._unfreeze_transformer_model_layers(
            n_unfreeze_layers=n_unfreeze_layers,
            unfreeze_word_embeddings=unfreeze_word_embeddings,
            unfreeze_from_bottom=unfreeze_from_bottom,
        )

    def _validate_and_setup_components(self) -> None:
        components = [
            self.projection.weight,
            self.categories_embeddings_l1.weight,
            self.categories_embeddings_l2.weight,
            self.l1_scale,
            self.l2_scale,
        ]
        all_same_device = all(component.device == self.device for component in components)
        if not all_same_device:
            raise ValueError("All components must be on the same device as the transformer model.")

    def _extract_dims(self) -> None:
        self.text_dim = self.projection.out_features
        self.categories_dim = self.categories_embeddings_l1.embedding_dim
        assert self.categories_dim == self.categories_embeddings_l2.embedding_dim
        self.embedding_dim = self.text_dim + self.categories_dim

    def forward(
        self,
        input_ids_tensor: torch.Tensor,
        attention_mask_tensor: torch.Tensor,
        category_l1_tensor: torch.Tensor,
        category_l2_tensor: torch.Tensor,
        normalize_embeddings: bool,
    ) -> torch.Tensor:
        papers_embeddings = self.transformer_model(
            input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
        ).last_hidden_state[:, 0, :]
        papers_embeddings = self.projection(papers_embeddings)
        if normalize_embeddings:
            papers_embeddings = F.normalize(papers_embeddings, p=2, dim=1)
        categories_embeddings = self.l1_scale * self.categories_embeddings_l1(category_l1_tensor)
        categories_embeddings_l2 = self.l2_scale * self.categories_embeddings_l2(category_l2_tensor)
        categories_embeddings = categories_embeddings + categories_embeddings_l2
        papers_embeddings = torch.cat((papers_embeddings, categories_embeddings), dim=1)
        return papers_embeddings

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

    def save_model(self, path: Path = None) -> None:
        if path is None:
            path = ProjectPaths.sequence_data_model_state_dicts_paper_encoder_path()
        if not isinstance(path, Path):
            path = Path(path).resolve()
        os.makedirs(path.parent, exist_ok=True)
        self.transformer_model.save_pretrained(path / "transformer_model")
        non_transformer_state = {
            k: v for k, v in self.state_dict().items() if not k.startswith("transformer_model.")
        }
        torch.save(non_transformer_state, path / "components.pt")


def load_projection(
    projection_weight: torch.Tensor, projection_bias: torch.Tensor, device: torch.device = None
) -> nn.Linear:
    output_dim, input_dim = projection_weight.shape
    projection = nn.Linear(input_dim, output_dim, bias=True)
    projection.load_state_dict({"weight": projection_weight, "bias": projection_bias})
    if device is not None:
        projection = projection.to(device)
    return projection


def load_embeddings(embedding_weight: torch.Tensor, device: torch.device = None) -> nn.Embedding:
    num_embeddings, embedding_dim = embedding_weight.shape
    embedding = nn.Embedding(num_embeddings, embedding_dim)
    embedding.load_state_dict({"weight": embedding_weight})
    if device is not None:
        embedding = embedding.to(device)
    return embedding


def initialize_categories_embeddings_l2(embedding_dim: int, seed: int = None) -> nn.Embedding:
    categories_to_idxs = load_categories_to_idxs(level="l2")
    embedding = nn.Embedding(num_embeddings=len(categories_to_idxs), embedding_dim=embedding_dim)
    if seed is not None:
        torch.manual_seed(seed)
        nn.init.normal_(embedding.weight, std=0.01)
    return embedding


def load_paper_encoder_trained(
    device: torch.device,
    path: Path,
    n_unfreeze_layers: int = 0,
    unfreeze_word_embeddings: bool = False,
    unfreeze_from_bottom: bool = False,
) -> PaperEncoder:
    if not isinstance(path, Path):
        path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Paper Encoder directory not found: {path}")

    transformer_path = path / "transformer_model"
    if not transformer_path.exists():
        raise FileNotFoundError(f"Transformer model directory not found: {transformer_path}")
    transformer_model = load_transformer_model(transformer_path=transformer_path, device=device)

    components_path = path / "components.pt"
    if not components_path.exists():
        raise FileNotFoundError(f"Components file not found: {components_path}")
    components_dict = torch.load(components_path, map_location=device, weights_only=True)
    projection = load_projection(
        projection_weight=components_dict["projection.weight"],
        projection_bias=components_dict["projection.bias"],
        device=device,
    )
    categories_embeddings_l1 = load_embeddings(
        embedding_weight=components_dict["categories_embeddings_l1.weight"], device=device
    )
    categories_embeddings_l2 = load_embeddings(
        embedding_weight=components_dict["categories_embeddings_l2.weight"], device=device
    )
    l1_scale = components_dict["l1_scale"].item()
    l2_scale = components_dict["l2_scale"].item()

    return PaperEncoder(
        transformer_model=transformer_model,
        projection=projection,
        categories_embeddings_l1=categories_embeddings_l1,
        categories_embeddings_l2=categories_embeddings_l2,
        l1_scale=l1_scale,
        l2_scale=l2_scale,
        n_unfreeze_layers=n_unfreeze_layers,
        unfreeze_word_embeddings=unfreeze_word_embeddings,
        unfreeze_from_bottom=unfreeze_from_bottom,
    )


def load_paper_encoder_initial(
    device: torch.device,
    path: Path = None,
    seed: int = None,
    l1_scale: float = 1.0,
    l2_scale: float = 1.0,
    n_unfreeze_layers: int = 0,
    unfreeze_word_embeddings: bool = False,
    unfreeze_from_bottom: bool = False,
) -> PaperEncoder:
    if path is None:
        path = ProjectPaths.sequence_data_model_state_dicts_paper_encoder_path()
    if not isinstance(path, Path):
        path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Paper Encoder directory not found: {path}")

    transformer_path = path / "transformer_model"
    if not transformer_path.exists():
        raise FileNotFoundError(f"Transformer model directory not found: {transformer_path}")
    transformer_model = load_transformer_model(transformer_path=transformer_path, device=device)

    projection_path = path / "projection.pt"
    if not projection_path.exists():
        raise FileNotFoundError(f"Projection file not found: {projection_path}")
    projection_dict = torch.load(projection_path, map_location=device, weights_only=True)
    projection = load_projection(
        projection_weight=projection_dict["weight"],
        projection_bias=projection_dict["bias"],
        device=device,
    )

    categories_embeddings_l1_path = path / "categories_embeddings_l1.pt"
    if not categories_embeddings_l1_path.exists():
        raise FileNotFoundError(
            f"Categories embeddings L1 file not found: {categories_embeddings_l1_path}"
        )
    categories_l1_dict = torch.load(
        categories_embeddings_l1_path, map_location=device, weights_only=True
    )
    categories_embeddings_l1 = load_embeddings(
        embedding_weight=categories_l1_dict["weight"], device=device
    )
    categories_embeddings_l2 = initialize_categories_embeddings_l2(
        embedding_dim=categories_embeddings_l1.embedding_dim, seed=seed
    ).to(device)

    return PaperEncoder(
        transformer_model=transformer_model,
        projection=projection,
        categories_embeddings_l1=categories_embeddings_l1,
        categories_embeddings_l2=categories_embeddings_l2,
        l1_scale=l1_scale,
        l2_scale=l2_scale,
        n_unfreeze_layers=n_unfreeze_layers,
        unfreeze_word_embeddings=unfreeze_word_embeddings,
        unfreeze_from_bottom=unfreeze_from_bottom,
    )


def _verify_paper_encoder(paper_encoder: PaperEncoder, embeddings_path: Path) -> None:
    if not isinstance(paper_encoder, PaperEncoder):
        return False
    if not embeddings_path.exists():
        return False

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

    paper_encoder.eval()
    with torch.autocast(device_type=paper_encoder.device.type, dtype=torch.float16):
        with torch.inference_mode():
            papers_encoded = paper_encoder(
                input_ids_tensor=input_ids_tensor,
                attention_mask_tensor=attention_mask_tensor,
                category_l1_tensor=category_l1_tensor,
                category_l2_tensor=category_l2_tensor,
                normalize_embeddings=True,
            )
    assert papers_encoded.shape == papers_embeddings.shape
    assert torch.allclose(papers_encoded, papers_embeddings, atol=1e-2)


def verify_paper_encoder(model_path: Path, device: torch.device) -> None:
    state_path = model_path / "state_dicts"
    paper_encoder = load_paper_encoder_initial(
        device=device,
        path=state_path,
        l1_scale=1.0,
        l2_scale=1.0,
        n_unfreeze_layers=0,
        unfreeze_word_embeddings=False,
        unfreeze_from_bottom=False,
    )
    l2_path = state_path / "categories_embeddings_l2.pt"
    if l2_path.exists():
        l2_dict = torch.load(l2_path, map_location=device, weights_only=True)
        paper_encoder.categories_embeddings_l2.load_state_dict(l2_dict)
    else:
        paper_encoder.categories_embeddings_l2.weight.data.zero_()
    embeddings_path = model_path / "embeddings"
    _verify_paper_encoder(paper_encoder=paper_encoder, embeddings_path=embeddings_path)
    print(f"Finished verifying paper encoder for model at {model_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_before_finetuning_path = ProjectPaths.finetuning_data_model_path()
    verify_paper_encoder(model_path=model_before_finetuning_path, device=device)
    model_after_finetuning_path = ProjectPaths.finetuning_data_checkpoints_path() / "cat_loss"
    verify_paper_encoder(model_path=model_after_finetuning_path, device=device)
