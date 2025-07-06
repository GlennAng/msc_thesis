import json
from pathlib import Path

import mteb
import numpy as np
import torch
from adapters import AutoAdapterModel
from transformers import AutoModel, AutoTokenizer

from ...logreg.src.embeddings.compute_embeddings import (
    extract_embeddings,
    tokenize_papers,
)
from ...src.project_paths import ProjectPaths
from .finetuning_model import FinetuningModel, load_finetuning_model


class CustomEmbeddingWrapper:
    def __init__(self, model_choice, device, model=None):
        self.model_choice = model_choice
        self.device = device

        if isinstance(model, FinetuningModel):
            self.model = model.to(device)
            self.tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5")
            return

        if model_choice == "gte-base-en-v1.5":
            hf_code = "Alibaba-NLP/gte-base-en-v1.5"
            self.model = AutoModel.from_pretrained(
                hf_code, trust_remote_code=True, unpad_inputs=True, torch_dtype="auto"
            ).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(hf_code)

        elif model_choice == "specter2_base":
            model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
            model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
            self.model = model.to(device)
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")

        elif model_choice.startswith("Qwen"):
            self.model = AutoModel.from_pretrained(
                model_choice, trust_remote_code=True, torch_dtype="auto"
            ).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_choice, padding_side="left")
        else:
            raise ValueError(f"Unsupported model type: {model}")

    def encode(self, sentences, batch_size=512, max_length=512, **kwargs):
        papers_texts = []
        for sentence in sentences:
            if "   " in sentence:
                parts = sentence.split("   ", 1)
                papers_texts.append((0, parts[0].strip(), parts[1].strip()))
            else:
                raise ValueError(f"Sentence does not contain a valid split point: {sentence}")
        tokenized_papers = tokenize_papers(
            papers_texts,
            tokenizer=self.tokenizer,
            max_sequence_length=max_length,
            model_path=self.model_choice,
        )[1]
        with torch.no_grad():
            with torch.inference_mode():
                if isinstance(self.model, FinetuningModel):
                    embeddings = self.model.perform_inference(
                        max_batch_size=batch_size,
                        input_ids_tensor=tokenized_papers["input_ids"].to(self.device),
                        attention_mask_tensor=tokenized_papers["attention_mask"].to(self.device),
                        category_l1_tensor=None,
                        eval_type="test",
                    )
                else:
                    embeddings = []
                    for i in range(0, len(tokenized_papers["input_ids"]), batch_size):
                        tokenized_papers_batch = {
                            key: value[i:i + batch_size] for key, value in tokenized_papers.items()
                        }
                        tokenized_papers_batch = torch.tensor(tokenized_papers_batch)
                        batch_outputs = self.model(**tokenized_papers_batch.to(self.device))
                        embeddings.append(extract_embeddings(
                            batch_outputs=batch_outputs,
                            attention_mask=tokenized_papers_batch["attention_mask"].to(self.device),
                            model_path=self.model_choice,
                        ))
                    embeddings = torch.cat(embeddings, dim=0)
        result = embeddings.to(torch.float32).cpu().numpy()
        print(f"Final result shape: {result.shape}, dtype: {result.dtype}")
        return result


MODELS_CHOICES = [
    "gte-base-en-v1.5",
    "gte-large-en-v1.5",
    "gte-large-en-v1.5_no_projection",
    "specter2_base",
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
    ProjectPaths.finetuning_data_path() / "checkpoints" / "best_batch_125",
]
MODELS_TO_LOAD = ["gte-large-en-v1.5", "gte-large-en-v1.5_no_projection"]
MODELS_TO_LOAD += [model for model in MODELS_CHOICES if isinstance(model, Path)]
MODELS_NAMES = [model if isinstance(model, str) else model.stem for model in MODELS_CHOICES]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evaluation = mteb.MTEB(tasks=mteb.get_tasks(tasks=["ArXivHierarchicalClusteringP2P"]))
for i, model_choice in enumerate(MODELS_CHOICES):
    model = None
    print(f"Evaluation Model: {MODELS_NAMES[i]}.")
    if model_choice in MODELS_TO_LOAD:
        if isinstance(model_choice, Path):
            model_path = model_choice / "state_dicts"
        else:
            model_path = ProjectPaths.finetuning_data_model_path() / "state_dicts"
        model = load_finetuning_model(
            finetuning_model_path=model_path,
            device=device,
            mode="eval",
            n_unfreeze_layers=0,
        )
        model.categories_embeddings_l1 = None
        if model == "gte-large-en-v1.5_no_projection":
            model.projection = None
    embedding_wrapper = CustomEmbeddingWrapper(
        model_choice=model_choice, device=device, model=model
    )

    evaluation.run(
        model=embedding_wrapper,
        output_folder=ProjectPaths.data_path(),
        overwrite_results=True,
        encode_kwargs={"batch_size": 15, "max_length": 512},
    )
