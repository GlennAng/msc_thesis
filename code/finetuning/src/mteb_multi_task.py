import json
from pathlib import Path

import mteb
import numpy as np
import torch
from transformers import AutoTokenizer

from ...src.project_paths import ProjectPaths
from .finetuning_model import FinetuningModel, load_finetuning_model


class CustomEmbeddingWrapper:
    def __init__(self, model: FinetuningModel, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def encode(
        self, sentences: list, batch_size: int = 512, max_length: int = 512, **kwargs
    ) -> np.ndarray:
        print(
            f"Encoding {len(sentences)} sentences with batch size {batch_size} and max length {max_length}."
        )
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        print(f"Input IDs shape: {input_ids.shape}, Attention mask shape: {attention_mask.shape}")
        embeddings = self.model.perform_inference(
            max_batch_size=batch_size,
            input_ids_tensor=input_ids,
            attention_mask_tensor=attention_mask,
            category_l1_tensor=None,
            eval_type="test",
        )
        result = embeddings.cpu().numpy()
        print(f"Final result shape: {result.shape}, dtype: {result.dtype}")
        return result


def get_tasks():
    return [
        "ArXivHierarchicalClusteringP2P",
        "ArXivHierarchicalClusteringS2S",
        "BigPatentClustering.v2",
        "BiorxivClusteringP2P.v2",
        "CLSClusteringP2P.v2",
        "MedrxivClusteringP2P.v2",
        "StackExchangeClusteringP2P.v2",
        "StackExchangeClustering.v2",
        "ToxicConversationsClassification",
    ]


def extract_results(path: Path) -> dict:
    results = {}
    files = [f for f in path.iterdir() if f.is_file() and f.suffix == ".json"]
    for f in files:
        task_name = f.stem
        with open(f, "r") as file:
            task_results = json.load(file)
        if (
            "scores" in task_results
            and "test" in task_results["scores"]
            and len(task_results["scores"]["test"]) > 0
        ):
            results[task_name] = task_results["scores"]["test"][0]["main_score"]
    return results


if __name__ == "__main__":
    model_path = ProjectPaths.finetuning_data_model_path() / "state_dicts"
    tokenizer = AutoTokenizer.from_pretrained((ProjectPaths.finetuning_data_model_hf()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = get_tasks()
    evaluation = mteb.MTEB(tasks=mteb.get_tasks(tasks=tasks))

    finetuning_model_with_projection = load_finetuning_model(
        finetuning_model_path=model_path,
        device=device,
        mode="eval",
        n_unfreeze_layers=0,
    )
    finetuning_model_with_projection.categories_embeddings_l1 = None
    embedding_wrapper_with_projection = CustomEmbeddingWrapper(
        model=finetuning_model_with_projection, tokenizer=tokenizer
    )
    evaluation.run(
        model=embedding_wrapper_with_projection,
        output_folder=ProjectPaths.data_path(),
        overwrite_results=True,
        encode_kwargs={"batch_size": 512, "max_length": 512},
    )
    results_with_projection = extract_results(
        ProjectPaths.data_path() / "no_model_name_available" / "no_revision_available"
    )
    print("FINISHED EVALUATION WITH PROJECTION")
    finetuning_model_without_projection = load_finetuning_model(
        finetuning_model_path=model_path,
        device=device,
        mode="eval",
        n_unfreeze_layers=0,
    )
    finetuning_model_without_projection.categories_embeddings_l1 = None
    finetuning_model_without_projection.projection = None
    embedding_wrapper_without_projection = CustomEmbeddingWrapper(
        model=finetuning_model_without_projection, tokenizer=tokenizer
    )
    evaluation.run(
        model=embedding_wrapper_without_projection,
        output_folder=ProjectPaths.data_path(),
        overwrite_results=True,
        encode_kwargs={"batch_size": 512, "max_length": 512},
    )
    results_without_projection = extract_results(
        ProjectPaths.data_path() / "no_model_name_available" / "no_revision_available"
    )

    assert sorted(list(results_with_projection.keys())) == sorted(
        list(results_without_projection.keys())
    )
    for task in results_with_projection.keys():
        print(f"Task: {task}")
        print(f"Without Projection: {results_without_projection[task]}")
        print(f"With Projection: {results_with_projection[task]}")
        print()

# Performance worse in different languages and domains
