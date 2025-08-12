import argparse
import gc
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from adapters import AutoAdapterModel
from transformers import AutoModel, AutoTokenizer

from ....src.load_files import load_papers_texts, load_relevant_papers_ids
from .embedding import Embedding

MAX_TRIES = 20

QWEN_TASK_INSTRUCTION = (
    "Given a scientific paper, retrieve relevant papers based on title and abstract."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embeddings Computation Parameters")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--embeddings_folder", type=str)
    parser.add_argument("--max_batch_size", type=int)
    parser.add_argument("--max_sequence_length", type=int)
    parser.add_argument("--all_papers", action="store_true", default=False)
    parser.add_argument("--existing_embedding", type=str, required=False)
    return parser.parse_args()


def is_qwen_instruct(model_path: str) -> bool:
    return model_path.startswith("Qwen/")


def load_model_and_tokenizer(model_path: str) -> tuple:
    if is_qwen_instruct(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype="auto"
        ).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if model_path == "allenai/specter2_base":
            model = AutoAdapterModel.from_pretrained(model_path)
            model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
            model = model.to(device)
        else:
            model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True, unpad_inputs=True, torch_dtype="auto"
            ).to(device)
    model.eval()
    return model, tokenizer


def get_gpu_info() -> str:
    if not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    s = "Memory: "
    s += f"Allocated {allocated:.2f} GB, "
    s += f"Reserved {reserved:.2f} GB."
    return s


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def tokenize_papers(
    batch_papers: list, tokenizer: AutoTokenizer, max_sequence_length: int, model_path: str
) -> dict:
    is_qwen_model = is_qwen_instruct(model_path)
    batch_papers_ids, batch_papers_titles, batch_papers_abstracts = zip(*batch_papers)
    batch_papers_ids, batch_papers_titles, batch_papers_abstracts = (
        list(batch_papers_ids),
        list(batch_papers_titles),
        list(batch_papers_abstracts),
    )
    if is_qwen_model:
        batch_papers_texts = [
            f"Title: {title}. Abstract: {abstract}"
            for title, abstract in zip(batch_papers_titles, batch_papers_abstracts)
        ]
        batch_papers_texts = [
            get_detailed_instruct(QWEN_TASK_INSTRUCTION, text) for text in batch_papers_texts
        ]
    else:
        sep_token = tokenizer.sep_token
        batch_papers_texts = [
            f"{title} {sep_token} {abstract}"
            for title, abstract in zip(batch_papers_titles, batch_papers_abstracts)
        ]
    batch_tokenized_papers = tokenizer(
        text=batch_papers_texts,
        max_length=max_sequence_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return batch_papers_ids, batch_tokenized_papers


def extract_embeddings(
    batch_outputs: dict, attention_mask: torch.Tensor, model_path: str
) -> torch.Tensor:
    if is_qwen_instruct(model_path):
        embeddings = last_token_pool(batch_outputs.last_hidden_state, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    else:
        return batch_outputs.last_hidden_state[:, 0]


def write_additions_to_file(
    embeddings_folder: Path,
    batch_papers_ids: list,
    batch_papers_embeddings: torch.Tensor,
    batch_num: int,
):
    assert isinstance(batch_papers_ids, list), "batch_papers_ids must be a list."
    assert batch_papers_embeddings.shape[0] == len(batch_papers_ids)
    batch_folder = embeddings_folder / "batches" / f"batch_{batch_num}"
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)
    with open(f"{batch_folder / 'abs_X.npy'}", "wb") as batch_papers_embeddings_file:
        batch_papers_embeddings_np = batch_papers_embeddings.cpu().detach().numpy()
        np.save(batch_papers_embeddings_file, batch_papers_embeddings_np)
    with open(f"{batch_folder / 'abs_ids.pkl'}", "wb") as batch_papers_ids_file:
        pickle.dump(batch_papers_ids, batch_papers_ids_file)


def tokenize_and_encode_papers_in_batches(
    embeddings_folder: Path,
    papers: list,
    device: torch.device,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    max_batch_size: int,
    max_sequence_length: int,
    model_path: str,
):
    n_papers, n_papers_processed, batch_num = len(papers), 0, 1
    while n_papers_processed < n_papers:
        batch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        n_tries_so_far, batch_size, successfully_processed = 0, max_batch_size, False
        while n_tries_so_far < MAX_TRIES:
            try:
                upper_bound = min(n_papers, n_papers_processed + batch_size)
                batch_papers = papers[n_papers_processed:upper_bound]
                batch_papers_ids, batch_tokenized_papers = tokenize_papers(
                    batch_papers, tokenizer, max_sequence_length, model_path
                )
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    with torch.no_grad():
                        with torch.inference_mode():
                            batch_papers_outputs = model(**batch_tokenized_papers.to(device))
                            batch_papers_embeddings = extract_embeddings(
                                batch_papers_outputs,
                                batch_tokenized_papers["attention_mask"].to(device),
                                model_path,
                            )
                            gpu_info = get_gpu_info()
                write_additions_to_file(
                    embeddings_folder, batch_papers_ids, batch_papers_embeddings, batch_num
                )
                n_papers_processed = min(n_papers, n_papers_processed + batch_size)
                print(
                    f"Finished Batch {batch_num} in {time.time() - batch_start_time:.2f} Seconds: Papers {n_papers_processed} / {n_papers}. {gpu_info}"
                )
                batch_num += 1
                successfully_processed = True
            except Exception as e:
                print(f"Error in encoding Batch {batch_num}: \n{e}")
                n_tries_so_far += 1
                batch_size = batch_size // 2
            if "batch_papers_outputs" in locals():
                del batch_papers_outputs
            if "batch_papers_embeddings" in locals():
                del batch_papers_embeddings
            if "batch_tokenized_papers" in locals():
                del batch_tokenized_papers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if successfully_processed:
                break
        if n_tries_so_far >= MAX_TRIES:
            os._exit(0)


if __name__ == "__main__":
    args = parse_args()
    embeddings_folder = Path(args.embeddings_folder).resolve()
    if os.path.exists(embeddings_folder):
        raise ValueError("Embeddings Folder already exists.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}.")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    papers_texts = load_papers_texts(relevant_columns=["paper_id", "title", "abstract"])
    relevant_papers_ids = papers_texts["paper_id"].tolist()
    if not args.all_papers:
        print("Finding relevant papers IDs...")
        relevant_papers_ids = load_relevant_papers_ids()
        print(f"Found {len(relevant_papers_ids)} relevant papers.")
    if args.existing_embedding:
        embedding = Embedding(Path(args.existing_embedding).resolve())
        available_papers_ids = list(embedding.papers_ids_to_idxs.keys())
        missing_papers_ids = set(relevant_papers_ids) - set(available_papers_ids)
        relevant_papers_ids = sorted(list(missing_papers_ids))
        print(f"Found {len(relevant_papers_ids)} missing papers.")
    papers_texts = papers_texts[papers_texts["paper_id"].isin(relevant_papers_ids)]
    papers_texts = papers_texts[["paper_id", "title", "abstract"]].values.tolist()
    max_sequence_length = args.max_sequence_length + (20 if is_qwen_instruct else 0)
    tokenize_and_encode_papers_in_batches(
        embeddings_folder,
        papers_texts,
        device,
        model,
        tokenizer,
        args.max_batch_size,
        max_sequence_length,
        args.model_path,
    )
    print(f"Embeddings computation finished. Results saved in {embeddings_folder}.")
