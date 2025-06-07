from data_handling import get_titles_and_abstracts
from transformers import AutoModel, AutoTokenizer
from transformers import pipeline
import argparse
import gc
import numpy as np
from adapters import AutoAdapterModel
import os
import pickle
import time
import torch

MAX_TRIES = 20

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Embeddings Computation Parameters")
    parser.add_argument("--model_path", type = str)
    parser.add_argument("--embeddings_folder", type = str)
    parser.add_argument("--max_batch_size", type = int)
    parser.add_argument("--max_sequence_length", type = int)
    return parser.parse_args()

def load_model_and_tokenizer(model_path : str) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if model_path == "allenai/specter2_base":
        model = AutoAdapterModel.from_pretrained(model_path)
        model.load_adapter("allenai/specter2", source = "hf", load_as = "specter2", set_active = True)
        model = model.to(device)
    else:
        model = AutoModel.from_pretrained(model_path, trust_remote_code = True, unpad_inputs = True, torch_dtype = "auto").to(device)
    model.eval()
    return model, tokenizer

def get_gpu_info() -> str:
    if not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    s = "Memory: "
    s += f"Allocated {allocated:.2f} GB, "
    s += f"Reserved {reserved:.2f} GB."
    return s

def tokenize_papers(batch_papers : list, tokenizer : AutoTokenizer, max_sequence_length : int) -> dict:
    sep_token = tokenizer.sep_token
    if len(batch_papers[0]) == 3:
        batch_papers_ids, batch_papers_titles, batch_papers_abstracts = zip(*batch_papers)
        batch_papers_ids, batch_papers_titles, batch_papers_abstracts = list(batch_papers_ids), list(batch_papers_titles), list(batch_papers_abstracts)
        batch_papers_titles_abstracts = [f"{title} {sep_token} {abstract}" for title, abstract in zip(batch_papers_titles, batch_papers_abstracts)]
    else:
        batch_papers_ids, batch_papers_titles, batch_papers_abstracts, batch_papers_categories = zip(*batch_papers)
        batch_papers_ids, batch_papers_titles, batch_papers_abstracts, batch_papers_categories = list(batch_papers_ids), list(batch_papers_titles), list(batch_papers_abstracts), list(batch_papers_categories)
        batch_papers_titles_abstracts = [f"{title} {sep_token} {abstract} {sep_token} {category}" for title, abstract, category in zip(batch_papers_titles, batch_papers_abstracts, batch_papers_categories)]
    batch_tokenized_papers = tokenizer(text = batch_papers_titles_abstracts, max_length = max_sequence_length, padding = True, truncation = True, return_tensors = "pt")
    return batch_papers_ids, batch_tokenized_papers

def write_additions_to_file(embeddings_folder : str, batch_papers_ids : list, batch_papers_embeddings : torch.Tensor, batch_num : int):
    assert isinstance(batch_papers_ids, list), 'batch_papers_ids must be a list.'
    assert batch_papers_embeddings.shape[0] == len(batch_papers_ids), f'Number of papers and number of embeddings do not match {batch_papers_embeddings.shape[0]} != {len(batch_papers_ids)}'
    batch_folder = f"{embeddings_folder}/batches/batch_{batch_num}"
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)
    with open(f"{batch_folder}/abs_X.npy", 'wb') as batch_papers_embeddings_file:
        batch_papers_embeddings_np = batch_papers_embeddings.cpu().detach().numpy()
        np.save(batch_papers_embeddings_file, batch_papers_embeddings_np)
    with open(f"{batch_folder}/abs_ids.pkl", 'wb') as batch_papers_ids_file:
        pickle.dump(batch_papers_ids, batch_papers_ids_file)

def tokenize_and_encode_papers_in_batches(embeddings_folder : str, papers : list, device : torch.device, model : AutoModel, tokenizer : AutoTokenizer, 
                                          max_batch_size : int, max_sequence_length : int):
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
                batch_papers = papers[n_papers_processed : upper_bound]
                batch_papers_ids, batch_tokenized_papers = tokenize_papers(batch_papers, tokenizer, max_sequence_length)
                
                with torch.autocast(device_type = device.type, dtype = torch.float16):
                    with torch.inference_mode():
                        batch_papers_outputs = model(**batch_tokenized_papers.to(device))
                        batch_papers_embeddings = batch_papers_outputs.last_hidden_state[:, 0].cpu()
                        gpu_info = get_gpu_info()
                write_additions_to_file(embeddings_folder, batch_papers_ids, batch_papers_embeddings, batch_num)
                n_papers_processed = min(n_papers, n_papers_processed + batch_size)
                print(f"Finished Batch {batch_num} in {time.time() - batch_start_time:.2f} Seconds: Papers {n_papers_processed} / {n_papers}. {gpu_info}")
                batch_num += 1
                successfully_processed = True
            except Exception as e:
                print(f"Error in encoding Batch {batch_num}: \n{e}")
                n_tries_so_far += 1
                batch_size = batch_size // 2
            if 'batch_outputs' in locals():
                del batch_outputs
            if 'batch_embeddings' in locals():
                del batch_embeddings
            if 'batch_tokenized_papers' in locals():
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
    if os.path.exists(args.embeddings_folder):
        raise ValueError("Embeddings Folder already exists.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    papers = get_titles_and_abstracts()
    tokenize_and_encode_papers_in_batches(args.embeddings_folder, papers, device, model, tokenizer, args.max_batch_size, args.max_sequence_length)