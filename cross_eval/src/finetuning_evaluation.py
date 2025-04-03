from compute_embeddings import get_gpu_info
from finetuning_model import FinetuningModel
from finetuning_preprocessing import *
import gc
import numpy as np
import os
import pickle
import time

def load_users_and_papers(overlap_users : bool, no_overlap_users : bool) -> tuple:
    overlap_users_ids = load_overlap_users_ids() if overlap_users else None
    no_overlap_users_ids = load_no_overlap_users_ids() if no_overlap_users else None
    if overlap_users and not no_overlap_users:
        papers_tokenized = load_overlap_papers_tokenized()
    elif no_overlap_users and not overlap_users:
        papers_tokenized = load_no_overlap_papers_tokenized()
    elif overlap_users and no_overlap_users:
        papers_tokenized = load_validation_papers_tokenized()
    else:
        raise ValueError("At least one of overlap_users or no_overlap_users must be True.")
    return overlap_users_ids, no_overlap_users_ids, papers_tokenized

def get_embedding_name(finetuning_model : FinetuningModel) -> str:
    if finetuning_model.transformer_model_name == GTE_BASE_PATH:
        embedding_name = "gte_base"
    elif finetuning_model.transformer_model_name == GTE_LARGE_PATH:
        embedding_name = "gte_large"
    else:
        raise ValueError(f"Model {finetuning_model.transformer_model_name} not supported.")
    embedding_name += f"_{time.strftime('%Y-%m-%d-%H-%M')}"
    return embedding_name

def perform_embeddings_inference(finetuning_model : FinetuningModel, device : torch.device, input_ids : torch.tensor, attention_masks : torch.tensor, max_batch_size : int) -> torch.tensor:
    MAX_TRIES = 10
    embeddings_total = []
    n_papers, n_papers_processed, batch_num = len(input_ids), 0, 1
    start_time = time.time()
    while n_papers_processed < n_papers:
        batch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        n_tries_so_far, batch_size, successfully_processed = 0, max_batch_size, False
        while n_tries_so_far < MAX_TRIES:
            try:
                upper_bound = min(n_papers, n_papers_processed + batch_size)
                batch_input_ids = input_ids[n_papers_processed : upper_bound].to(device)
                batch_attention_masks = attention_masks[n_papers_processed : upper_bound].to(device)
                with torch.autocast(device_type = device.type, dtype = torch.float16):
                    with torch.inference_mode():
                        embeddings = finetuning_model(input_ids = batch_input_ids, attention_mask = batch_attention_masks).cpu()
                        gpu_info = get_gpu_info()
                embeddings_total.append(embeddings)
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
    embeddings_total = torch.cat(embeddings_total, dim = 0)
    print(f"Finished all batches in {time.time() - start_time:.2f} seconds. Total Papers: {n_papers_processed}.")
    return embeddings_total

def compute_embeddings(finetuning_model : FinetuningModel, device : torch.device, input_ids : torch.tensor, attention_masks : torch.tensor) -> torch.tensor:
    training_mode = finetuning_model.training
    finetuning_model.eval()
    if finetuning_model.transformer_model_name == GTE_BASE_PATH:
        max_batch_size = 1024
    elif finetuning_model.transformer_model_name == GTE_LARGE_PATH:
        max_batch_size = 512
    embeddings = perform_embeddings_inference(finetuning_model, device, input_ids, attention_masks, max_batch_size)
    if training_mode:
        finetuning_model.train()
    return embeddings

def generate_config(file_path : str, embedding_folder : str, users_ids : list, evaluation : str) -> None:
    if not file_path.endswith(".json"):
        file_path += ".json"
    with open(f"{FILES_SAVE_PATH}/example_config.json", "r") as config_file:
        example_config = json.load(config_file)
    example_config["embedding_folder"] = embedding_folder
    example_config["users_selection"] = users_ids
    example_config["evaluation"] = evaluation
    with open(file_path, "w") as config_file:
        json.dump(example_config, config_file, indent = 3)
    
def generate_configs(embedding_folder : str, overlap_users_ids : list = None, no_overlap_users_ids : list = None) -> None:
    embedding_name = embedding_folder.split("/")[-1]
    if overlap_users_ids is not None:
        generate_config(f"{embedding_folder}/{embedding_name}_overlap.json", embedding_folder, overlap_users_ids, "train_test_split")
    if no_overlap_users_ids is not None:
        generate_config(f"{embedding_folder}/{embedding_name}_no_overlap.json", embedding_folder, no_overlap_users_ids, "cross_validation")

def run_evaluation(finetuning_model : FinetuningModel, device : torch.device, overlap_users : bool = True, no_overlap_users : bool = True) -> None:
    overlap_users_ids, no_overlap_users_ids, papers_tokenized = load_users_and_papers(overlap_users, no_overlap_users)
    embedding_name = get_embedding_name(finetuning_model)
    embedding_folder = f"{FILES_SAVE_PATH}/experiments/{embedding_name}"
    papers_ids_to_idxs = {}
    for idx, paper_id in enumerate(papers_tokenized["papers_ids"]):
        papers_ids_to_idxs[paper_id.item()] = idx
    embeddings = compute_embeddings(finetuning_model, device, papers_tokenized["input_ids"], papers_tokenized["attention_masks"])
    os.makedirs(embedding_folder, exist_ok = True)
    np.save(f"{embedding_folder}/abs_X.npy", embeddings)
    with open(f"{embedding_folder}/abs_paper_ids_to_idx.pkl", "wb") as f:
        pickle.dump(papers_ids_to_idxs, f)
    generate_configs(embedding_folder, overlap_users_ids, no_overlap_users_ids)
    os.system(f"python run_cross_eval.py --config_path {embedding_folder}")

def test_evaluation() -> None:
    from finetuning_model import load_finetuning_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetuning_model = load_finetuning_model(GTE_BASE_PATH, device, 256, "/home/scholar/glenn_rp/msc_thesis/data/finetuning/gte_base_256_projection.pt")
    run_evaluation(finetuning_model, device, overlap_users = True, no_overlap_users = True)

test_evaluation()