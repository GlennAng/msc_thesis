from compute_embeddings import get_gpu_info
from finetuning_preprocessing import GTE_BASE_PATH, GTE_LARGE_PATH
from transformers import AutoModel, AutoTokenizer
import gc
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_SAVE_PATH = "../data/models/"

class FinetuningModel(nn.Module):
    def __init__(self, transformer_model : AutoModel, projection : nn.Linear, users_embeddings : nn.Embedding, val_users_embeddings_idxs : torch.tensor = None) -> None:
        super().__init__()
        self.transformer_model = transformer_model
        self.transformer_model_name = transformer_model.config._name_or_path
        self.projection = projection
        self.users_embeddings = users_embeddings
        assert next(transformer_model.parameters()).device == self.projection.weight.device == self.users_embeddings.weight.device
        self.device = self.projection.weight.device
        if val_users_embeddings_idxs is not None:
            self.val_users_embeddings_idxs = val_users_embeddings_idxs.to(self.device)
        
    def forward(self, eval_type : str, input_ids_tensor : torch.tensor, attention_mask_tensor : torch.tensor, user_idx_tensor : torch.tensor = None) -> torch.tensor:
        eval_type = eval_type.lower()
        assert eval_type in ["train", "val", "negative_samples", "test"]
        papers_embeddings = self.transformer_model(input_ids = input_ids_tensor, attention_mask = attention_mask_tensor).last_hidden_state[:, 0, :]
        papers_embeddings = self.projection(papers_embeddings)
        papers_embeddings = F.normalize(papers_embeddings, p = 2, dim = 1)
        
        if eval_type == "test":
            return papers_embeddings
        if eval_type == "negative_samples":
            val_users_embeddings = self.users_embeddings(self.val_users_embeddings_idxs)
            dot_products = torch.matmul(val_users_embeddings[:, :-1], papers_embeddings.transpose(0, 1))
            dot_products = dot_products + val_users_embeddings[:, -1].unsqueeze(1)
            return dot_products
        if eval_type == "val":
            users_embeddings = self.users_embeddings(user_idx_tensor)
            dot_products = torch.sum(papers_embeddings * users_embeddings[:, :-1], dim = 1) + self.users_embeddings(user_idx_tensor)[:, -1]
            return dot_products

    def get_embedding_name(self) -> str:
        transformer_model_name = self.transformer_model.config._name_or_path
        if transformer_model_name == GTE_BASE_PATH:
            embedding_name = "gte_base"
        elif transformer_model_name == GTE_LARGE_PATH:
            embedding_name = "gte_large"
        else:
            raise ValueError(f"Model {transformer_model_name} not supported.")
        embedding_name += f"_{time.strftime('%Y-%m-%d-%H-%M')}"
        return embedding_name

    def compute_papers_embeddings(self, input_ids_tensor : torch.tensor, attention_mask_tensor : torch.tensor) -> torch.tensor:
        if self.transformer_model_name == GTE_BASE_PATH:
            max_batch_size = 1024
        elif self.transformer_model_name == GTE_LARGE_PATH:
            max_batch_size = 512
        papers_embeddings = self.perform_inference(max_batch_size, input_ids_tensor, attention_mask_tensor, eval_type = "test")
        return papers_embeddings

    def perform_inference(self, max_batch_size : int, input_ids_tensor : torch.tensor, attention_mask_tensor : torch.tensor, 
                          user_idx_tensor : torch.tensor = None, eval_type : str = None) -> torch.tensor:
        MAX_TRIES = 10
        results_total = []
        n_samples, n_samples_processed, batch_num = len(input_ids_tensor), 0, 1
        start_time = time.time()
        while n_samples_processed < n_samples:
            batch_start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            n_tries_so_far, batch_size, successfully_processed = 0, max_batch_size, False
            while n_tries_so_far < MAX_TRIES:
                try:
                    upper_bound = min(n_samples, n_samples_processed + batch_size)
                    batch_input_ids_tensor = input_ids_tensor[n_samples_processed : upper_bound].to(self.device)
                    batch_attention_mask_tensor = attention_mask_tensor[n_samples_processed : upper_bound].to(self.device)
                    batch_user_idx_tensor = None
                    if user_idx_tensor is not None:
                        batch_user_idx_tensor = user_idx_tensor[n_samples_processed : upper_bound].to(self.device)
                    with torch.autocast(device_type = self.device.type, dtype = torch.float16):
                        with torch.inference_mode():
                            results = self(eval_type = eval_type, input_ids_tensor = batch_input_ids_tensor, attention_mask_tensor = batch_attention_mask_tensor, 
                                           user_idx_tensor = batch_user_idx_tensor).cpu()
                            gpu_info = get_gpu_info()
                    results_total.append(results)
                    n_samples_processed = min(n_samples, n_samples_processed + batch_size)
                    print(f"Finished Batch {batch_num} in {time.time() - batch_start_time:.2f} Seconds: Samples {n_samples_processed} / {n_samples}. {gpu_info}")
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
        results_total = torch.cat(results_total, dim = 0)
        print(f"Finished all batches in {time.time() - start_time:.2f} seconds. Total Samples: {n_samples_processed}.")
        return results_total

    def compute_negative_samples_scores(self, negative_samples : torch.tensor) -> torch.tensor:
        with torch.autocast(device_type = self.device.type, dtype = torch.float16):
            with torch.inference_mode():
                input_ids_tensor, attention_mask_tensor = negative_samples["input_ids"].to(self.device), negative_samples["attention_mask"].to(self.device)
                negative_samples_scores = self(eval_type = "negative_samples", input_ids_tensor = input_ids_tensor, attention_mask_tensor = attention_mask_tensor).cpu()
        return negative_samples_scores

    def compute_val_scores(self, user_idx_tensor : torch.tensor, input_ids_tensor : torch.tensor, attention_mask_tensor : torch.tensor, max_batch_size : int = None) -> torch.tensor:
        if max_batch_size is None:
            if self.transformer_model_name == GTE_BASE_PATH:
                max_batch_size = 1024
            elif self.transformer_model_name == GTE_LARGE_PATH:
                max_batch_size = 512
        users_scores = self.perform_inference(max_batch_size, input_ids_tensor, attention_mask_tensor, user_idx_tensor, eval_type = "val")
        return users_scores

def load_transformer_model(transformer_path : str, device : torch.device) -> AutoModel:
    return AutoModel.from_pretrained(transformer_path, trust_remote_code = True, unpad_inputs = True, torch_dtype = "auto").to(device)

def load_transformer_tokenizer(transformer_path : str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(transformer_path)

def save_transformer_model(transformer_model : AutoModel, model_path : str) -> None:
    transformer_model.save_pretrained(model_path)

def get_transformer_embedding_dim(transformer_model : AutoModel) -> int:
    return transformer_model.config.hidden_size

def load_projection(embedding_dim : int, projection_dim : int, device : torch.device, projection_state_dict : dict = None, seed : int = None) -> nn.Linear:
    if seed is not None:
        torch.manual_seed(seed)
    projection = nn.Linear(embedding_dim, projection_dim).to(device)
    if projection_state_dict is not None:
        projection.load_state_dict(projection_state_dict)
    return projection

def load_users_embeddings(num_embeddings : int, embedding_dim : int, device : torch.device, users_embeddings_state_dict : dict = None, seed : int = None) -> nn.Embedding:
    if seed is not None:
        torch.manual_seed(seed)
    users_embeddings = nn.Embedding(num_embeddings, embedding_dim).to(device)
    if users_embeddings_state_dict is not None:
        users_embeddings.load_state_dict(users_embeddings_state_dict)
    return users_embeddings

def load_finetuning_model(transformer_path : str, device : torch.device, projection_state_dict : dict = None, users_embeddings_state_dict : dict = None,
                          val_users_embeddings_idxs : torch.tensor = None, projection_dim : int = 256, n_users : int = 4111, seed : int = None) -> FinetuningModel:
    transformer_model = load_transformer_model(transformer_path, device)
    embedding_dim = get_transformer_embedding_dim(transformer_model)
    if type(projection_state_dict) == str:
        projection_state_dict = torch.load(projection_state_dict, map_location = device, weights_only = True)
    if projection_state_dict is not None:
        projection_dim = projection_state_dict["weight"].shape[0]
    if type(users_embeddings_state_dict) == str:
        users_embeddings_state_dict = torch.load(users_embeddings_state_dict, map_location = device, weights_only = True)
    if users_embeddings_state_dict is not None:
        n_users = users_embeddings_state_dict["weight"].shape[0]
    projection = load_projection(embedding_dim, projection_dim, device, projection_state_dict, seed)
    users_embeddings = load_users_embeddings(n_users, projection_dim + 1, device, users_embeddings_state_dict, seed)
    return FinetuningModel(transformer_model, projection, users_embeddings, val_users_embeddings_idxs)