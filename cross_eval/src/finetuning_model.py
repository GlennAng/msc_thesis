from compute_embeddings import get_gpu_info
from transformers import AutoModel, AutoTokenizer
import json
import gc
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_SAVE_PATH = "../data/models/"
N_CATEGORIES = 18
CATEGORIES_EMBEDDINGS_DIM = 100

class FinetuningModel(nn.Module):
    def __init__(self, transformer_model : AutoModel, projection : nn.Linear, users_embeddings : nn.Embedding, 
                val_users_embeddings_idxs : torch.tensor = None, categories_embeddings : nn.Embedding = None, n_unfreeze_layers : int = 4) -> None:
        super().__init__()
        self.transformer_model = transformer_model
        self.set_transformer_model_name()
        self.projection = projection
        self.users_embeddings = users_embeddings
        assert next(transformer_model.parameters()).device == self.projection.weight.device == self.users_embeddings.weight.device
        self.device = self.projection.weight.device
        if val_users_embeddings_idxs is not None:
            self.val_users_embeddings_idxs = val_users_embeddings_idxs.to(self.device)
        self.categories_embeddings = categories_embeddings
        self.unfreeze_layers(n_unfreeze_layers)
        
    def forward(self, eval_type : str, input_ids_tensor : torch.tensor, attention_mask_tensor : torch.tensor,
                user_idx_tensor : torch.tensor = None, category_idx_tensor : torch.tensor = None) -> torch.tensor:
        eval_type = eval_type.lower()
        assert eval_type in ["train", "val", "negative_samples", "test"]
        papers_embeddings = self.transformer_model(input_ids = input_ids_tensor, attention_mask = attention_mask_tensor).last_hidden_state[:, 0, :]
        papers_embeddings = self.projection(papers_embeddings)
        papers_embeddings = F.normalize(papers_embeddings, p = 2, dim = 1)

        if self.categories_embeddings is not None:
            assert category_idx_tensor is not None
            categories_embeddings = self.categories_embeddings(category_idx_tensor)
            papers_embeddings = torch.cat((papers_embeddings, categories_embeddings), dim = 1)
        
        if eval_type == "test":
            return papers_embeddings
        if eval_type == "negative_samples":
            val_users_embeddings = self.users_embeddings(self.val_users_embeddings_idxs)
            dot_products = torch.matmul(val_users_embeddings[:, :-1], papers_embeddings.transpose(0, 1))
            dot_products = dot_products + val_users_embeddings[:, -1].unsqueeze(1)
            return dot_products
        if eval_type == "val" or eval_type == "train":
            users_embeddings = self.users_embeddings(user_idx_tensor)
            dot_products = torch.sum(papers_embeddings * users_embeddings[:, :-1], dim = 1) + self.users_embeddings(user_idx_tensor)[:, -1]
            return dot_products

    def save_finetuning_model(self, model_path : str) -> None:
        model_path = model_path.rstrip("/") + "/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.transformer_model.save_pretrained(model_path + "transformer_model")
        torch.save(self.projection.state_dict(), model_path + "projection.pt")
        if self.categories_embeddings is not None:
            torch.save(self.users_embeddings.state_dict(), model_path + "users_embeddings_categories.pt")
        else:
            torch.save(self.users_embeddings.state_dict(), model_path + "users_embeddings.pt")
        if self.categories_embeddings is not None:
            torch.save(self.categories_embeddings.state_dict(), model_path + "categories_embeddings.pt")

    def unfreeze_layers(self, n_unfreeze_layers : int) -> None:
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
        
    def set_transformer_model_name(self) -> None:
        if "gte_base_256" in self.transformer_model.config._name_or_path:
            self.transformer_model_name = "gte_base_256"
        elif "gte_large_256" in self.transformer_model.config._name_or_path:
            self.transformer_model_name = "gte_large_256"
        else:
            raise ValueError(f"Unknown transformer model name: {self.transformer_model.config._name_or_path}")

    def compute_papers_embeddings(self, input_ids_tensor : torch.tensor, attention_mask_tensor : torch.tensor, category_idx_tensor : torch.tensor) -> torch.tensor:
        if self.transformer_model_name.startswith("gte_base"):
            max_batch_size = 1024
        elif self.transformer_model_name.startswith("gte_large"):
            max_batch_size = 512
        papers_embeddings = self.perform_inference(max_batch_size, input_ids_tensor, attention_mask_tensor, category_idx_tensor = category_idx_tensor, eval_type = "test")
        return papers_embeddings

    def perform_inference(self, max_batch_size : int, input_ids_tensor : torch.tensor, attention_mask_tensor : torch.tensor, user_idx_tensor : torch.tensor = None, 
                                category_idx_tensor : torch.tensor = None, eval_type : str = None) -> torch.tensor:
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
                    batch_category_idx_tensor = None
                    if category_idx_tensor is not None:
                        batch_category_idx_tensor = category_idx_tensor[n_samples_processed : upper_bound].to(self.device)
                    with torch.autocast(device_type = self.device.type, dtype = torch.float16):
                        with torch.inference_mode():
                            results = self(eval_type = eval_type, input_ids_tensor = batch_input_ids_tensor, attention_mask_tensor = batch_attention_mask_tensor, 
                                           user_idx_tensor = batch_user_idx_tensor, category_idx_tensor = batch_category_idx_tensor).cpu()
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
        print(f"Finished all Batches in {time.time() - start_time:.2f} seconds. Total Samples: {n_samples_processed}.")
        return results_total

    def compute_negative_samples_scores(self, negative_samples : torch.tensor) -> torch.tensor:
        with torch.autocast(device_type = self.device.type, dtype = torch.float16):
            with torch.inference_mode():
                input_ids_tensor, attention_mask_tensor = negative_samples["input_ids"].to(self.device), negative_samples["attention_mask"].to(self.device)
                if self.categories_embeddings is not None:
                    category_idx_tensor = negative_samples["category_idx"].to(self.device)
                else:
                    category_idx_tensor = None
                negative_samples_scores = self(eval_type = "negative_samples", input_ids_tensor = input_ids_tensor, attention_mask_tensor = attention_mask_tensor,
                                               category_idx_tensor = category_idx_tensor).cpu()
        return negative_samples_scores

    def compute_val_scores(self, user_idx_tensor : torch.tensor, input_ids_tensor : torch.tensor, attention_mask_tensor : torch.tensor, category_idx_tensor : torch.tensor = None,
                           max_batch_size : int = None) -> torch.tensor:
        if max_batch_size is None:
            if self.transformer_model_name.startswith("gte_base"):
                max_batch_size = 1024
            elif self.transformer_model_name.startswith("gte_large"):
                max_batch_size = 512
        users_scores = self.perform_inference(max_batch_size, input_ids_tensor, attention_mask_tensor, user_idx_tensor, category_idx_tensor, eval_type = "val")
        return users_scores

def load_transformer_model(transformer_path : str, device : torch.device) -> AutoModel:
    return AutoModel.from_pretrained(transformer_path, trust_remote_code = True, unpad_inputs = True, torch_dtype = "auto").to(device)

def load_transformer_tokenizer(transformer_path : str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(transformer_path)

def save_transformer_model(transformer_model : AutoModel, model_path : str) -> None:
    transformer_model.save_pretrained(model_path)

def get_transformer_embedding_dim(transformer_model : AutoModel) -> int:
    return transformer_model.config.hidden_size

def load_projection(embedding_dim : int, projection_dim : int, device : torch.device, projection_state_dict : dict = None, dtype = torch.float32) -> nn.Linear:
    projection = nn.Linear(embedding_dim, projection_dim)
    if projection_state_dict is not None:
        projection.load_state_dict(projection_state_dict)
    projection = projection.to(device, dtype = dtype)
    return projection

def load_users_embeddings(num_embeddings : int, embedding_dim : int, device : torch.device, users_embeddings_state_dict : dict = None, dtype = torch.float32) -> nn.Embedding:
    users_embeddings = nn.Embedding(num_embeddings, embedding_dim)
    if users_embeddings_state_dict is not None:
        users_embeddings.load_state_dict(users_embeddings_state_dict)
    users_embeddings = users_embeddings.to(device, dtype = dtype)
    return users_embeddings

def load_categories_embeddings(device : torch.device, categories_embeddings_state_dict : dict = None, dtype = torch.float32) -> nn.Embedding:
    categories_embeddings = nn.Embedding(N_CATEGORIES, CATEGORIES_EMBEDDINGS_DIM)
    if categories_embeddings_state_dict is not None:
        categories_embeddings.load_state_dict(categories_embeddings_state_dict)
    categories_embeddings = categories_embeddings.to(device, dtype = dtype)
    return categories_embeddings

def load_finetuning_model(transformer_path : str, device : torch.device, projection_state_dict : dict = None, users_embeddings_state_dict : dict = None,
                          categories_attached : bool = True, categories_embeddings_state_dict : dict = None, val_users_embeddings_idxs : torch.tensor = None,
                          projection_dim : int = 256, n_users : int = 4111, n_unfreeze_layers : int = 4) -> FinetuningModel:
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
    if type(categories_embeddings_state_dict) == str:
        categories_embeddings_state_dict = torch.load(categories_embeddings_state_dict, map_location = device, weights_only = True)
    projection = load_projection(embedding_dim, projection_dim, device, projection_state_dict)
    users_embeddings = load_users_embeddings(n_users, projection_dim + 1 + (CATEGORIES_EMBEDDINGS_DIM if categories_attached else 0), device, users_embeddings_state_dict)
    categories_embeddings = None
    if categories_attached:
        categories_embeddings = load_categories_embeddings(device, categories_embeddings_state_dict)
    return FinetuningModel(transformer_model, projection, users_embeddings, val_users_embeddings_idxs, categories_embeddings, n_unfreeze_layers = n_unfreeze_layers)

def fix_gte_config(model_path : str) -> None:
    config_path = os.path.join(model_path, "config.json")
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

def load_finetuning_model_full(finetuning_model_path : str, device : torch.device, val_users_embeddings_idxs : torch.tensor = None, categories_attached : bool = True,
                               pretrained_projection : bool = True, pretrained_users_embeddings : bool = True, pretrained_categories_embeddings : bool = True, 
                               n_unfreeze_layers : int = 4) -> FinetuningModel:
    finetuning_model_path = finetuning_model_path.rstrip("/") + "/"
    transformer_model_path = finetuning_model_path + "transformer_model"
    fix_gte_config(transformer_model_path)
    if pretrained_projection:
        projection_state_dict = torch.load(finetuning_model_path + "projection.pt", map_location = device, weights_only = True)
    else:
        projection_state_dict = None
    if pretrained_users_embeddings:
        users_embeddings_path = finetuning_model_path + f"{'users_embeddings_categories.pt' if categories_attached else 'users_embeddings.pt'}"
        users_embeddings_state_dict = torch.load(users_embeddings_path, map_location = device, weights_only = True)
    else:
        users_embeddings_state_dict = None
    if categories_attached and pretrained_categories_embeddings:
        pretrained_categories_state_dict = torch.load(finetuning_model_path + "categories_embeddings.pt", map_location = device, weights_only = True)
    else:
        pretrained_categories_state_dict = None
    return load_finetuning_model(transformer_model_path, device, projection_state_dict, users_embeddings_state_dict, categories_attached, pretrained_categories_state_dict,
                                 val_users_embeddings_idxs, n_unfreeze_layers = n_unfreeze_layers)