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

class FinetuningModel(nn.Module):
    def __init__(self, transformer_model : AutoModel, projection : nn.Linear, users_embeddings : nn.Embedding, categories_embeddings : nn.Embedding,
                 val_users_embeddings_idxs : torch.tensor = None, categories_embeddings_l2 : nn.Embedding = None, n_unfreeze_layers : int = 4) -> None:
        super().__init__()
        self.transformer_model = transformer_model
        self.set_transformer_model_name()
        self.set_eval_batch_size()
        self.projection = projection
        self.users_embeddings = users_embeddings
        self.categories_embeddings = categories_embeddings
        self.categories_embeddings_l2 = categories_embeddings_l2
        assert next(transformer_model.parameters()).device == self.projection.weight.device == self.users_embeddings.weight.device == self.categories_embeddings.weight.device
        self.device = self.projection.weight.device
        if val_users_embeddings_idxs is not None:
            self.val_users_embeddings_idxs = val_users_embeddings_idxs.to(self.device)
            assert self.val_users_embeddings_idxs.tolist() == sorted(self.val_users_embeddings_idxs.tolist())
        self.unfreeze_layers(n_unfreeze_layers)
        
    def forward(self, eval_type : str, input_ids_tensor : torch.tensor, attention_mask_tensor : torch.tensor, category_idx_tensor : torch.tensor,
                user_idx_tensor : torch.tensor = None, category_idx_l2_tensor : torch.tensor = None,
                sorted_unique_user_idx_tensor : torch.tensor = None, batch_negatives_indices : torch.tensor = None) -> torch.tensor:
        eval_type = eval_type.lower()
        assert eval_type in ["train", "negative_samples", "val", "test"]
        input_ids_tensor, attention_mask_tensor, category_idx_tensor, user_idx_tensor, category_idx_l2_tensor, sorted_unique_user_idx_tensor, batch_negatives_indices = self.check_devices(
            input_ids_tensor, attention_mask_tensor, category_idx_tensor, user_idx_tensor, category_idx_l2_tensor, sorted_unique_user_idx_tensor, batch_negatives_indices)
        
        papers_embeddings = self.transformer_model(input_ids = input_ids_tensor, attention_mask = attention_mask_tensor).last_hidden_state[:, 0, :]
        papers_embeddings = self.projection(papers_embeddings)
        papers_embeddings = F.normalize(papers_embeddings, p = 2, dim = 1)
        categories_embeddings = self.categories_embeddings(category_idx_tensor)
        if category_idx_l2_tensor is not None:
            categories_embeddings_l2 = self.categories_embeddings_l2(category_idx_l2_tensor)
            categories_embeddings = categories_embeddings + categories_embeddings_l2
        papers_embeddings = torch.cat((papers_embeddings, categories_embeddings), dim = 1)

        if sorted_unique_user_idx_tensor is not None:
            assert sorted_unique_user_idx_tensor.tolist() == sorted(sorted_unique_user_idx_tensor.tolist())
            sorted_unique_users_embeddings = self.users_embeddings(sorted_unique_user_idx_tensor)
        
        if eval_type == "test":
            return papers_embeddings
        if eval_type == "negative_samples":
            dot_products = torch.matmul(sorted_unique_users_embeddings[:, :-1], papers_embeddings.transpose(0, 1))
            dot_products = dot_products + sorted_unique_users_embeddings[:, -1].unsqueeze(1)
            return dot_products
        if eval_type == "val" or eval_type == "train":
            users_embeddings = self.users_embeddings(user_idx_tensor)
            dot_products = torch.sum(papers_embeddings * users_embeddings[:, :-1], dim = 1) + self.users_embeddings(user_idx_tensor)[:, -1]
            if eval_type == "val":
                return dot_products
            batch_negatives_scores = None
            if batch_negatives_indices is not None:
                batch_negatives_papers = papers_embeddings[batch_negatives_indices]
                batch_negatives_scores = torch.bmm(batch_negatives_papers, sorted_unique_users_embeddings[:, :-1].unsqueeze(-1)).squeeze(-1)
                batch_negatives_scores = batch_negatives_scores + sorted_unique_users_embeddings[:, -1].unsqueeze(1)
            return dot_products, batch_negatives_scores

    def check_devices(self, input_ids_tensor : torch.tensor, attention_mask_tensor : torch.tensor, category_idx_tensor : torch.tensor, 
                            user_idx_tensor : torch.tensor = None, category_idx_l2_tensor : torch.tensor = None, sorted_unique_user_idx_tensor : torch.tensor = None, 
                            batch_negatives_indices : torch.tensor = None) -> tuple:
        if input_ids_tensor.device != self.device:
            input_ids_tensor = input_ids_tensor.to(self.device)
        if attention_mask_tensor.device != self.device:
            attention_mask_tensor = attention_mask_tensor.to(self.device)
        if category_idx_tensor.device != self.device:
            category_idx_tensor = category_idx_tensor.to(self.device)
        if user_idx_tensor is not None:
            if user_idx_tensor.device != self.device:
                user_idx_tensor = user_idx_tensor.to(self.device)
        if category_idx_l2_tensor is not None:
            if category_idx_l2_tensor.device != self.device:
                category_idx_l2_tensor = category_idx_l2_tensor.to(self.device)
        if sorted_unique_user_idx_tensor is not None:
            if sorted_unique_user_idx_tensor.device != self.device:
                sorted_unique_user_idx_tensor = sorted_unique_user_idx_tensor.to(self.device)
        if batch_negatives_indices is not None:
            if batch_negatives_indices.device != self.device:
                batch_negatives_indices = batch_negatives_indices.to(self.device)
        return (input_ids_tensor, attention_mask_tensor, category_idx_tensor, user_idx_tensor, category_idx_l2_tensor, sorted_unique_user_idx_tensor, batch_negatives_indices)
        
    def save_finetuning_model(self, model_path : str) -> None:
        model_path = model_path.rstrip("/") + "/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.transformer_model.save_pretrained(model_path + "transformer_model")
        torch.save(self.projection.state_dict(), model_path + "projection.pt")
        torch.save(self.users_embeddings.state_dict(), model_path + "users_embeddings.pt")
        if self.categories_embeddings is not None:
            torch.save(self.categories_embeddings.state_dict(), model_path + "categories_embeddings.pt")
        if self.categories_embeddings_l2 is not None:
            torch.save(self.categories_embeddings_l2.state_dict(), model_path + "categories_embeddings_l2.pt")

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

    def set_eval_batch_size(self) -> None:
        if self.transformer_model_name.startswith("gte_base"):
            self.eval_batch_size = 1024
        elif self.transformer_model_name.startswith("gte_large"):
            self.eval_batch_size = 512
        else:
            raise ValueError(f"Unknown transformer model name: {self.transformer_model_name}")

    def perform_inference(self, max_batch_size : int, input_ids_tensor : torch.tensor, attention_mask_tensor : torch.tensor, category_idx_tensor : torch.tensor, 
                          user_idx_tensor : torch.tensor = None, category_idx_l2_tensor : torch.tensor = None, eval_type : str = None) -> torch.tensor:
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
            n_tries_so_far, batch_size, successfully_processed = 0, max_batch_size, False
            while n_tries_so_far < MAX_TRIES:
                try:
                    upper_bound = min(n_samples, n_samples_processed + batch_size)
                    batch_input_ids_tensor = input_ids_tensor[n_samples_processed : upper_bound]
                    batch_attention_mask_tensor = attention_mask_tensor[n_samples_processed : upper_bound]
                    batch_category_idx_tensor = category_idx_tensor[n_samples_processed : upper_bound]
                    batch_user_idx_tensor = None
                    if user_idx_tensor is not None:
                        batch_user_idx_tensor = user_idx_tensor[n_samples_processed : upper_bound]
                    batch_category_idx_l2_tensor = None
                    if category_idx_l2_tensor is not None:
                        batch_category_idx_l2_tensor = category_idx_l2_tensor[n_samples_processed : upper_bound]
                    with torch.autocast(device_type = self.device.type, dtype = torch.float16):
                        with torch.inference_mode():
                            results = self(eval_type = eval_type, input_ids_tensor = batch_input_ids_tensor, attention_mask_tensor = batch_attention_mask_tensor, 
                                           category_idx_tensor = batch_category_idx_tensor, user_idx_tensor = batch_user_idx_tensor, 
                                           category_idx_l2_tensor = batch_category_idx_l2_tensor).cpu()
                            gpu_info = get_gpu_info()
                    results_total.append(results)
                    n_samples_processed = min(n_samples, n_samples_processed + batch_size)
                    if batch_num == 1 or batch_num % 10 == 0:
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
        print(f"Finished all Batches in {time.time() - start_time:.2f} seconds. Total Samples: {n_samples_processed}.\n")
        return results_total

    def compute_papers_embeddings(self, input_ids_tensor : torch.tensor, attention_mask_tensor : torch.tensor, category_idx_tensor : torch.tensor, 
                                  category_idx_l2_tensor : torch.tensor = None) -> torch.tensor:
        return self.perform_inference(self.eval_batch_size, input_ids_tensor, attention_mask_tensor, category_idx_tensor, category_idx_l2_tensor = category_idx_l2_tensor, eval_type = "test")

    def compute_val_ratings_scores(self, input_ids_tensor : torch.tensor, attention_mask_tensor : torch.tensor, category_idx_tensor : torch.tensor, user_idx_tensor : torch.tensor,
                                   category_idx_l2_tensor : torch.tensor = None) -> torch.tensor:
        return self.perform_inference(self.eval_batch_size, input_ids_tensor, attention_mask_tensor, category_idx_tensor, user_idx_tensor, category_idx_l2_tensor, eval_type = "val")

    def compute_val_negative_samples_scores(self, input_ids_tensor : torch.tensor, attention_mask_tensor : torch.tensor, category_idx_tensor : torch.tensor, 
                                            category_idx_l2_tensor : torch.tensor = None) -> torch.tensor:
        with torch.autocast(device_type = self.device.type, dtype = torch.float16):
            with torch.inference_mode():
                return self(eval_type = "negative_samples", input_ids_tensor = input_ids_tensor, attention_mask_tensor = attention_mask_tensor,
                            category_idx_tensor = category_idx_tensor, category_idx_l2_tensor = category_idx_l2_tensor, sorted_unique_user_idx_tensor = self.val_users_embeddings_idxs).cpu()

def load_transformer_model(transformer_path : str, device : torch.device) -> AutoModel:
    return AutoModel.from_pretrained(transformer_path, trust_remote_code = True, unpad_inputs = True, torch_dtype = "auto").to(device)

def save_transformer_model(transformer_model : AutoModel, model_path : str) -> None:
    transformer_model.save_pretrained(model_path)

def get_transformer_embeddings_dim(transformer_model : AutoModel) -> int:
    return transformer_model.config.hidden_size

def load_projection(device : torch.device, transformer_embeddings_dim : int = 1024, projection_dim : int = 256, projection_state_dict : dict = None, dtype = torch.float32) -> nn.Linear:
    if projection_state_dict is not None:
        projection_dim, transformer_embeddings_dim = projection_state_dict["weight"].shape
    projection = nn.Linear(transformer_embeddings_dim, projection_dim)
    if projection_state_dict is not None:
        projection.load_state_dict(projection_state_dict)
    projection = projection.to(device, dtype = dtype)
    return projection

def load_embeddings(device : torch.device, num_embeddings : int = None, embeddings_dim : int = None, embeddings_state_dict : dict = None, dtype = torch.float32) -> nn.Embedding:
    if embeddings_state_dict is not None:
        num_embeddings, embeddings_dim = embeddings_state_dict["weight"].shape
    embeddings = nn.Embedding(num_embeddings, embeddings_dim)
    if embeddings_state_dict is not None:
        embeddings.load_state_dict(embeddings_state_dict)
    embeddings = embeddings.to(device, dtype = dtype)
    return embeddings

def load_categories_embeddings_l2(device : torch.device, num_embeddings : int, embeddings_dim : int, dtype = torch.float32) -> nn.Embedding:
    categories_embeddings_l2 = nn.Embedding(num_embeddings, embeddings_dim, dtype=dtype, device=device)
    nn.init.normal_(categories_embeddings_l2.weight, std=0.01)
    return categories_embeddings_l2

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

def load_finetuning_model(finetuning_model_path : str, device : torch.device, val_users_embeddings_idxs : torch.tensor = None, n_unfreeze_layers : int = 4,
                          projection_state_dict : dict = None, users_embeddings_state_dict : dict = None, categories_embeddings_state_dict : dict = None,
                          pretrained_projection : bool = True, pretrained_users_embeddings : bool = True, pretrained_categories_embeddings : bool = True,
                          projection_dim : int = 256, n_users : int = 4111, n_categories : int = 18, categories_embeddings_dim : int = 100,
                          categories_embeddings_l2_state_dict : dict = None, n_categories_l2 : int = 0) -> FinetuningModel:
    finetuning_model_path = finetuning_model_path.rstrip("/") + "/"
    transformer_model_path = finetuning_model_path + "transformer_model"
    fix_gte_config(transformer_model_path)
    transformer_model = load_transformer_model(transformer_model_path, device)
    transformer_embeddings_dim = get_transformer_embeddings_dim(transformer_model)

    if projection_state_dict is None and pretrained_projection:
        projection_state_dict = torch.load(finetuning_model_path + "projection.pt", map_location = device, weights_only = True)
    projection = load_projection(device, transformer_embeddings_dim, projection_dim, projection_state_dict)
    if users_embeddings_state_dict is None and pretrained_users_embeddings:
        users_embeddings_state_dict = torch.load(finetuning_model_path + "users_embeddings.pt", map_location = device, weights_only = True)
    users_embeddings = load_embeddings(device, n_users, transformer_embeddings_dim + categories_embeddings_dim + 1, users_embeddings_state_dict)
    if categories_embeddings_state_dict is None and pretrained_categories_embeddings:
        categories_embeddings_state_dict = torch.load(finetuning_model_path + "categories_embeddings.pt", map_location = device, weights_only = True)
    categories_embeddings = load_embeddings(device, n_categories, categories_embeddings_dim, categories_embeddings_state_dict)
    if n_categories_l2 > 0:
        if categories_embeddings_l2_state_dict is None:
            if os.path.exists(finetuning_model_path + "categories_embeddings_l2.pt"):
                categories_embeddings_l2_state_dict = torch.load(finetuning_model_path + "categories_embeddings_l2.pt", map_location = device, weights_only = True)
                categories_embeddings_l2 = load_embeddings(device, n_categories_l2, categories_embeddings_dim, categories_embeddings_l2_state_dict)
            else:
                categories_embeddings_l2 = load_categories_embeddings_l2(device, n_categories_l2, categories_embeddings_dim, categories_embeddings_l2_state_dict)
        else:
            categories_embeddings_l2 = load_embeddings(device, n_categories_l2, categories_embeddings_dim, categories_embeddings_l2_state_dict)
    else:
        categories_embeddings_l2 = None
    return FinetuningModel(transformer_model, projection, users_embeddings, categories_embeddings, val_users_embeddings_idxs, categories_embeddings_l2, n_unfreeze_layers = n_unfreeze_layers)