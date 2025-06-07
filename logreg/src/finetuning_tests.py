from data_handling import *
from main import get_users_ids
from embedding import Embedding
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
import torch.nn as nn
import time

def are_models_equal(model1 : nn.Module, model2 : nn.Module, rtol = 1e-5, atol = 1e-8) -> bool:
    model1_state_dict, model2_state_dict = model1.state_dict(), model2.state_dict()
    if model1_state_dict.keys() != model2_state_dict.keys():
        return False
    for key in model1_state_dict.keys():
        if model1_state_dict[key].shape != model2_state_dict[key].shape:
            return False
        if not torch.allclose(model1_state_dict[key], model2_state_dict[key], rtol = rtol, atol = atol):
            return False
    return True

def change_first_model_param(model : nn.Module) -> None:
    with torch.no_grad():
        name, param = next(model.named_parameters())
        current_value = param[0, 0].item()
        param[0, 0] = current_value + 1

def load_papers_embeddings(embedding_path : str, papers_ids : list = [5, 10049, 101808]) -> torch.Tensor:
    embedding = Embedding(embedding_path)
    papers_idxs = embedding.get_idxs(papers_ids)
    papers_embeddings = embedding.matrix[papers_idxs]
    return torch.tensor(papers_embeddings, dtype = torch.float32)

def recompute_papers_embeddings(transformer_model : AutoModel, tokenizer : AutoTokenizer, papers_ids : list = [5, 10049, 101808],
                                projection : nn.Linear = None, l2_normalization : bool = False) -> torch.Tensor:
    #device = transformer_model.device
    #training_mode = transformer_model.training
    #if training_mode:
    #    transformer_model.eval()
    papers = get_titles_and_abstracts(papers_ids)
    papers_ids, papers_titles, papers_abstracts = zip(*papers)
    papers_ids, papers_titles, papers_abstracts = list(papers_ids), list(papers_titles), list(papers_abstracts)
    papers = [f"{title} {tokenizer.sep_token} {abstract}" for title, abstract in zip(papers_titles, papers_abstracts)]
    tokenized_papers = tokenizer(text = papers, max_length = 512, padding = True, truncation = True, return_tensors = "pt")
    return tokenized_papers
    with torch.autocast(device_type = device.type, dtype = torch.float16):
        with torch.inference_mode():
            papers_embeddings = transformer_model(**tokenized_papers.to(device))
            papers_embeddings = papers_embeddings.last_hidden_state[:, 0]
    if projection is not None:
        projection_training = projection.training
        if projection_training:
            projection.eval()
        with torch.no_grad():
            papers_embeddings = projection(papers_embeddings)
        if projection_training:
            projection.train()
    if l2_normalization:
        papers_embeddings = nn.functional.normalize(papers_embeddings, p = 2, dim = 1)
    if training_mode:
        transformer_model.train()
    return papers_embeddings.cpu()

def are_papers_embeddings_equal(papers_embeddings1 : torch.Tensor, papers_embeddings2 : torch.Tensor, rtol = 1e-5, atol = 1e-8) -> bool:
    if papers_embeddings1.shape != papers_embeddings2.shape:
        return False
    return torch.allclose(papers_embeddings1, papers_embeddings2, rtol = rtol, atol = atol)


def get_holdout_users_ids(max_users : int = 500, min_n_posrated : int = 20, min_n_negrated : int = 20, random_state : int = 42) -> list:
    users_ids = get_users_ids(users_selection = "random", max_users = max_users, min_n_posrated = min_n_posrated, 
                              min_n_negrated = min_n_negrated, random_state = random_state)
    return users_ids["user_id"].tolist()

def get_relevant_papers_ids(users_ids : list, random_state : int = 42) -> list:
    papers_ids = set(get_global_cache_papers_ids(5000, random_state))
    papers_ids.update(get_negative_samples_ids(100, random_state))
    for user_id in users_ids:
        papers_ids.update(get_rated_papers_ids_for_user(user_id, 1))
        papers_ids.update(get_rated_papers_ids_for_user(user_id, -1))
    return sorted(list(papers_ids))

def get_papers_texts(papers_ids : list, tokenizer : AutoTokenizer) -> list:
    papers = get_titles_and_abstracts()
    _, papers_titles, papers_abstracts = zip(*papers)
    papers_titles, papers_abstracts = list(papers_titles), list(papers_abstracts)
    return [f"{title} {tokenizer.sep_token} {abstract}" for title, abstract in zip(papers_titles, papers_abstracts)]