from data_handling import get_negative_samples_ids, get_rated_papers_ids_for_user, get_titles_and_abstracts, get_cache_papers_ids_for_user
from main import get_users_ids
from transformers import AutoTokenizer
import json
import numpy as np
import os
import pickle
import random
import torch

GTE_BASE_PATH = "Alibaba-NLP/gte-base-en-v1.5"
GTE_LARGE_PATH = "Alibaba-NLP/gte-large-en-v1.5"
FILES_SAVE_PATH = "/home/scholar/glenn_rp/msc_thesis/data/finetuning"

def save_validation_users_ids(n_overlap_users : int = 500, n_no_overlap_users : int = 500, min_n_posrated : int = 20, min_n_negrated : int = 20, random_state : int = 42) -> None:
    overlap_users = get_users_ids(users_selection = "random", max_users = n_no_overlap_users, min_n_posrated = min_n_posrated, min_n_negrated = min_n_negrated, random_state = random_state, take_complement = True)
    overlap_users = overlap_users.sort_values(by = "user_id")
    overlap_users = overlap_users.sample(n = n_overlap_users, random_state = random_state)
    no_overlap_users = get_users_ids(users_selection = "random", max_users = n_no_overlap_users, min_n_posrated = min_n_posrated, min_n_negrated = min_n_negrated, random_state = random_state)
    overlap_users, no_overlap_users = sorted(overlap_users["user_id"].tolist()), sorted(no_overlap_users["user_id"].tolist())
    with open(f"{FILES_SAVE_PATH}/overlap_users_ids.pkl", "wb") as overlap_file:
        pickle.dump(overlap_users, overlap_file)
    with open(f"{FILES_SAVE_PATH}/no_overlap_users_ids.pkl", "wb") as no_overlap_file:
        pickle.dump(no_overlap_users, no_overlap_file)

def load_overlap_users_ids() -> list:
    with open(f"{FILES_SAVE_PATH}/overlap_users_ids.pkl", "rb") as overlap_file:
        overlap_users_ids = pickle.load(overlap_file)
    return overlap_users_ids

def load_no_overlap_users_ids() -> list:
    with open(f"{FILES_SAVE_PATH}/no_overlap_users_ids.pkl", "rb") as no_overlap_file:
        no_overlap_users_ids = pickle.load(no_overlap_file)
    return no_overlap_users_ids

def save_validation_papers_ids(overlap_users_ids : list, no_overlap_users_ids : list, n_cache : int = 5000, n_negative_samples : int = 100, 
                               random_state : int = 42) -> None:
    negative_samples_ids = set(get_negative_samples_ids(n_negative_samples, random_state))
    overlap_papers_ids, no_overlap_papers_ids = set(), set()
    for user_id in overlap_users_ids:
        overlap_papers_ids.update(get_rated_papers_ids_for_user(user_id, 1))
        overlap_papers_ids.update(get_rated_papers_ids_for_user(user_id, -1))
        overlap_papers_ids.update(get_cache_papers_ids_for_user(user_id, n_cache, random_state))
    for user_id in no_overlap_users_ids:
        no_overlap_papers_ids.update(get_rated_papers_ids_for_user(user_id, 1))
        no_overlap_papers_ids.update(get_rated_papers_ids_for_user(user_id, -1))
        no_overlap_papers_ids.update(get_cache_papers_ids_for_user(user_id, n_cache, random_state))
    overlap_papers_ids = sorted(list(negative_samples_ids | overlap_papers_ids))
    no_overlap_papers_ids = sorted(list(negative_samples_ids | no_overlap_papers_ids))
    validation_papers_ids = sorted(list(set(overlap_papers_ids) | set(no_overlap_papers_ids)))
    with open(f"{FILES_SAVE_PATH}/overlap_papers_ids.pkl", "wb") as overlap_file:
        pickle.dump(overlap_papers_ids, overlap_file)
    with open(f"{FILES_SAVE_PATH}/no_overlap_papers_ids.pkl", "wb") as no_overlap_file:
        pickle.dump(no_overlap_papers_ids, no_overlap_file)
    with open(f"{FILES_SAVE_PATH}/validation_papers_ids.pkl", "wb") as validation_file:
        pickle.dump(validation_papers_ids, validation_file)

def load_overlap_papers_ids() -> list:
    with open(f"{FILES_SAVE_PATH}/overlap_papers_ids.pkl", "rb") as overlap_file:
        overlap_papers_ids = pickle.load(overlap_file)
    return overlap_papers_ids

def load_no_overlap_papers_ids() -> list:
    with open(f"{FILES_SAVE_PATH}/no_overlap_papers_ids.pkl", "rb") as no_overlap_file:
        no_overlap_papers_ids = pickle.load(no_overlap_file)
    return no_overlap_papers_ids

def load_validation_papers_ids() -> list:
    with open(f"{FILES_SAVE_PATH}/validation_papers_ids.pkl", "rb") as validation_file:
        validation_papers_ids = pickle.load(validation_file)
    return validation_papers_ids

def save_papers_tokenized_individually(file_path : str, papers_ids : list, tokenizer : AutoTokenizer, max_sequence_length : int = 512) -> None:
    papers = get_titles_and_abstracts(papers_ids)
    papers_ids, papers_titles, papers_abstracts = zip(*papers)
    papers_ids, papers_titles, papers_abstracts = list(papers_ids), list(papers_titles), list(papers_abstracts)
    papers = [f"{title} {tokenizer.sep_token} {abstract}" for title, abstract in zip(papers_titles, papers_abstracts)]
    input_ids, attention_masks = [], []
    for i in range(len(papers)):
        encoding = tokenizer(papers[i], max_length = max_sequence_length, padding = "max_length", truncation = True, return_tensors = "pt")
        input_ids.append(encoding["input_ids"].squeeze(0))
        attention_masks.append(encoding["attention_mask"].squeeze(0))
    papers_dict = {"papers_ids": torch.tensor(papers_ids), "input_ids": torch.stack(input_ids), "attention_masks": torch.stack(attention_masks)}
    torch.save(papers_dict, file_path)

def save_validation_papers_tokenized(overlap_papers_ids : list, no_overlap_papers_ids : list, validation_papers_ids : list, 
                                     tokenizer : AutoTokenizer, max_sequence_length : int = 512) -> None:
    save_papers_tokenized_individually(FILES_SAVE_PATH + "/overlap_papers_tokenized.pt", overlap_papers_ids, tokenizer, max_sequence_length)
    save_papers_tokenized_individually(FILES_SAVE_PATH + "/no_overlap_papers_tokenized.pt", no_overlap_papers_ids, tokenizer, max_sequence_length)
    save_papers_tokenized_individually(FILES_SAVE_PATH + "/validation_papers_tokenized.pt", validation_papers_ids, tokenizer, max_sequence_length)

def load_overlap_papers_tokenized() -> dict:
    return torch.load(FILES_SAVE_PATH + "/overlap_papers_tokenized.pt", weights_only = True)

def load_no_overlap_papers_tokenized() -> dict:
    return torch.load(FILES_SAVE_PATH + "/no_overlap_papers_tokenized.pt", weights_only = True)

def load_validation_papers_tokenized() -> dict:
    return torch.load(FILES_SAVE_PATH + "/validation_papers_tokenized.pt", weights_only = True)

def save_early_stopping_papers_tokenized(overlap_users_ids : list, tokenizer : AutoTokenizer, n_negatively_rated : int = 4, n_negative_samples : int = 100,
                                         max_sequence_length : int = 512, random_state : 42) -> None:
    early_stopping_papers_dict = {}
    negative_samples_ids = get_negative_samples_ids(n_negative_samples, random_state)
    early_stopping_papers_dict["negative_samples_ids"] = negative_samples_ids
    early_stopping_papers_ids = set(negative_samples_ids)
    for user_id in overlap_users_ids:
        positively_rated_papers_ids = get_rated_papers_ids_for_user(user_id, 1)
        early_stopping_papers_ids.update(get_rated_papers_ids_for_user(user_id, 1))
        negatively_rated_papers_ids = np.array([get_rated_papers_ids_for_user(user_id, -1)])
        np.random.seed(random_state)
        np.random.shuffle(negatively_rated_papers_ids)
        negatively_rated_papers_ids = negatively_rated_papers_ids[:n_negatively_rated].tolist()
        early_stopping_papers_dict[user_id] = {"pos": sorted(positively_rated_papers_ids), "neg": sorted(negatively_rated_papers_ids)}
        early_stopping_papers_ids.update(set(positively_rated_papers_ids))
        early_stopping_papers_ids.update(set(negatively_rated_papers_ids))
    early_stopping_papers_ids = sorted(list(early_stopping_papers_ids))
    save_papers_tokenized_individually(FILES_SAVE_PATH + "/early_stopping_papers_tokenized.pt", early_stopping_papers_ids, tokenizer, max_sequence_length)
    with open(f"{FILES_SAVE_PATH}/early_stopping_papers_dict.pkl", "wb") as early_stopping_file:
        pickle.dump(early_stopping_papers_dict, early_stopping_file)
    
def save_projection_tensor(pca_components : np.ndarray, pca_mean : np.ndarray, file_path : str) -> None:
    if type(pca_components) == str:
        pca_components = np.load(pca_components)
    if type(pca_mean) == str:
        pca_mean = np.load(pca_mean)
    bias = -(pca_mean @ pca_components.T)
    pca_components = torch.from_numpy(pca_components).to(torch.float32)
    pca_bias = torch.from_numpy(bias).to(torch.float32)
    projection = torch.nn.Linear(pca_components.shape[1], pca_components.shape[0], bias = True, dtype = torch.float32)
    with torch.no_grad():
        projection.weight.copy_(pca_components)
        projection.bias.copy_(pca_bias)
    torch.save(projection.state_dict(), file_path)