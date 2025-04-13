from data_handling import get_negative_samples_ids, get_rated_papers_ids_for_user, get_titles_and_abstracts, get_cache_papers_ids_for_user
from main import get_users_ids
from training_data import load_negrated_ranking_ids_for_user
from sklearn.model_selection import train_test_split
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

def save_projection_tensor(pca_components : np.ndarray, pca_mean : np.ndarray, file_name : str = "gte_base_256_projection") -> None:
    file_name = file_name.rstrip(".pt")
    bias = -(pca_mean @ pca_components.T)
    pca_components = torch.from_numpy(pca_components).to(torch.float32)
    pca_bias = torch.from_numpy(bias).to(torch.float32)
    projection = torch.nn.Linear(pca_components.shape[1], pca_components.shape[0], bias = True, dtype = torch.float32)
    with torch.no_grad():
        projection.weight.copy_(pca_components)
        projection.bias.copy_(pca_bias)
    torch.save(projection.state_dict(), f"{FILES_SAVE_PATH}/parameters/{file_name}.pt")

def save_users_embeddings_tensor(train_users_ids : list, users_coefs : np.ndarray, users_ids_to_idxs : dict, file_name : str = "gte_base_256_users_embeddings") -> None:
    file_name = file_name.rstrip(".pt")
    users_embeddings_ids_to_idxs = {}
    num_embeddings, embedding_dim = len(train_users_ids), users_coefs.shape[1]
    users_embeddings = torch.nn.Embedding(num_embeddings, embedding_dim, dtype = torch.float32)
    for idx, user_id in enumerate(train_users_ids):
        with torch.no_grad():
            users_embeddings.weight.data[idx] = torch.from_numpy(users_coefs[users_ids_to_idxs[user_id]]).to(torch.float32)
        users_embeddings_ids_to_idxs[user_id] = idx
    torch.save(users_embeddings.state_dict(), f"{FILES_SAVE_PATH}/parameters/{file_name}.pt")
    with open(f"{FILES_SAVE_PATH}/users/users_embeddings_ids_to_idxs.pkl", "wb") as f:
        pickle.dump(users_embeddings_ids_to_idxs, f)

def load_users_embeddings_ids_to_idxs() -> dict:
    with open(f"{FILES_SAVE_PATH}/users/users_embeddings_ids_to_idxs.pkl", "rb") as f:
        users_embeddings_ids_to_idxs = pickle.load(f)
    return users_embeddings_ids_to_idxs

def save_finetuning_users_ids(n_val_users : int = 500, n_test_users_no_overlap : int = 500, min_n_posrated : int = 20, min_n_negrated : int = 20, random_state : int = 42) -> None:
    all_users = get_users_ids(users_selection = "random", max_users = None, min_n_posrated = min_n_posrated, min_n_negrated = min_n_negrated)
    val_users = get_users_ids("random", n_test_users_no_overlap, min_n_posrated, min_n_negrated, random_state = random_state, take_complement = True)
    val_users = val_users.sort_values(by = "user_id")
    val_users = val_users.sample(n = n_val_users, random_state = random_state)
    test_users_no_overlap = get_users_ids("random", n_test_users_no_overlap, min_n_posrated, min_n_negrated, random_state = random_state, take_complement = False)
    val_users, test_users_no_overlap = sorted(val_users["user_id"].tolist()), sorted(test_users_no_overlap["user_id"].tolist())
    train_users = sorted(list(set(all_users["user_id"].tolist()) - set(test_users_no_overlap)))
    with open(f"{FILES_SAVE_PATH}/users/train_users_ids.pkl", "wb") as train_file:
        pickle.dump(train_users, train_file)
    with open(f"{FILES_SAVE_PATH}/users/val_users_ids.pkl", "wb") as val_file:
        pickle.dump(val_users, val_file)
    with open(f"{FILES_SAVE_PATH}/users/test_users_no_overlap_ids.pkl", "wb") as test_file:
        pickle.dump(test_users_no_overlap, test_file)

def load_finetuning_users_ids() -> tuple:
    with open(f"{FILES_SAVE_PATH}/users/train_users_ids.pkl", "rb") as train_file:
        train_users_ids = pickle.load(train_file)
    with open(f"{FILES_SAVE_PATH}/users/val_users_ids.pkl", "rb") as val_file:
        val_users_ids = pickle.load(val_file)
    with open(f"{FILES_SAVE_PATH}/users/test_users_no_overlap_ids.pkl", "rb") as test_file:
        test_users_no_overlap_ids = pickle.load(test_file)
    return train_users_ids, val_users_ids, test_users_no_overlap_ids

def tokenize_papers(papers_ids : list, tokenizer : AutoTokenizer, max_sequence_length : int = 512) -> tuple:
    papers = get_titles_and_abstracts(papers_ids)
    papers_ids, papers_titles, papers_abstracts = zip(*papers)
    papers_ids, papers_titles, papers_abstracts = list(papers_ids), list(papers_titles), list(papers_abstracts)
    assert papers_ids == sorted(papers_ids)
    papers = [f"{title} {tokenizer.sep_token} {abstract}" for title, abstract in zip(papers_titles, papers_abstracts)]
    input_ids_list, attention_mask_list = [], []
    for i in range(len(papers)):
        encoding = tokenizer(papers[i], max_length = max_sequence_length, padding = "max_length", truncation = True, return_tensors = "pt")
        input_ids_list.append(encoding["input_ids"].squeeze(0))
        attention_mask_list.append(encoding["attention_mask"].squeeze(0))
    papers_dict = {"paper_id": torch.tensor(papers_ids), "input_ids": torch.stack(input_ids_list), "attention_mask": torch.stack(attention_mask_list)}
    papers_ids_to_idxs = {}
    for idx, paper_id in enumerate(papers_ids):
        papers_ids_to_idxs[paper_id] = idx
    return papers_dict, papers_ids_to_idxs

def save_test_papers(val_users_ids : list, test_users_no_overlap_ids : list, tokenizer : AutoTokenizer, max_sequence_length : int = 512, 
                     n_cache : int = 5000, n_negative_samples : int = 100, random_state : int = 42) -> None:
    negative_samples_ids = set(get_negative_samples_ids(n_negative_samples, random_state))
    users_ids = val_users_ids + test_users_no_overlap_ids
    assert len(users_ids) == len(set(users_ids))
    test_papers_ids = set()
    for user_id in users_ids:
        test_papers_ids.update(get_rated_papers_ids_for_user(user_id, 1))
        test_papers_ids.update(get_rated_papers_ids_for_user(user_id, -1))
        test_papers_ids.update(get_cache_papers_ids_for_user(user_id, n_cache, random_state))
    test_papers_ids = sorted(list(negative_samples_ids | test_papers_ids))
    test_papers, _ = tokenize_papers(test_papers_ids, tokenizer, max_sequence_length)
    torch.save(test_papers, FILES_SAVE_PATH + "/datasets/test_papers.pt")

def load_test_papers() -> dict:
    return torch.load(FILES_SAVE_PATH + "/datasets/test_papers.pt", weights_only = True)

def tokenize_train_val_papers(tokenizer : AutoTokenizer, train_users_ids : list, max_sequence_length : int = 512, n_negative_samples : int = 100, random_state : int = 42) -> None:
    papers_ids = set(get_negative_samples_ids(n_negative_samples, random_state))
    for user_id in train_users_ids:
        papers_ids.update(get_rated_papers_ids_for_user(user_id, 1))
        papers_ids.update(get_rated_papers_ids_for_user(user_id, -1))
    papers_ids = sorted(list(papers_ids))
    papers_dict, papers_ids_to_idxs = tokenize_papers(papers_ids, tokenizer, max_sequence_length)
    torch.save(papers_dict, FILES_SAVE_PATH + "/datasets/train_val_papers.pt")
    with open(f"{FILES_SAVE_PATH}/datasets/train_val_papers_ids_to_idxs.pkl", "wb") as f:
        pickle.dump(papers_ids_to_idxs, f)

def load_train_val_papers() -> tuple:
    train_val_papers = torch.load(FILES_SAVE_PATH + "/datasets/train_val_papers.pt", weights_only = True)
    with open(f"{FILES_SAVE_PATH}/datasets/train_val_papers_ids_to_idxs.pkl", "rb") as f:
        train_val_papers_ids_to_idxs = pickle.load(f)
    return train_val_papers, train_val_papers_ids_to_idxs

def save_negative_samples(tokenizer : AutoTokenizer, n_negative_samples : int = 100, random_state : int = 42) -> None:
    negative_samples_ids = get_negative_samples_ids(n_negative_samples, random_state)
    negative_samples_ids = sorted(list(negative_samples_ids))
    negative_samples = tokenize_papers(negative_samples_ids, tokenizer)[0]
    torch.save(negative_samples, FILES_SAVE_PATH + "/datasets/negative_samples.pt")

def load_negative_samples() -> dict:
    return torch.load(FILES_SAVE_PATH + "/datasets/negative_samples.pt", weights_only = True)

def save_val_dataset(val_users_ids : list, train_val_papers : dict, train_val_papers_ids_to_idxs : dict, random_state : int = 42) -> None:
    user_id_list, paper_id_list, label_list, input_ids_list, attention_mask_list = [], [], [], [], []
    for user_id in val_users_ids:
        posrated_ids, negrated_ids = get_rated_papers_ids_for_user(user_id, 1), get_rated_papers_ids_for_user(user_id, -1)
        posrated_n, negrated_n = len(posrated_ids), len(negrated_ids)
        rated_ids, rated_labels = posrated_ids + negrated_ids, np.concatenate((np.ones(posrated_n, dtype = np.int32), np.zeros(negrated_n, dtype = np.int32)))
        train_rated_ids, val_rated_ids, y_train_rated, y_val = train_test_split(rated_ids, rated_labels, test_size = 0.2, random_state = random_state, stratify = rated_labels)
        posrated_validation_ids, negrated_validation_ids = [id for id in val_rated_ids if id in posrated_ids], [id for id in val_rated_ids if id in negrated_ids]
        negrated_ranking_ids = load_negrated_ranking_ids_for_user(negrated_validation_ids, random_state)
        papers_ids = posrated_validation_ids + negrated_ranking_ids
        user_id_list += [user_id] * len(papers_ids)
        papers_idxs = [train_val_papers_ids_to_idxs[paper_id] for paper_id in papers_ids]
        paper_id_list += papers_ids
        label_list += [1] * len(posrated_validation_ids) + [0] * len(negrated_ranking_ids)
        input_ids_list += [train_val_papers["input_ids"][idx] for idx in papers_idxs]
        attention_mask_list += [train_val_papers["attention_mask"][idx] for idx in papers_idxs]
    user_id_list = torch.tensor(user_id_list)
    paper_id_list = torch.tensor(paper_id_list)
    label_list = torch.tensor(label_list)
    input_ids_list = torch.stack(input_ids_list)
    attention_mask_list = torch.stack(attention_mask_list)
    val_dataset = {"user_id": user_id_list, "paper_id": paper_id_list, "label": label_list, "input_ids": input_ids_list, "attention_mask": attention_mask_list}
    torch.save(val_dataset, FILES_SAVE_PATH + "/datasets/val_dataset.pt")

def load_val_dataset() -> dict:
    val_dataset = torch.load(FILES_SAVE_PATH + "/datasets/val_dataset.pt", weights_only = True)
    return val_dataset

def save_train_dataset(train_users_ids : list, train_val_papers : dict, train_val_papers_ids_to_idxs : dict, random_state : int = 42) -> None:
    user_id_list, paper_id_list, label_list, input_ids_list, attention_mask_list = [], [], [], [], []
    for user_id in train_users_ids:
        posrated_ids, negrated_ids = get_rated_papers_ids_for_user(user_id, 1), get_rated_papers_ids_for_user(user_id, -1)
        posrated_n, negrated_n = len(posrated_ids), len(negrated_ids)
        rated_ids, rated_labels = posrated_ids + negrated_ids, np.concatenate((np.ones(posrated_n, dtype = np.int32), np.zeros(negrated_n, dtype = np.int32)))
        train_rated_ids, val_rated_ids, y_train_rated, y_val = train_test_split(rated_ids, rated_labels, test_size = 0.2, random_state = random_state, stratify = rated_labels)
        posrated_training_ids, negrated_training_ids = [id for id in train_rated_ids if id in posrated_ids], [id for id in train_rated_ids if id in negrated_ids]
        papers_ids = posrated_training_ids + negrated_training_ids
        user_id_list += [user_id] * len(papers_ids)
        papers_idxs = [train_val_papers_ids_to_idxs[paper_id] for paper_id in papers_ids]
        paper_id_list += papers_ids
        label_list += [1] * len(posrated_training_ids) + [0] * len(negrated_training_ids)
        input_ids_list += [train_val_papers["input_ids"][idx] for idx in papers_idxs]
        attention_mask_list += [train_val_papers["attention_mask"][idx] for idx in papers_idxs]
    user_id_list = torch.tensor(user_id_list)
    paper_id_list = torch.tensor(paper_id_list)
    label_list = torch.tensor(label_list)
    input_ids_list = torch.stack(input_ids_list)
    attention_mask_list = torch.stack(attention_mask_list)
    train_dataset = {"user_id": user_id_list, "paper_id": paper_id_list, "label": label_list, "input_ids": input_ids_list, "attention_mask": attention_mask_list}
    torch.save(train_dataset, FILES_SAVE_PATH + "/datasets/train_dataset.pt")

def load_train_dataset() -> dict:
    train_dataset = torch.load(FILES_SAVE_PATH + "/datasets/train_dataset.pt", weights_only = True)
    return train_dataset

def load_val_users_embeddings_idxs(val_users_ids : list, users_embeddings_ids_to_idxs : dict) -> torch.tensor:
    val_users_embeddings_idxs = [users_embeddings_ids_to_idxs[user_id] for user_id in val_users_ids]
    return torch.tensor(val_users_embeddings_idxs, dtype = torch.int64)