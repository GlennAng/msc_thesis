from data_handling import get_negative_samples_ids, get_rated_papers_ids_for_user, get_titles_and_abstracts, get_cache_papers_ids_for_user
from main import get_users_ids
from sklearn.model_selection import train_test_split
from training_data import load_negrated_ranking_ids_for_user
from transformers import AutoTokenizer
import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
from tqdm import tqdm

GTE_BASE_PATH = "Alibaba-NLP/gte-base-en-v1.5"
GTE_LARGE_PATH = "Alibaba-NLP/gte-large-en-v1.5"
FILES_SAVE_PATH = "/home/scholar/glenn_rp/msc_thesis/data/finetuning"
TRANSFORMER_MODEL_NAME = "gte_large_256"
USERS_SELECTION_RANDOM_STATE = 42
TRAIN_TEST_SPLIT_RANDOM_STATE = 42
VALIDATION_RANDOM_STATE = 42
TESTING_NO_OVERLAP_RANDOM_STATES = [1, 2, 25, 26, 75, 76, 100, 101, 150, 151]



def save_users_embeddings_tensor(train_users_ids : list, users_coefs : np.ndarray, users_ids_to_idxs : dict, save_users_embeddings_ids_to_idxs : bool = True) -> None:
    users_embeddings_ids_to_idxs = {}
    num_embeddings, embedding_dim = len(train_users_ids), users_coefs.shape[1]
    users_embeddings = torch.nn.Embedding(num_embeddings, embedding_dim, dtype = torch.float32)
    for idx, user_id in enumerate(train_users_ids):
        with torch.no_grad():
            users_embeddings.weight.data[idx] = torch.from_numpy(users_coefs[users_ids_to_idxs[user_id]]).to(torch.float32)
        users_embeddings_ids_to_idxs[user_id] = idx
    users_embeddings_path = f"{FILES_SAVE_PATH}/{TRANSFORMER_MODEL_NAME}/state_dicts/users_embeddings.pt"
    torch.save(users_embeddings.state_dict(), users_embeddings_path)
    if save_users_embeddings_ids_to_idxs:
        with open(f"{FILES_SAVE_PATH}/users/users_embeddings_ids_to_idxs.pkl", "wb") as f:
            pickle.dump(users_embeddings_ids_to_idxs, f)

def load_users_embeddings_ids_to_idxs() -> dict:
    with open(f"{FILES_SAVE_PATH}/users/users_embeddings_ids_to_idxs.pkl", "rb") as f:
        users_embeddings_ids_to_idxs = pickle.load(f)
    assert list(users_embeddings_ids_to_idxs.values()) == sorted(list(users_embeddings_ids_to_idxs.values()))
    return users_embeddings_ids_to_idxs

def save_categories_embeddings(papers_ids_to_categories : dict = None, glove_categories_embeddings : dict = None) -> None:
    if papers_ids_to_categories is None:
        from papers_categories import load_papers_ids_to_categories
        papers_ids_to_categories = load_papers_ids_to_categories(level = "l1")
    if glove_categories_embeddings is None:
        from papers_categories import get_glove_categories_embeddings_l1
        glove_categories_embeddings = get_glove_categories_embeddings_l1()
    keys_sorted = sorted([category for category in glove_categories_embeddings.keys() if category is not None])
    if None in glove_categories_embeddings:
        keys_sorted = [None] + keys_sorted
    n_embeddings, dim = len(glove_categories_embeddings), len(glove_categories_embeddings[keys_sorted[0]])
    categories_embeddings = torch.nn.Embedding(n_embeddings, dim, dtype = torch.float32)
    for idx, category in enumerate(keys_sorted):
        with torch.no_grad():
            categories_embeddings.weight.data[idx] = torch.from_numpy(glove_categories_embeddings[category]).to(torch.float32)
    categories_to_idxs = {category : idx for idx, category in enumerate(keys_sorted)}
    papers_ids_to_categories_idxs = {paper_id : categories_to_idxs[papers_ids_to_categories[paper_id]] for paper_id in papers_ids_to_categories}
    torch.save(categories_embeddings.state_dict(), f"{FILES_SAVE_PATH}/{TRANSFORMER_MODEL_NAME}/state_dicts/categories_embeddings.pt")
    with open(f"{FILES_SAVE_PATH}/{TRANSFORMER_MODEL_NAME}/state_dicts/categories_to_idxs.pkl", "wb") as f:
        pickle.dump(categories_to_idxs, f)
    with open(f"{FILES_SAVE_PATH}/{TRANSFORMER_MODEL_NAME}/state_dicts/papers_ids_to_categories_idxs.pkl", "wb") as f:
        pickle.dump(papers_ids_to_categories_idxs, f)

def load_categories_to_idxs() -> dict:
    with open(f"{FILES_SAVE_PATH}/{TRANSFORMER_MODEL_NAME}/state_dicts/categories_to_idxs.pkl", "rb") as f:
        categories_to_idxs = pickle.load(f)
    assert list(categories_to_idxs.values()) == sorted(list(categories_to_idxs.values()))
    return categories_to_idxs

def load_papers_ids_to_categories_idxs() -> dict:
    with open(f"{FILES_SAVE_PATH}/{TRANSFORMER_MODEL_NAME}/state_dicts/papers_ids_to_categories_idxs.pkl", "rb") as f:
        papers_ids_to_categories_idxs = pickle.load(f)
    assert list(papers_ids_to_categories_idxs.keys()) == sorted(list(papers_ids_to_categories_idxs.keys()))
    return papers_ids_to_categories_idxs

def tokenize_papers(papers_ids : list, tokenizer : AutoTokenizer, max_sequence_length : int = 512) -> tuple:
    papers = get_titles_and_abstracts(papers_ids)
    papers_ids, papers_titles, papers_abstracts = zip(*papers)
    papers_ids, papers_titles, papers_abstracts = list(papers_ids), list(papers_titles), list(papers_abstracts)
    assert papers_ids == sorted(papers_ids)
    papers = [f"{title} {tokenizer.sep_token} {abstract}" for title, abstract in zip(papers_titles, papers_abstracts)]
    input_ids_list, attention_mask_list = [], []
    for i in tqdm(range(len(papers)), desc = "Tokenizing Papers", unit = "Paper"):
        encoding = tokenizer(papers[i], max_length = max_sequence_length, padding = "max_length", truncation = True, return_tensors = "pt")
        input_ids_list.append(encoding["input_ids"].squeeze(0))
        attention_mask_list.append(encoding["attention_mask"].squeeze(0))
    papers_dict = {"paper_id": torch.tensor(papers_ids), "input_ids": torch.stack(input_ids_list), "attention_mask": torch.stack(attention_mask_list)}
    papers_ids_to_idxs = {paper_id: idx for idx, paper_id in enumerate(papers_ids)}
    return papers_dict, papers_ids_to_idxs

def save_train_val_papers(papers_type : str, tokenizer : AutoTokenizer, users_ids : list, max_sequence_length : int = 512) -> None:
    valid_papers_types = ["train", "val"]
    if papers_type not in valid_papers_types:
        raise ValueError(f"Invalid papers type. Choose from {valid_papers_types}.")
    papers_ids = set()
    for user_id in tqdm(users_ids, desc = f"Getting {papers_type.capitalize()} Papers", unit = f"{papers_type.capitalize()} User"):
        papers_ids.update(get_rated_papers_ids_for_user(user_id, 1))
        papers_ids.update(get_rated_papers_ids_for_user(user_id, -1))
    papers_ids = sorted(list(papers_ids))
    papers_dict, papers_ids_to_idxs = tokenize_papers(papers_ids, tokenizer, max_sequence_length)
    torch.save(papers_dict, FILES_SAVE_PATH + f"/datasets/{papers_type}_papers.pt")
    with open(f"{FILES_SAVE_PATH}/datasets/{papers_type}_papers_ids_to_idxs.pkl", "wb") as f:
        pickle.dump(papers_ids_to_idxs, f)

def save_test_papers(val_users_ids : list, test_users_no_overlap_ids : list, tokenizer : AutoTokenizer, max_sequence_length : int = 512, n_cache : int = 5000, n_cache_attached : int = 5000,
                     n_negative_samples : int = 100, validation_random_state : int = VALIDATION_RANDOM_STATE, testing_no_overlap_random_states : list = TESTING_NO_OVERLAP_RANDOM_STATES) -> None:
    assert validation_random_state not in testing_no_overlap_random_states
    negative_samples_ids = get_negative_samples_ids(n_negative_samples, validation_random_state)
    cache_attached_papers_ids = get_negative_samples_ids(n_cache_attached, validation_random_state, papers_to_exclude = negative_samples_ids)
    test_papers_ids = set(negative_samples_ids + cache_attached_papers_ids)
    for random_state in testing_no_overlap_random_states:
        negative_samples_ids = get_negative_samples_ids(n_negative_samples, random_state)
        cache_attached_papers_ids = get_negative_samples_ids(n_cache_attached, random_state, papers_to_exclude = negative_samples_ids)
        test_papers_ids.update(negative_samples_ids + cache_attached_papers_ids)
    for val_user_id in tqdm(val_users_ids, desc = "Getting Test Papers for Val Users", unit = "Val User"):
        test_papers_ids.update(get_rated_papers_ids_for_user(val_user_id, 1))
        test_papers_ids.update(get_rated_papers_ids_for_user(val_user_id, -1))
        test_papers_ids.update(get_cache_papers_ids_for_user(val_user_id, n_cache, validation_random_state))
    for test_user_no_overlap_id in tqdm(test_users_no_overlap_ids, desc = "Getting Test Papers for Test Users No Overlap", unit = "Test User No Overlap"):
        test_papers_ids.update(get_rated_papers_ids_for_user(test_user_no_overlap_id, 1))
        test_papers_ids.update(get_rated_papers_ids_for_user(test_user_no_overlap_id, -1))
        for random_state in testing_no_overlap_random_states:
            test_papers_ids.update(get_cache_papers_ids_for_user(test_user_no_overlap_id, n_cache, random_state))
    test_papers_ids = sorted(list(test_papers_ids))
    assert len(test_papers_ids) == len(set(test_papers_ids))
    test_papers, _ = tokenize_papers(test_papers_ids, tokenizer, max_sequence_length)
    torch.save(test_papers, FILES_SAVE_PATH + "/datasets/test_papers.pt")

def save_val_negative_samples(tokenizer : AutoTokenizer, n_negative_samples : int = 100, random_state : int = VALIDATION_RANDOM_STATE) -> None:
    negative_samples_ids = get_negative_samples_ids(n_negative_samples, random_state)
    negative_samples_ids = sorted(list(negative_samples_ids))
    negative_samples = tokenize_papers(negative_samples_ids, tokenizer)[0]
    torch.save(negative_samples, FILES_SAVE_PATH + "/datasets/val_negative_samples.pt")

def save_papers(papers_type : str, tokenizer : AutoTokenizer, max_sequence_length : int = 512) -> None:
    valid_papers_types = ["train", "val", "test", "val_negative_samples"]
    if papers_type not in valid_papers_types:
        raise ValueError(f"Invalid papers type. Choose from {valid_papers_types}.")
    if papers_type == "val_negative_samples":
        save_val_negative_samples(tokenizer, max_sequence_length)
    else:
        train_users_ids, val_users_ids, test_users_no_overlap_ids = load_finetuning_users_ids()
        if papers_type == "train":
            save_train_val_papers("train", tokenizer, train_users_ids, max_sequence_length)
        elif papers_type == "val":
            save_train_val_papers("val", tokenizer, val_users_ids, max_sequence_length)
        elif papers_type == "test":
            save_test_papers(val_users_ids, test_users_no_overlap_ids, tokenizer, max_sequence_length)

def load_papers(path : str, papers_ids_to_categories_idxs : dict = None, papers_ids_to_categories_idxs_l2 : dict = None) -> dict:
    if papers_ids_to_categories_idxs is None:
        assert papers_ids_to_categories_idxs_l2 is None
    if path == "train":
        path = FILES_SAVE_PATH + "/datasets/train_papers.pt"
    elif path == "val":
        path = FILES_SAVE_PATH + "/datasets/val_papers.pt"
    elif path == "test":
        path = FILES_SAVE_PATH + "/datasets/test_papers.pt"
    elif path == "val_negative_samples":
        path = FILES_SAVE_PATH + "/datasets/val_negative_samples.pt"
    papers = torch.load(path, weights_only = True)
    if papers_ids_to_categories_idxs is not None:
        papers["category_idx"] = torch.tensor([papers_ids_to_categories_idxs[paper_id.item()] for paper_id in papers["paper_id"]], dtype = torch.int64)
    if papers_ids_to_categories_idxs_l2 is not None:
        papers["category_idx_l2"] = torch.tensor([papers_ids_to_categories_idxs_l2[paper_id.item()] for paper_id in papers["paper_id"]], dtype = torch.int64)
    return papers

def load_papers_ids_to_idxs(path : str) -> dict:
    if path == "train":
        path = FILES_SAVE_PATH + "/datasets/train_papers_ids_to_idxs.pkl"
    elif path == "val":
        path = FILES_SAVE_PATH + "/datasets/val_papers_ids_to_idxs.pkl"
    with open(path, "rb") as f:
        papers_ids_to_idxs = pickle.load(f)
    return papers_ids_to_idxs

def save_train_val_dataset(papers_type : str, users_ids : list, papers : dict, papers_ids_to_idxs : dict, random_state : int = TRAIN_TEST_SPLIT_RANDOM_STATE) -> None:
    valid_papers_types = ["train", "val"]
    if papers_type not in valid_papers_types:
        raise ValueError(f"Invalid papers type. Choose from {valid_papers_types}.")
    user_id_list, paper_id_list, label_list, input_ids_list, attention_mask_list = [], [], [], [], []
    for user_id in tqdm(users_ids, desc = f"Getting {papers_type.capitalize()} Dataset", unit = f"{papers_type.capitalize()} User"):
        posrated_ids, negrated_ids = get_rated_papers_ids_for_user(user_id, 1), get_rated_papers_ids_for_user(user_id, -1)
        posrated_n, negrated_n = len(posrated_ids), len(negrated_ids)
        rated_ids, rated_labels = posrated_ids + negrated_ids, np.concatenate((np.ones(posrated_n, dtype = np.int32), np.zeros(negrated_n, dtype = np.int32)))
        train_rated_ids, val_rated_ids, y_train_rated, y_val = train_test_split(rated_ids, rated_labels, test_size = 0.2, random_state = random_state, stratify = rated_labels)
        relevant_rated_ids = train_rated_ids if papers_type == "train" else val_rated_ids
        relevant_posrated_ids, relevant_negrated_ids = [id for id in relevant_rated_ids if id in posrated_ids], [id for id in relevant_rated_ids if id in negrated_ids]
        if papers_type == "val":
            negrated_ranking_ids = load_negrated_ranking_ids_for_user(relevant_negrated_ids, random_state)
            non_negrated_ranking_ids = [id for id in relevant_negrated_ids if id not in negrated_ranking_ids]
            papers_ids = relevant_posrated_ids + non_negrated_ranking_ids + negrated_ranking_ids
        else:
            papers_ids = relevant_posrated_ids + relevant_negrated_ids
        user_id_list += [user_id] * len(papers_ids)
        paper_id_list += papers_ids
        papers_idxs = [papers_ids_to_idxs[paper_id] for paper_id in papers_ids]
        label_list += [1] * len(relevant_posrated_ids) + [0] * len(relevant_negrated_ids)
        input_ids_list += [papers["input_ids"][idx] for idx in papers_idxs]
        attention_mask_list += [papers["attention_mask"][idx] for idx in papers_idxs]
    user_id_list = torch.tensor(user_id_list)
    paper_id_list = torch.tensor(paper_id_list)
    label_list = torch.tensor(label_list)
    input_ids_list = torch.stack(input_ids_list)
    attention_mask_list = torch.stack(attention_mask_list)
    dataset = {"user_id": user_id_list, "paper_id": paper_id_list, "label": label_list, "input_ids": input_ids_list, "attention_mask": attention_mask_list}
    torch.save(dataset, FILES_SAVE_PATH + f"/datasets/{papers_type}_dataset.pt")

def load_dataset(path : str, users_embeddings_ids_to_idxs : dict, papers_ids_to_categories_idxs : dict, papers_ids_to_categories_idxs_l2 : dict = None) -> dict:
    if path == "train":
        path = FILES_SAVE_PATH + "/datasets/train_dataset.pt"
    elif path == "val":
        path = FILES_SAVE_PATH + "/datasets/val_dataset.pt"
    dataset = torch.load(path, weights_only = True)
    dataset["user_idx"] = torch.tensor([users_embeddings_ids_to_idxs[user_id.item()] for user_id in dataset["user_id"]], dtype = torch.int32)
    del dataset["user_id"]
    dataset["category_idx"] = torch.tensor([papers_ids_to_categories_idxs[paper_id.item()] for paper_id in dataset["paper_id"]], dtype = torch.int32)
    if papers_ids_to_categories_idxs_l2 is not None:
        dataset["category_idx_l2"] = torch.tensor([papers_ids_to_categories_idxs_l2[paper_id.item()] for paper_id in dataset["paper_id"]], dtype = torch.int32)
    return dataset

def load_val_users_embeddings_idxs(val_users_ids : list, users_embeddings_ids_to_idxs : dict) -> torch.tensor:
    val_users_embeddings_idxs = [users_embeddings_ids_to_idxs[user_id] for user_id in val_users_ids]
    return torch.tensor(val_users_embeddings_idxs, dtype = torch.int64)

def save_papers_ids_to_categories_idxs_l2(papers_ids_to_categories_idxs : dict = None, papers_ids_to_categories_l2 : dict = None) -> None:
    if papers_ids_to_categories_idxs is None:
        papers_ids_to_categories_idxs = load_papers_ids_to_categories_idxs()
    if papers_ids_to_categories_l2 is None:
        with open(f"../data/categories/papers_ids_to_categories_l2.pkl", "rb") as f:
            papers_ids_to_categories_l2 = pickle.load(f)
    df = pd.DataFrame({"paper_id": list(papers_ids_to_categories_idxs.keys()), "category_l1_idx": list(papers_ids_to_categories_idxs.values()), 
                       "category_l2": [papers_ids_to_categories_l2[paper_id] for paper_id in papers_ids_to_categories_idxs.keys()]})
    unique_combinations = df[['category_l1_idx', 'category_l2']].drop_duplicates().reset_index(drop = True)
    combination_to_idx = {(row['category_l1_idx'], row['category_l2']): idx for idx, row in unique_combinations.iterrows()}
    papers_ids_to_categories_idxs_l2 = {}
    for _, row in tqdm(df.iterrows(), desc = "Mapping Papers to Category Indices L2", total = len(df), unit = "Paper"):
        papers_ids_to_categories_idxs_l2[row["paper_id"]] = combination_to_idx[(row['category_l1_idx'], row['category_l2'])]
    with open(f"{FILES_SAVE_PATH}/{TRANSFORMER_MODEL_NAME}/state_dicts/papers_ids_to_categories_idxs_l2.pkl", "wb") as f:
        pickle.dump(papers_ids_to_categories_idxs_l2, f)

def save_papers_ids_to_categories_idxs_l2_duplicates(papers_ids_to_categories_l2 : dict = None) -> None:
    if papers_ids_to_categories_l2 is None:
        with open(f"../data/categories/papers_ids_to_categories_l2.pkl", "rb") as f:
            papers_ids_to_categories_l2 = pickle.load(f)
    unique_categories_l2 = list(set(papers_ids_to_categories_l2.values()))
    categories_l2_to_idxs = {category: idx for idx, category in enumerate(unique_categories_l2)}
    papers_ids_to_categories_idxs_l2 = {paper_id: categories_l2_to_idxs[papers_ids_to_categories_l2[paper_id]] for paper_id in papers_ids_to_categories_l2}
    with open(f"{FILES_SAVE_PATH}/{TRANSFORMER_MODEL_NAME}/state_dicts/papers_ids_to_categories_idxs_l2_duplicates.pkl", "wb") as f:
        pickle.dump(papers_ids_to_categories_idxs_l2, f)

def load_papers_ids_to_categories_idxs_l2(s : str = "papers_ids_to_categories_idxs_l2_duplicates.pkl") -> dict:
    with open(f"{FILES_SAVE_PATH}/{TRANSFORMER_MODEL_NAME}/state_dicts/{s}", "rb") as f:
        papers_ids_to_categories_idxs_l2 = pickle.load(f)
    assert list(papers_ids_to_categories_idxs_l2.keys()) == sorted(list(papers_ids_to_categories_idxs_l2.keys()))
    return papers_ids_to_categories_idxs_l2

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(GTE_LARGE_PATH)
    train_users_ids, val_users_ids, test_users_no_overlap_ids = load_finetuning_users_ids()
    save_papers_ids_to_categories_idxs_l2_duplicates()
    