from arxiv import attach_arxiv_categories
from finetuning_preprocessing import FILES_SAVE_PATH
from finetuning_model import FinetuningModel
from finetuning_data import FinetuningDataset
from sklearn.metrics import roc_auc_score, ndcg_score
import json
import numpy as np
import os
import pickle
import torch

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
    
def generate_configs(embedding_folder : str, val_users_ids : list = None, test_users_no_overlap_ids : list = None) -> None:
    embedding_name = embedding_folder.split("/")[-1]
    if val_users_ids is not None:
        generate_config(f"{embedding_folder}/{embedding_name}_overlap.json", embedding_folder, val_users_ids, "train_test_split")
    if test_users_no_overlap_ids is not None:
        generate_config(f"{embedding_folder}/{embedding_name}_no_overlap.json", embedding_folder, test_users_no_overlap_ids, "cross_validation")

def run_evaluation(finetuning_model : FinetuningModel, val_users_ids : list, test_users_no_overlap_ids : list, test_papers : dict) -> None:
    training_mode = finetuning_model.training
    finetuning_model.eval()
    embedding_name = finetuning_model.get_embedding_name()
    embedding_folder = f"{FILES_SAVE_PATH}/experiments/{embedding_name}"
    papers_ids_to_idxs = {}
    for idx, paper_id in enumerate(test_papers["paper_id"]):
        papers_ids_to_idxs[paper_id.item()] = idx
    embeddings = finetuning_model.compute_papers_embeddings(test_papers["input_ids"], test_papers["attention_mask"])
    embeddings = attach_arxiv_categories(embeddings, papers_ids_to_idxs)
    os.makedirs(embedding_folder, exist_ok = True)
    np.save(f"{embedding_folder}/abs_X.npy", embeddings)
    with open(f"{embedding_folder}/abs_paper_ids_to_idx.pkl", "wb") as f:
        pickle.dump(papers_ids_to_idxs, f)
    generate_configs(embedding_folder, val_users_ids, test_users_no_overlap_ids)
    os.system(f"python run_cross_eval.py --config_path {embedding_folder}")
    if training_mode:
        finetuning_model.train()

def get_users_starting_ending_idxs(user_idx_tensor : torch.tensor, offset : int) -> torch.tensor:
    counts = torch.unique(user_idx_tensor, return_inverse = False, return_counts = True)
    ending_idxs = torch.cumsum(counts[1], dim = 0)
    starting_idxs = torch.cat((torch.tensor([0]), ending_idxs[:-1]), dim = 0)
    ending_idxs = ending_idxs - offset
    return starting_idxs, ending_idxs

def compute_ranking_scores(y_true : torch.tensor, y_proba : torch.tensor) -> torch.tensor:
    ranking_scores = np.zeros(4, dtype = np.float32)
    y_true, y_proba = y_true.numpy(), y_proba.numpy()
    ranking_scores[0] = roc_auc_score(y_true, y_proba)
    ranking_scores[1] = ndcg_score(y_true.reshape(1, -1), y_proba.reshape(1, -1), k = 5)
    sorted_indices = np.argsort(y_proba)[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        if y_true[idx] > 0:
            ranking_scores[2] = 1.0 / rank
            break
    top_1_idx = np.argmax(y_proba)
    ranking_scores[3] = float(y_true[top_1_idx] > 0)
    return torch.tensor(ranking_scores, dtype = torch.float32)

def get_ranking_scores(users_scores : torch.tensor, explicit_negatives_scores : torch.tensor, negative_samples_scores : torch.tensor, 
                       users_starting_idxs : torch.tensor, users_ending_idxs : torch.tensor) -> tuple:
    n_users = explicit_negatives_scores.shape[0]
    explicit_negatives_labels = torch.cat((torch.tensor([1]), torch.zeros(explicit_negatives_scores.shape[1], dtype = torch.int32)))
    negative_samples_labels = torch.cat((torch.tensor([1]), torch.zeros(negative_samples_scores.shape[1], dtype = torch.int32)))
    explicit_negatives_ranking_scores, negative_samples_ranking_scores = torch.zeros(size = (n_users, 4), dtype = torch.float32), torch.zeros(size = (n_users, 4), dtype = torch.float32)
    for i in range(n_users):
        explicit_negatives_scores_user, negative_samples_scores_user = explicit_negatives_scores[i], negative_samples_scores[i]
        positives_scores_user = users_scores[users_starting_idxs[i]:users_ending_idxs[i]]
        n_positives_scores = positives_scores_user.shape[0]
        explicit_negatives_ranking_scores_user = torch.zeros(size = (n_positives_scores, 4), dtype = torch.float32)
        negative_samples_ranking_scores_user = torch.zeros(size = (n_positives_scores, 4), dtype = torch.float32)
        for j, positive_score in enumerate(positives_scores_user):
            positive_score = positive_score.unsqueeze(0)
            explicit_negatives_scores_cat = torch.cat((positive_score, explicit_negatives_scores_user))
            negative_samples_scores_cat = torch.cat((positive_score, negative_samples_scores_user))
            explicit_negatives_ranking_scores_user[j] = compute_ranking_scores(explicit_negatives_labels, explicit_negatives_scores_cat)
            negative_samples_ranking_scores_user[j] = compute_ranking_scores(negative_samples_labels, negative_samples_scores_cat)
        explicit_negatives_ranking_scores[i] = torch.mean(explicit_negatives_ranking_scores_user, dim = 0)
        negative_samples_ranking_scores[i] = torch.mean(negative_samples_ranking_scores_user, dim = 0)
    explicit_negatives_ranking_scores = torch.mean(explicit_negatives_ranking_scores, dim = 0)
    negative_samples_ranking_scores = torch.mean(negative_samples_ranking_scores, dim = 0)
    return explicit_negatives_ranking_scores, negative_samples_ranking_scores
            
def run_validation(finetuning_model : FinetuningModel, val_dataset : FinetuningDataset, negative_samples : dict, train_val_dataset : FinetuningDataset = None) -> tuple:
    assert val_dataset.user_idx_tensor.unique().tolist() == finetuning_model.val_users_embeddings_idxs.tolist()
    training_mode = finetuning_model.training
    finetuning_model.eval()
    negative_samples_scores = finetuning_model.compute_negative_samples_scores(negative_samples)
    val_users_scores = finetuning_model.compute_users_scores(val_dataset.user_idx_tensor, val_dataset.input_ids_tensor, val_dataset.attention_mask_tensor, training = False)
    label_0_idxs = torch.where(val_dataset.label_tensor == 0)[0]
    explicit_negatives_scores = val_users_scores[label_0_idxs].reshape(-1, 4)
    val_users_starting_idxs, val_users_ending_idxs = get_users_starting_ending_idxs(val_dataset.user_idx_tensor, 4)
    val_ranking_scores = get_ranking_scores(val_users_scores, explicit_negatives_scores, negative_samples_scores, val_users_starting_idxs, val_users_ending_idxs)
    train_val_ranking_scores = None
    if train_val_dataset is not None:
        assert train_val_dataset.user_idx_tensor.unique().tolist() == finetuning_model.val_users_embeddings_idxs.tolist()
        train_val_users_scores = finetuning_model.compute_users_scores(train_val_dataset.user_idx_tensor, train_val_dataset.input_ids_tensor, train_val_dataset.attention_mask_tensor, training = False)
        train_val_users_starting_idxs, train_val_users_ending_idxs = get_users_starting_ending_idxs(train_val_dataset.user_idx_tensor, 0)
        train_val_ranking_scores = get_ranking_scores(train_val_users_scores, explicit_negatives_scores, negative_samples_scores, train_val_users_starting_idxs, train_val_users_ending_idxs)
    if training_mode:
        finetuning_model.train()
    return val_ranking_scores, train_val_ranking_scores