from arxiv import attach_arxiv_categories
from finetuning_preprocessing import FILES_SAVE_PATH
from finetuning_model import FinetuningModel, load_finetuning_model_full
from finetuning_data import FinetuningDataset
from finetuning_preprocessing import load_finetuning_users_ids, load_test_papers, load_users_embeddings_ids_to_idxs
from sklearn.metrics import roc_auc_score, ndcg_score
import json
import numpy as np
import os
import pickle
import torch

RANKING_METRICS = ["auc", "ndcg@5", "mrr", "hr@1"]
CLASSIFICATION_METRICS = ["bcel", "recall", "specificity", "balanced_accuracy"]
EXPLICIT_NEGATIVES_METRICS = CLASSIFICATION_METRICS + [metric + "_explicit_negatives" for metric in RANKING_METRICS]
NEGATIVE_SAMPLES_METRICS = [metric + "_negative_samples" for metric in RANKING_METRICS]

def generate_config(file_path : str, users_ids : list, evaluation : str, users_coefs_path : str = None, arxiv : bool = False) -> str:
    file_path = file_path.rstrip(".json")
    with open(f"{FILES_SAVE_PATH}/example_config.json", "r") as config_file:
        example_config = json.load(config_file)
    example_config["embedding_folder"] = embeddings_folder + ("/arxiv" if arxiv else "")
    example_config["users_selection"] = users_ids
    if users_coefs_path is not None:
        example_config["users_coefs_path"] = users_coefs_path
        example_config["load_users_coefs"] = True
    example_config["evaluation"] = evaluation
    file_path += ("_arxiv" if arxiv else "") + ".json"
    with open(file_path, "w") as config_file:
        json.dump(example_config, config_file, indent = 3)
    return file_path.split("/")[-1].split(".")[0]
    
def generate_configs(transformer_model_name : str, val_users_ids : list = None, test_users_no_overlap_ids : list = None, load_coefs : bool = False, arxiv : bool = False) -> list:
    configs_names = []
    if val_users_ids is not None:
        configs_names.append(generate_config(f"{configs_folder}/{transformer_model_name}_overlap", val_users_ids, "train_test_split", arxiv = arxiv))
        if load_coefs:
            configs_names.append(generate_config(f"{configs_folder}/{transformer_model_name}_overlap_users_coefs", val_users_ids, "train_test_split", 
                                 users_coefs_path = embeddings_folder, arxiv = arxiv))
    if test_users_no_overlap_ids is not None:
        configs_names.append(generate_config(f"{configs_folder}/{transformer_model_name}_no_overlap", test_users_no_overlap_ids, "cross_validation", arxiv = arxiv))
    return configs_names

def save_users_coefs(finetuning_model : FinetuningModel, val_users_ids : list, users_embeddings_ids_to_idxs : dict) -> None:
    users_coefs = finetuning_model.users_embeddings.weight.detach().cpu().numpy().astype(np.float64)
    np.save(f"{embeddings_folder}/users_coefs.npy", users_coefs)
    with open(f"{embeddings_folder}/users_coefs_ids_to_idxs.pkl", "wb") as f:
        pickle.dump(users_embeddings_ids_to_idxs, f)

def run_evaluation(finetuning_model : FinetuningModel, val_users_ids : list, test_users_no_overlap_ids : list, test_papers : dict, users_embeddings_ids_to_idxs : dict, 
                   attach_arxiv : bool = False) -> None:
    training_mode = finetuning_model.training
    finetuning_model.eval()
    papers_ids_to_idxs = {}
    for idx, paper_id in enumerate(test_papers["paper_id"]):
        papers_ids_to_idxs[paper_id.item()] = idx
    embeddings = finetuning_model.compute_papers_embeddings(test_papers["input_ids"], test_papers["attention_mask"])
    np.save(f"{embeddings_folder}/abs_X.npy", embeddings)
    with open(f"{embeddings_folder}/abs_paper_ids_to_idx.pkl", "wb") as f:
        pickle.dump(papers_ids_to_idxs, f)
    save_users_coefs(finetuning_model, val_users_ids, users_embeddings_ids_to_idxs)
    configs_names = generate_configs(finetuning_model.transformer_model_name, val_users_ids, test_users_no_overlap_ids, load_coefs = True, arxiv = False)
    if attach_arxiv:
        embeddings_folder_arxiv = f"{embeddings_folder}/arxiv"
        os.makedirs(embeddings_folder_arxiv, exist_ok = True)
        embeddings_arxiv = attach_arxiv_categories(embeddings, papers_ids_to_idxs)
        np.save(f"{embeddings_folder_arxiv}/abs_X.npy", embeddings_arxiv)
        with open(f"{embeddings_folder_arxiv}/abs_paper_ids_to_idx.pkl", "wb") as f:
            pickle.dump(papers_ids_to_idxs, f)
        configs_names.extend(generate_configs(finetuning_model.transformer_model_name, val_users_ids, test_users_no_overlap_ids, load_coefs = False, arxiv = True))
    os.system(f"python run_cross_eval.py --config_path {configs_folder}")
    for config_name in configs_names:
        os.system(f"mv outputs/{config_name} {outputs_folder}/{config_name}")
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
    for i, metric in enumerate(RANKING_METRICS):
        if metric == "auc":
            ranking_scores[i] = roc_auc_score(y_true, y_proba)
        elif metric == "ndcg@5":
            ranking_scores[i] = ndcg_score(y_true.reshape(1, -1), y_proba.reshape(1, -1), k = 5)
        elif metric == "mrr":
            sorted_indices = np.argsort(y_proba)[::-1]
            for rank, idx in enumerate(sorted_indices, 1):
                if y_true[idx] > 0:
                    ranking_scores[i] = 1.0 / rank
                    break
        elif metric == "hr@1":
            top_1_idx = np.argmax(y_proba)
            ranking_scores[i] = float(y_true[top_1_idx] > 0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return torch.tensor(ranking_scores, dtype = torch.float32)

def get_ranking_scores_for_user(pos_scores : torch.tensor, neg_ranking_scores : torch.tensor) -> torch.tensor:
    user_ranking_scores = torch.zeros(size = (len(pos_scores), len(RANKING_METRICS)), dtype = torch.float32)
    y_true = torch.cat((torch.tensor([1]), torch.zeros(len(neg_ranking_scores), dtype = torch.int32)))
    for i, pos_score in enumerate(pos_scores):
        pos_score = pos_score.unsqueeze(0)
        y_pred = torch.cat((pos_score, neg_ranking_scores))
        user_ranking_scores[i] = compute_ranking_scores(y_true, y_pred)
    return torch.mean(user_ranking_scores, dim = 0)

def compute_classification_scores(user_labels : torch.tensor, user_scores : torch.tensor) -> torch.tensor:
    bcel = torch.nn.BCEWithLogitsLoss()(user_scores, user_labels)
    recall = torch.sum(user_scores[user_labels == 1] > 0) / torch.sum(user_labels == 1)
    specificity = torch.sum(user_scores[user_labels == 0] <= 0) / torch.sum(user_labels == 0)
    balanced_accuracy = (recall + specificity) / 2
    return torch.tensor([bcel, recall, specificity, balanced_accuracy], dtype = torch.float32)

def get_user_scores(dataset : FinetuningDataset, user_idx : int, users_scores : torch.tensor) -> tuple:
    user_starting_idx, user_ending_idx = dataset.users_pos_starting_idxs[user_idx], dataset.users_pos_starting_idxs[user_idx] + dataset.users_counts[user_idx]
    user_labels = dataset.label_tensor[user_starting_idx:user_ending_idx].to(torch.float32)
    user_scores = users_scores[user_starting_idx:user_ending_idx]
    user_pos_scores = user_scores[:dataset.users_pos_counts[user_idx]]
    return user_labels, user_scores, user_pos_scores

def run_validation(finetuning_model : FinetuningModel, val_dataset : FinetuningDataset, negative_samples : torch.tensor = None, train_val_dataset : FinetuningDataset = None) -> dict:
    scores_dict = {}
    assert val_dataset.user_idx_tensor.unique().tolist() == finetuning_model.val_users_embeddings_idxs.tolist()
    if train_val_dataset is not None:
        assert train_val_dataset.user_idx_tensor.unique().tolist() == finetuning_model.val_users_embeddings_idxs.tolist()
    training_mode = finetuning_model.training
    finetuning_model.eval()

    val_users_scores = finetuning_model.compute_val_scores(val_dataset.user_idx_tensor, val_dataset.input_ids_tensor, val_dataset.attention_mask_tensor)
    val_classification_scores = torch.zeros(size = (val_dataset.n_users, len(CLASSIFICATION_METRICS)), dtype = torch.float32)
    val_explicit_negatives_ranking_scores = torch.zeros(size = (val_dataset.n_users, len(RANKING_METRICS)), dtype = torch.float32)
    if train_val_dataset is not None:
        train_val_users_scores = finetuning_model.compute_val_scores(train_val_dataset.user_idx_tensor, train_val_dataset.input_ids_tensor, train_val_dataset.attention_mask_tensor)
        train_val_classification_scores = torch.zeros(size = (train_val_dataset.n_users, len(CLASSIFICATION_METRICS)), dtype = torch.float32)
        train_val_explicit_negatives_ranking_scores = torch.zeros(size = (train_val_dataset.n_users, len(RANKING_METRICS)), dtype = torch.float32)
    if negative_samples is not None:
        negative_samples_scores = finetuning_model.compute_negative_samples_scores(negative_samples)
        val_negative_samples_ranking_scores = torch.zeros(size = (val_dataset.n_users, len(RANKING_METRICS)), dtype = torch.float32)
        if train_val_dataset is not None:
            train_val_negative_samples_ranking_scores = torch.zeros(size = (train_val_dataset.n_users, len(RANKING_METRICS)), dtype = torch.float32)

    for i in range(val_dataset.n_users):
        val_user_labels, val_user_scores, val_user_pos_scores = get_user_scores(val_dataset, i, val_users_scores)
        user_neg_ranking_scores = val_user_scores[-4:]
        val_classification_scores[i] = compute_classification_scores(val_user_labels, val_user_scores)
        val_explicit_negatives_ranking_scores[i] = get_ranking_scores_for_user(val_user_pos_scores, user_neg_ranking_scores)
        if train_val_dataset is not None:
            train_val_user_labels, train_val_user_scores, train_val_user_pos_scores = get_user_scores(train_val_dataset, i, train_val_users_scores)
            train_val_classification_scores[i] = compute_classification_scores(train_val_user_labels, train_val_user_scores)
            train_val_explicit_negatives_ranking_scores[i] = get_ranking_scores_for_user(train_val_user_pos_scores, user_neg_ranking_scores)
        if negative_samples is not None:
            val_negative_samples_ranking_scores[i] = get_ranking_scores_for_user(val_user_pos_scores, negative_samples_scores[i])
            if train_val_dataset is not None:
                train_val_negative_samples_ranking_scores[i] = get_ranking_scores_for_user(train_val_user_pos_scores, negative_samples_scores[i])

    val_classification_scores = torch.mean(val_classification_scores, dim = 0)
    val_explicit_negatives_ranking_scores = torch.mean(val_explicit_negatives_ranking_scores, dim = 0)
    if train_val_dataset is not None:
        train_val_classification_scores = torch.mean(train_val_classification_scores, dim = 0)
        train_val_explicit_negatives_ranking_scores = torch.mean(train_val_explicit_negatives_ranking_scores, dim = 0)
    if negative_samples is not None:
        val_negative_samples_ranking_scores = torch.mean(val_negative_samples_ranking_scores, dim = 0)
        if train_val_dataset is not None:
            train_val_negative_samples_ranking_scores = torch.mean(train_val_negative_samples_ranking_scores, dim = 0)

    for i, metric in enumerate(CLASSIFICATION_METRICS):
        scores_dict[f"{metric}_val"] = val_classification_scores[i].item()
        if train_val_dataset is not None:
            scores_dict[f"{metric}_train"] = train_val_classification_scores[i].item()
    for i, metric in enumerate(RANKING_METRICS):
        scores_dict[f"{metric}_explicit_negatives_val"] = val_explicit_negatives_ranking_scores[i].item()
        if train_val_dataset is not None:
            scores_dict[f"{metric}_explicit_negatives_train"] = train_val_explicit_negatives_ranking_scores[i].item()
        if negative_samples is not None:
            scores_dict[f"{metric}_negative_samples_val"] = val_negative_samples_ranking_scores[i].item()
            if train_val_dataset is not None:
                scores_dict[f"{metric}_negative_samples_train"] = train_val_negative_samples_ranking_scores[i].item()
    return scores_dict

if __name__ == "__main__":
    import sys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetuning_model_path = sys.argv[1].rstrip("/")
    state_dicts_folder = finetuning_model_path + "/state_dicts"
    embeddings_folder = finetuning_model_path + "/embeddings"
    os.makedirs(embeddings_folder, exist_ok = True)
    configs_folder = finetuning_model_path + "/configs"
    os.makedirs(configs_folder, exist_ok = True)
    outputs_folder = finetuning_model_path + "/outputs"
    os.makedirs(outputs_folder, exist_ok = True)

    finetuning_model = load_finetuning_model_full(state_dicts_folder, device)
    train_users_ids, val_users_ids, test_users_no_overlap_ids = load_finetuning_users_ids()
    test_papers = load_test_papers()
    users_embeddings_ids_to_idxs = load_users_embeddings_ids_to_idxs()
    run_evaluation(finetuning_model, val_users_ids, test_users_no_overlap_ids, test_papers, users_embeddings_ids_to_idxs, attach_arxiv = True)