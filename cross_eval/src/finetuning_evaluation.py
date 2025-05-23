from algorithm import Score, SCORES_DICT
from finetuning_preprocessing import FILES_SAVE_PATH, VALIDATION_RANDOM_STATE, TESTING_NO_OVERLAP_RANDOM_STATES
from finetuning_preprocessing import load_finetuning_users_ids, load_papers, load_users_embeddings_ids_to_idxs, load_papers_ids_to_categories_idxs
from finetuning_model import FinetuningModel, load_finetuning_model
from finetuning_data import FinetuningDataset
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
from sklearn.metrics import roc_auc_score, ndcg_score
from torch.nn.functional import log_softmax
from visualization_tools import print_table, PLOT_CONSTANTS, PRINT_SCORES, format_number
import argparse
import json
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import time
import torch

CLASSIFICATION_METRICS = ["bcel", "recall", "specificity", "balacc"]
RANKING_METRICS = ["ndcg", "mrr", "hr@1", "infonce"]

def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description = "Evaluation script.")
    parser.add_argument("--finetuning_model_path", type = str, required = True)
    parser.add_argument("--not_perform_evaluation", action = "store_false", dest = "perform_evaluation")
    parser.add_argument("--allow_configs", action = "store_true", default = False)
    parser.add_argument("--allow_outputs", action = "store_true", default = False)
    args_dict = vars(parser.parse_args())
    args_dict["finetuning_model_path"] = args_dict["finetuning_model_path"].rstrip("/")
    args_dict["state_dicts_folder"] = args_dict["finetuning_model_path"] + "/state_dicts"
    args_dict["embeddings_folder"] = args_dict["finetuning_model_path"] + "/embeddings"
    args_dict["compute_test_embeddings"] = (not os.path.exists(args_dict["embeddings_folder"])) or len(os.listdir(args_dict["embeddings_folder"])) == 0
    os.makedirs(args_dict["embeddings_folder"], exist_ok = True)
    args_dict["configs_folder"] = args_dict["finetuning_model_path"] + "/configs"
    os.makedirs(args_dict["configs_folder"], exist_ok = True)
    if len(os.listdir(args_dict["configs_folder"])) and not args_dict["allow_configs"]:
        raise ValueError(f"Folder {args_dict['configs_folder']} is not empty. Please remove the files inside it.")
    args_dict["outputs_folder"] = args_dict["finetuning_model_path"] + "/outputs"
    os.makedirs(args_dict["outputs_folder"], exist_ok = True)
    if len(os.listdir(args_dict["outputs_folder"])) > 0 and not args_dict["allow_outputs"]:
        raise ValueError(f"Folder {args_dict['outputs_folder']} is not empty. Please remove the files inside it.")
    return args_dict

def save_users_coefs(finetuning_model : FinetuningModel, val_users_ids : list, users_embeddings_ids_to_idxs : dict, embeddings_folder : str) -> None:
    users_coefs = finetuning_model.users_embeddings.weight.detach().cpu().numpy().astype(np.float64)
    np.save(f"{embeddings_folder}/users_coefs.npy", users_coefs)
    with open(f"{embeddings_folder}/users_coefs_ids_to_idxs.pkl", "wb") as f:
        pickle.dump(users_embeddings_ids_to_idxs, f)

def compute_test_embeddings(finetuning_model : FinetuningModel, test_papers : dict, embeddings_folder : str) -> None:
    training_mode = finetuning_model.training
    finetuning_model.eval()
    papers_ids_to_idxs = {paper_id.item() : idx for idx, paper_id in enumerate(test_papers["paper_id"])}
    embeddings = finetuning_model.compute_papers_embeddings(test_papers["input_ids"], test_papers["attention_mask"], test_papers["category_idx"])
    np.save(f"{embeddings_folder}/abs_X.npy", embeddings)
    with open(f"{embeddings_folder}/abs_paper_ids_to_idx.pkl", "wb") as f:
        pickle.dump(papers_ids_to_idxs, f)
    save_users_coefs(finetuning_model, val_users_ids, users_embeddings_ids_to_idxs, embeddings_folder)
    if training_mode:
        finetuning_model.train()

def generate_config(file_path : str, embeddings_folder : str, users_ids : list, evaluation : str, random_state : int = 42, users_coefs_path : str = None) -> str:
    with open(f"{FILES_SAVE_PATH}/example_config.json", "r") as config_file:
        example_config = json.load(config_file)
    example_config["embedding_folder"] = embeddings_folder
    example_config["users_selection"] = users_ids
    example_config["evaluation"] = evaluation
    example_config["model_random_state"] = random_state
    example_config["cache_random_state"] = random_state
    example_config["ranking_random_state"] = random_state
    if users_coefs_path is not None:
        example_config["users_coefs_path"] = users_coefs_path
        example_config["load_users_coefs"] = True
    file_path = file_path.rstrip(".json") + ".json"
    with open(file_path, "w") as config_file:
        json.dump(example_config, config_file, indent = 3)
    return file_path.split("/")[-1].split(".")[0]

def run_testing_single(name : str, embeddings_folder : str, configs_folder : str, outputs_folder : str, users_ids : list, evaluation : str, random_states : list, 
                       users_coefs_path : str = None) -> None:
    configs_names, configs_folder_single, outputs_folder_single = [], f"{configs_folder}/{name}", f"{outputs_folder}/{name}"
    os.makedirs(configs_folder_single, exist_ok = True)
    os.makedirs(outputs_folder_single, exist_ok = True)
    for random_state in random_states:
        configs_names.append(generate_config(f"{configs_folder_single}/{name}_s{random_state}", embeddings_folder, users_ids, evaluation, random_state, users_coefs_path))
    os.system(f"python run_cross_eval.py --config_path {configs_folder_single} --save_scores_tables")
    for config_name in configs_names:
        os.system(f"mv outputs/{config_name} {outputs_folder_single}/{config_name}")

def run_testing(finetuning_model : FinetuningModel, embeddings_folder : str, configs_folder : str, outputs_folder : str, val_users_ids : list, test_users_no_overlap_ids : list,
                validation_random_state : int = VALIDATION_RANDOM_STATE, testing_no_overlap_random_states : list = TESTING_NO_OVERLAP_RANDOM_STATES) -> None:
    assert validation_random_state not in testing_no_overlap_random_states
    if val_users_ids is not None:
        run_testing_single("overlap", embeddings_folder, configs_folder, outputs_folder, val_users_ids, "train_test_split", [validation_random_state])
        run_testing_single("overlap_users_coefs", embeddings_folder, configs_folder, outputs_folder, val_users_ids, "train_test_split", [validation_random_state],
                           users_coefs_path = embeddings_folder)
    if test_users_no_overlap_ids is not None:
        run_testing_single("no_overlap", embeddings_folder, configs_folder, outputs_folder, test_users_no_overlap_ids, "cross_validation", testing_no_overlap_random_states)

def scores_table_to_df(scores_table : list) -> pd.DataFrame:
    scores_names, val_scores, train_scores = [], [], []
    scores_table_first_row = scores_table.pop(0)
    index_score, index_val, index_train = scores_table_first_row.index("Score"), scores_table_first_row.index("All"), scores_table_first_row.index("All_T")
    for i, row in enumerate(scores_table):
        scores_names.append(row[index_score])
        val_scores.append(row[index_val])
        train_scores.append(row[index_train])
    scores_df = pd.DataFrame({"Score": scores_names, "Validation": val_scores, "Train": train_scores})
    scores_df["Validation"] = pd.to_numeric(scores_df["Validation"], errors = "coerce")
    scores_df["Train"] = pd.to_numeric(scores_df["Train"], errors = "coerce")
    return scores_df
        
def get_scores_dict(outputs_folder : str) -> dict:
    scores_dict = {}
    for folder in os.listdir(outputs_folder):
        folder_name = " ".join([w.capitalize() for w in folder.split("_")])
        scores_dfs = []
        random_states = []
        for random_state in os.listdir(f"{outputs_folder}/{folder}"):
            random_states.append(random_state.split("_")[-1][1:])
            with open(f"{outputs_folder}/{folder}/{random_state}/scores_table_1.pkl", "rb") as f:
                scores_table_1 = pickle.load(f)
                scores_table_1 = scores_table_to_df(scores_table_1)
            with open(f"{outputs_folder}/{folder}/{random_state}/scores_table_2.pkl", "rb") as f:
                scores_table_2 = pickle.load(f)
                scores_table_2 = scores_table_to_df(scores_table_2)
            scores_df = pd.concat([scores_table_1, scores_table_2[1:]], axis = 0)
            scores_df.set_index("Score", inplace = True)
            scores_dfs.append(scores_df)
        mean_scores_df = pd.concat(scores_dfs).groupby(level = 0).mean()
        std_scores_df = pd.concat(scores_dfs).groupby(level = 0).std()
        std_scores_df.fillna(0, inplace = True)
        random_states = sorted(random_states, key = lambda x : int(x))
        scores_dict[folder_name] = {"mean": mean_scores_df, "std": std_scores_df, "random_states": random_states}
    return scores_dict

def scores_df_to_table(mean_df : pd.DataFrame, std_df : pd.DataFrame, random_states : list) -> list:
    mean_row_val, std_row_val, mean_row_train, std_row_train = ["Val_μ"], ["Val_σ"], ["Train_μ"], ["Train_σ"]
    for i in range(len(mean_df)):
        mean_row_val.append(format_number(mean_df.iloc[i, 0]))
        std_row_val.append(format_number(std_df.iloc[i, 0]))
        mean_row_train.append(format_number(mean_df.iloc[i, 1]))
        std_row_train.append(format_number(std_df.iloc[i, 1]))
    if len(random_states) > 1:
        return [mean_row_val, std_row_val, mean_row_train, std_row_train]
    else:
        return [mean_row_val, mean_row_train]

def visualize_testing(finetuning_model_path : str, outputs_folder : str, testing_metrics : list = PRINT_SCORES) -> None:
    scores_dict = get_scores_dict(outputs_folder)
    file_name = f"{finetuning_model_path}/visualization.pdf"
    with PdfPages(file_name) as pdf:
        try:
            config, train_losses, val_scores = load_visualization_files(finetuning_model_path)
        except Exception as e:
            config, train_losses, val_scores = None, None, None
        if config is not None:
            visualize_config(pdf, config)
        visualize_scores(pdf, scores_dict, testing_metrics)
        if config is not None:
            visualize_training_curve(pdf, train_losses, val_scores, config["loss_function"], config["n_batches_per_val"])

def visualize_scores(pdf : PdfPages, scores_dict : dict, testing_metrics : list) -> None:
    fig, ax = plt.subplots(figsize = PLOT_CONSTANTS["FIG_SIZE"])
    ax.axis("off")
    ax.text(0.5, 1.11, "Testing Scores:", fontsize = 16, ha = 'center', va = 'center', fontweight = 'bold')
    testing_metrics_abbreviations = [SCORES_DICT[metric]["abbreviation"] for metric in testing_metrics]
    scores_dict_keys_sorted = sorted(scores_dict.keys())
    text_positions = [1.061, 0.68, 0.458]
    columns = [""]
    for i, key in enumerate(scores_dict_keys_sorted):
        mean_df, std_df = scores_dict[key]["mean"], scores_dict[key]["std"]
        mean_df, std_df = mean_df[mean_df.index.isin(testing_metrics_abbreviations)], std_df[std_df.index.isin(testing_metrics_abbreviations)]
        mean_df, std_df = mean_df.reindex(testing_metrics_abbreviations), std_df.reindex(testing_metrics_abbreviations)
        if columns == [""]:
            columns += mean_df.index.tolist()
        random_states = scores_dict[key]["random_states"]
        scores_table = scores_df_to_table(mean_df, std_df, random_states)
        n_rows = len(scores_table)
        if len(random_states) > 1:
            seeds_title = f" (Seeds {', '.join(random_states)}):"
        else:
            seeds_title = f" (Seed {random_states[0]}):"
        ax.text(0.5, text_positions[i], key + seeds_title, fontsize = 11, ha = 'center', va = 'center', fontweight = 'bold')
        print_table(scores_table, [-0.14, 0.72 - (i * n_rows * 0.11), 1.25, 0.08 * n_rows], columns, [0.1] + len(testing_metrics) * [0.15], grey_row = [1, 3])
    pdf.savefig(fig)
    plt.close(fig)

def visualize_config(pdf : PdfPages, config : dict) -> None:
    fig, ax = plt.subplots(figsize = PLOT_CONSTANTS["FIG_SIZE"])
    ax.axis("off")
    ax.text(0.5, 1.1, "Finetuning Configuration:", fontsize = 16, ha = 'center', va = 'center', fontweight = 'bold')
    ax.text(0.5, 1.0, get_config_string(config), fontsize = 9, ha = 'center', va = 'top', wrap = True)
    pdf.savefig(fig)
    plt.close(fig)

def get_config_string(config : dict) -> str:
    config_string = []
    hours, minutes = divmod(config["time_elapsed"], 3600)
    config_string.append(f"Folder Name: {config['outputs_folder'].split('/')[-1]}   |   Random Seed: {config['seed']}")
    frozen_string = f"Number of unfrozen Transformer Layers: {config['n_unfreeze_layers']} (out of {config['n_transformer_layers']})"
    n_transformer_params, n_unfrozen_transformer_params = round(config["n_transformer_parameters"] / 1000000), round(config["n_unfrozen_transformer_parameters"] / 1000000)
    frozen_string += f"   |   Number of unfrozen Transformer Parameters: {n_unfrozen_transformer_params}M (out of {n_transformer_params}M)"
    config_string.append(frozen_string)
    pretrained_string = "Layers pretrained?   "
    pretrained_string += f"Projection: {'Yes' if config['pretrained_projection'] else 'No'}"
    pretrained_string += f"   |   Users Embeddings: {'Yes' if config['pretrained_users_embeddings'] else 'No'}"
    pretrained_string += f"   |   Categories Embeddings: {'Yes' if config['pretrained_categories_embeddings'] else 'No'}"
    config_string.append(pretrained_string)
    config_string.append("\n")

    config_string.append(f"Training Time: {int(hours)}h {int(minutes)%60}m   |   Max Number of Batches: {config['n_batches_total']//1000}K")
    val_string = f"Validation Metric: {config['val_metric']}"
    val_string += f"   |   Number of Batches per Validation Check: {config['n_batches_per_val']}   |   Early Stopping Patience: {config['early_stopping_patience']}"
    config_string.append(val_string)
    loss_string = f"Train Loss Function: {config['loss_function']}"
    if config["loss_function"] == "info_nce": loss_string += f" (Temperate: {config['info_nce_temperature']}   |   Log-Q Correction? {config['info_nce_log_q_correction']})"
    config_string.append(loss_string)
    config_string.append(f"Number of negative Samples during Training: {config['n_train_negative_samples']}")
    config_string.append("\n")

    batch_string = f"Batch Size: {config['batch_size']}   |   Users Sampling Strategy: {config['users_sampling_strategy']}"
    batch_string += f"   |   Number of Samples in Batch per selected User: {config['n_samples_per_user']}"
    config_string.append(batch_string)
    min_max_string = "Min / Max Number of Samples for each User?   "
    min_max_string += f"Positive: {config['n_min_positive_samples_per_user']} / {config['n_max_positive_samples_per_user']}"
    min_max_string += f"   |   Negative: {config['n_min_negative_samples_per_user']} / {config['n_max_negative_samples_per_user']}"
    config_string.append(min_max_string)
    config_string.append("\n")

    scheduler_string = f"Learning Rate Scheduler: {config['lr_scheduler']}"
    if config["lr_scheduler"] == "linear_decay":
        scheduler_string += f"   |   Percentage of Warmup Steps: {config['percentage_warmup_steps']}"
    config_string.append(scheduler_string)
    config_string.append(f"Learning Rates:   Transformer: {config['lr_transformer_model']}   |   Other: {config['lr_other']}")
    config_string.append(f"L2 Regularization:   Transformer: {config['l2_regularization_transformer_model']}   |   Other: {config['l2_regularization_other']}")
    return "\n\n".join(config_string)

def load_visualization_files(finetuning_model_path : str) -> tuple:
    with open(f"{finetuning_model_path}/config.json", "r") as f:
        config = json.load(f)
    with open(f"{finetuning_model_path}/train_losses.pkl", "rb") as f:
        train_losses = pickle.load(f)
    with open(f"{finetuning_model_path}/val_scores.pkl", "rb") as f:
        val_scores = pickle.load(f)
    return config, train_losses, val_scores

def visualize_training_curve(pdf : PdfPages, train_losses : list, val_scores : list, loss_function : str, n_batches_per_val : int) -> None:
    train_idxs, train_losses = zip(*train_losses)
    train_idxs, train_losses = np.array(train_idxs), np.array(train_losses)
    assert len(train_losses) % n_batches_per_val == 0
    val_idxs, val_losses = zip(*val_scores)
    val_losses = [val_dict[f"val_{loss_function}"] for val_dict in val_losses]
    val_idxs, val_losses = np.array(val_idxs), np.array(val_losses)

    num_chunks = len(train_losses) // n_batches_per_val
    chunks = [train_losses[i * n_batches_per_val:(i + 1) * n_batches_per_val] for i in range(num_chunks)]
    chunks_idxs, chunks_means, chunks_stds = [], [], []
    for i in range(num_chunks):
        chunks_idxs.append(i * n_batches_per_val + n_batches_per_val // 2)
        chunks_means.append(np.mean(chunks[i]))
        chunks_stds.append(np.std(chunks[i]))
    chunks_idxs, chunks_means, chunks_stds = np.array(chunks_idxs), np.array(chunks_means), np.array(chunks_stds)
    min_train_loss_idx = np.argmin(chunks_means)
    min_val_loss_idx = np.argmin(val_losses)
    min_train_loss, min_val_loss = chunks_means[min_train_loss_idx], val_losses[min_val_loss_idx]
    x, y = PLOT_CONSTANTS["FIG_SIZE"]
    fig, ax = plt.subplots(figsize = (x * 1.25, y))
    very_light_grey = "#e6e6e6"
    ax.set_facecolor(very_light_grey)
    formatted_labels = []
    for x in val_idxs:
        if x % 1000 == 0:
            formatted_labels.append(f"{int(x / 1000)}")
        else:
            formatted_labels.append("")
    ax.set_xticks(val_idxs)
    ax.set_xticklabels(formatted_labels, fontsize = 10)
    all_yticks = np.arange(0, 0.65, 0.05)
    if all_yticks[-1] != 0.6:
        all_yticks = np.append(all_yticks, 0.6)
    y_tick_labels = [f"{y:.1f}" if y*10 % 1 == 0 else "" for y in all_yticks]
    ax.set_yticks(all_yticks)
    ax.set_yticklabels(y_tick_labels, fontsize = 10)
    ax.set_xlim(-50, len(train_losses) + 50)
    ax.set_ylim(0, 0.6)
    ax2 = ax.twinx()
    if loss_function == "bcel":
        ax2.set_ylim(0, 0.6)
        ax.set_ylabel("BCEL", fontsize = 13)
    ax2.set_yticks([min_train_loss, min_val_loss])
    ax2.set_yticklabels([f"{min_train_loss:.3f}", f"{min_val_loss:.3f}"], fontsize=10)
    ax.plot(train_idxs, train_losses, label = "Train Loss (per Batch)", linewidth = 2, color = "blue", alpha = 0.25)
    train_line, = ax.plot(chunks_idxs, chunks_means, label = "Train Loss (Mean of each Chunk)", linewidth = 2, color = "royalblue", marker = "o", markersize = 3)
    val_line, = ax.plot(val_idxs, val_losses, label = "Validation Loss", linewidth = 2, color = "darkorange", marker = "o", markersize = 3)
    ax.scatter(chunks_idxs[min_train_loss_idx], chunks_means[min_train_loss_idx], s = 50, zorder = 5, facecolor = train_line.get_color(), edgecolor = 'black')
    ax.scatter(val_idxs[min_val_loss_idx], val_losses[min_val_loss_idx], s = 50, zorder = 5, facecolor = val_line.get_color(), edgecolor = 'black')
    ax.set_xlabel("Batch Number (in K)", fontsize = 13)
    
    ax.legend(fontsize = 12)
    ax.grid(True, linestyle = "-", alpha = 0.4)
    pdf.savefig(fig)
    plt.close(fig)

def get_user_scores(dataset : FinetuningDataset, user_idx : int, users_scores : torch.tensor) -> tuple:
    user_starting_idx, user_ending_idx = dataset.users_pos_starting_idxs[user_idx], dataset.users_pos_starting_idxs[user_idx] + dataset.users_counts[user_idx]
    user_labels = dataset.label_tensor[user_starting_idx:user_ending_idx].to(torch.float32)
    user_scores = users_scores[user_starting_idx:user_ending_idx]
    user_scores_pos = user_scores[:dataset.users_pos_counts[user_idx]]
    return user_labels, user_scores, user_scores_pos

def compute_user_classification_metrics(user_labels : torch.tensor, user_scores : torch.tensor) -> torch.tensor:
    bcel = torch.nn.BCEWithLogitsLoss()(user_scores, user_labels)
    recall = torch.sum(user_scores[user_labels == 1] > 0) / torch.sum(user_labels == 1)
    specificity = torch.sum(user_scores[user_labels == 0] < 0) / torch.sum(user_labels == 0)
    balanced_accuracy = (recall + specificity) / 2
    return torch.tensor([bcel, recall, specificity, balanced_accuracy], dtype = torch.float32)

def compute_user_ranking_metrics_single(pos_rank : int) -> torch.tensor:
    ndcg = 1 / np.log2(pos_rank + 1)
    mrr = 1 / pos_rank
    hr_at_1 = 1.0 if pos_rank == 1 else 0.0
    results = [ndcg, mrr, hr_at_1]
    if len(results) < len(RANKING_METRICS):
        results += [0.0] * (len(RANKING_METRICS) - len(results))
    return torch.tensor(results, dtype = torch.float32)

def compute_user_ranking_metrics(user_scores_pos : torch.tensor, user_scores_explicit_negatives : torch.tensor, user_scores_negative_samples : torch.tensor, info_nce_temperature : float) -> tuple:
    user_ranking_metrics_explicit_negatives = torch.zeros(size = (len(user_scores_pos), len(RANKING_METRICS)), dtype = torch.float32)
    user_ranking_metrics_negative_samples = torch.zeros(size = (len(user_scores_pos), len(RANKING_METRICS)), dtype = torch.float32)
    user_ranking_metrics_all = torch.zeros(size = (len(user_scores_pos), len(RANKING_METRICS)), dtype = torch.float32)

    info_nce_tensor_explicit_negatives = user_scores_explicit_negatives / info_nce_temperature
    info_nce_tensor_negative_samples = user_scores_negative_samples / info_nce_temperature
    info_nce_tensor_all = torch.cat((info_nce_tensor_explicit_negatives, info_nce_tensor_negative_samples))

    for i, pos_score in enumerate(user_scores_pos):
        pos_score_unsqueezed = pos_score.unsqueeze(0)
        pos_rank_explicit_negatives = torch.sum(user_scores_explicit_negatives >= pos_score).item() + 1
        pos_rank_negative_samples = torch.sum(user_scores_negative_samples >= pos_score).item() + 1
        pos_rank_all = torch.sum(torch.cat((user_scores_explicit_negatives, user_scores_negative_samples)) >= pos_score).item() + 1

        user_ranking_metrics_explicit_negatives[i] = compute_user_ranking_metrics_single(pos_rank_explicit_negatives)
        user_ranking_metrics_explicit_negatives[i, -1] = -torch.log_softmax(torch.cat((pos_score_unsqueezed, info_nce_tensor_explicit_negatives)) + 1e-10, dim = 0)[0].item()
        user_ranking_metrics_negative_samples[i] = compute_user_ranking_metrics_single(pos_rank_negative_samples)
        user_ranking_metrics_negative_samples[i, -1] = -torch.log_softmax(torch.cat((pos_score_unsqueezed, info_nce_tensor_negative_samples)) + 1e-10, dim = 0)[0].item()
        user_ranking_metrics_all[i] = compute_user_ranking_metrics_single(pos_rank_all)
        user_ranking_metrics_all[i, -1] = -torch.log_softmax(torch.cat((pos_score_unsqueezed, info_nce_tensor_all)) + 1e-10, dim = 0)[0].item()
    user_ranking_metrics_explicit_negatives = torch.mean(user_ranking_metrics_explicit_negatives, dim = 0)
    user_ranking_metrics_negative_samples = torch.mean(user_ranking_metrics_negative_samples, dim = 0)
    user_ranking_metrics_all = torch.mean(user_ranking_metrics_all, dim = 0)
    return user_ranking_metrics_explicit_negatives, user_ranking_metrics_negative_samples, user_ranking_metrics_all

def print_metrics(scores_dict : dict, metrics : list) -> str:
    METRIC_STRINGS = {"bcel": "BCEL", "recall": "Recall", "specificity": "Specificity", "balacc": "Balanced Accuracy",
                      "ndcg": "NDCG", "mrr": "MRR", "hr@1": "HR@1", "infonce": "InfoNCE"}
    metrics_string = ""
    for i, metric in enumerate(metrics):
        metric_string = metric.split("_")[1]
        if i > 0:
            metrics_string += ", "
        metrics_string += f"{METRIC_STRINGS[metric_string]}: {format_number(scores_dict[metric])}"
    return metrics_string

def print_validation(scores_dict : dict) -> str:
    validation_str = ""
    validation_str += "\nClassification: " + print_metrics(scores_dict, [f"val_{metric}" for metric in CLASSIFICATION_METRICS])
    validation_str += "\nRanking (Explicit Negatives): " + print_metrics(scores_dict, [f"val_{metric}_explicit_negatives" for metric in RANKING_METRICS])
    validation_str += "\nRanking (Negative Samples): " + print_metrics(scores_dict, [f"val_{metric}_negative_samples" for metric in RANKING_METRICS])
    validation_str += "\nRanking (All): " + print_metrics(scores_dict, [f"val_{metric}_all" for metric in RANKING_METRICS])
    return validation_str
        
def run_validation(finetuning_model : FinetuningModel, val_ratings : FinetuningDataset, val_negative_samples : dict, info_nce_temperature : float, print_results : bool = True) -> tuple:
    scores_dict = {}
    assert val_ratings.user_idx_tensor.unique().tolist() == finetuning_model.val_users_embeddings_idxs.tolist()
    training_mode = finetuning_model.training
    finetuning_model.eval()
    val_users_scores = finetuning_model.compute_val_ratings_scores(val_ratings.input_ids_tensor, val_ratings.attention_mask_tensor, val_ratings.category_idx_tensor, val_ratings.user_idx_tensor)
    val_negative_samples_scores = finetuning_model.compute_val_negative_samples_scores(val_negative_samples["input_ids"], val_negative_samples["attention_mask"], val_negative_samples["category_idx"])

    val_classification_metrics = torch.zeros(size = (val_ratings.n_users, len(CLASSIFICATION_METRICS)), dtype = torch.float32)
    val_ranking_metrics_explicit_negatives = torch.zeros(size = (val_ratings.n_users, len(RANKING_METRICS)), dtype = torch.float32)
    val_ranking_metrics_negative_samples = torch.zeros(size = (val_ratings.n_users, len(RANKING_METRICS)), dtype = torch.float32)
    val_ranking_metrics_all = torch.zeros(size = (val_ratings.n_users, len(RANKING_METRICS)), dtype = torch.float32)

    for i in range(val_ratings.n_users):
        val_user_labels, val_user_scores, val_user_scores_pos = get_user_scores(val_ratings, i, val_users_scores)
        val_classification_metrics[i] = compute_user_classification_metrics(val_user_labels, val_user_scores)
        val_user_scores_explicit_negatives = val_user_scores[-4:]
        val_user_scores_negative_samples = val_negative_samples_scores[i]
        val_user_ranking_metrics = compute_user_ranking_metrics(val_user_scores_pos, val_user_scores_explicit_negatives, val_user_scores_negative_samples, info_nce_temperature)
        val_ranking_metrics_explicit_negatives[i] = val_user_ranking_metrics[0]
        val_ranking_metrics_negative_samples[i] = val_user_ranking_metrics[1]
        val_ranking_metrics_all[i] = val_user_ranking_metrics[2]

    val_classification_metrics = torch.mean(val_classification_metrics, dim = 0)
    val_ranking_metrics_explicit_negatives = torch.mean(val_ranking_metrics_explicit_negatives, dim = 0)
    val_ranking_metrics_negative_samples = torch.mean(val_ranking_metrics_negative_samples, dim = 0)
    val_ranking_metrics_all = torch.mean(val_ranking_metrics_all, dim = 0)

    for i, metric in enumerate(CLASSIFICATION_METRICS):
        scores_dict[f"val_{metric}"] = val_classification_metrics[i].item()
    for i, metric in enumerate(RANKING_METRICS):
        scores_dict[f"val_{metric}_explicit_negatives"] = val_ranking_metrics_explicit_negatives[i].item()
        scores_dict[f"val_{metric}_negative_samples"] = val_ranking_metrics_negative_samples[i].item()
        scores_dict[f"val_{metric}_all"] = val_ranking_metrics_all[i].item()
    validation_str = print_validation(scores_dict)
    if print_results:
        print(validation_str)
    if training_mode:
        finetuning_model.train()
    return scores_dict, validation_str
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation_config = parse_arguments()
    train_users_ids, val_users_ids, test_users_no_overlap_ids = load_finetuning_users_ids()
    users_embeddings_ids_to_idxs = load_users_embeddings_ids_to_idxs()
    papers_ids_to_categories_idxs = load_papers_ids_to_categories_idxs()
    test_papers = load_papers(path = "test", papers_ids_to_categories_idxs = papers_ids_to_categories_idxs)
    finetuning_model = load_finetuning_model(evaluation_config["state_dicts_folder"], device, n_unfreeze_layers = 0)
    if evaluation_config["compute_test_embeddings"]:
        compute_test_embeddings(finetuning_model, test_papers, evaluation_config["embeddings_folder"])
    run_testing(finetuning_model, evaluation_config["embeddings_folder"], evaluation_config["configs_folder"], evaluation_config["outputs_folder"], 
                val_users_ids, test_users_no_overlap_ids)        
    visualize_testing(evaluation_config["finetuning_model_path"], evaluation_config["outputs_folder"])