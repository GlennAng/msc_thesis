from finetuning_preprocessing import *
from finetuning_data import *
from finetuning_model import *
from finetuning_evaluation import *
from torch.utils.data import DataLoader
import argparse
import json
import numpy as np
import logging
import os
import pickle
import time
import torch

def log_string(string : str) -> None:
    print(string)
    logger.info(string)

def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def args_dict_assertions(args_dict : dict) -> None:
    assert args_dict["n_samples_per_user"] > 0 and args_dict["n_samples_per_user"] <= min(32, args_dict["batch_size"])
    assert args_dict["batch_size"] % args_dict["n_samples_per_user"] == 0

def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description = "Finetuning script")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--n_epochs", type = int, default = 10)
    parser.add_argument("--n_samples_per_user", type = int, default = 16)
    parser.add_argument("--users_sampling_strategy", type = str, default = "uniform", choices = ["uniform", "proportional", "square_root_proportional", "cube_root_proportional"])
    parser.add_argument("--class_balancing", action = "store_true", default = False)
    parser.add_argument("--n_batches_per_val", type = int, default = 500)
    parser.add_argument("--val_measure_training", action = "store_true", default = False)
    parser.add_argument("--val_measure_negative_samples", action = "store_true", default = False)
    parser.add_argument("--val_metric", type = str, default = "bcel", choices = EXPLICIT_NEGATIVES_METRICS)
    parser.add_argument("--model_path", type = str, default = "/home/scholar/glenn_rp/msc_thesis/data/finetuning/gte_base_256/state_dicts")
    parser.add_argument("--scores_path", type = str, default = "/home/scholar/glenn_rp/msc_thesis/data/finetuning/gte_base_256/scores.json")
    parser.add_argument("--n_unfreeze_layers", type = int, default = 4)
    parser.add_argument("--transformer_model_lr", type = float, default = 1e-5)
    parser.add_argument("--projection_lr", type = float, default = 1e-4)
    parser.add_argument("--users_embeddings_lr", type = float, default = 1e-4)
    args_dict = vars(parser.parse_args())
    args_dict_assertions(args_dict)
    return args_dict

def get_baseline_values(scores_dict : dict) -> tuple:
    if type(scores_dict) == str:
        with open(scores_dict, "r") as f:
            scores_dict = json.load(f)
    baseline_loss = scores_dict["bcel_val"]
    baseline_metric = scores_dict[f"{args_dict['val_metric']}_val"]
    return baseline_loss, baseline_metric

def get_baseline_metric(scores_dict : dict, metric : str, transformer_model_name : str) -> float:
    if type(scores_dict) == str:
        with open(scores_dict, "r") as f:
            scores_dict = json.load(f)
    baseline_metric = scores_dict[metric + "_val"]
    all_experiments = os.listdir(f"{FILES_SAVE_PATH}/experiments")
    all_experiments = [x for x in all_experiments if x.startswith(transformer_model_name)]
    for experiment in all_experiments:
        if os.path.exists(f"{FILES_SAVE_PATH}/experiments/{experiment}/scores.json"):
            with open(f"{FILES_SAVE_PATH}/experiments/{experiment}/scores.json", "r") as f:
                scores_dict_experiment = json.load(f)
            new_metric = scores_dict_experiment[metric + "_val"]
            if metric == "bcel":
                if new_metric < baseline_metric:
                    baseline_metric = new_metric
            else:
                if new_metric > baseline_metric:
                    baseline_metric = new_metric
    return baseline_metric

def save_config(args_dict : dict) -> None:
    os.makedirs(args_dict["outputs_folder"], exist_ok = True)
    with open(f"{args_dict['outputs_folder']}/config.json", "w") as f:
        json.dump(args_dict, f, indent = 4)

def init_logging(args_dict : dict) -> logging.Logger:
    logging.basicConfig(filename = f"{args_dict['outputs_folder']}/logger.log", filemode = "w", format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s", level = logging.INFO)
    logger = logging.getLogger("finetuning")
    logger.info(f"Finetuning script started with config: {args_dict}.")
    return logger

def print_batch(batch : dict) -> str:
    s = ""
    user_idx_list, paper_id_list, label_list = batch["user_idx"][:24].tolist(), batch["paper_id"][:24].tolist(), batch["label"][:24].tolist()
    s += f"User IDXs: {user_idx_list}.\n"
    s += f"Paper IDs: {paper_id_list}.\n"
    s += f"Labels: {label_list}."
    return s

def load_dataloader(train_dataset : FinetuningDataset, batch_size : int, n_samples_per_user : int, users_sampling_strategy : str, class_balancing : bool, seed : int) -> DataLoader:
    sampler = FinetuningSampler(train_dataset, batch_size, n_samples_per_user, users_sampling_strategy, class_balancing, seed)
    dataloader = DataLoader(train_dataset, sampler = sampler, batch_size = batch_size, drop_last = False, num_workers = 4, pin_memory = True)
    batch = next(iter(dataloader))
    first_batch_test = sampler.run_test(batch)
    if first_batch_test:
        logger.info(f"First Batch passed the test:\n{print_batch(batch)}")
    return dataloader
    
def load_optimizer(finetuning_model : FinetuningModel, transformer_model_lr : float, projection_lr : float, users_embeddings_lr : float) -> torch.optim.Optimizer:
    param_groups = [{'params': finetuning_model.transformer_model.parameters(), 'lr': transformer_model_lr}, 
                    {'params': finetuning_model.projection.parameters(), 'lr': projection_lr}, 
                    {'params': finetuning_model.users_embeddings.parameters(), 'lr': users_embeddings_lr}]
    return torch.optim.Adam(param_groups)

def round_number(number : float, decimal_places : int = 4) -> float:
    return round(number, decimal_places)

def process_val_explicit_negatives(scores_dict : dict, measure_training : bool) -> str:
    explicit_negatives_metrics = ["bcel", "balanced_accuracy", "ndcg@5_explicit_negatives", "mrr_explicit_negatives"]
    s = "\nExplicit Negatives Scores: "
    for i, score in enumerate(explicit_negatives_metrics):
        if i > 0:
            s += ", "
        s += f"{score.upper()}: {round_number(scores_dict[score + '_val'])}"
        if measure_training:
            s += f" ({round_number(scores_dict[score + '_train'])})"
    s += "."
    return s

def process_val_negative_samples(scores_dict : dict, measure_training : bool) -> str:
    negative_samples_metrics = ["ndcg@5_negative_samples", "mrr_negative_samples"]
    s = "\nNegative Samples Scores: "
    for i, score in enumerate(negative_samples_metrics):
        if i > 0:
            s += ", "
        s += f"{score.upper()}: {round_number(scores_dict[score + '_val'])}"
        if measure_training:
            s += f" ({round_number(scores_dict[score + '_train'])})"
    s += "."
    return s

def process_val(finetuning_model : FinetuningModel, args_dict : dict, datasets_dict : dict, baseline_metric : float, n_batches_processed_so_far : int, n_batches_total : int) -> tuple:
    negative_samples = datasets_dict["negative_samples"] if args_dict["val_measure_negative_samples"] else None
    train_val_dataset = datasets_dict["train_val_dataset"] if args_dict["val_measure_training"] else None
    scores_dict = run_validation(finetuning_model, datasets_dict["val_dataset"], negative_samples, train_val_dataset)
    new_baseline_metric = scores_dict[args_dict["val_metric"] + "_val"]
    s = f"Validation after having processed {n_batches_processed_so_far}/{n_batches_total} Batches:"
    s += process_val_explicit_negatives(scores_dict, args_dict["val_measure_training"])
    if args_dict["val_measure_negative_samples"]:
        s += process_val_negative_samples(scores_dict, args_dict["val_measure_training"])
    if args_dict["val_metric"] == "bcel":
        improvement = (new_baseline_metric < baseline_metric)
    else:
        improvement = (new_baseline_metric > baseline_metric)
    if improvement:
        s += f"\nNew Optimal Value for Metric {args_dict['val_metric'].upper()}: {round_number(new_baseline_metric)} after previously {round_number(baseline_metric)}."
        baseline_metric = new_baseline_metric
        finetuning_model.save_finetuning_model(args_dict["outputs_folder"] + "/state_dicts")
    log_string(s)
    return scores_dict["bcel_val"], baseline_metric, improvement

def process_batch(finetuning_model : FinetuningModel, optimizer : torch.optim.Optimizer, batch : dict, n_batches_processed_so_far : int) -> float:
    optimizer.zero_grad()
    batch = {k: batch[k].to(finetuning_model.device) for k in ["user_idx", "label", "input_ids", "attention_mask"]}
    with torch.autocast(device_type = 'cuda', dtype = torch.float16):
        dot_products = finetuning_model(eval_type = "train", input_ids_tensor = batch["input_ids"], attention_mask_tensor = batch["attention_mask"], user_idx_tensor = batch["user_idx"])
        loss = torch.nn.BCEWithLogitsLoss()(dot_products, batch["label"].float())
        if n_batches_processed_so_far % 250 == 0:
            print(get_gpu_info())
    loss.backward()
    optimizer.step()
    return loss.item()

def run_training(finetuning_model : FinetuningModel, optimizer : torch.optim.Optimizer, dataloader : DataLoader, args_dict : dict, datasets_dict : dict, baseline_metric : float) -> None:
    early_stopping_counter = 0
    finetuning_model.train()
    n_epochs = args_dict["n_epochs"]
    log_string(f"Starting training for {n_epochs} epochs.")
    n_batches_processed_so_far = 0
    n_batches_total = len(dataloader) * n_epochs
    for i in range(n_epochs):
        epoch_start_time = time.time()
        dataloader.sampler.epoch = i
        for j, batch in enumerate(dataloader):
            if j == 0:
                log_string(f"Starting Epoch {i + 1} of {n_epochs}. First Batch:\n{print_batch(batch)}")
            train_loss = process_batch(finetuning_model, optimizer, batch, n_batches_processed_so_far)
            print(f"Epoch {i+1}/{n_epochs}, Batch {j + 1}/{len(dataloader)}. Loss: {train_loss:.4f}.")
            train_losses.append((n_batches_processed_so_far, train_loss))
            n_batches_processed_so_far += 1
            if n_batches_processed_so_far in [1, 10, 100] or n_batches_processed_so_far % args_dict["n_batches_per_val"] == 0:
                val_loss, baseline_metric, improvement = process_val(finetuning_model, args_dict, datasets_dict, baseline_metric, n_batches_processed_so_far, n_batches_total)
                if improvement:
                    early_stopping_counter = 0
                else:
                    if n_batches_processed_so_far > 100:
                        early_stopping_counter += 1
                val_losses.append((n_batches_processed_so_far, val_loss))
                if early_stopping_counter >= 5:
                    log_string(f"Early stopping after {i + 1} Epochs and {n_batches_processed_so_far} Batches.")
                    return
            if n_batches_processed_so_far % len(dataloader) == 0:
                log_string(f"Finished Epoch {i + 1} of {args_dict['n_epochs']}. Time: {time.time() - epoch_start_time:.2f} seconds.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args_dict = parse_arguments()
    set_all_seeds(args_dict["seed"])

    datasets_dict = load_datasets_dict()
    finetuning_model = load_finetuning_model_full(args_dict["model_path"], device, datasets_dict["val_users_embeddings_idxs"])
    baseline_loss, baseline_metric = get_baseline_values(args_dict["scores_path"])
    train_losses, val_losses = [], [(0, baseline_loss)]
    args_dict["outputs_folder"] = FILES_SAVE_PATH + f"/experiments/{finetuning_model.transformer_model_name}_{time.strftime('%Y-%m-%d-%H-%M')}"
    save_config(args_dict)
    logger = init_logging(args_dict)

    dataloader = load_dataloader(datasets_dict["train_dataset"], args_dict["batch_size"], args_dict["n_samples_per_user"], 
                                 args_dict["users_sampling_strategy"], args_dict["class_balancing"], args_dict["seed"])
    optimizer = load_optimizer(finetuning_model, args_dict["transformer_model_lr"], args_dict["projection_lr"], args_dict["users_embeddings_lr"])
    run_training(finetuning_model, optimizer, dataloader, args_dict, datasets_dict, baseline_metric)
    with open(f"{args_dict['outputs_folder']}/train_losses.pkl", "wb") as f:
        pickle.dump(train_losses, f)
    with open(f"{args_dict['outputs_folder']}/val_losses.pkl", "wb") as f:
        pickle.dump(val_losses, f)
    
    print(f"Finished Finetuning:     {args_dict['outputs_folder']}.")
    if os.path.exists(args_dict["outputs_folder"] + "/state_dicts"):
        os.system(f"python src/finetuning_evaluation.py {args_dict['outputs_folder']}")