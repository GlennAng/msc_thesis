from finetuning_preprocessing import *
from finetuning_data import *
from finetuning_model import *
from finetuning_evaluation import *
from tqdm import tqdm
import argparse
import json
import numpy as np
import logging
import os
import pickle
import time
import torch

def log_string(logger : logging.Logger, string : str) -> None:
    print(string)
    logger.info(string)

def args_dict_assertions(args_dict : dict) -> None:
    assert args_dict["n_samples_per_user"] > 0 and args_dict["n_samples_per_user"] <= min(32, args_dict["batch_size"])
    assert args_dict["batch_size"] % args_dict["n_samples_per_user"] == 0
    assert args_dict["n_max_positive_samples_per_user"] >= args_dict["n_min_positive_samples_per_user"]
    assert args_dict["n_max_negative_samples_per_user"] >= args_dict["n_min_negative_samples_per_user"]
    assert args_dict["n_max_positive_samples_per_user"] <= args_dict["n_samples_per_user"]
    assert args_dict["n_max_negative_samples_per_user"] <= args_dict["n_samples_per_user"]
    assert args_dict["n_min_positive_samples_per_user"] + args_dict["n_min_negative_samples_per_user"] <= args_dict["n_samples_per_user"]
    assert args_dict["n_max_positive_samples_per_user"] + args_dict["n_max_negative_samples_per_user"] >= args_dict["n_samples_per_user"]

def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description = "Finetuning script")
    parser.add_argument("--seed", type = int, default = 42)

    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--users_sampling_strategy", type = str, default = "uniform", choices = ["uniform", "proportional", "square_root_proportional", "cube_root_proportional"])
    parser.add_argument("--n_samples_per_user", type = int, default = 8)
    parser.add_argument("--n_min_positive_samples_per_user", type = int, default = 3)
    parser.add_argument("--n_min_negative_samples_per_user", type = int, default = 3)
    parser.add_argument("--n_max_positive_samples_per_user", type = int, default = None)
    parser.add_argument("--n_max_negative_samples_per_user", type = int, default = None)
    parser.add_argument("--n_train_negative_samples", type = int, default = 100)

    parser.add_argument("--loss_function", type = str, default = "bcel", choices = ["bcel", "info_nce"])
    parser.add_argument("--info_nce_temperature", type = float, default = 1.0)
    parser.add_argument("--info_nce_log_q_correction", action = "store_true", default = False)
    parser.add_argument("--bcel_pos_weight", type = float, default = 1.0)
    parser.add_argument("--n_batches_total", type = int, default = 10000)
    parser.add_argument("--n_batches_per_val", type = int, default = 500)
    parser.add_argument("--val_metric", type = str, default = "infonce_all")
    parser.add_argument("--early_stopping_patience", type = int, default = None)

    parser.add_argument("--model_path", type = str, default = "/home/scholar/glenn_rp/msc_thesis/data/finetuning/gte_large_256/state_dicts")
    parser.add_argument("--n_unfreeze_layers", type = int, default = 2)
    parser.add_argument("--not_pretrained_projection", action = "store_false", dest = "pretrained_projection")
    parser.add_argument("--not_pretrained_users_embeddings", action = "store_false", dest = "pretrained_users_embeddings")
    parser.add_argument("--not_pretrained_categories_embeddings", action = "store_false", dest = "pretrained_categories_embeddings")

    parser.add_argument("--lr_transformer_model", type = float, default = 1e-5)
    parser.add_argument("--lr_other", type = float, default = 1e-4)
    parser.add_argument("--lr_scheduler", type = str, default = "linear_decay", choices = ["constant", "linear_decay"])
    parser.add_argument("--l2_regularization_transformer_model", type = float, default = 0)
    parser.add_argument("--l2_regularization_other", type = float, default = 0)
    parser.add_argument("--percentage_warmup_steps", type = float, default = 0.1)
    
    parser.add_argument("--testing", action = "store_true", default = False)
    args_dict = vars(parser.parse_args())
    if args_dict["n_max_positive_samples_per_user"] is None:
        args_dict["n_max_positive_samples_per_user"] = args_dict["n_samples_per_user"]
    if args_dict["n_max_negative_samples_per_user"] is None:
        args_dict["n_max_negative_samples_per_user"] = args_dict["n_samples_per_user"]
    args_dict_assertions(args_dict)
    return args_dict

def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def get_dataloaders(args_dict : dict, users_embeddings_ids_to_idxs : dict, papers_ids_to_categories_idxs : dict) -> tuple:
    train_ratings_dataloader = get_train_ratings_dataloader(args_dict, users_embeddings_ids_to_idxs, papers_ids_to_categories_idxs)
    assert len(train_ratings_dataloader) == args_dict["n_batches_total"]
    if args_dict["n_train_negative_samples"] > 0:
        train_negative_samples_dataloader = get_train_negative_samples_dataloader(args_dict["n_train_negative_samples"], 
                                                                                papers_ids_to_categories_idxs, args_dict["n_batches_total"], args_dict["seed"])
        assert len(train_negative_samples_dataloader) == len(train_ratings_dataloader)
    else:
        train_negative_samples_dataloader = None
    return train_ratings_dataloader, train_negative_samples_dataloader

def load_optimizer(finetuning_model : FinetuningModel, lr_transformer_model, l2_regularization_transformer_model, lr_other, l2_regularization_other,
                   lr_scheduler : str, n_batches_total : int, percentage_warmup_steps : float) -> torch.optim.Optimizer:
    param_groups = [{'params': finetuning_model.transformer_model.parameters(), 'lr': lr_transformer_model, 'weight_decay': l2_regularization_transformer_model}, 
                    {'params': finetuning_model.projection.parameters(), 'lr': lr_other, 'weight_decay': l2_regularization_other}, 
                    {'params': finetuning_model.users_embeddings.parameters(), 'lr': lr_other, 'weight_decay': l2_regularization_other},
                    {'params': finetuning_model.categories_embeddings.parameters(), 'lr': lr_other, 'weight_decay': l2_regularization_other}]
    optimizer = torch.optim.Adam(param_groups)
    if lr_scheduler == "constant":
        return optimizer, None
    elif lr_scheduler == "linear_decay":
        n_warmup_steps = round(n_batches_total * percentage_warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, n_total_steps = n_batches_total, n_warmup_steps = n_warmup_steps)
    return optimizer, scheduler

def get_linear_schedule_with_warmup(optimizer : torch.optim.Optimizer, n_total_steps : int, n_warmup_steps : int) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step : int):
        if current_step < n_warmup_steps:
            return float(current_step) / float(n_warmup_steps)
        else:
            return max(0.0, float(n_total_steps - current_step) / float(n_total_steps - n_warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch = -1)

def init_logging(args_dict : dict) -> logging.Logger:
    logging.basicConfig(filename = f"{args_dict['outputs_folder']}/logger.log", filemode = "w", format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s", level = logging.INFO)
    logger = logging.getLogger("finetuning")
    logger.info(f"Finetuning script started with config: {args_dict}.")
    return logger

def save_config(args_dict : dict) -> None:
    with open(f"{args_dict['outputs_folder']}/config.json", "w") as f:
        json.dump(args_dict, f, indent = 4)

def compute_info_nce_loss(train_ratings_scores : torch.tensor, user_idx_tensor : torch.tensor, label_tensor : torch.tensor, 
                          negative_samples_scores : torch.tensor, sorted_unique_user_idx_tensor : torch.tensor, args_dict : dict) -> float:
    info_nce_temperature = args_dict["info_nce_temperature"]
    enumerated_users_dict = {user_idx.item(): i for i, user_idx in enumerate(sorted_unique_user_idx_tensor)}
    negatives_users_dict = {}
    for user_idx in enumerated_users_dict.keys():
        negative_train_ratings_scores_user = train_ratings_scores[(user_idx_tensor == user_idx) & (label_tensor == 0)]
        if negative_samples_scores is not None:
            negative_samples_scores_user = negative_samples_scores[(user_idx_tensor == user_idx) & (label_tensor == 0)]
            negatives_users_dict[user_idx] = torch.cat((negative_train_ratings_scores_user, negative_samples_scores_user))
        else:
            negatives_users_dict[user_idx] = negative_train_ratings_scores_user
        negatives_users_dict[user_idx] /= info_nce_temperature

    positive_indices = torch.where(label_tensor == 1)[0]
    positive_scores, positive_users_idxs = train_ratings_scores[positive_indices], user_idx_tensor[positive_indices]
    losses = []
    for i, pos_idx in enumerate(positive_indices):
        pos_score, pos_user_idx = positive_scores[i], positive_users_idxs[i].item()
        pos_score = pos_score / info_nce_temperature
        losses.append(-torch.log_softmax(torch.cat((pos_score.unsqueeze(0), negatives_users_dict[pos_user_idx])), dim = 0)[0])
    return torch.mean(torch.stack(losses))

def process_val(finetuning_model : FinetuningModel, args_dict : dict, val_data : dict, n_updates_so_far : int, logger : logging.Logger,
                baseline_metric : float = None, early_stopping_counter : int = None) -> tuple:
    if baseline_metric is None or early_stopping_counter is None:
        assert n_updates_so_far == 0
    baseline_metric_string = "None" if baseline_metric is None else round_number(baseline_metric)
    val_scores_batch, val_scores_batch_string = run_validation(finetuning_model, val_data["val_ratings"], val_data["val_negative_samples"], info_nce_temperature = args_dict["info_nce_temperature"], print_results = False)
    val_scores_batch_string = f"VALIDATION METRICS AFTER UPDATE {n_updates_so_far} / {args_dict['n_batches_total']}:" + val_scores_batch_string
    baseline_metric_batch = val_scores_batch[f"val_{args_dict['val_metric']}"]
    if baseline_metric is None:
        improvement = True
    else:
        if args_dict["val_metric"] == "bcel" or args_dict["val_metric"].startswith("infonce"):
            improvement = (baseline_metric_batch < baseline_metric)
        else:
            improvement = (baseline_metric_batch > baseline_metric)
    if improvement:
        if baseline_metric is not None:
            finetuning_model.save_finetuning_model(args_dict["outputs_folder"] + "/state_dicts")
        val_scores_batch_string += f"\nNew Optimal Value for Metric {args_dict['val_metric'].upper()}: {round_number(baseline_metric_batch)} after previously {baseline_metric_string}."
        baseline_metric, early_stopping_counter = baseline_metric_batch, 0
    else:
        val_scores_batch_string += f"\nNo Improvement for Metric {args_dict['val_metric'].upper()}: {round_number(baseline_metric_batch)} after previously {baseline_metric_string}."
        early_stopping_counter += 1
    val_scores_batch_string += f" Early Stopping Counter: {early_stopping_counter} / {args_dict['early_stopping_patience']}.\n"
    log_string(logger, val_scores_batch_string)
    return val_scores_batch, baseline_metric, early_stopping_counter

def process_batch(finetuning_model : FinetuningModel, args_dict : dict, optimizer : torch.optim.Optimizer, scheduler : torch.optim.lr_scheduler, train_ratings_batch : dict,
                 train_negative_samples_batch : dict) -> float:
    user_idx_tensor = train_ratings_batch["user_idx"].to(finetuning_model.device)
    sorted_unique_user_idx_tensor = torch.unique(user_idx_tensor, sorted = True).to(finetuning_model.device)
    optimizer.zero_grad()
    with torch.autocast(device_type = 'cuda', dtype = torch.float16):
        train_ratings_scores = finetuning_model(eval_type = "train", input_ids_tensor = train_ratings_batch["input_ids"], attention_mask_tensor = train_ratings_batch["attention_mask"],
                                                user_idx_tensor = user_idx_tensor, category_idx_tensor = train_ratings_batch["category_idx"])
        if train_negative_samples_batch is not None:
            train_negative_samples_scores = finetuning_model(eval_type = "negative_samples", input_ids_tensor = train_negative_samples_batch["input_ids"],
                                                            attention_mask_tensor = train_negative_samples_batch["attention_mask"], user_idx_tensor = sorted_unique_user_idx_tensor, 
                                                            category_idx_tensor = train_negative_samples_batch["category_idx"])
        else:
            train_negative_samples_scores = None
        if args_dict["loss_function"] == "bcel":
            label_tensor = train_ratings_batch["label"].float().to(finetuning_model.device)
            pos_weight = torch.tensor(args_dict["bcel_pos_weight"]).to(finetuning_model.device)
            loss = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)(train_ratings_scores, label_tensor)
        elif args_dict["loss_function"] == "info_nce":
            label_tensor = train_ratings_batch["label"].to(finetuning_model.device)
            loss = compute_info_nce_loss(train_ratings_scores, user_idx_tensor, label_tensor, train_negative_samples_scores, sorted_unique_user_idx_tensor, args_dict)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return loss.item()

def run_training(finetuning_model : FinetuningModel, optimizer : torch.optim.Optimizer, scheduler : torch.optim.lr_scheduler, train_ratings_dataloader : DataLoader, 
                 train_negative_samples_dataloader : DataLoader, args_dict : dict, val_data : dict, logger : logging.Logger) -> tuple:
    val_scores_batch, baseline_metric, early_stopping_counter = process_val(finetuning_model, args_dict, val_data, 0, logger)
    train_losses, val_scores = [], [(0, val_scores_batch)]
    finetuning_model.train()
    train_ratings_iter = iter(train_ratings_dataloader)
    if train_negative_samples_dataloader is not None:
        train_negative_samples_iter = iter(train_negative_samples_dataloader)
    start_time = time.time()

    for i in tqdm(range(args_dict['n_batches_total']), desc = "Training Batches", unit = "Batch"):
        train_ratings_batch = next(train_ratings_iter)
        train_negative_samples_batch = next(train_negative_samples_iter) if train_negative_samples_dataloader is not None else None
        train_loss = process_batch(finetuning_model, args_dict, optimizer, scheduler, train_ratings_batch, train_negative_samples_batch)
        train_losses.append((i, train_loss))
        n_batches_processed_so_far = i + 1
        if n_batches_processed_so_far % args_dict["n_batches_per_val"] == 0 or n_batches_processed_so_far == args_dict['n_batches_total']:
            train_ratings_dataloader.batch_sampler.run_test(train_ratings_batch)
            log_string(logger, f"\nTRAIN RATINGS BATCH BEFORE UPDATE {n_batches_processed_so_far} / {args_dict['n_batches_total']}\n{print_train_ratings_batch(train_ratings_batch)}")
            if train_negative_samples_dataloader is not None:
                train_negative_samples_dataloader.batch_sampler.run_test(train_negative_samples_batch)
            train_losses_chunk = [loss for _, loss in train_losses[-args_dict["n_batches_per_val"] :]]
            log_string(logger, f"AVERAGED TRAIN LOSS: {round_number(np.mean(train_losses_chunk))}.\n")
            val_scores_batch, baseline_metric, early_stopping_counter = process_val(finetuning_model, args_dict, val_data, n_batches_processed_so_far, logger,
                                                                                    baseline_metric, early_stopping_counter)
            val_scores.append((n_batches_processed_so_far, val_scores_batch))
            if args_dict["early_stopping_patience"] is not None and early_stopping_counter >= args_dict["early_stopping_patience"]:
                log_string(logger, f"\nEarly stopping after {n_batches_processed_so_far} Batches.")
                break
    args_dict["time_elapsed"] = time.time() - start_time
    return train_losses, val_scores

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args_dict = parse_arguments()
    set_all_seeds(args_dict["seed"])

    users_embeddings_ids_to_idxs, papers_ids_to_categories_idxs = load_users_embeddings_ids_to_idxs(), load_papers_ids_to_categories_idxs()
    val_data = load_val_data(users_embeddings_ids_to_idxs, papers_ids_to_categories_idxs)
    train_ratings_dataloader, train_negative_samples_dataloader = get_dataloaders(args_dict, users_embeddings_ids_to_idxs, papers_ids_to_categories_idxs)
    finetuning_model = load_finetuning_model(args_dict["model_path"], device, val_data["val_users_embeddings_idxs"], n_unfreeze_layers = args_dict["n_unfreeze_layers"],
                                            pretrained_projection = args_dict["pretrained_projection"], pretrained_users_embeddings = args_dict["pretrained_users_embeddings"],
                                            pretrained_categories_embeddings = args_dict["pretrained_categories_embeddings"])
    args_dict["n_transformer_layers"] = finetuning_model.count_transformer_layers()
    args_dict["n_transformer_parameters"], args_dict["n_unfrozen_transformer_parameters"] = finetuning_model.count_transformer_parameters()
    optimizer, scheduler = load_optimizer(finetuning_model, args_dict["lr_transformer_model"], args_dict["l2_regularization_transformer_model"], 
                                          args_dict["lr_other"], args_dict["l2_regularization_other"], args_dict["lr_scheduler"], args_dict["n_batches_total"], args_dict["percentage_warmup_steps"])
    if not args_dict["testing"]:
        args_dict["outputs_folder"] = FILES_SAVE_PATH + f"/experiments/{finetuning_model.transformer_model_name}_{time.strftime('%Y-%m-%d-%H-%M')}"
        os.makedirs(args_dict["outputs_folder"], exist_ok = True)
        logger = init_logging(args_dict)

        train_losses, val_scores = run_training(finetuning_model, optimizer, scheduler, train_ratings_dataloader, train_negative_samples_dataloader, args_dict, val_data, logger)
        with open(f"{args_dict['outputs_folder']}/train_losses.pkl", "wb") as f:
            pickle.dump(train_losses, f)
        with open(f"{args_dict['outputs_folder']}/val_scores.pkl", "wb") as f:
            pickle.dump(val_scores, f)
        print(f"Finished Finetuning:     {args_dict['outputs_folder']}.")
        save_config(args_dict)
        if os.path.exists(args_dict["outputs_folder"] + "/state_dicts"):
            os.system(f"python src/finetuning_evaluation.py --finetuning_model_path {args_dict['outputs_folder']}")