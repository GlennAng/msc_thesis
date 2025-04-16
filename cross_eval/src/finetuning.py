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
import time
import torch

def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description = "Finetuning script")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--n_epochs", type = int, default = 3)
    parser.add_argument("--n_samples_per_user", type = int, default = 16)
    parser.add_argument("--users_sampling_strategy", type = str, default = "uniform", choices = ["uniform", "proportional", "square_root_proportional", "cube_root_proportional"])
    parser.add_argument("--class_balancing", action = "store_true", default = False)
    parser.add_argument("--n_batches_per_val", type = int, default = 500)
    parser.add_argument("--metric", type = str, default = "ndcg@5", choices = ["auc", "ndcg@5", "mrr", "hr@1"])

    parser.add_argument("--transformer_model", type = str, default = GTE_BASE_PATH)
    parser.add_argument("--projection_state_dict", type = str, default = "/home/scholar/glenn_rp/msc_thesis/data/finetuning/parameters/gte_base_256_projection.pt")
    parser.add_argument("--users_embeddings_state_dict", type = str, default = "/home/scholar/glenn_rp/msc_thesis/data/finetuning/parameters/gte_base_256_users_embeddings.pt")

    parser.add_argument("--transformer_model_lr", type = float, default = 1e-5)
    parser.add_argument("--projection_lr", type = float, default = 1e-4)
    parser.add_argument("--users_embeddings_lr", type = float, default = 1e-4)

    args_dict = vars(parser.parse_args())
    assert args_dict["batch_size"] % args_dict["n_samples_per_user"] == 0, "Batch size must be divisible by n_samples_per_user"
    args_dict["time"] = time.strftime("%Y-%m-%d_%H-%M-%S")
    args_dict["outputs_folder"] = FILES_SAVE_PATH + f"/experiments/{args_dict['time']}"
    return args_dict

def save_config(args_dict : dict) -> None:
    os.makedirs(args_dict["outputs_folder"], exist_ok = True)
    with open(f"{args_dict['outputs_folder']}/config.json", "w") as f:
        json.dump(args_dict, f, indent = 4)

def load_optimizer(finetuning_model : FinetuningModel, transformer_model_lr : float, projection_lr : float, users_embeddings_lr : float) -> torch.optim.Optimizer:
    param_groups = [{'params': finetuning_model.transformer_model.parameters(), 'lr': transformer_model_lr}, 
                    {'params': finetuning_model.projection.parameters(), 'lr': projection_lr}, 
                    {'params': finetuning_model.users_embeddings.parameters(), 'lr': users_embeddings_lr}]
    return torch.optim.Adam(param_groups)

if __name__ == "__main__":
    args_dict = parse_arguments()
    save_config(args_dict)
    set_all_seeds(args_dict["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.basicConfig(filename = f"{args_dict['outputs_folder']}/finetuning.log", filemode = "w", format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s", level = logging.INFO)
    logger = logging.getLogger("finetuning")
    logger.info(f"Finetuning script started with config: {args_dict}.")

    datasets_dict = load_datasets_dict()
    finetuning_model = load_finetuning_model(args_dict["transformer_model"], device, args_dict["projection_state_dict"], args_dict["users_embeddings_state_dict"], 
                                             datasets_dict["val_users_embeddings_idxs"])
    optimizer = load_optimizer(finetuning_model, args_dict["transformer_model_lr"], args_dict["projection_lr"], args_dict["users_embeddings_lr"])

    baseline_values = run_validation(finetuning_model, datasets_dict["val_dataset"], datasets_dict["negative_samples"])[0][0]
    logger.info(f"Baseline values after first Validation: {baseline_values}.")
    BASELINE_VALUE = baseline_values[METRICS_LIST.index(args_dict["metric"])]
    logger.info(f"Baseline value for {args_dict['metric']}: {BASELINE_VALUE}.")

    sampler = FinetuningSampler(datasets_dict["train_dataset"], args_dict["batch_size"], args_dict["n_samples_per_user"], 
                                args_dict["users_sampling_strategy"], args_dict["class_balancing"], args_dict["seed"])
    dataloader = DataLoader(datasets_dict["train_dataset"], sampler = sampler, batch_size = args_dict["batch_size"], drop_last = False)
    batch = next(iter(dataloader))
    sampler.run_test(batch)
    logger.info(f"Sampler test: {batch}.")


    print(f"Finished Finetuning:     {args_dict['outputs_folder']}.")
