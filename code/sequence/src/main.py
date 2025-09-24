import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...finetuning.src.finetuning_data import round_number
from ...finetuning.src.finetuning_main import log_string, set_all_seeds
from ...src.load_files import FINETUNING_MODEL
from ...src.project_paths import ProjectPaths
from .data.eval_data import load_val_data
from .data.train_data_negative_samples import get_train_negative_samples_dataloader
from .data.train_data_sessions import get_train_sessions_dataloader
from .eval.sequence_val import process_validation
from .eval.train_utils import load_optimizer, process_batch, compute_info_nce_loss
from .models.recommender import Recommender, load_recommender_from_scratch


def get_embeddings_path(use_finetuned_embeddings: bool) -> Path:
    if use_finetuned_embeddings:
        return ProjectPaths.sequence_finetuned_embeddings_path()
    else:
        return ProjectPaths.sequence_non_finetuned_embeddings_path()


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description="Sequence script")
    parser.add_argument("--users_encoder_type", type=str, required=True)
    parser.add_argument(
        "--not_use_finetuned_embeddings",
        action="store_false",
        dest="use_finetuned_embeddings",
        default=True,
    )
    parser.add_argument("--info_nce_temperature", type=float, default=2.5)

    parser.add_argument("--n_candidates_per_batch", type=int, default=16)
    parser.add_argument("--n_negrated_per_candidate", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_score_name", type=str, default="ndcg_all")

    parser.add_argument("--nrms_num_heads", type=int, default=4)
    parser.add_argument("--nrms_query_dim", type=int, default=256)

    parser.add_argument("--n_batches_total", type=int, default=50000)
    parser.add_argument("--n_batches_per_val", type=int, default=5000)
    parser.add_argument("--n_warmup_steps", type=int, default=500)

    parser.add_argument("--recommender_lr", type=float, default=3e-6)
    parser.add_argument(
        "--recommender_lr_scheduler",
        type=str,
        default="constant",
        choices=["constant", "linear_decay"],
    )
    parser.add_argument("--recommender_weight_decay", type=float, default=0.0)

    parser.add_argument("--negrated_hard_constraint_max_n_ratings", type=int, default=15)
    parser.add_argument("--negrated_hard_constraint_max_n_sessions", type=int, default=None)
    parser.add_argument("--negrated_hard_constraint_max_n_days", type=int, default=None)

    parser.add_argument("--histories_hard_constraint_min_n_train_posrated", type=int, default=0)
    parser.add_argument("--histories_hard_constraint_max_n_train_rated", type=int, default=None)
    parser.add_argument("--histories_soft_constraint_max_n_train_sessions", type=int, default=None)
    parser.add_argument("--histories_soft_constraint_max_n_train_days", type=int, default=None)
    parser.add_argument(
        "--histories_remove_negrated_from_history", action="store_true", default=False
    )
    parser.add_argument("--testing", action="store_true", default=False)

    parser.add_argument("--early_stopping_patience", type=int, default=None)

    parser.add_argument("--n_train_negative_samples", type=int, default=20)

    args = vars(parser.parse_args())
    args["embeddings_path"] = get_embeddings_path(args["use_finetuned_embeddings"])
    args["outputs_folder"] = (
        ProjectPaths.sequence_data_experiments_path()
        / f"{FINETUNING_MODEL}_{time.strftime('%Y-%m-%d-%H-%M')}"
    )
    return args


def get_users_encoder_type_specific_args(args_dict: dict) -> dict:
    users_encoder_type = args_dict["users_encoder_type"]
    if users_encoder_type == "MeanPoolingUsersEncoder":
        return {}
    elif users_encoder_type == "NRMSUsersEncoder":
        return {
            "num_heads": args_dict["nrms_num_heads"],
            "query_dim": args_dict["nrms_query_dim"],
        }
    elif users_encoder_type == "GRUUsersEncoder":
        return {
            "hidden_dim": args_dict.get("gru_hidden_dim", 356),
            "num_layers": args_dict.get("gru_num_layers", 1),
            "dropout": args_dict.get("gru_dropout", 0.2),
        }
    elif users_encoder_type == "MultiLayerNRMSUsersEncoder":
        return {
        }
    else:
        raise ValueError(f"Unknown users_encoder_type: {users_encoder_type}")


def load_recommender_main(args_dict: dict, device: torch.device) -> Recommender:
    recommender_args = {
        "users_encoder_type": args_dict["users_encoder_type"],
        "embeddings_path": args_dict["embeddings_path"],
        "random_seed": args_dict["seed"],
        "device": device,
    }
    recommender_args.update(get_users_encoder_type_specific_args(args_dict))
    recommender = load_recommender_from_scratch(**recommender_args).train()
    print(recommender.get_memory_footprint())
    return recommender


def load_val_data_main(args_dict: dict) -> dict:
    return load_val_data(
        histories_hard_constraint_min_n_train_posrated=args_dict[
            "histories_hard_constraint_min_n_train_posrated"
        ],
        histories_hard_constraint_max_n_train_rated=args_dict[
            "histories_hard_constraint_max_n_train_rated"
        ],
        histories_soft_constraint_max_n_train_sessions=args_dict[
            "histories_soft_constraint_max_n_train_sessions"
        ],
        histories_soft_constraint_max_n_train_days=args_dict[
            "histories_soft_constraint_max_n_train_days"
        ],
        histories_remove_negrated_from_history=args_dict["histories_remove_negrated_from_history"],
    )


def load_optimizer_main(recommender: Recommender, args_dict: dict) -> tuple:
    return load_optimizer(
        recommender=recommender,
        lr=args_dict["recommender_lr"],
        lr_scheduler=args_dict["recommender_lr_scheduler"],
        weight_decay=args_dict["recommender_weight_decay"],
        n_batches_total=args_dict["n_batches_total"],
        n_warmup_steps=args_dict["n_warmup_steps"],
    )


def load_train_dataloaders_main(recommender: Recommender, args_dict: dict) -> tuple:
    train_negative_samples_dataloader = get_train_negative_samples_dataloader(
        papers_embeddings=recommender.papers_embeddings,
        papers_ids_to_idxs=recommender.papers_ids_to_idxs,
        n_train_negative_samples=args_dict["n_train_negative_samples"],
        n_batches_total=args_dict["n_batches_total"],
        seed=args_dict["seed"],
    )
    train_sessions_dataloader = get_train_sessions_dataloader(
        papers_embeddings=recommender.papers_embeddings,
        papers_ids_to_idxs=recommender.papers_ids_to_idxs,
        n_candidates_per_batch=args_dict["n_candidates_per_batch"],
        n_negrated_per_candidate=args_dict["n_negrated_per_candidate"],
        n_batches_total=args_dict["n_batches_total"],
        seed=args_dict["seed"],
        negrated_hard_constraint_max_n_ratings=args_dict["negrated_hard_constraint_max_n_ratings"],
        negrated_hard_constraint_max_n_sessions=args_dict[
            "negrated_hard_constraint_max_n_sessions"
        ],
        negrated_hard_constraint_max_n_days=args_dict["negrated_hard_constraint_max_n_days"],
        histories_hard_constraint_min_n_train_posrated=args_dict[
            "histories_hard_constraint_min_n_train_posrated"
        ],
        histories_hard_constraint_max_n_train_rated=args_dict[
            "histories_hard_constraint_max_n_train_rated"
        ],
        histories_soft_constraint_max_n_train_sessions=args_dict[
            "histories_soft_constraint_max_n_train_sessions"
        ],
        histories_soft_constraint_max_n_train_days=args_dict[
            "histories_soft_constraint_max_n_train_days"
        ],
        histories_remove_negrated_from_history=args_dict["histories_remove_negrated_from_history"],
    )
    return train_sessions_dataloader, train_negative_samples_dataloader


def init_logging(args_dict: dict) -> logging.Logger:
    filename = Path(args_dict["outputs_folder"]).resolve() / "logger.log"
    logging.basicConfig(
        filename=filename,
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger("sequence")
    logger.info(f"Sequence script started with config: {args_dict}.")
    return logger


def save_config(args_dict: dict) -> None:
    copy_args_dict = args_dict.copy()
    for key in args_dict:
        if isinstance(args_dict[key], Path):
            copy_args_dict[key] = str(args_dict[key])
    with open(args_dict["outputs_folder"] / "config.json", "w") as f:
        json.dump(copy_args_dict, f, indent=4)


def save_values(
    train_info_nce_losses: list,
    train_cat_losses: list,
    val_scores: list,
    outputs_folder: Path,
) -> None:
    with open(outputs_folder / "train_info_nce_losses.pkl", "wb") as f:
        pickle.dump(train_info_nce_losses, f)
    with open(outputs_folder / "train_cat_losses.pkl", "wb") as f:
        pickle.dump(train_cat_losses, f)
    with open(outputs_folder / "val_scores.pkl", "wb") as f:
        pickle.dump(val_scores, f)


def process_validation_main(
    recommender: Recommender,
    args_dict: dict,
    val_data: dict,
    previous_best_score: float,
    early_stopping_counter: int,
    logger: logging.Logger,
    n_batches_processed_so_far: int,
    train_info_nce_losses: list,
    train_cat_losses: list,
) -> tuple:
    log_string(
        logger,
        f"\nProcessing Validation after {n_batches_processed_so_far} Batches.",
    )

    best_val_score, early_stopping_counter, batch_val_score = process_validation(
        recommender=recommender,
        args_dict=args_dict,
        val_data=val_data,
        previous_best_score=previous_best_score,
        early_stopping_counter=early_stopping_counter,
        logger=logger,
    )

    if train_info_nce_losses:
        train_info_nce_losses_chunk = [
            loss for _, loss in train_info_nce_losses[-args_dict["n_batches_per_val"] :]
        ]
        info_nce_s = (
            f"Train InfoNCE Loss over the previous {args_dict['n_batches_per_val']} Batches:"
        )
        info_nce_s += f" {round_number(np.mean(train_info_nce_losses_chunk))}."
        log_string(logger, info_nce_s)

    if train_cat_losses:
        if train_cat_losses[0][1] is not None:
            train_cat_losses_chunk = [
                loss for _, loss in train_cat_losses[-args_dict["n_batches_per_val"] :]
            ]
            cat_s = f"Train Cat Loss over the previous {args_dict['n_batches_per_val']} Batches:"
            cat_s += f" {round_number(np.mean(train_cat_losses_chunk))}."
            log_string(logger, cat_s)
    log_string(logger, "----------------\n")

    return best_val_score, early_stopping_counter, batch_val_score


def check_early_stopping(
    early_stopping_counter: int,
    early_stopping_patience: int,
    logger: logging.Logger,
    n_batches_processed_so_far: int,
) -> bool:
    if early_stopping_patience is None:
        return False
    if early_stopping_counter >= early_stopping_patience:
        log_string(
            logger,
            f"Early stopping after {n_batches_processed_so_far} batches.",
        )
        return True
    return False


def run_training_main(
    recommender: Recommender,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_sessions_dataloader: DataLoader,
    train_negative_samples_dataloader: DataLoader,
    args_dict: dict,
    val_data: dict,
    logger: logging.Logger,
) -> None:
    start_time = time.time()
    train_info_nce_losses, train_cat_losses, val_scores = [], [], []

    best_val_score, early_stopping_counter, _ = process_validation(
        recommender=recommender, args_dict=args_dict, val_data=val_data, logger=logger
    )
    val_scores.append((0, best_val_score))

    train_sessions_iter = iter(train_sessions_dataloader)
    train_negative_samples_iter = iter(train_negative_samples_dataloader)

    for i in tqdm(range(args_dict["n_batches_total"]), desc="Training Batches", unit="Batch"):
        train_cat_loss = None
        optimizer.zero_grad()
        batch = {**next(train_sessions_iter), **next(train_negative_samples_iter)}
        dot_products, mask_negrated = process_batch(batch=batch, recommender=recommender)
        train_info_nce_loss = compute_info_nce_loss(
            dot_products, mask_negrated, args_dict["info_nce_temperature"]
        )
        train_info_nce_losses.append((i, train_info_nce_loss.item()))

        total_loss = train_info_nce_loss
        total_loss.backward()
        optimizer.step() 
        if scheduler:
            scheduler.step()

        train_cat_losses.append((i, train_cat_loss))
        n_batches_processed_so_far = i + 1
        check_val = (
            n_batches_processed_so_far % args_dict["n_batches_per_val"] == 0
            or n_batches_processed_so_far == args_dict["n_batches_total"]
        )
        if check_val:
            best_val_score, early_stopping_counter, batch_val_score = process_validation_main(
                recommender=recommender,
                args_dict=args_dict,
                val_data=val_data,
                previous_best_score=best_val_score,
                early_stopping_counter=early_stopping_counter,
                logger=logger,
                n_batches_processed_so_far=n_batches_processed_so_far,
                train_info_nce_losses=train_info_nce_losses,
                train_cat_losses=train_cat_losses,
            )
            val_scores.append((n_batches_processed_so_far, batch_val_score))
            if check_early_stopping(
                early_stopping_counter=early_stopping_counter,
                early_stopping_patience=args_dict["early_stopping_patience"],
                logger=logger,
                n_batches_processed_so_far=n_batches_processed_so_far,
            ):
                break

    log_string(logger, f"Total Training Time: {time.time() - start_time:.1f} seconds.")
    save_values(train_info_nce_losses, train_cat_losses, val_scores, args_dict["outputs_folder"])


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args_dict = parse_arguments()
    set_all_seeds(args_dict["seed"])
    recommender = load_recommender_main(args_dict, device)
    val_data = load_val_data_main(args_dict)
    optimizer, scheduler = load_optimizer_main(recommender, args_dict)
    train_sessions_dataloader, train_negative_samples_dataloader = load_train_dataloaders_main(
        recommender=recommender, args_dict=args_dict
    )

    if not args_dict["testing"]:
        os.makedirs(args_dict["outputs_folder"], exist_ok=True)
        logger = init_logging(args_dict)
        save_config(args_dict)
        run_training_main(
            recommender=recommender,
            optimizer=optimizer,
            scheduler=scheduler,
            train_sessions_dataloader=train_sessions_dataloader,
            train_negative_samples_dataloader=train_negative_samples_dataloader,
            args_dict=args_dict,
            val_data=val_data,
            logger=logger,
        )
