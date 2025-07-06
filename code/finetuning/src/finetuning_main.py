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

from ...src.project_paths import ProjectPaths
from .finetuning_data import (
    get_train_dataset_dataloader,
    get_train_negative_samples_dataloader,
    load_val_data,
    print_train_ratings_batch,
    round_number,
)
from .finetuning_evaluation import run_validation
from .finetuning_model import FinetuningModel, load_finetuning_model
from .finetuning_preprocessing import (
    load_categories_to_idxs,
    load_val_users_embeddings_idxs,
)


def log_string(logger: logging.Logger, string: str) -> None:
    print(string)
    logger.info(string)


def args_dict_assertions(args_dict: dict) -> None:
    assert args_dict["n_samples_per_user"] > 0 and args_dict["n_samples_per_user"] <= min(
        32, args_dict["batch_size"]
    )
    assert args_dict["batch_size"] % args_dict["n_samples_per_user"] == 0
    assert (
        args_dict["n_max_positive_samples_per_user"] >= args_dict["n_min_positive_samples_per_user"]
    )
    assert (
        args_dict["n_max_negative_samples_per_user"] >= args_dict["n_min_negative_samples_per_user"]
    )
    assert args_dict["n_max_positive_samples_per_user"] <= args_dict["n_samples_per_user"]
    assert args_dict["n_max_negative_samples_per_user"] <= args_dict["n_samples_per_user"]
    assert (
        args_dict["n_min_positive_samples_per_user"] + args_dict["n_min_negative_samples_per_user"]
        <= args_dict["n_samples_per_user"]
    )
    assert (
        args_dict["n_max_positive_samples_per_user"] + args_dict["n_max_negative_samples_per_user"]
        >= args_dict["n_samples_per_user"]
    )
    assert args_dict["n_batch_negatives"] <= (
        args_dict["batch_size"] - args_dict["n_samples_per_user"]
    )


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description="Finetuning script")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--testing", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=125)
    parser.add_argument(
        "--users_sampling_strategy",
        type=str,
        default="uniform",
        choices=["uniform", "proportional", "square_root_proportional", "cube_root_proportional"],
    )
    parser.add_argument("--n_samples_per_user", type=int, default=5)
    parser.add_argument("--n_min_positive_samples_per_user", type=int, default=0)
    parser.add_argument("--n_min_negative_samples_per_user", type=int, default=4)
    parser.add_argument("--n_max_positive_samples_per_user", type=int, default=None)
    parser.add_argument("--n_max_negative_samples_per_user", type=int, default=4)
    parser.add_argument("--n_train_negative_samples", type=int, default=20)
    parser.add_argument("--n_batch_negatives", type=int, default=7)
    parser.add_argument(
        "--not_closest_temporal_samples",
        action="store_false",
        dest="closest_temporal_samples",
        default=True,
    )
    parser.add_argument("--n_samples_from_most_recent_positive_votes", type=int, default=7)
    parser.add_argument("--n_samples_from_closest_negative_votes", type=int, default=4)

    parser.add_argument(
        "--loss_function", type=str, default="info_nce", choices=["bcel", "info_nce"]
    )
    parser.add_argument("--info_nce_temperature_explicit_negatives", type=float, default=2.5)
    parser.add_argument("--info_nce_temperature_batch_negatives", type=float, default=2.5)
    parser.add_argument("--info_nce_temperature_negative_samples", type=float, default=2.5)
    parser.add_argument("--info_nce_log_q_correction", action="store_true", default=False)
    parser.add_argument("--n_batches_total", type=int, default=10000)
    parser.add_argument("--n_batches_per_val", type=int, default=500)
    parser.add_argument("--val_metric", type=str, default="ndcg_all")
    parser.add_argument("--early_stopping_patience", type=int, default=None)

    parser.add_argument(
        "--model_path", type=str, default=str(ProjectPaths.finetuning_data_model_state_dicts_path())
    )
    parser.add_argument("--unfreeze_from_bottom", action="store_true", default=False)
    parser.add_argument("--n_unfreeze_layers", type=int, default=4)
    parser.add_argument(
        "--not_pretrained_projection", action="store_false", dest="pretrained_projection"
    )
    parser.add_argument(
        "--not_pretrained_users_embeddings",
        action="store_false",
        dest="pretrained_users_embeddings",
    )
    parser.add_argument(
        "--not_pretrained_categories_embeddings_l1",
        action="store_false",
        dest="pretrained_categories_embeddings_l1",
    )
    parser.add_argument(
        "--not_include_l2_categories",
        action="store_false",
        dest="include_l2_categories",
        default=True,
    )
    parser.add_argument(
        "--unfreeze_word_embeddings",
        action="store_true",
        default=False,
    )

    parser.add_argument("--lr_transformer_model", type=float, default=1e-5)
    parser.add_argument("--lr_other", type=float, default=1e-5)
    parser.add_argument("--lr_projection", type=float, default=1e-5)
    parser.add_argument("--lr_users_embeddings", type=float, default=1e-5)
    parser.add_argument(
        "--lr_scheduler", type=str, default="linear_decay", choices=["constant", "linear_decay"]
    )
    parser.add_argument("--l2_regularization_transformer_model", type=float, default=0)
    parser.add_argument("--l2_regularization_other", type=float, default=0)
    parser.add_argument("--n_warmup_steps", type=float, default=500)

    args_dict = vars(parser.parse_args())
    args_dict["model_path"] = Path(args_dict["model_path"]).resolve()
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


def get_finetuning_dataloaders(args_dict: dict) -> tuple:
    train_dataset_dataloader = get_train_dataset_dataloader(args_dict)
    assert len(train_dataset_dataloader) == args_dict["n_batches_total"]
    train_negative_samples_dataloader = None
    if args_dict["n_train_negative_samples"] > 0:
        train_negative_samples_dataloader = get_train_negative_samples_dataloader(
            n_train_negative_samples=args_dict["n_train_negative_samples"],
            n_batches_total=args_dict["n_batches_total"],
            seed=args_dict["seed"],
        )
        assert len(train_negative_samples_dataloader) == len(train_dataset_dataloader)
    return train_dataset_dataloader, train_negative_samples_dataloader


def load_optimizer(
    finetuning_model: FinetuningModel,
    lr_transformer_model: float,
    l2_regularization_transformer_model: float,
    lr_other: float,
    l2_regularization_other: float,
    lr_projection: float,
    lr_users_embeddings: float,
    lr_scheduler: str,
    n_batches_total: int,
    n_warmup_steps: int,
) -> torch.optim.Optimizer:
    param_groups = [
        {
            "params": finetuning_model.transformer_model.parameters(),
            "lr": lr_transformer_model,
            "weight_decay": l2_regularization_transformer_model,
        },
        {
            "params": finetuning_model.projection.parameters(),
            "lr": lr_projection,
            "weight_decay": l2_regularization_other,
        },
        {
            "params": finetuning_model.users_embeddings.parameters(),
            "lr": lr_users_embeddings,
            "weight_decay": l2_regularization_other,
        },
        {
            "params": finetuning_model.categories_embeddings_l1.parameters(),
            "lr": lr_transformer_model,
            "weight_decay": l2_regularization_other,
        },
    ]
    if finetuning_model.categories_embeddings_l2 is not None:
        param_groups.append(
            {
                "params": finetuning_model.categories_embeddings_l2.parameters(),
                "lr": lr_transformer_model,
                "weight_decay": l2_regularization_other,
            }
        )
    optimizer = torch.optim.Adam(param_groups)
    if lr_scheduler == "constant":
        return optimizer, None
    elif lr_scheduler == "linear_decay":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, n_total_steps=n_batches_total, n_warmup_steps=n_warmup_steps
        )
    return optimizer, scheduler


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, n_total_steps: int, n_warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < n_warmup_steps:
            return float(current_step) / float(n_warmup_steps)
        else:
            return max(
                0.0, float(n_total_steps - current_step) / float(n_total_steps - n_warmup_steps)
            )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def init_logging(args_dict: dict) -> logging.Logger:
    filename = Path(args_dict["outputs_folder"]).resolve() / "logger.log"
    logging.basicConfig(
        filename=filename,
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger("finetuning")
    logger.info(f"Finetuning script started with config: {args_dict}.")
    return logger


def save_config(args_dict: dict) -> None:
    outputs_folder = Path(args_dict["outputs_folder"]).resolve()
    for key in args_dict:
        if isinstance(args_dict[key], Path):
            args_dict[key] = str(args_dict[key])
    with open(outputs_folder / "config.json", "w") as f:
        json.dump(args_dict, f, indent=4)


def compute_info_nce_loss(
    train_dataset_scores: torch.Tensor,
    user_idx_tensor: torch.Tensor,
    rating_tensor: torch.Tensor,
    negative_samples_scores: torch.Tensor,
    batch_negatives_scores: torch.Tensor,
    sorted_unique_user_idx_tensor: torch.Tensor,
    args_dict: dict,
) -> float:
    temperature_explicit_negatives = args_dict["info_nce_temperature_explicit_negatives"]
    temperature_batch_negatives = args_dict["info_nce_temperature_batch_negatives"]
    temperature_negative_samples = args_dict["info_nce_temperature_negative_samples"]
    enumerated_users_dict = {
        user_idx.item(): i for i, user_idx in enumerate(sorted_unique_user_idx_tensor)
    }
    negatives_users_dict = {}
    for user_idx in enumerated_users_dict.keys():
        negative_train_ratings_scores_user = train_dataset_scores[
            (user_idx_tensor == user_idx) & (rating_tensor == 0)
        ]
        negative_samples_scores_user = (
            None
            if negative_samples_scores is None
            else negative_samples_scores[enumerated_users_dict[user_idx]]
        )
        batch_negatives_scores_user = (
            None
            if batch_negatives_scores is None
            else batch_negatives_scores[enumerated_users_dict[user_idx]]
        )
        negative_users_dict_user = (
            negative_train_ratings_scores_user / temperature_explicit_negatives
        )
        if negative_samples_scores_user is not None:
            negative_samples_scores_user = (
                negative_samples_scores_user / temperature_negative_samples
            )
            negative_users_dict_user = torch.cat(
                (negative_users_dict_user, negative_samples_scores_user), dim=0
            )
        if batch_negatives_scores_user is not None:
            batch_negatives_scores_user = batch_negatives_scores_user / temperature_batch_negatives
            negative_users_dict_user = torch.cat(
                (negative_users_dict_user, batch_negatives_scores_user), dim=0
            )
        negatives_users_dict[user_idx] = negative_users_dict_user
    positive_indices = torch.where(rating_tensor == 1)[0]
    positive_scores, positive_users_idxs = (
        train_dataset_scores[positive_indices],
        user_idx_tensor[positive_indices],
    )
    losses = []
    for i, _ in enumerate(positive_indices):
        pos_score, pos_user_idx = positive_scores[i], positive_users_idxs[i].item()
        pos_score = pos_score / temperature_explicit_negatives
        losses.append(
            -torch.log_softmax(
                torch.cat((pos_score.unsqueeze(0), negatives_users_dict[pos_user_idx])), dim=0
            )[0]
        )
    return torch.mean(torch.stack(losses))


def generate_batch_negatives_indices(
    user_idx_tensor: torch.Tensor,
    sorted_unique_user_idx_tensor: torch.Tensor,
    n_batch_negatives: int,
    random_state: int,
) -> torch.Tensor:
    batch_negatives_indices = torch.zeros(
        (len(sorted_unique_user_idx_tensor), n_batch_negatives), dtype=torch.long
    )
    for i, user_idx in enumerate(sorted_unique_user_idx_tensor):
        non_user_indices = torch.where(user_idx_tensor != user_idx)[0]
        if len(non_user_indices) < n_batch_negatives:
            raise ValueError(
                f"Not enough negative samples for user {user_idx.item()}. Required: {n_batch_negatives}, Available: {len(non_user_indices)}."
            )
        sampled_indices = np.random.RandomState(random_state).choice(
            non_user_indices.cpu().numpy(), size=n_batch_negatives, replace=False
        )
        batch_negatives_indices[i] = torch.from_numpy(sampled_indices).to(user_idx_tensor.device)
    assert batch_negatives_indices.shape == (len(sorted_unique_user_idx_tensor), n_batch_negatives)
    return batch_negatives_indices


def process_val(
    finetuning_model: FinetuningModel,
    args_dict: dict,
    val_data: dict,
    n_updates_so_far: int,
    logger: logging.Logger,
    baseline_metric: float = None,
    early_stopping_counter: int = None,
) -> tuple:
    if baseline_metric is None or early_stopping_counter is None:
        assert n_updates_so_far == 0
    baseline_metric_string = "None" if baseline_metric is None else round_number(baseline_metric)
    val_scores_batch, val_scores_batch_string = run_validation(
        finetuning_model,
        val_data["val_dataset"],
        val_data["val_negative_samples"],
        print_results=False,
    )
    val_scores_batch_string = (
        f"VALIDATION METRICS AFTER UPDATE {n_updates_so_far} / {args_dict['n_batches_total']}:"
        + val_scores_batch_string
    )
    baseline_metric_batch = val_scores_batch[f"val_{args_dict['val_metric']}"]
    if baseline_metric is None:
        improvement = True
    else:
        if args_dict["val_metric"] == "bcel" or args_dict["val_metric"].startswith("infonce"):
            improvement = baseline_metric_batch < baseline_metric
        else:
            improvement = baseline_metric_batch > baseline_metric
    if improvement:
        if baseline_metric is not None:
            finetuning_model.save_finetuning_model(args_dict["outputs_folder"] / "state_dicts")
        val_scores_batch_string += (
            f"\nNew Optimal Value for Metric {args_dict['val_metric'].upper()}: "
            f"{round_number(baseline_metric_batch)} after previously {baseline_metric_string}."
        )
        baseline_metric, early_stopping_counter = baseline_metric_batch, 0
    else:
        val_scores_batch_string += (
            f"\nNo Improvement for Metric {args_dict['val_metric'].upper()}: "
            f"{round_number(baseline_metric_batch)} after previously {baseline_metric_string}."
        )
        early_stopping_counter += 1
    val_scores_batch_string += (
        f" Early Stopping Counter: {early_stopping_counter} / "
        f"{args_dict['early_stopping_patience']}.\n"
    )
    log_string(logger, val_scores_batch_string)
    return val_scores_batch, baseline_metric, early_stopping_counter


def process_batch(
    finetuning_model: FinetuningModel,
    args_dict: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_dataset_batch: dict,
    train_negative_samples_batch: dict,
    i: int,
) -> float:
    user_idx_tensor = train_dataset_batch["user_idx"].to(finetuning_model.device)
    sorted_unique_user_idx_tensor = torch.unique(user_idx_tensor, sorted=True).to(
        finetuning_model.device
    )
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        batch_negatives_indices = None
        if args_dict["n_batch_negatives"] > 0:
            batch_negatives_indices = generate_batch_negatives_indices(
                user_idx_tensor,
                sorted_unique_user_idx_tensor,
                args_dict["n_batch_negatives"],
                args_dict["seed"] + i,
            )
        train_dataset_scores, batch_negatives_scores = finetuning_model(
            eval_type="train",
            input_ids_tensor=train_dataset_batch["input_ids"],
            attention_mask_tensor=train_dataset_batch["attention_mask"],
            category_l1_tensor=train_dataset_batch["category_l1"],
            category_l2_tensor=train_dataset_batch.get("category_l2", None),
            user_idx_tensor=user_idx_tensor,
            sorted_unique_user_idx_tensor=sorted_unique_user_idx_tensor,
            batch_negatives_indices=batch_negatives_indices,
        )
        train_negative_samples_scores = None
        if train_negative_samples_batch is not None:
            train_negative_samples_scores = finetuning_model(
                eval_type="negative_samples",
                input_ids_tensor=train_negative_samples_batch["input_ids"],
                attention_mask_tensor=train_negative_samples_batch["attention_mask"],
                sorted_unique_user_idx_tensor=sorted_unique_user_idx_tensor,
                category_l1_tensor=train_negative_samples_batch["category_l1"],
                category_l2_tensor=train_negative_samples_batch["category_l2"],
            )
        if args_dict["loss_function"] == "bcel":
            rating_tensor = train_dataset_batch["rating"].float().to(finetuning_model.device)
            pos_weight = torch.tensor(2.0).to(finetuning_model.device)
            loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
                train_dataset_scores, rating_tensor
            )
        elif args_dict["loss_function"] == "info_nce":
            rating_tensor = train_dataset_batch["rating"].to(finetuning_model.device)
            loss = compute_info_nce_loss(
                train_dataset_scores,
                user_idx_tensor,
                rating_tensor,
                train_negative_samples_scores,
                batch_negatives_scores,
                sorted_unique_user_idx_tensor,
                args_dict,
            )
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return loss.item()


def run_training(
    finetuning_model: FinetuningModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_dataset_dataloader: DataLoader,
    train_negative_samples_dataloader: DataLoader,
    args_dict: dict,
    val_data: dict,
    logger: logging.Logger,
) -> tuple:
    val_scores_batch, baseline_metric, early_stopping_counter = process_val(
        finetuning_model, args_dict, val_data, 0, logger
    )
    train_losses, val_scores = [], [(0, val_scores_batch)]

    finetuning_model.train()
    train_dataset_iter = iter(train_dataset_dataloader)
    if train_negative_samples_dataloader is not None:
        train_negative_samples_iter = iter(train_negative_samples_dataloader)
    start_time = time.time()
    for i in tqdm(range(args_dict["n_batches_total"]), desc="Training Batches", unit="Batch"):
        train_dataset_batch = next(train_dataset_iter)
        train_negative_samples_batch = (
            next(train_negative_samples_iter)
            if train_negative_samples_dataloader is not None
            else None
        )
        train_loss = process_batch(
            finetuning_model,
            args_dict,
            optimizer,
            scheduler,
            train_dataset_batch,
            train_negative_samples_batch,
            i,
        )
        train_losses.append((i, train_loss))
        n_batches_processed_so_far = i + 1
        if (
            n_batches_processed_so_far % args_dict["n_batches_per_val"] == 0
            or n_batches_processed_so_far == args_dict["n_batches_total"]
        ):
            train_dataset_dataloader.batch_sampler.run_test(train_dataset_batch)
            log_string(
                logger,
                f"\nTRAIN RATINGS BATCH BEFORE UPDATE {n_batches_processed_so_far} / "
                f"{args_dict['n_batches_total']}\n{print_train_ratings_batch(train_dataset_batch)}",
            )
            if train_negative_samples_dataloader is not None:
                train_negative_samples_dataloader.batch_sampler.run_test(
                    train_negative_samples_batch
                )
            train_losses_chunk = [
                loss for _, loss in train_losses[-args_dict["n_batches_per_val"] :]
            ]
            log_string(logger, f"AVERAGED TRAIN LOSS: {round_number(np.mean(train_losses_chunk))}.")
            log_string(logger,
                       f"LEARNING RATE: {optimizer.param_groups[0]['lr']}\n")
            val_scores_batch, baseline_metric, early_stopping_counter = process_val(
                finetuning_model,
                args_dict,
                val_data,
                n_batches_processed_so_far,
                logger,
                baseline_metric,
                early_stopping_counter,
            )
            val_scores.append((n_batches_processed_so_far, val_scores_batch))
            if (
                args_dict["early_stopping_patience"] is not None
                and early_stopping_counter >= args_dict["early_stopping_patience"]
            ):
                log_string(logger, f"\nEarly stopping after {n_batches_processed_so_far} Batches.")
                break
            if i >= 20000:
                break
    args_dict["time_elapsed"] = time.time() - start_time
    return train_losses, val_scores


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args_dict = parse_arguments()
    set_all_seeds(args_dict["seed"])
    val_data = load_val_data()

    finetuning_model = load_finetuning_model(
        finetuning_model_path=args_dict["model_path"],
        device=device,
        mode="train",
        n_unfreeze_layers=args_dict["n_unfreeze_layers"],
        pretrained_projection=args_dict["pretrained_projection"],
        pretrained_users_embeddings=args_dict["pretrained_users_embeddings"],
        pretrained_categories_embeddings_l1=args_dict["pretrained_categories_embeddings_l1"],
        n_categories_l2=(
            len(load_categories_to_idxs("l2")) if args_dict["include_l2_categories"] else None
        ),
        val_users_embeddings_idxs=load_val_users_embeddings_idxs(),
        unfreeze_word_embeddings=args_dict["unfreeze_word_embeddings"],
        unfreeze_from_bottom=args_dict["unfreeze_from_bottom"],
    )
    print(finetuning_model.get_memory_footprint())
    args_dict["n_transformer_layers"] = finetuning_model.count_transformer_layers()
    args_dict["n_transformer_parameters"], args_dict["n_unfrozen_transformer_parameters"] = (
        finetuning_model.count_transformer_parameters()
    )

    train_dataset_dataloader, train_negative_samples_dataloader = get_finetuning_dataloaders(
        args_dict
    )
    optimizer, scheduler = load_optimizer(
        finetuning_model=finetuning_model,
        lr_transformer_model=args_dict["lr_transformer_model"],
        l2_regularization_transformer_model=args_dict["l2_regularization_transformer_model"],
        lr_other=args_dict["lr_other"],
        l2_regularization_other=args_dict["l2_regularization_other"],
        lr_projection=args_dict["lr_projection"],
        lr_users_embeddings=args_dict["lr_users_embeddings"],
        lr_scheduler=args_dict["lr_scheduler"],
        n_batches_total=args_dict["n_batches_total"],
        n_warmup_steps=args_dict["n_warmup_steps"],
    )

    if not args_dict["testing"]:
        args_dict["outputs_folder"] = (
            ProjectPaths.finetuning_data_experiments_path()
            / f"{finetuning_model.transformer_model_name}_{time.strftime('%Y-%m-%d-%H-%M')}"
        )
        os.makedirs(args_dict["outputs_folder"], exist_ok=True)
        logger = init_logging(args_dict)

        train_losses, val_scores = run_training(
            finetuning_model=finetuning_model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataset_dataloader=train_dataset_dataloader,
            train_negative_samples_dataloader=train_negative_samples_dataloader,
            args_dict=args_dict,
            val_data=val_data,
            logger=logger,
        )

        train_losses_path, val_scores_path = (
            args_dict["outputs_folder"] / "train_losses.pkl",
            args_dict["outputs_folder"] / "val_scores.pkl",
        )
        with open(train_losses_path, "wb") as f:
            pickle.dump(train_losses, f)
        with open(val_scores_path, "wb") as f:
            pickle.dump(val_scores, f)
        print(f"Finished Finetuning:     {args_dict['outputs_folder']}.")
        save_config(args_dict)
