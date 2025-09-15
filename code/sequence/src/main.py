import argparse
from pathlib import Path

import torch

from ...finetuning.src.finetuning_main import set_all_seeds
from ...src.project_paths import ProjectPaths
from .data.eval_papers_dataset import get_eval_papers_dataloader
from .data.sequence_preprocessing import (
    load_negative_samples_matrix_val,
    load_ranking_matrix_val,
)
from .models.recommender import Recommender, load_recommender


def select_model(start_from_finetuning: bool) -> Path:
    if start_from_finetuning:
        model_path = ProjectPaths.finetuning_data_checkpoints_path() / "no_seq_eval" / "state_dicts"
    else:
        model_path = ProjectPaths.sequence_data_model_state_dicts_papers_encoder_path()
    return model_path


def extract_papers_encoder_dict(args_dict: dict, device: torch.device) -> dict:
    return {
        "path": args_dict["model_path"],
        "overwrite_l1_scale": args_dict["l1_scale"],
        "overwrite_use_l2_embeddings": args_dict["use_l2_embeddings"],
        "overwrite_l2_scale": args_dict["l2_scale"],
        "l2_init_seed": args_dict["seed"],
        "unfreeze_l1_scale": not args_dict["freeze_l1_scale"],
        "unfreeze_l2_scale": not args_dict["freeze_l2_scale"],
        "n_unfreeze_layers": args_dict["n_unfreeze_layers"],
        "unfreeze_word_embeddings": args_dict["unfreeze_word_embeddings"],
        "unfreeze_from_bottom": args_dict["unfreeze_from_bottom"],
        "device": device,
    }


def extract_users_encoder_dict(args_dict: dict, device: torch.device) -> dict:
    return {"device": device}


def load_recommender_main(args_dict: dict) -> Recommender:
    recommender = load_recommender(
        papers_encoder_dict=extract_papers_encoder_dict(args_dict, device),
        users_encoder_dict=extract_users_encoder_dict(args_dict, device),
    ).train()
    print(recommender.get_memory_footprint())
    return recommender





def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description="Sequence script")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_from_finetuning", action="store_true", default=False)

    parser.add_argument("--l1_scale", type=float, default=1.0)
    parser.add_argument("--freeze_l1_scale", action="store_true", default=False)
    parser.add_argument(
        "--not_use_l2_embeddings", action="store_false", dest="use_l2_embeddings", default=True
    )
    parser.add_argument("--l2_scale", type=float, default=1.0)
    parser.add_argument("--freeze_l2_scale", action="store_true", default=False)
    parser.add_argument("--n_unfreeze_layers", type=int, default=4)
    parser.add_argument("--unfreeze_word_embeddings", action="store_true", default=False)
    parser.add_argument("--unfreeze_from_bottom", action="store_true", default=False)

    args = vars(parser.parse_args())
    args["model_path"] = select_model(args["start_from_finetuning"])
    return args


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args_dict = parse_arguments()
    set_all_seeds(args_dict["seed"])
    recommender = load_recommender_main(args_dict)
    val_data = load_val_data()
