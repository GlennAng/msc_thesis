import argparse
import json
import os
from pathlib import Path

import torch

from ....scripts.sequence_eval import get_output_folder
from ....src.load_files import VAL_RANDOM_STATE
from ..data.sequence_dataset import load_sequence_dataset_by_split, load_val_dataloader
from ..models.recommender import Recommender, load_recommender
from ..models.users_encoder import save_users_embeddings_as_pickle


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--embeddings_path", type=str, required=False)
    args = vars(parser.parse_args())
    args["model_path"] = Path(args["model_path"]).resolve()
    if args.get("config_path", None) is not None:
        args["config_path"] = Path(args["config_path"]).resolve()
    if args.get("embeddings_path", None) is not None:
        args["embeddings_path"] = Path(args["embeddings_path"]).resolve()
    return args


def get_config(args_dict: dict) -> dict:
    if args_dict["test"]:
        return {
            "hard_constraint_min_n_train_posrated": 0,
            "hard_constraint_max_n_train_rated": None,
            "soft_constraint_max_n_train_sessions": None,
            "soft_constraint_max_n_train_days": None,
            "remove_negrated_from_history": True,
        }
    config_path = args_dict.get("config_path", None)
    if config_path is None:
        config_path = args_dict["model_path"].parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


def load_recommender_for_testing(args_dict: dict, device: torch.device) -> Recommender:
    users_encoder_dict = {"device": device}
    if args_dict["embeddings_path"] is None:
        papers_encoder_dict = {
            "path": args_dict["model_path"] / "papers_encoder",
            "device": device,
        }
        recommender = load_recommender(
            users_encoder_dict=users_encoder_dict,
            papers_encoder_dict=papers_encoder_dict,
            use_papers_encoder=True,
        )
    else:
        papers_encoder_dict = {
            "papers_embeddings_path": args_dict["embeddings_path"],
            "device": device,
        }
        recommender = load_recommender(
            users_encoder_dict=users_encoder_dict,
            papers_encoder_dict=papers_encoder_dict,
            use_papers_encoder=False,
        )
    recommender.eval()
    return recommender


def get_output_folder_for_testing(config: dict) -> Path:
    output_folder = get_output_folder(
        {
            **config,
            "embed_function": "neural",
            "users_selection": "sequence_test",
            "single_val_session": False,
        }
    )
    output_folder = output_folder / "users_embeddings" / f"s_{VAL_RANDOM_STATE}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Users embeddings will be saved to {output_folder}.")
    return output_folder


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args_dict = parse_args()
    config = get_config(args_dict)
    recommender = load_recommender_for_testing(args_dict=args_dict, device=device)

    if recommender.use_papers_encoder:
        pass
    else:
        embeddings = recommender.papers_embeddings
        papers_ids_to_idxs = recommender.papers_ids_to_idxs
    test_dataset = load_sequence_dataset_by_split(
        split="test",
        hard_constraint_min_n_train_posrated=config["hard_constraint_min_n_train_posrated"],
        hard_constraint_max_n_train_rated=config["hard_constraint_max_n_train_rated"],
        soft_constraint_max_n_train_sessions=config["soft_constraint_max_n_train_sessions"],
        soft_constraint_max_n_train_days=config["soft_constraint_max_n_train_days"],
        remove_negrated_from_history=config["remove_negrated_from_history"],
    )
    test_dataloader = load_val_dataloader(
        dataset=test_dataset,
        papers_embeddings=embeddings,
        papers_ids_to_idxs=papers_ids_to_idxs,
    )
    users_embeddings, users_sessions_ids_to_idxs = (
        recommender.users_encoder.compute_users_embeddings(dataloader=test_dataloader)
    )
    output_folder = get_output_folder_for_testing(config=config)
    save_users_embeddings_as_pickle(
        path=output_folder,
        embeddings=users_embeddings,
        users_sessions_ids_to_idxs=users_sessions_ids_to_idxs,
    )
    print("Testing run completed.")
