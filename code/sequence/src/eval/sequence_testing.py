import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

import torch

from ....scripts.sequence_eval import get_output_folder
from ....src.load_files import VAL_RANDOM_STATE
from ....src.project_paths import ProjectPaths
from ..data.eval_data import load_eval_dataloader
from ..data.sessions_dataset import load_sessions_dataset_by_split
from ..data.eval_papers import get_eval_papers_ids
from ..models.recommender import Recommender, load_recommender_pretrained
from ..models.users_encoder import save_users_embeddings_as_pickle

TEST_MODEL_PATH = ProjectPaths.data_path() / "test_recommender_model"
TEST_EMBEDDINGS_PATH = ProjectPaths.sequence_finetuned_embeddings_path()


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--papers_embeddings_path", type=str, required=False)
    parser.add_argument("--config_path", type=str, required=False)
    parser.add_argument("--test", action="store_true", default=False)
    args = vars(parser.parse_args())
    if not args["test"]:
        args["model_path"] = Path(args["model_path"]).resolve()
        args["papers_embeddings_path"] = Path(args["papers_embeddings_path"]).resolve()
    return args


def get_config(args_dict: dict) -> dict:
    if args_dict["test"]:
        return {
            "histories_hard_constraint_min_n_train_posrated": 10,
            "histories_hard_constraint_max_n_train_rated": None,
            "histories_soft_constraint_max_n_train_sessions": None,
            "histories_soft_constraint_max_n_train_days": None,
            "histories_remove_negrated_from_history": True,
        }
    config_path = args_dict.get("config_path", None)
    if config_path is None:
        config_path = args_dict["model_path"].parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


def create_recommender_for_testing(args_dict: dict) -> None:
    from ..models.recommender import load_recommender_from_scratch

    recommender = load_recommender_from_scratch(
        users_encoder_type="MeanPoolingUsersEncoder",
        embeddings_path=TEST_EMBEDDINGS_PATH,
    )
    recommender.save_model(TEST_MODEL_PATH)
    args_dict["model_path"] = TEST_MODEL_PATH
    args_dict["papers_embeddings_path"] = TEST_EMBEDDINGS_PATH


def load_recommender_for_testing(args_dict: dict, device: torch.device) -> Recommender:
    recommender = load_recommender_pretrained(
        recommender_model_path=args_dict["model_path"],
        embeddings_path=args_dict["papers_embeddings_path"],
        device=device,
    )
    if args_dict["test"]:
        import shutil
        
        shutil.rmtree(TEST_MODEL_PATH, ignore_errors=True)
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

    if args_dict["test"]:
        create_recommender_for_testing(args_dict=args_dict)
    recommender = load_recommender_for_testing(args_dict=args_dict, device=device)
    test_dataset = load_sessions_dataset_by_split(
        split="test",
        histories_hard_constraint_min_n_train_posrated=config[
            "histories_hard_constraint_min_n_train_posrated"
        ],
        histories_hard_constraint_max_n_train_rated=config[
            "histories_hard_constraint_max_n_train_rated"
        ],
        histories_soft_constraint_max_n_train_sessions=config[
            "histories_soft_constraint_max_n_train_sessions"
        ],
        histories_soft_constraint_max_n_train_days=config[
            "histories_soft_constraint_max_n_train_days"
        ],
        histories_remove_negrated_from_history=config["histories_remove_negrated_from_history"],
    )
    papers_embeddings, papers_ids_to_idxs = recommender.extract_papers_embeddings(
        papers_ids=get_eval_papers_ids(papers_type="eval_test_users")
    )
    test_dataloader = load_eval_dataloader(
        dataset=test_dataset,
        papers_embeddings=papers_embeddings,
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


    args = [
        sys.executable,
        "-m",
        "code.scripts.sequence_eval",
        "--embed_function",
        "neural",
        "--embedding_path",
        str(TEST_EMBEDDINGS_PATH),
        "--histories_hard_constraint_min_n_train_posrated",
        str(config["histories_hard_constraint_min_n_train_posrated"]),
        "--use_existing_embeddings",
        "--users_selection",
        "sequence_test",
    ]
    if config["histories_hard_constraint_max_n_train_rated"] is not None:
        args += [
            "--histories_hard_constraint_max_n_train_rated",
            str(config["histories_hard_constraint_max_n_train_rated"]),
        ]
    if config["histories_soft_constraint_max_n_train_sessions"] is not None:
        args += [
            "--histories_soft_constraint_max_n_train_sessions",
            str(config["histories_soft_constraint_max_n_train_sessions"]),
        ]
    if config["histories_soft_constraint_max_n_train_days"] is not None:
        args += [
            "--histories_soft_constraint_max_n_train_days",
            str(config["histories_soft_constraint_max_n_train_days"]),
        ]
    if config["histories_remove_negrated_from_history"]:
        args += ["--histories_remove_negrated_from_history"]
    subprocess.run(args, check=True)
