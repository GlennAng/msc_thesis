import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ....src.load_files import VAL_RANDOM_STATE
from ....src.project_paths import ProjectPaths
from ..data.eval_data import load_eval_dataloader
from ..data.eval_papers import get_eval_papers_ids
from ..data.sessions_dataset import load_sessions_dataset_by_split
from ..models.recommender import Recommender, load_recommender_pretrained
from ..models.users_encoder import (
    get_users_encoder_type_from_arg,
    save_users_embeddings_as_pickle,
)
from .compute_users_embeddings_utils import get_eval_data_folder

TRIAL_MODEL_PATH = ProjectPaths.data_path() / "trial_recommender_model"
TRIAL_EMBEDDINGS_PATH = ProjectPaths.sequence_non_finetuned_embeddings_path()


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--training_config_path", type=str, required=False)
    parser.add_argument("--trial", action="store_true", default=False)
    args = vars(parser.parse_args())
    if args["trial"]:
        args["model_path"] = TRIAL_MODEL_PATH
    else:
        path = Path(args["model_path"]).resolve()
        if path.stem != "model":
            path = path / "model"
        args["model_path"] = path

    return args


def get_training_config(args_dict: dict) -> dict:
    if args_dict["trial"]:
        return {
            "embeddings_path": TRIAL_EMBEDDINGS_PATH,
            "histories_hard_constraint_min_n_train_posrated": 10,
            "histories_hard_constraint_max_n_train_rated": None,
            "histories_soft_constraint_max_n_train_sessions": None,
            "histories_soft_constraint_max_n_train_days": None,
            "histories_remove_negrated_from_history": True,
        }
    training_config_path = args_dict.get("training_config_path", None)
    if training_config_path is None:
        training_config_path = args_dict["model_path"].parent / "config.json"
    else:
        training_config_path = Path(training_config_path).resolve()
    with open(training_config_path) as f:
        training_config = json.load(f)
    training_config["embeddings_path"] = Path(training_config["embeddings_path"]).resolve()
    return training_config


def create_trial_recommender() -> None:
    from ..models.recommender import load_recommender_from_scratch

    recommender = load_recommender_from_scratch(
        users_encoder_type="mean_pos_pooling",
        embeddings_path=TRIAL_EMBEDDINGS_PATH,
    )
    recommender.save_model(TRIAL_MODEL_PATH)


def load_testing_recommender(
    model_path: Path,
    embeddings_path: Path,
    users_encoder_type: str,
    device: torch.device,
    trial: bool = False,
) -> Recommender:
    recommender = load_recommender_pretrained(
        recommender_model_path=model_path,
        embeddings_path=embeddings_path,
        users_encoder_type_str=users_encoder_type,
        device=device,
    )
    if trial:
        import shutil

        shutil.rmtree(TRIAL_MODEL_PATH, ignore_errors=True)
    recommender.eval()
    return recommender


def load_testing_dataset(training_config: dict) -> Dataset:
    return load_sessions_dataset_by_split(
        split="test",
        histories_hard_constraint_min_n_train_posrated=training_config[
            "histories_hard_constraint_min_n_train_posrated"
        ],
        histories_hard_constraint_max_n_train_rated=training_config[
            "histories_hard_constraint_max_n_train_rated"
        ],
        histories_soft_constraint_max_n_train_sessions=training_config[
            "histories_soft_constraint_max_n_train_sessions"
        ],
        histories_soft_constraint_max_n_train_days=training_config[
            "histories_soft_constraint_max_n_train_days"
        ],
        histories_remove_negrated_from_history=training_config[
            "histories_remove_negrated_from_history"
        ],
    )


def load_testing_dataloader(recommender: Recommender, dataset: Dataset) -> DataLoader:
    papers_embeddings, papers_ids_to_idxs = recommender.extract_papers_embeddings(
        papers_ids=get_eval_papers_ids(papers_type="eval_test_users")
    )
    return load_eval_dataloader(
        dataset=dataset,
        papers_embeddings=papers_embeddings,
        papers_ids_to_idxs=papers_ids_to_idxs,
    )


def get_testing_data_folder() -> Path:
    return get_eval_data_folder(
        embed_function="neural_precomputed",
        users_selection="sequence_test",
        single_val_session=False,
    )


def run_sequence_testing(args_dict: dict, training_config: dict, data_folder: Path) -> None:
    args = [
        sys.executable,
        "-m",
        "code.scripts.sliding_window_eval",
        "--embed_function",
        "neural_precomputed",
        "--papers_embedding_path",
        str(training_config["embeddings_path"]),
        "--histories_hard_constraint_min_n_train_posrated",
        str(training_config["histories_hard_constraint_min_n_train_posrated"]),
        "--use_existing_users_embeddings",
        "--existing_users_embeddings_path",
        str(data_folder),
        "--users_selection",
        "sequence_test",
    ]
    if training_config["histories_hard_constraint_max_n_train_rated"] is not None:
        args += [
            "--histories_hard_constraint_max_n_train_rated",
            str(training_config["histories_hard_constraint_max_n_train_rated"]),
        ]
    if training_config["histories_soft_constraint_max_n_train_sessions"] is not None:
        args += [
            "--histories_soft_constraint_max_n_train_sessions",
            str(training_config["histories_soft_constraint_max_n_train_sessions"]),
        ]
    if training_config["histories_soft_constraint_max_n_train_days"] is not None:
        args += [
            "--histories_soft_constraint_max_n_train_days",
            str(training_config["histories_soft_constraint_max_n_train_days"]),
        ]
    if training_config["histories_remove_negrated_from_history"]:
        args += ["--histories_remove_negrated_from_history"]
    if not args_dict["trial"]:
        args += ["--move_outputs_folder", str(args_dict["model_path"].parent / "outputs")]
    subprocess.run(args, check=True)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args_dict = parse_args()
    training_config = get_training_config(args_dict)

    if args_dict["trial"]:
        create_trial_recommender()
    recommender = load_testing_recommender(
        model_path=args_dict["model_path"],
        embeddings_path=training_config["embeddings_path"],
        users_encoder_type=training_config["users_encoder_type"],
        device=device,
        trial=args_dict["trial"],
    )
    dataset = load_testing_dataset(training_config)
    dataloader = load_testing_dataloader(recommender, dataset)
    users_embeddings, users_sessions_ids_to_idxs = (
        recommender.users_encoder.compute_users_embeddings(dataloader=dataloader)
    )
    data_folder = get_testing_data_folder()
    save_users_embeddings_as_pickle(
        path=data_folder / "users_embeddings" / f"s_{VAL_RANDOM_STATE}",
        embeddings=users_embeddings,
        users_sessions_ids_to_idxs=users_sessions_ids_to_idxs,
    )
    run_sequence_testing(args_dict, training_config, data_folder)
