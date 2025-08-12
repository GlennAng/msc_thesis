import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ..src.project_paths import ProjectPaths
from .create_example_configs import (
    create_example_config,
    create_example_config_cross_val,
)


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description="Evaluation script.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--not_val_users_session_based", action="store_false", dest="val_users_session_based"
    )
    parser.add_argument(
        "--not_test_users_session_based", action="store_false", dest="test_users_session_based"
    )
    parser.add_argument(
        "--use_existing_embeddings", action="store_true", dest="use_existing_embeddings"
    )
    parser.add_argument("--test_users_cross_val", action="store_true", dest="test_users_cross_val")
    args = vars(parser.parse_args())
    args["model_path"] = Path(args["model_path"])
    if not args["model_path"].exists():
        raise FileNotFoundError(f"Model path {args['model_path']} does not exist.")
    return args


def check_embeddings_exist(embeddings_path: Path, use_existing_embeddings: bool) -> bool:
    embeddings_exist = (embeddings_path / "abs_X.npy").exists() and (
        embeddings_path / "abs_paper_ids_to_idx.pkl"
    ).exists()
    if embeddings_exist and not use_existing_embeddings:
        raise FileExistsError(
            f"Embeddings already exist at {embeddings_path}. "
            "Use --use_existing_embeddings to use existing embeddings."
        )
    if not embeddings_exist and use_existing_embeddings:
        raise FileNotFoundError(
            f"Embeddings do not exist at {embeddings_path}. "
            "Please run the embedding generation script first."
        )
    return embeddings_exist


def compute_embeddings(args: dict) -> None:
    val_users = args["val_users_session_based"]
    test_users = args["test_users_session_based"] or args["test_users_cross_val"]
    args_list = [
        sys.executable,
        "-m",
        "code.finetuning.src.finetuning_eval_embeddings",
        "--model_path",
        str(args["model_path"]),
    ]
    if not val_users:
        args_list.append("--not_val_users")
    if not test_users:
        args_list.append("--not_test_users")
    subprocess.run(
        args_list,
        check=True,
    )


def run_val_users(embeddings_path: Path, selection: str = "session_based") -> None:
    if selection not in ["session_based", "cross_validation"]:
        raise ValueError("Selection must be either 'session_based' or 'cross_validation'.")
    name = f"val_users_{selection}"
    config_val_users_path = embeddings_path / f"config_{name}.json"
    if selection == "session_based":
        config_val_users = create_example_config(embeddings_path)
    else:
        config_val_users = create_example_config_cross_val(embeddings_path)
    config_val_users["users_selection"] = "finetuning_val"
    with open(config_val_users_path, "w") as f:
        json.dump(config_val_users, f, indent=4)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "code.scripts.logreg_run",
            "--config_path",
            str(config_val_users_path),
        ],
        check=True,
    )
    config_val_users_path.unlink()
    outputs_path = ProjectPaths.logreg_outputs_path() / f"config_{name}" / "global_visu_ndcg_all.pdf"
    shutil.move(outputs_path, embeddings_path.parent / f"visu_{name}.pdf")


def run_test_users(embeddings_path: Path, selection: str = "session_based") -> None:
    if selection not in ["session_based", "cross_validation"]:
        raise ValueError("Selection must be either 'session_based' or 'cross_validation'.")
    name = f"test_users_{selection}"
    config_test_users_path = embeddings_path / f"config_{name}.json"
    if selection == "session_based":
        config_test_users = create_example_config(embeddings_path)
    else:
        config_test_users = create_example_config_cross_val(embeddings_path)
    config_test_users["users_selection"] = "finetuning_test"
    with open(config_test_users_path, "w") as f:
        json.dump(config_test_users, f, indent=4)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "code.scripts.average_seeds",
            "--config_path",
            str(config_test_users_path),
        ],
        check=True,
    )
    config_test_users_path.unlink()
    outputs_path = ProjectPaths.logreg_outputs_path() / f"config_{name}_averaged" / "global_visu_ndcg_all.pdf"
    shutil.move(outputs_path, embeddings_path.parent / f"visu_{name}.pdf")


args = parse_arguments()
embeddings_path = args["model_path"] / "embeddings"
embeddings_exist = check_embeddings_exist(embeddings_path, args["use_existing_embeddings"])
if not embeddings_exist:
    compute_embeddings(args)

if args["val_users_session_based"]:
    run_val_users(embeddings_path, selection="session_based")
if args["test_users_session_based"]:
    run_test_users(embeddings_path, selection="session_based")
if args["test_users_cross_val"]:
    run_test_users(embeddings_path, selection="cross_validation")
