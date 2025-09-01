import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from .finetuning_model import FinetuningModel, load_finetuning_model
from .finetuning_preprocessing import (
    load_finetuning_papers_tokenized,
    load_users_coefs_ids_to_idxs,
    load_val_users_embeddings_idxs,
)


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description="Evaluation script.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--not_val_users", action="store_false", dest="val_users")
    parser.add_argument("--not_test_users", action="store_false", dest="test_users")
    args = vars(parser.parse_args())
    args["model_path"] = Path(args["model_path"])
    if not args["model_path"].exists():
        raise FileNotFoundError(f"Model path {args['model_path']} does not exist.")
    return args


def get_eval_papers_tokenized(val_users: bool, test_users: bool) -> dict:
    if not val_users and not test_users:
        raise ValueError("At least one of val_users or test_users must be True.")
    elif val_users and not test_users:
        return load_finetuning_papers_tokenized("eval_val_users")
    elif not val_users and test_users:
        return load_finetuning_papers_tokenized("eval_test_users")
    elif val_users and test_users:
        eval_papers_tokenized_val_users = load_finetuning_papers_tokenized("eval_val_users")
        eval_papers_tokenized_test_users = load_finetuning_papers_tokenized("eval_test_users")
        merged_dict = {}
        for key in eval_papers_tokenized_val_users.keys():
            merged_dict[key] = torch.cat(
                [eval_papers_tokenized_val_users[key], eval_papers_tokenized_test_users[key]], dim=0
            )
        paper_ids = merged_dict["paper_id"]
        unique_mask = torch.zeros(paper_ids.size(0), dtype=torch.bool)
        seen_ids = set()
        for i, paper_id in enumerate(paper_ids):
            paper_id_item = paper_id.item()
            if paper_id_item not in seen_ids:
                unique_mask[i] = True
                seen_ids.add(paper_id_item)
        for key in merged_dict.keys():
            merged_dict[key] = merged_dict[key][unique_mask]
        return merged_dict


def save_users_coefs(
    finetuning_model: FinetuningModel,
    embeddings_folder: Path,
    users_embeddings_ids_to_idxs: dict,
) -> None:
    if not isinstance(embeddings_folder, Path):
        embeddings_folder = Path(embeddings_folder).resolve()
    users_coefs = finetuning_model.users_embeddings.weight.detach().cpu().numpy().astype(np.float64)
    np.save(embeddings_folder / "users_coefs.npy", users_coefs)
    with open(embeddings_folder / "users_coefs_ids_to_idxs.pkl", "wb") as f:
        pickle.dump(users_embeddings_ids_to_idxs, f)


def compute_eval_embeddings(
    finetuning_model: FinetuningModel,
    eval_papers_tokenized: dict,
    embeddings_folder: Path,
    users_embeddings_ids_to_idxs: dict,
) -> None:
    training_mode = finetuning_model.training
    finetuning_model.eval()
    papers_ids_to_idxs = {
        paper_id.item(): idx for idx, paper_id in enumerate(eval_papers_tokenized["paper_id"])
    }
    embeddings = finetuning_model.compute_papers_embeddings(
        input_ids_tensor=eval_papers_tokenized["input_ids"],
        attention_mask_tensor=eval_papers_tokenized["attention_mask"],
        category_l1_tensor=eval_papers_tokenized["l1"],
        category_l2_tensor=eval_papers_tokenized["l2"],
    )
    np.save(f"{embeddings_folder / 'abs_X.npy'}", embeddings)
    with open(f"{embeddings_folder / 'abs_paper_ids_to_idx.pkl'}", "wb") as f:
        pickle.dump(papers_ids_to_idxs, f)
    save_users_coefs(finetuning_model, embeddings_folder, users_embeddings_ids_to_idxs)
    if training_mode:
        finetuning_model.train()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()

    users_embeddings_ids_to_idxs = load_users_coefs_ids_to_idxs()
    val_users_embeddings_idxs = load_val_users_embeddings_idxs()
    include_l2 = (args["model_path"] / "state_dicts" / "categories_embeddings_l2.pt").exists()
    finetuning_model = load_finetuning_model(
        args["model_path"] / "state_dicts",
        device=device,
        mode="eval",
        val_users_embeddings_idxs=val_users_embeddings_idxs,
        include_categories_embeddings_l2=include_l2,
    )
    eval_papers_tokenized = get_eval_papers_tokenized(args["val_users"], args["test_users"])
    embeddings_path = args["model_path"] / "embeddings"
    os.makedirs(embeddings_path, exist_ok=True)
    print("Computing evaluation embeddings...")
    compute_eval_embeddings(
        finetuning_model,
        eval_papers_tokenized,
        embeddings_path,
        users_embeddings_ids_to_idxs,
    )
