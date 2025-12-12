import os
import pickle

import numpy as np
import pandas as pd
import torch

from ....finetuning.src.finetuning_preprocessing import (
    attach_categories_to_papers_tensor,
    save_eval_papers_tokenized,
    save_negative_samples_val,
)
from ....logreg.src.training.users_ratings import (
    N_NEGRATED_RANKING,
    UsersRatingsSelection,
    load_users_ratings_from_selection,
)
from ....src.load_files import load_sequence_users_ids
from ....src.project_paths import ProjectPaths


def get_users_ratings_for_ranking_matrix_val() -> tuple:
    users_ratings_full = load_users_ratings_from_selection(
        users_ratings_selection=UsersRatingsSelection.SESSION_BASED_FILTERING,
        relevant_users_ids="sequence_val",
    )
    pos_val_ratings = users_ratings_full[users_ratings_full["rating"] == 1]
    pos_val_ratings = pos_val_ratings[pos_val_ratings["split"] == "val"].reset_index(drop=True)
    assert pos_val_ratings.equals(
        pos_val_ratings.sort_values(by=["user_id", "session_id", "time"]).reset_index(drop=True)
    )
    neg_ratings = users_ratings_full[users_ratings_full["rating"] == 0].reset_index(drop=True)
    return pos_val_ratings, neg_ratings


def get_user_ratings_for_ranking_matrix_val(
    pos_val_ratings: pd.DataFrame, neg_ratings: pd.DataFrame, user_id: int, session_id: int
) -> tuple:
    pos_val_user_session = pos_val_ratings[
        (pos_val_ratings["user_id"] == user_id) & (pos_val_ratings["session_id"] == session_id)
    ].reset_index(drop=True)
    neg_user_sessions = neg_ratings[
        (neg_ratings["user_id"] == user_id) & (neg_ratings["session_id"] >= session_id)
    ].reset_index(drop=True)
    return pos_val_user_session, neg_user_sessions


def save_ranking_matrix_val() -> None:
    matrix_path = ProjectPaths.sequence_data_model_datasets_path() / "ranking_matrix_val.pt"
    dict_path = (
        ProjectPaths.sequence_data_model_datasets_path()
        / "ranking_matrix_val_pairs_endings_indices.pt"
    )
    if matrix_path.exists() and dict_path.exists():
        print(f"Ranking matrix already exists at {matrix_path}. Skipping save.")
        return
    papers_tensor = load_sequence_papers_tokenized(papers_type="eval_val_users")
    papers_ids_to_idxs = {pid: idx for idx, pid in enumerate(papers_tensor["paper_id"].tolist())}
    pos_val_ratings, neg_ratings = get_users_ratings_for_ranking_matrix_val()

    ranking_matrix_val_shape = (len(pos_val_ratings), N_NEGRATED_RANKING + 1)
    ranking_matrix_val = np.zeros(ranking_matrix_val_shape, dtype=np.int64)
    pos_val_users_sessions_pairs = pos_val_ratings[["user_id", "session_id"]].drop_duplicates()
    users_sessions_starting_indices, users_sessions_ending_indices = [], []

    for pair_idx, (user_id, session_id) in pos_val_users_sessions_pairs.iterrows():
        pos_val_user_session, neg_user_sessions = get_user_ratings_for_ranking_matrix_val(
            pos_val_ratings, neg_ratings, user_id, session_id
        )
        neg_user_times = np.array(neg_user_sessions["time"])

        if pair_idx == 0:
            user_session_starting_index = 0
        else:
            user_session_starting_index = users_sessions_ending_indices[-1]
        users_sessions_starting_indices.append(user_session_starting_index)
        user_session_ending_index = user_session_starting_index + len(pos_val_user_session)
        users_sessions_ending_indices.append(user_session_ending_index)

        ranking_matrix_user = np.zeros(
            (len(pos_val_user_session), N_NEGRATED_RANKING + 1), dtype=np.int64
        )
        ranking_matrix_user[:, 0] = pos_val_user_session["paper_id"].values
        for i, row in pos_val_user_session.iterrows():
            rating_time = np.datetime64(row["time"])
            time_diffs = np.abs(neg_user_times - rating_time)
            closest_idxs = np.argsort(time_diffs)[:N_NEGRATED_RANKING]
            ranking_matrix_user[i, 1:] = neg_user_sessions.iloc[closest_idxs]["paper_id"].values
        ranking_matrix_val[user_session_starting_index:user_session_ending_index, :] = (
            ranking_matrix_user
        )
    ranking_matrix_val_idxs = np.vectorize(papers_ids_to_idxs.get)(ranking_matrix_val)
    ranking_matrix_val_tensor = torch.tensor(ranking_matrix_val_idxs, dtype=torch.long)
    torch.save(ranking_matrix_val_tensor, matrix_path)
    torch.save(torch.tensor(users_sessions_ending_indices, dtype=torch.long), dict_path)
    print(f"Saved ranking matrix of shape {ranking_matrix_val_tensor.shape} at {matrix_path}.")
    print(
        f"Saved users_sessions_ending_indices of length {len(users_sessions_ending_indices)} at {dict_path}."
    )


def load_sequence_papers_tokenized(
    papers_type: str,
    attach_l1: bool = True,
    attach_l2: bool = True,
    papers: pd.DataFrame = None,
    categories_to_idxs_l1: dict = None,
    categories_to_idxs_l2: dict = None,
) -> dict:
    papers_types = [
        "eval_val_users",
        "eval_test_users",
        "negative_samples_val",
    ]
    if papers_type not in papers_types:
        raise ValueError(f"Invalid papers type: {papers_type}. Choose from {papers_types}.")
    if papers_type == "eval_val_users":
        tensor_path = (
            ProjectPaths.sequence_data_model_datasets_path() / "eval_papers_tokenized_val_users.pt"
        )
    elif papers_type == "eval_test_users":
        tensor_path = (
            ProjectPaths.sequence_data_model_datasets_path() / "eval_papers_tokenized_test_users.pt"
        )
    elif papers_type == "negative_samples_val":
        tensor_path = (
            ProjectPaths.sequence_data_model_datasets_path() / "negative_samples_tokenized_val.pt"
        )
    papers_tensor = torch.load(tensor_path, weights_only=True)
    papers_tensor = attach_categories_to_papers_tensor(
        papers_tensor,
        attach_l1=attach_l1,
        attach_l2=attach_l2,
        papers=papers,
        categories_to_idxs_l1=categories_to_idxs_l1,
        categories_to_idxs_l2=categories_to_idxs_l2,
    )
    assert papers_tensor["paper_id"].tolist() == sorted(papers_tensor["paper_id"].tolist())
    assert len(papers_tensor["paper_id"].tolist()) == len(set(papers_tensor["paper_id"].tolist()))
    return papers_tensor


def load_negative_samples_matrix_val() -> torch.Tensor:
    from ....finetuning.src.finetuning_preprocessing import (
        load_negative_samples_matrix_val,
    )

    return load_negative_samples_matrix_val(
        matrix_path=ProjectPaths.sequence_data_model_datasets_path()
        / "negative_samples_matrix_val.pt"
    )


def load_ranking_matrix_val() -> tuple:
    matrix_path = ProjectPaths.sequence_data_model_datasets_path() / "ranking_matrix_val.pt"
    dict_path = (
        ProjectPaths.sequence_data_model_datasets_path()
        / "ranking_matrix_val_pairs_endings_indices.pt"
    )
    ranking_matrix_val = torch.load(matrix_path, weights_only=True)
    users_sessions_endings_indices = torch.load(dict_path, weights_only=True)
    assert users_sessions_endings_indices[-1] == ranking_matrix_val.shape[0]
    return ranking_matrix_val, users_sessions_endings_indices


def save_sequence_datasets_standard_train() -> None:
    from .sessions_dataset import (
        get_standard_train_sessions_dataset_params,
        load_sessions_dataset_by_split,
    )

    standard_train_sessions_dataset_params = get_standard_train_sessions_dataset_params()
    train_set_path = ProjectPaths.sequence_data_model_datasets_standard_train_path()
    if train_set_path.exists():
        print(f"Standard train dataset already exists at {train_set_path}. Skipping save.")
    else:
        train_set = load_sessions_dataset_by_split(
            split="train",
            **standard_train_sessions_dataset_params,
            histories_remove_negrated_from_history=False,
        )
        with open(train_set_path, "wb") as f:
            pickle.dump(train_set, f)
        print(f"Saved standard train dataset at {train_set_path}.")
    train_set_no_neg_path = ProjectPaths.sequence_data_model_datasets_standard_train_no_neg_path()
    if train_set_no_neg_path.exists():
        print(
            f"Standard train no neg dataset already exists at {train_set_no_neg_path}. Skipping save."
        )
    else:
        train_set_no_neg = load_sessions_dataset_by_split(
            split="train",
            **standard_train_sessions_dataset_params,
            histories_remove_negrated_from_history=True,
        )
        with open(train_set_no_neg_path, "wb") as f:
            pickle.dump(train_set_no_neg, f)
        print(f"Saved standard train no neg dataset at {train_set_no_neg_path}.")
    train_set_no_neg_no_causal_path = (
        ProjectPaths.sequence_data_model_datasets_standard_train_no_neg_no_causal_path()
    )
    if train_set_no_neg_no_causal_path.exists():
        print(
            f"Standard train no neg no causal dataset already exists at {train_set_no_neg_no_causal_path}. Skipping save."
        )
    else:
        train_set_no_neg_no_causal = load_sessions_dataset_by_split(
            split="train",
            **standard_train_sessions_dataset_params,
            histories_remove_negrated_from_history=True,
            negrated_causal_mask=False,
        )
        with open(train_set_no_neg_no_causal_path, "wb") as f:
            pickle.dump(train_set_no_neg_no_causal, f)
        print(f"Saved standard train no neg no causal dataset at {train_set_no_neg_no_causal_path}.")


if __name__ == "__main__":
    os.makedirs(ProjectPaths.sequence_data_model_path(), exist_ok=True)
    os.makedirs(ProjectPaths.sequence_data_model_datasets_path(), exist_ok=True)
    os.makedirs(ProjectPaths.sequence_data_model_state_dicts_path(), exist_ok=True)
    save_eval_papers_tokenized(mode="sequence")
    sequence_val_users_ids = load_sequence_users_ids("val")
    save_negative_samples_val(
        tensor_path=ProjectPaths.sequence_data_model_datasets_path()
        / "negative_samples_tokenized_val.pt",
        matrix_path=ProjectPaths.sequence_data_model_datasets_path()
        / "negative_samples_matrix_val.pt",
        users_ids=sequence_val_users_ids,
    )
    save_ranking_matrix_val()
    save_sequence_datasets_standard_train()
