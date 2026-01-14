import pickle

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from ....logreg.src.training.users_ratings import (
    UsersRatingsSelection,
    load_users_ratings_from_selection,
)
from ....src.project_paths import ProjectPaths
from ..eval.compute_users_embeddings import get_user_train_set
from .users_embeddings_data import get_users_val_sessions_ids


def get_standard_train_sessions_dataset_params() -> dict:
    return {
        "histories_hard_constraint_min_n_train_posrated": 0,
        "histories_hard_constraint_max_n_train_rated": 1024,
        "histories_soft_constraint_max_n_train_sessions": None,
        "histories_soft_constraint_max_n_train_days": None,
        "negrated_hard_constraint_max_n_ratings": 15,
        "negrated_hard_constraint_max_n_sessions": None,
        "negrated_hard_constraint_max_n_days": None,
    }


class SessionsDataset(Dataset):
    def __init__(self, split: str, sessions_list: list) -> None:
        self.split = split
        self.sessions_list = sessions_list
        self.set_lists()

    def __len__(self) -> int:
        return len(self.sessions_list)

    def __getitem__(self, idx: int) -> dict:
        return self.sessions_list[idx]

    def set_lists(self) -> None:
        histories_lengths, candidates_lengths, negrated_yet_to_come_lengths = [], [], []
        histories_max_length, candidates_max_length, negrated_yet_to_come_max_length = 0, 0, 0

        for item in self.sessions_list:
            history_length = len(item["history_dict"]["history_papers_ids"])
            histories_lengths.append(history_length)
            histories_max_length = max(histories_max_length, history_length)
            if self.split == "train":
                candidates_length = len(item["candidates_dict"]["candidates_papers_ids"])
                candidates_lengths.append(candidates_length)
                candidates_max_length = max(candidates_max_length, candidates_length)
                negrated_yet_to_come_length = len(
                    item["negrated_yet_to_come_dict"]["negrated_yet_to_come_papers_ids"]
                )
                negrated_yet_to_come_lengths.append(negrated_yet_to_come_length)
                negrated_yet_to_come_max_length = max(
                    negrated_yet_to_come_max_length, negrated_yet_to_come_length
                )

        self.histories_lengths = histories_lengths
        self.histories_max_length = histories_max_length
        self.candidates_lengths = candidates_lengths
        self.candidates_max_length = candidates_max_length
        self.negrated_yet_to_come_lengths = negrated_yet_to_come_lengths
        self.negrated_yet_to_come_max_length = negrated_yet_to_come_max_length


def load_standard_train_sessions_dataset(
    histories_remove_negrated_from_history: bool,
) -> SessionsDataset:
    if histories_remove_negrated_from_history:
        dataset_path = ProjectPaths.sequence_data_model_datasets_standard_train_no_neg_path()
    else:
        dataset_path = ProjectPaths.sequence_data_model_datasets_standard_train_path()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Standard train sessions dataset not found at {dataset_path}. Please generate it first."
        )
    with open(dataset_path, "rb") as f:
        sessions_dataset = pickle.load(f)
    return sessions_dataset


def load_users_ratings_full_by_split(split: str) -> pd.DataFrame:
    if split == "train":
        urs = UsersRatingsSelection.SEQUENCE_TRAIN
        relevant_users_ids = None
    elif split == "val":
        urs = UsersRatingsSelection.SESSION_BASED_FILTERING
        relevant_users_ids = "sequence_val"
    elif split == "test":
        urs = UsersRatingsSelection.SESSION_BASED_FILTERING
        relevant_users_ids = "sequence_test"
    return load_users_ratings_from_selection(
        users_ratings_selection=urs,
        relevant_users_ids=relevant_users_ids,
    )


def get_users_val_sessions_ids_for_sessions_dataset_by_split(
    split: str, users_ratings_full: pd.DataFrame
) -> dict:
    if split in ["train", "val"]:
        users_ratings = users_ratings_full[users_ratings_full["rating"] == 1].reset_index(drop=True)
    else:
        users_ratings = users_ratings_full
    assert users_ratings["user_id"].nunique() == users_ratings_full["user_id"].nunique()
    return get_users_val_sessions_ids(users_ratings=users_ratings)


def extract_history_dict_for_sessions_dataset(
    session_train_set: pd.DataFrame, session_start_time: pd.Timestamp
) -> dict:
    history_dict = {
        "history_labels": session_train_set["rating"].values,
        "history_papers_ids": session_train_set["paper_id"].values,
        "history_sessions_ids": session_train_set["session_id"].values,
        "history_times": session_train_set["time"].values,
    }
    session_start = pd.to_datetime(session_start_time)
    times_diffs = session_start - pd.to_datetime(history_dict["history_times"])
    days_diffs = times_diffs.days
    history_dict["history_days_diffs"] = days_diffs.tolist()
    return history_dict


def extract_candidates_dict_for_sessions_dataset(
    user_ratings: pd.DataFrame,
    session_id: int,
) -> dict:
    pos_ratings_current_session = user_ratings[
        (user_ratings["session_id"] == session_id) & (user_ratings["rating"] == 1)
    ].reset_index(drop=True)
    candidates_dict = {
        "candidates_papers_ids": pos_ratings_current_session["paper_id"].values,
        "candidates_times": pos_ratings_current_session["time"].values,
    }
    return candidates_dict


def extract_negrated_yet_to_come_dict_for_sessions_dataset(
    user_ratings: pd.DataFrame,
    session_id: int,
    session_start_time: pd.Timestamp,
    hard_constraint_max_n_ratings: int,
    hard_constraint_max_n_sessions: int,
    hard_constraint_max_n_days: int,
    causal_mask: bool,
) -> dict:
    neg_ratings = user_ratings[user_ratings["rating"] == 0].reset_index(drop=True)
    if causal_mask:
        neg_ratings = neg_ratings[neg_ratings["session_id"] > session_id].reset_index(drop=True)
    negrated_yet_to_come_dict = {
        "negrated_yet_to_come_papers_ids": neg_ratings["paper_id"].values,
        "negrated_yet_to_come_times": neg_ratings["time"].values,
        "negrated_yet_to_come_sessions_ids": neg_ratings["session_id"].values,
    }
    times_series = pd.Series(negrated_yet_to_come_dict["negrated_yet_to_come_times"])
    times_diffs_seconds = (times_series - session_start_time).dt.total_seconds()
    if causal_mask:
        assert np.all(np.diff(negrated_yet_to_come_dict["negrated_yet_to_come_sessions_ids"]) >= 0)
        assert np.all(times_diffs_seconds >= 0)
    else:
        sorted_indices = np.argsort(times_diffs_seconds.abs().values)
        negrated_yet_to_come_dict = {
            key: value[sorted_indices] for key, value in negrated_yet_to_come_dict.items()
        }
    if hard_constraint_max_n_ratings is not None:
        negrated_yet_to_come_dict = {
            key: value[:hard_constraint_max_n_ratings]
            for key, value in negrated_yet_to_come_dict.items()
        }
    if hard_constraint_max_n_sessions is not None:
        sessions_diffs = np.abs(negrated_yet_to_come_dict["negrated_yet_to_come_sessions_ids"] - session_id)
        mask = sessions_diffs <= hard_constraint_max_n_sessions
        negrated_yet_to_come_dict = {
            key: value[mask] for key, value in negrated_yet_to_come_dict.items()
        }
    if hard_constraint_max_n_days is not None:
        times_series = pd.Series(negrated_yet_to_come_dict["negrated_yet_to_come_times"])
        time_diffs_days = (times_series - session_start_time).dt.days.abs()
        mask = time_diffs_days.values <= hard_constraint_max_n_days
        negrated_yet_to_come_dict = {
            key: value[mask] for key, value in negrated_yet_to_come_dict.items()
        }
    return negrated_yet_to_come_dict


def get_sessions_list_by_split(
    split: str,
    users_ratings_full: pd.DataFrame,
    val_sessions_ids: dict,
    histories_hard_constraint_min_n_train_posrated: int,
    histories_hard_constraint_max_n_train_rated: int,
    histories_soft_constraint_max_n_train_sessions: int,
    histories_soft_constraint_max_n_train_days: int,
    histories_remove_negrated_from_history: bool,
    negrated_hard_constraint_max_n_ratings: int,
    negrated_hard_constraint_max_n_sessions: int,
    negrated_hard_constraint_max_n_days: int,
    negrated_causal_mask: bool,
) -> list:
    sessions_list = []
    users_ids = users_ratings_full["user_id"].unique().tolist()
    for uid in tqdm(users_ids):
        user_ratings = users_ratings_full[users_ratings_full["user_id"] == uid]
        user_sessions_ids = val_sessions_ids[uid]
        for session_id in user_sessions_ids:
            session_start_time = user_ratings[user_ratings["session_id"] == session_id][
                "time"
            ].min()
            session_dict = {
                "user_id": uid,
                "session_id": session_id,
                "session_start_time": session_start_time,
            }
            session_train_set = get_user_train_set(
                user_ratings=user_ratings,
                session_id=session_id,
                hard_constraint_min_n_train_posrated=histories_hard_constraint_min_n_train_posrated,
                hard_constraint_max_n_train_rated=histories_hard_constraint_max_n_train_rated,
                soft_constraint_max_n_train_sessions=histories_soft_constraint_max_n_train_sessions,
                soft_constraint_max_n_train_days=histories_soft_constraint_max_n_train_days,
                remove_negrated_from_history=histories_remove_negrated_from_history,
                ignore_hard_constraint_min_n_train_posrated=(split == "train"),
            )
            session_dict["history_dict"] = extract_history_dict_for_sessions_dataset(
                session_train_set, session_start_time
            )
            if split == "train":
                session_dict["candidates_dict"] = extract_candidates_dict_for_sessions_dataset(
                    user_ratings=user_ratings, session_id=session_id
                )
                session_dict["negrated_yet_to_come_dict"] = (
                    extract_negrated_yet_to_come_dict_for_sessions_dataset(
                        user_ratings=user_ratings,
                        session_id=session_id,
                        session_start_time=session_start_time,
                        hard_constraint_max_n_ratings=negrated_hard_constraint_max_n_ratings,
                        hard_constraint_max_n_sessions=negrated_hard_constraint_max_n_sessions,
                        hard_constraint_max_n_days=negrated_hard_constraint_max_n_days,
                        causal_mask=negrated_causal_mask,
                    )
                )
            sessions_list.append(session_dict)
    return sessions_list


def check_standard_train_sessions_dataset(
    histories_hard_constraint_min_n_train_posrated: int,
    histories_hard_constraint_max_n_train_rated: int,
    histories_soft_constraint_max_n_train_sessions: int,
    histories_soft_constraint_max_n_train_days: int,
    negrated_hard_constraint_max_n_ratings: int,
    negrated_hard_constraint_max_n_sessions: int,
    negrated_hard_constraint_max_n_days: int,
) -> bool:
    standard_params = get_standard_train_sessions_dataset_params()
    return (
        histories_hard_constraint_min_n_train_posrated
        == standard_params["histories_hard_constraint_min_n_train_posrated"]
        and histories_hard_constraint_max_n_train_rated
        == standard_params["histories_hard_constraint_max_n_train_rated"]
        and histories_soft_constraint_max_n_train_sessions
        == standard_params["histories_soft_constraint_max_n_train_sessions"]
        and histories_soft_constraint_max_n_train_days
        == standard_params["histories_soft_constraint_max_n_train_days"]
        and negrated_hard_constraint_max_n_ratings
        == standard_params["negrated_hard_constraint_max_n_ratings"]
        and negrated_hard_constraint_max_n_sessions
        == standard_params["negrated_hard_constraint_max_n_sessions"]
        and negrated_hard_constraint_max_n_days
        == standard_params["negrated_hard_constraint_max_n_days"]
    )


def load_sessions_dataset_by_split(
    split: str,
    histories_hard_constraint_min_n_train_posrated: int = 0,
    histories_hard_constraint_max_n_train_rated: int = None,
    histories_soft_constraint_max_n_train_sessions: int = None,
    histories_soft_constraint_max_n_train_days: int = None,
    histories_remove_negrated_from_history: bool = False,
    negrated_hard_constraint_max_n_ratings: int = 15,
    negrated_hard_constraint_max_n_sessions: int = None,
    negrated_hard_constraint_max_n_days: int = None,
    negrated_causal_mask: bool = True,
) -> SessionsDataset:
    if not histories_remove_negrated_from_history and not negrated_causal_mask:
        raise ValueError(
            "If histories_remove_negrated_from_history is False, negrated_causal_mask must be True."
        )
    valid_splits = ["train", "val", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split: {split}. Choose from {valid_splits}.")
    if split == "train" and check_standard_train_sessions_dataset(
        histories_hard_constraint_min_n_train_posrated,
        histories_hard_constraint_max_n_train_rated,
        histories_soft_constraint_max_n_train_sessions,
        histories_soft_constraint_max_n_train_days,
        negrated_hard_constraint_max_n_ratings,
        negrated_hard_constraint_max_n_sessions,
        negrated_hard_constraint_max_n_days,
    ):
        if histories_remove_negrated_from_history:
            if negrated_causal_mask:
                path = ProjectPaths.sequence_data_model_datasets_standard_train_no_neg_path()
            else:
                path = ProjectPaths.sequence_data_model_datasets_standard_train_no_neg_no_causal_path()
        else:
            path = ProjectPaths.sequence_data_model_datasets_standard_train_path()
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)

    users_ratings_full = load_users_ratings_full_by_split(split=split)
    users_val_sessions_ids = get_users_val_sessions_ids_for_sessions_dataset_by_split(
        split=split, users_ratings_full=users_ratings_full
    )
    sessions_list = get_sessions_list_by_split(
        split=split,
        users_ratings_full=users_ratings_full,
        val_sessions_ids=users_val_sessions_ids,
        histories_hard_constraint_max_n_train_rated=histories_hard_constraint_max_n_train_rated,
        histories_hard_constraint_min_n_train_posrated=histories_hard_constraint_min_n_train_posrated,
        histories_soft_constraint_max_n_train_sessions=histories_soft_constraint_max_n_train_sessions,
        histories_soft_constraint_max_n_train_days=histories_soft_constraint_max_n_train_days,
        histories_remove_negrated_from_history=histories_remove_negrated_from_history,
        negrated_hard_constraint_max_n_ratings=negrated_hard_constraint_max_n_ratings,
        negrated_hard_constraint_max_n_sessions=negrated_hard_constraint_max_n_sessions,
        negrated_hard_constraint_max_n_days=negrated_hard_constraint_max_n_days,
        negrated_causal_mask=negrated_causal_mask,
    )
    return SessionsDataset(split=split, sessions_list=sessions_list)
