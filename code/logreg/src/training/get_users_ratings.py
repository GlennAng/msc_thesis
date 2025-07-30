import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd

from ....src.load_files import (
    load_finetuning_users_ids,
    load_session_based_users_ids,
    load_users_ratings,
)
from ....src.project_paths import ProjectPaths
from .algorithm import Evaluation

LABEL_DTYPE = np.int64

pd.set_option("display.max_rows", None)


def get_train_test_split(
    users_ratings: pd.DataFrame, test_size: float, random_state: int, stratify: bool = True
) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    if test_size == 0.0 or test_size == 1.0:
        users_ratings_copy = users_ratings.copy()
        users_ratings_copy["split"] = "train" if test_size == 0.0 else "val"
        return users_ratings_copy

    def split_user_data(user_ratings: pd.DataFrame) -> pd.DataFrame:
        posrated_ids = user_ratings[user_ratings["rating"] == 1]["paper_id"].values
        negrated_ids = user_ratings[user_ratings["rating"] == 0]["paper_id"].values

        rated_ids = np.concatenate((posrated_ids, negrated_ids))
        posrated_n, negrated_n = len(posrated_ids), len(negrated_ids)
        rated_labels = np.concatenate(
            (np.ones(posrated_n, dtype=LABEL_DTYPE), np.zeros(negrated_n, dtype=LABEL_DTYPE))
        )
        train_rated_ids, _, _, _ = train_test_split(
            rated_ids,
            rated_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=rated_labels if stratify else None,
        )
        user_ratings_copy = user_ratings.copy()
        user_ratings_copy["split"] = user_ratings_copy["paper_id"].apply(
            lambda x: "train" if x in train_rated_ids else "val"
        )
        return user_ratings_copy

    result_df = users_ratings.groupby("user_id", group_keys=False).apply(
        split_user_data, include_groups=True
    )
    return result_df.reset_index(drop=True)


def filter_users_ratings_with_sufficient_votes(
    users_ratings: pd.DataFrame, min_n_posrated: int, min_n_negrated: int
) -> pd.DataFrame:
    users_ratings = users_ratings.copy()
    min_n_posrated = 0 if (min_n_posrated < 0 or min_n_posrated is None) else min_n_posrated
    min_n_negrated = 0 if (min_n_negrated < 0 or min_n_negrated is None) else min_n_negrated
    users_ratings_counts = (
        users_ratings.groupby("user_id")
        .agg(
            n_posrated=("rating", lambda x: (x == 1).sum()),
            n_negrated=("rating", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )
    users_ratings_counts_sufficient_votes = users_ratings_counts[
        (users_ratings_counts["n_posrated"] >= min_n_posrated)
        & (users_ratings_counts["n_negrated"] >= min_n_negrated)
    ]
    users_ratings_filtered = users_ratings[
        users_ratings["user_id"].isin(users_ratings_counts_sufficient_votes["user_id"])
    ]
    return users_ratings_filtered.reset_index(drop=True)


def check_single_split(
    user_ratings: pd.DataFrame,
    split_session_id: int,
    min_n_posrated_train: int,
    min_n_negrated_train: int,
    min_n_posrated_val: int,
    min_n_negrated_val: int,
) -> tuple:
    train_mask, val_mask = (
        user_ratings["session_id"] < split_session_id,
        user_ratings["session_id"] >= split_session_id,
    )
    train_ratings, val_ratings = user_ratings[train_mask], user_ratings[val_mask]
    n_posrated_train, n_posrated_val = train_ratings["rating"].sum(), val_ratings["rating"].sum()
    n_negrated_train, n_negrated_val = (
        len(train_ratings) - n_posrated_train,
        len(val_ratings) - n_posrated_val,
    )
    valid_split_train = (
        n_posrated_train >= min_n_posrated_train and n_negrated_train >= min_n_negrated_train
    )
    valid_split_val = n_posrated_val >= min_n_posrated_val and n_negrated_val >= min_n_negrated_val
    valid_split = valid_split_train and valid_split_val
    test_size_split = len(val_ratings) / len(user_ratings)
    return valid_split, test_size_split


def split_single_user(
    user_ratings: pd.DataFrame,
    test_size: float,
    min_n_posrated_train: int,
    min_n_negrated_train: int,
    min_n_posrated_val: int,
    min_n_negrated_val: int,
) -> pd.DataFrame:
    n_sessions = user_ratings["session_id"].nunique()
    best_session_id, best_session_test_size_diff = None, float("inf")
    for split_session_id in range(n_sessions, -1, -1):
        valid_split, test_size_split = check_single_split(
            user_ratings,
            split_session_id,
            min_n_posrated_train,
            min_n_negrated_train,
            min_n_posrated_val,
            min_n_negrated_val,
        )
        if valid_split and test_size_split <= 0.3:
            test_size_diff = abs(test_size_split - test_size)
            if test_size_diff < best_session_test_size_diff:
                best_session_id, best_session_test_size_diff = split_session_id, test_size_diff
            else:
                if test_size_split > test_size:
                    break
    if best_session_id is None:
        return None
    user_ratings["split"] = user_ratings["session_id"].apply(
        lambda x: "train" if x < best_session_id else "val"
    )
    return user_ratings


def filter_users_ratings_with_sufficient_votes_session_based(
    users_ratings: pd.DataFrame,
    test_size: float,
    min_n_posrated_train: int,
    min_n_negrated_train: int,
    min_n_posrated_val: int,
    min_n_negrated_val: int,
) -> pd.DataFrame:
    assert users_ratings.groupby("user_id")["time"].is_monotonic_increasing.all()
    assert users_ratings.groupby("user_id")["session_id"].is_monotonic_increasing.all()
    users_ratings = users_ratings.copy()
    users_ratings_with_sufficient_votes = []
    users_ids = users_ratings["user_id"].unique()
    for user_id in users_ids:
        user_ratings = users_ratings[users_ratings["user_id"] == user_id].reset_index(drop=True)
        user_ratings_split = split_single_user(
            user_ratings,
            test_size,
            min_n_posrated_train,
            min_n_negrated_train,
            min_n_posrated_val,
            min_n_negrated_val,
        )
        if user_ratings_split is not None:
            users_ratings_with_sufficient_votes.append(user_ratings_split)
    if not users_ratings_with_sufficient_votes:
        return pd.DataFrame(columns=users_ratings.columns)
    users_ratings = pd.concat(users_ratings_with_sufficient_votes, ignore_index=True)
    return users_ratings


def get_users_ids_with_sufficient_votes(users_ratings: pd.DataFrame) -> np.ndarray:
    users_ids_with_sufficient_votes = users_ratings["user_id"].unique()
    assert np.all(users_ids_with_sufficient_votes[:-1] <= users_ids_with_sufficient_votes[1:])
    print(len(users_ids_with_sufficient_votes), "Users with sufficient Votes.")
    return users_ids_with_sufficient_votes


def process_selected_users(
    users_ids_selected: np.ndarray,
    users_ids_with_sufficient_votes: np.ndarray,
    take_complement: bool,
) -> np.ndarray:
    assert np.all(users_ids_selected[:-1] <= users_ids_selected[1:])
    assert len(users_ids_selected) == len(set(users_ids_selected))
    assert set(users_ids_selected) <= set(users_ids_with_sufficient_votes)
    n_users_not_selected = len(users_ids_with_sufficient_votes) - len(users_ids_selected)
    if take_complement:
        assert n_users_not_selected > 0
        users_ids_selected = users_ids_with_sufficient_votes[
            ~np.isin(users_ids_with_sufficient_votes, users_ids_selected)
        ]
        assert np.all(users_ids_selected[:-1] <= users_ids_selected[1:])
    print(len(users_ids_selected), "Users selected.")
    return users_ids_selected


def select_users_explicit(
    users_selection: str, users_ids_with_sufficient_votes: np.ndarray, take_complement: bool
) -> np.ndarray:
    users_ids_selected = np.array(users_selection, dtype=np.int64)
    return process_selected_users(
        users_ids_selected, users_ids_with_sufficient_votes, take_complement
    )


def select_users_random(
    users_ids_with_sufficient_votes: np.ndarray,
    max_users: int = None,
    random_state: int = None,
    take_complement: bool = False,
) -> np.ndarray:
    if max_users is not None and max_users < len(users_ids_with_sufficient_votes):
        series = pd.Series(users_ids_with_sufficient_votes)
        sampled_series = series.sample(n=max_users, random_state=random_state)
        users_ids_selected = sampled_series.values
        users_ids_selected.sort()
    else:
        users_ids_selected = users_ids_with_sufficient_votes
    return process_selected_users(
        users_ids_selected, users_ids_with_sufficient_votes, take_complement
    )


def get_session_based_users_ratings(users_selection: str) -> tuple:
    if users_selection == "finetuning_val":
        users_ids = load_finetuning_users_ids(selection="val")
        users_ratings = pd.read_parquet(
            ProjectPaths.data_session_based_ratings_val_users_path(), engine="pyarrow"
        )
        assert np.all(users_ratings["user_id"].unique() == users_ids)
        print(f"{len(users_ids)} Users selected for Validation Users Session-Based Evaluation.")
        return users_ratings, users_ids
    elif users_selection == "finetuning_test":
        users_ids = load_finetuning_users_ids(selection="test")
        users_ratings = pd.read_parquet(
            ProjectPaths.data_session_based_ratings_test_users_path(), engine="pyarrow"
        )
        assert np.all(users_ratings["user_id"].unique() == users_ids)
        print(f"{len(users_ids)} Users selected for Test Users Session-Based Evaluation.")
        return users_ratings, users_ids
    elif users_selection == "session_based":
        users_ids = load_session_based_users_ids()
        users_ratings = pd.read_parquet(
            ProjectPaths.data_session_based_ratings_session_based_users_path(), engine="pyarrow"
        )
        assert np.all(users_ratings["user_id"].unique() == users_ids)
        print(f"{len(users_ids)} Users selected for Session-Based Users Session-Based Evaluation.")
        return users_ratings, users_ids
    else:
        raise ValueError(f"Invalid users_selection: {users_selection}.")


def get_users_ratings(
    users_selection: str,
    evaluation: Evaluation,
    train_size: float,
    max_users: int = None,
    users_random_state: int = 42,
    model_random_state: int = 42,
    stratify: bool = True,
    min_n_posrated: int = 0,
    min_n_negrated: int = 0,
    take_complement: bool = False,
    min_n_posrated_train: int = 0,
    min_n_negrated_train: int = 0,
    min_n_posrated_val: int = 0,
    min_n_negrated_val: int = 0,
) -> tuple:

    if evaluation == Evaluation.SESSION_BASED:
        path = None
        if users_selection == "finetuning_val":
            path = ProjectPaths.data_session_based_ratings_val_users_path()
        elif users_selection == "finetuning_test":
            path = ProjectPaths.data_session_based_ratings_test_users_path()
        elif users_selection == "session_based":
            path = ProjectPaths.data_session_based_ratings_session_based_users_path()
        if path and path.exists():
            return get_session_based_users_ratings(users_selection)

    test_size = 1.0 - train_size
    users_ratings = load_users_ratings()
    if evaluation in [Evaluation.TRAIN_TEST_SPLIT, Evaluation.CROSS_VALIDATION]:
        users_ratings = filter_users_ratings_with_sufficient_votes(
            users_ratings, min_n_posrated, min_n_negrated
        )
    elif evaluation == Evaluation.SESSION_BASED:
        users_ratings = filter_users_ratings_with_sufficient_votes_session_based(
            users_ratings,
            test_size,
            min_n_posrated_train,
            min_n_negrated_train,
            min_n_posrated_val,
            min_n_negrated_val,
        )
    users_ids_with_sufficient_votes = get_users_ids_with_sufficient_votes(users_ratings)

    if users_selection != "random":
        if type(users_selection) is str:
            users_selection = convert_users_selection_string(users_selection)
        users_ids_selected = select_users_explicit(
            users_selection, users_ids_with_sufficient_votes, take_complement
        )
    else:
        users_ids_selected = select_users_random(
            users_ids_with_sufficient_votes, max_users, users_random_state, take_complement
        )

    users_ratings = users_ratings[users_ratings["user_id"].isin(users_ids_selected)].reset_index(
        drop=True
    )
    assert np.all(users_ratings["user_id"].unique() == users_ids_selected)
    if evaluation == Evaluation.TRAIN_TEST_SPLIT:
        users_ratings = get_train_test_split(users_ratings, test_size, model_random_state, stratify)
    return users_ratings, users_ids_selected.tolist()


def convert_users_selection_string(users_selection: str) -> list:
    if users_selection == "finetuning_train":
        return load_finetuning_users_ids(selection="train")
    elif users_selection == "finetuning_val":
        return load_finetuning_users_ids(selection="val")
    elif users_selection == "finetuning_test":
        return load_finetuning_users_ids(selection="test")
    elif users_selection == "session_based":
        return load_session_based_users_ids()
    else:
        raise ValueError(f"Invalid users_selection: {users_selection}.")


def assert_sorted_and_unique(users_ids: list) -> None:
    assert users_ids == sorted(users_ids), "Users IDs must be sorted."
    assert len(users_ids) == len(set(users_ids)), "Users IDs must be unique."


def users_query(
    take_complement: bool,
    max_users: int,
    random_state: int,
) -> list:
    _, users_ids = get_users_ratings(
        users_selection="random",
        evaluation=Evaluation.SESSION_BASED,
        train_size=1.0,
        max_users=max_users,
        users_random_state=random_state,
        min_n_posrated_train=16,
        min_n_negrated_train=16,
        min_n_posrated_val=4,
        min_n_negrated_val=4,
        take_complement=take_complement,
    )
    return users_ids


def save_finetuning_users_ids(path: Path, random_state: int) -> None:
    test_users_ids = users_query(
        take_complement=False,
        max_users=500,
        random_state=random_state,
    )
    assert_sorted_and_unique(test_users_ids)
    complement_users_ids = users_query(
        take_complement=True,
        max_users=500,
        random_state=random_state,
    )
    assert_sorted_and_unique(complement_users_ids)
    assert set(test_users_ids).isdisjoint(set(complement_users_ids))
    assert len(complement_users_ids) >= 500
    random.seed(random_state)
    val_users_ids = random.sample(complement_users_ids, k=500)
    val_users_ids = sorted(val_users_ids)
    assert_sorted_and_unique(val_users_ids)
    _, full_users_ids = get_users_ratings(
        users_selection="random",
        evaluation=Evaluation.CROSS_VALIDATION,
        train_size=1.0,
        max_users=None,
        users_random_state=random_state,
        min_n_posrated=20,
        min_n_negrated=20,
        take_complement=False,
    )
    assert_sorted_and_unique(full_users_ids)
    assert set(test_users_ids) <= set(full_users_ids) and set(val_users_ids) <= set(full_users_ids)
    train_users_ids = [
        user_id
        for user_id in full_users_ids
        if user_id not in test_users_ids and user_id not in val_users_ids
    ]
    assert_sorted_and_unique(train_users_ids)
    assert set(train_users_ids).isdisjoint(set(test_users_ids))
    assert set(train_users_ids).isdisjoint(set(val_users_ids))
    sets_joined = set(test_users_ids) | set(val_users_ids) | set(train_users_ids)
    assert sets_joined == set(full_users_ids)
    users_dict = {"test": test_users_ids, "val": val_users_ids, "train": train_users_ids}
    with open(path, "wb") as f:
        pickle.dump(users_dict, f)


def save_session_based_users_ids(path: Path) -> None:
    session_based_users_ids = users_query(
        take_complement=False,
        max_users=None,
        random_state=None,
    )
    assert_sorted_and_unique(session_based_users_ids)
    finetuning_users_ids = load_finetuning_users_ids(selection="all")
    assert set(finetuning_users_ids["test"]) <= set(session_based_users_ids)
    assert set(finetuning_users_ids["val"]) <= set(session_based_users_ids)
    with open(path, "wb") as f:
        pickle.dump(session_based_users_ids, f)


if __name__ == "__main__":
    random_state = 42

    finetuning_users_ids_path = ProjectPaths.data_finetuning_users_ids_path()
    if not finetuning_users_ids_path.exists():
        save_finetuning_users_ids(
            path=ProjectPaths.data_finetuning_users_ids_path(),
            random_state=random_state,
        )
        print(f"Finetuning Users IDs saved to {finetuning_users_ids_path}.")
    else:
        print("Finetuning Users IDs already exist. Skipping saving.")

    session_based_users_ids_path = ProjectPaths.data_session_based_users_ids_path()
    if not session_based_users_ids_path.exists():
        save_session_based_users_ids(path=session_based_users_ids_path)
        print(f"Session-Based Users IDs saved to {session_based_users_ids_path}.")
    else:
        print("Session-Based Users IDs already exist. Skipping saving.")

    users_types = ["finetuning_val", "finetuning_test", "session_based"]
    for users_type in users_types:
        users_type_string = users_type.replace("_", " ").title()
        if users_type == "finetuning_val":
            path = ProjectPaths.data_session_based_ratings_val_users_path()
        elif users_type == "finetuning_test":
            path = ProjectPaths.data_session_based_ratings_test_users_path()
        elif users_type == "session_based":
            path = ProjectPaths.data_session_based_ratings_session_based_users_path()
        if not path.exists():
            session_based_users_ratings = get_users_ratings(
                users_selection=users_type,
                evaluation=Evaluation.SESSION_BASED,
                train_size=1.0,
                min_n_posrated_train=16,
                min_n_negrated_train=16,
                min_n_posrated_val=4,
                min_n_negrated_val=4,
            )[0]
            session_based_users_ratings.to_parquet(path, index=False, compression="gzip")
            print(f"{users_type_string} Users Session-Based Ratings saved to {path}.")
        else:
            print(f"{users_type_string} Users Session-Based Ratings already exist. Skipping saving.")

    users_ids = load_session_based_users_ids(select_non_cs_users_only=True)
    print(users_ids)
    from ....src.load_files import load_users_significant_categories
    significant_categories = load_users_significant_categories(
        relevant_users_ids=users_ids
    )
    print(significant_categories)