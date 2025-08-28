import pickle
import random
from enum import Enum, auto

import pandas as pd

from ....src.load_files import load_finetuning_users_ids, load_users_ratings
from ....src.project_paths import ProjectPaths

pd.set_option("display.max_rows", None)
N_NEGRATED_RANKING = 4


class UsersRatingsSelection(Enum):
    FINETUNING_TRAIN = auto()
    SESSION_BASED_NO_FILTERING = auto()
    SESSION_BASED_NO_FILTERING_OLD = auto()
    SESSION_BASED_FILTERING = auto()
    SESSION_BASED_FILTERING_OLD = auto()


def get_users_ratings_selection_from_arg(urs_arg: str) -> UsersRatingsSelection:
    valid_urs_args = [urs.name.lower() for urs in UsersRatingsSelection]
    if urs_arg.lower() not in valid_urs_args:
        raise ValueError(
            f"Invalid argument {urs_arg} 'users_ratings_selection'. Possible values: {valid_urs_args}."
        )
    return UsersRatingsSelection[urs_arg.upper()]


def assert_sorted_and_unique(users_ids: list) -> None:
    assert users_ids == sorted(users_ids), "Users IDs must be sorted."
    assert len(users_ids) == len(set(users_ids)), "Users IDs must be unique."


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
    min_n_posrated_train: int,
    min_n_negrated_train: int,
    min_n_posrated_val: int,
    min_n_negrated_val: int,
    test_size: float,
) -> pd.DataFrame:
    n_sessions = user_ratings["session_id"].nunique()
    best_session_id, best_session_test_size_diff = None, float("inf")
    for split_session_id in range(n_sessions, -1, -1):
        valid_split, test_size_split = check_single_split(
            user_ratings=user_ratings,
            split_session_id=split_session_id,
            min_n_posrated_train=min_n_posrated_train,
            min_n_negrated_train=min_n_negrated_train,
            min_n_posrated_val=min_n_posrated_val,
            min_n_negrated_val=min_n_negrated_val,
        )
        if valid_split:
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
    min_n_posrated_train: int,
    min_n_negrated_train: int,
    min_n_posrated_val: int,
    min_n_negrated_val: int,
    min_n_sessions: int,
    test_size: float,
) -> pd.DataFrame:
    assert users_ratings.groupby("user_id")["time"].is_monotonic_increasing.all()
    assert users_ratings.groupby("user_id")["session_id"].is_monotonic_increasing.all()
    users_ratings = users_ratings.copy()
    users_ratings_with_sufficient_votes = []
    users_ids = users_ratings["user_id"].unique()
    for user_id in users_ids:
        user_ratings = users_ratings[users_ratings["user_id"] == user_id].reset_index(drop=True)
        n_sessions = user_ratings["session_id"].nunique()
        if n_sessions < min_n_sessions:
            continue
        user_ratings_split = split_single_user(
            user_ratings=user_ratings,
            min_n_posrated_train=min_n_posrated_train,
            min_n_negrated_train=min_n_negrated_train,
            min_n_posrated_val=min_n_posrated_val,
            min_n_negrated_val=min_n_negrated_val,
            test_size=test_size,
        )
        if user_ratings_split is not None:
            users_ratings_with_sufficient_votes.append(user_ratings_split)
    if not users_ratings_with_sufficient_votes:
        return pd.DataFrame(columns=users_ratings.columns)
    users_ratings = pd.concat(users_ratings_with_sufficient_votes, ignore_index=True)
    return users_ratings


def filter_users_ratings_with_sufficient_votes(
    users_ratings: pd.DataFrame, min_n_posrated: int, min_n_negrated: int, min_n_sessions: int
) -> pd.DataFrame:
    users_ratings = users_ratings.copy()
    min_n_posrated = 0 if (min_n_posrated < 0 or min_n_posrated is None) else min_n_posrated
    min_n_negrated = 0 if (min_n_negrated < 0 or min_n_negrated is None) else min_n_negrated
    users_ratings_counts = (
        users_ratings.groupby("user_id")
        .agg(
            n_posrated=("rating", lambda x: (x == 1).sum()),
            n_negrated=("rating", lambda x: (x == 0).sum()),
            n_distinct_sessions=("session_id", "nunique"),
        )
        .reset_index()
    )
    users_ratings_counts_sufficient_votes = users_ratings_counts[
        (users_ratings_counts["n_posrated"] >= min_n_posrated)
        & (users_ratings_counts["n_negrated"] >= min_n_negrated)
        & (users_ratings_counts["n_distinct_sessions"] >= min_n_sessions)
    ]
    users_ratings_filtered = users_ratings[
        users_ratings["user_id"].isin(users_ratings_counts_sufficient_votes["user_id"])
    ]
    return users_ratings_filtered.reset_index(drop=True)


def save_session_based_no_filtering_ratings(users_ratings: pd.DataFrame, params: dict) -> None:
    path = ProjectPaths.data_session_based_no_filtering_ratings_path()
    if path.exists():
        print(f"{path} already exists. Skipping saving.")
        return
    ratings = filter_users_ratings_with_sufficient_votes_session_based(
        users_ratings=users_ratings,
        min_n_posrated_train=params["min_n_posrated_train"],
        min_n_negrated_train=params["min_n_negrated_train"],
        min_n_posrated_val=params["min_n_posrated_val"],
        min_n_negrated_val=params["min_n_negrated_val"],
        min_n_sessions=params["min_n_sessions"],
        test_size=params["test_size"],
    )
    ratings.to_parquet(path, index=False, compression="gzip")
    print(f"Saved ratings to {path}. Total Users: {ratings['user_id'].nunique()}.")


def split_users_ids_into_val_test(users_ids: list, n_test_users: int, seed: int) -> tuple:
    random.seed(seed)
    random.shuffle(users_ids)
    val_users_ids = sorted(users_ids[:-n_test_users])
    test_users_ids = sorted(users_ids[-n_test_users:])
    assert_sorted_and_unique(val_users_ids)
    assert_sorted_and_unique(test_users_ids)
    assert set(val_users_ids).isdisjoint(set(test_users_ids))
    assert set(val_users_ids + test_users_ids) == set(users_ids)
    return val_users_ids, test_users_ids


def save_finetuning_users_ids(users_ratings: pd.DataFrame, params: dict) -> None:
    path = ProjectPaths.data_finetuning_users_ids_path()
    if path.exists():
        print(f"{path} already exists. Skipping saving.")
        return
    users_ids = load_users_ratings_from_selection(
        UsersRatingsSelection.SESSION_BASED_NO_FILTERING, ids_only=True
    )
    val_users_ids, test_users_ids = split_users_ids_into_val_test(
        users_ids=users_ids, n_test_users=params["n_test_users"], seed=params["val_test_split_seed"]
    )
    train_ratings = filter_users_ratings_with_sufficient_votes(
        users_ratings=users_ratings,
        min_n_posrated=params["train_users_min_n_posrated_total"],
        min_n_negrated=params["train_users_min_n_negrated_total"],
        min_n_sessions=params["train_users_min_n_sessions"],
    )
    potential_users_ids = train_ratings["user_id"].unique().tolist()
    assert set(users_ids).issubset(set(potential_users_ids))
    train_users_ids = [uid for uid in potential_users_ids if uid not in users_ids]
    assert_sorted_and_unique(train_users_ids)
    assert set(train_users_ids).isdisjoint(set(val_users_ids))
    assert set(train_users_ids).isdisjoint(set(test_users_ids))
    assert set(train_users_ids + val_users_ids + test_users_ids) == set(potential_users_ids)
    finetuning_users_ids = {"train": train_users_ids, "val": val_users_ids, "test": test_users_ids}
    n_train, n_val, n_test = len(train_users_ids), len(val_users_ids), len(test_users_ids)
    print(f"Finetuning: Train Users: {n_train}, Val Users: {n_val}, Test Users: {n_test}.")
    with open(path, "wb") as f:
        pickle.dump(finetuning_users_ids, f)


def filter_users_ratings_for_negrated_ranking(
    users_ratings: pd.DataFrame, n_negrated_ranking: int = N_NEGRATED_RANKING
) -> tuple:
    users_ratings_head = users_ratings[
        users_ratings["n_negrated_still_to_come"] >= n_negrated_ranking
    ]
    users_ratings_tail = users_ratings[
        users_ratings["n_negrated_still_to_come"] < n_negrated_ranking
    ]
    assert len(users_ratings_head) + len(users_ratings_tail) == len(users_ratings)
    return users_ratings_head, users_ratings_tail


def check_user_ratings_for_conditions(user_ratings: pd.DataFrame, params: dict) -> bool:
    train_ratings = user_ratings[user_ratings["split"] == "train"]
    val_ratings = user_ratings[user_ratings["split"] == "val"]
    pos_val_ratings = val_ratings[val_ratings["rating"] == 1]
    n_pos_val_sessions = pos_val_ratings["session_id"].nunique()
    n_posrated_train = train_ratings["rating"].sum()
    n_negrated_train = len(train_ratings) - n_posrated_train
    n_posrated_val = val_ratings["rating"].sum()
    n_negrated_val = len(val_ratings) - n_posrated_val
    assert n_posrated_train >= params["n_posrated_train_for_first_split"]
    conditions_met = (
        n_posrated_train <= params["max_n_posrated_train_in_first_split"]
        and n_negrated_train >= params["min_n_negrated_train_in_first_split"]
        and n_posrated_val >= params["min_n_posrated_val_in_first_split"]
        and n_negrated_val >= params["min_n_negrated_val_in_first_split"]
        and n_pos_val_sessions >= params["min_n_pos_val_sessions_in_first_split"]
    )
    return conditions_met


def append_removed_for_negrated_ranking(
    users_ratings: pd.DataFrame,
    users_ratings_removed_for_negrated_ranking: pd.DataFrame,
) -> pd.DataFrame:
    assert "split" in users_ratings.columns
    assert "split" not in users_ratings_removed_for_negrated_ranking.columns
    if len(users_ratings_removed_for_negrated_ranking) == 0:
        return users_ratings
    removed_df = users_ratings_removed_for_negrated_ranking.copy()
    removed_df["split"] = "removed"
    users_ratings = pd.concat([users_ratings, removed_df])
    users_ratings = users_ratings.sort_values(["user_id", "time"])
    return users_ratings


def save_session_based_filtering_ratings(
    users_ratings_head: pd.DataFrame, users_ratings_tail: pd.DataFrame, params: dict
) -> None:
    path = ProjectPaths.data_session_based_filtering_ratings_path()
    if path.exists():
        print(f"{path} already exists. Skipping saving.")
        return
    n_users_before_filtering = users_ratings_head["user_id"].nunique()
    users_ratings = filter_users_ratings_with_sufficient_votes_session_based(
        users_ratings=users_ratings_head,
        min_n_posrated_train=params["n_posrated_train_for_first_split"],
        min_n_negrated_train=0,
        min_n_posrated_val=0,
        min_n_negrated_val=0,
        min_n_sessions=0,
        test_size=1.0,
    )
    n_users_after_first_filtering = users_ratings["user_id"].nunique()
    print(f"First Filtering: {n_users_before_filtering} -> {n_users_after_first_filtering}.")
    users_ids = users_ratings["user_id"].unique()
    qualifying_users_ids = []
    for user_id in users_ids:
        user_ratings = users_ratings[users_ratings["user_id"] == user_id]
        if check_user_ratings_for_conditions(user_ratings, params):
            qualifying_users_ids.append(user_id)
    print(f"Second Filtering: {n_users_after_first_filtering} -> {len(qualifying_users_ids)}.")
    users_ratings = users_ratings[users_ratings["user_id"].isin(qualifying_users_ids)]
    users_ratings_tail = users_ratings_tail[
        users_ratings_tail["user_id"].isin(qualifying_users_ids)
    ]
    users_ratings = append_removed_for_negrated_ranking(
        users_ratings=users_ratings, users_ratings_removed_for_negrated_ranking=users_ratings_tail
    )
    users_ratings.to_parquet(path, engine="pyarrow")
    print(f"Saved ratings to {path}. Total Users: {users_ratings['user_id'].nunique()}.")


def load_users_ratings_from_selection(
    users_ratings_selection: UsersRatingsSelection,
    relevant_users_ids: list = None,
    ids_only: bool = False,
) -> pd.DataFrame:
    if users_ratings_selection == UsersRatingsSelection.SESSION_BASED_NO_FILTERING:
        path = ProjectPaths.data_session_based_no_filtering_ratings_path()
        users_ratings = pd.read_parquet(path, engine="pyarrow")
    elif users_ratings_selection == UsersRatingsSelection.SESSION_BASED_FILTERING:
        path = ProjectPaths.data_session_based_filtering_ratings_path()
        users_ratings = pd.read_parquet(path, engine="pyarrow")
    elif users_ratings_selection == UsersRatingsSelection.FINETUNING_TRAIN:
        users_ids = load_finetuning_users_ids(selection="train")
        users_ratings = load_users_ratings(relevant_users_ids=users_ids)
        users_ratings["split"] = "train"
    elif users_ratings_selection == UsersRatingsSelection.SESSION_BASED_NO_FILTERING_OLD:
        path = ProjectPaths.data_session_based_no_filtering_ratings_old_path()
        users_ratings = pd.read_parquet(path, engine="pyarrow")
    elif users_ratings_selection == UsersRatingsSelection.SESSION_BASED_FILTERING_OLD:
        path = ProjectPaths.data_session_based_filtering_ratings_old_path()
        users_ratings = pd.read_parquet(path, engine="pyarrow")

    assert users_ratings["user_id"].is_monotonic_increasing
    assert users_ratings.groupby("user_id")["time"].is_monotonic_increasing.all()
    assert users_ratings.groupby("user_id")["session_id"].is_monotonic_increasing.all()
    if relevant_users_ids is not None:
        old = users_ratings_selection in [
            UsersRatingsSelection.SESSION_BASED_NO_FILTERING_OLD,
            UsersRatingsSelection.SESSION_BASED_FILTERING_OLD,
        ]
        if relevant_users_ids == "finetuning_val":
            relevant_users_ids = load_finetuning_users_ids(selection="val", old=old)
        elif relevant_users_ids == "finetuning_test":
            relevant_users_ids = load_finetuning_users_ids(selection="test", old=old)
        assert_sorted_and_unique(relevant_users_ids)
        full_users_ids = users_ratings["user_id"].unique().tolist()
        assert set(relevant_users_ids).issubset(set(full_users_ids))
        users_ratings = users_ratings[users_ratings["user_id"].isin(relevant_users_ids)]
    if ids_only:
        users_ids = users_ratings["user_id"].unique().tolist()
        assert_sorted_and_unique(users_ids)
        return users_ids
    return users_ratings


if __name__ == "__main__":
    users_ratings = load_users_ratings()

    SESSION_BASED_NO_FILTERING_PARAMS = {
        "filter_for_negrated_ranking": False,
        "min_n_posrated_train": 20,
        "min_n_negrated_train": 20,
        "min_n_posrated_val": 5,
        "min_n_negrated_val": 5,
        "min_n_sessions": 5,
        "test_size": 0.2,
    }
    save_session_based_no_filtering_ratings(
        users_ratings=users_ratings, params=SESSION_BASED_NO_FILTERING_PARAMS
    )

    FINETUNING_PARAMS = {
        "n_test_users": 500,
        "val_test_split_seed": 42,
        "train_users_min_n_posrated_total": 25,
        "train_users_min_n_negrated_total": 25,
        "train_users_min_n_sessions": 5,
    }
    save_finetuning_users_ids(users_ratings=users_ratings, params=FINETUNING_PARAMS)

    test_users = load_finetuning_users_ids("test")
    val_users = load_finetuning_users_ids("val")
    train_users = load_finetuning_users_ids("train")

    users_ratings_head, users_ratings_tail = filter_users_ratings_for_negrated_ranking(
        users_ratings=users_ratings, n_negrated_ranking=N_NEGRATED_RANKING
    )
    SESSION_BASED_FILTERING_PARAMS = {
        "n_posrated_train_for_first_split": 20,
        "max_n_posrated_train_in_first_split": 40,
        "min_n_negrated_train_in_first_split": 8,
        "min_n_posrated_val_in_first_split": 10,
        "min_n_negrated_val_in_first_split": 4,
        "min_n_pos_val_sessions_in_first_split": 3,
    }
    save_session_based_filtering_ratings(
        users_ratings_head=users_ratings_head,
        users_ratings_tail=users_ratings_tail,
        params=SESSION_BASED_FILTERING_PARAMS,
    )        
