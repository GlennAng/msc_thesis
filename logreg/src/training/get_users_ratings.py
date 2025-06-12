import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[3]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_src_paths_to_sys()

import numpy as np, pandas as pd
pd.set_option("display.max_rows", None)

from algorithm import Evaluation
from load_files import load_users_ratings
from training_data import LABEL_DTYPE

def get_temporal_750_users_ratings() -> tuple:
    return get_users_ratings(users_selection = "random", evaluation = Evaluation.SESSION_BASED, test_size = 0.0, max_users = 750,
                        users_random_state = 42, model_random_state = 42, stratify = True, min_n_posrated = 20, min_n_negrated = 20,
                        take_complement = False, users_mapped = False, min_n_posrated_train = 16, min_n_negrated_train = 16,
                        min_n_posrated_val = 5, min_n_negrated_val = 5)

def get_train_test_split(users_ratings: pd.DataFrame, test_size: float, random_state: int, stratify: bool = True) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    def split_user_data(user_ratings: pd.DataFrame) -> pd.DataFrame:
        posrated_ids = user_ratings[user_ratings["rating"] == 1]["paper_id"].values
        negrated_ids = user_ratings[user_ratings["rating"] == 0]["paper_id"].values

        rated_ids = np.concatenate((posrated_ids, negrated_ids))
        posrated_n, negrated_n = len(posrated_ids), len(negrated_ids)
        rated_labels = np.concatenate((np.ones(posrated_n, dtype = LABEL_DTYPE), np.zeros(negrated_n, dtype = LABEL_DTYPE)))
        train_rated_ids, val_rated_ids, _, _ = train_test_split(rated_ids, rated_labels, test_size = test_size, random_state = random_state,
                                                                stratify = rated_labels if stratify else None)
        user_ratings_copy = user_ratings.copy()
        user_ratings_copy["split"] = user_ratings_copy["paper_id"].apply(lambda x: "train" if x in train_rated_ids else "val")
        return user_ratings_copy

    result_df = users_ratings.groupby("user_id", group_keys = False).apply(split_user_data, include_groups = False)
    return result_df.reset_index(drop = True)  

def load_users_ratings_from_backup(users_mapped: bool) -> pd.DataFrame:
    users_ratings_path = ProjectPaths.data_db_backup_date_path()
    if users_mapped:
        users_ratings_path /= "users_ratings_mapped.parquet"
    else:
        users_ratings_path /= "users_ratings.parquet"
    return load_users_ratings(users_ratings_path)

def filter_users_ratings_with_sufficient_votes(users_ratings: pd.DataFrame, min_n_posrated: int, min_n_negrated: int) -> pd.DataFrame:
    users_ratings = users_ratings.copy()
    min_n_posrated = 0 if (min_n_posrated < 0 or min_n_posrated is None) else min_n_posrated
    min_n_negrated = 0 if (min_n_negrated < 0 or min_n_negrated is None) else min_n_negrated
    users_ratings_counts = users_ratings.groupby("user_id").agg(
        n_posrated = ("rating", lambda x: (x == 1).sum()), n_negrated = ("rating", lambda x: (x == 0).sum())).reset_index()
    users_ratings_counts_sufficient_votes = users_ratings_counts[
        (users_ratings_counts["n_posrated"] >= min_n_posrated) & (users_ratings_counts["n_negrated"] >= min_n_negrated)]
    users_ratings_filtered = users_ratings[users_ratings["user_id"].isin(users_ratings_counts_sufficient_votes["user_id"])]
    return users_ratings_filtered.reset_index(drop = True)

def check_single_split(user_ratings: pd.DataFrame, split_session_id: int, min_n_posrated_train: int, min_n_negrated_train: int,
                       min_n_posrated_val: int, min_n_negrated_val: int) -> tuple:
    train_mask, val_mask = user_ratings["session_id"] < split_session_id, user_ratings["session_id"] >= split_session_id
    train_ratings, val_ratings = user_ratings[train_mask], user_ratings[val_mask]
    n_posrated_train, n_posrated_val = train_ratings["rating"].sum(), val_ratings["rating"].sum()
    n_negrated_train, n_negrated_val = len(train_ratings) - n_posrated_train, len(val_ratings) - n_posrated_val
    valid_split_train = n_posrated_train >= min_n_posrated_train and n_negrated_train >= min_n_negrated_train
    valid_split_val = n_posrated_val >= min_n_posrated_val and n_negrated_val >= min_n_negrated_val
    valid_split = valid_split_train and valid_split_val
    test_size_split = len(val_ratings) / len(user_ratings)
    return valid_split, test_size_split

def split_single_user(user_ratings: pd.DataFrame, test_size: float, min_n_posrated_train: int, min_n_negrated_train: int,
                      min_n_posrated_val: int, min_n_negrated_val: int) -> pd.DataFrame:
    n_sessions = user_ratings["session_id"].nunique()
    best_session_id, best_session_test_size_diff = None, float("inf")
    for split_session_id in range(n_sessions, -1, -1):
        valid_split, test_size_split = check_single_split(user_ratings, split_session_id, min_n_posrated_train, min_n_negrated_train,
                                                          min_n_posrated_val, min_n_negrated_val)
        if valid_split:
            test_size_diff = abs(test_size_split - test_size)
            if test_size_diff < best_session_test_size_diff:
                best_session_id, best_session_test_size_diff = split_session_id, test_size_diff
            else:
                if test_size_split > test_size:
                    break
    if best_session_id is None:
        return None
    user_ratings["split"] = user_ratings["session_id"].apply(lambda x: "train" if x < best_session_id else "val")
    return user_ratings

def filter_users_ratings_with_sufficient_votes_session_based(users_ratings: pd.DataFrame, test_size: float, min_n_posrated_train: int, min_n_negrated_train: int,
                                                             min_n_posrated_val: int, min_n_negrated_val: int) -> pd.DataFrame:
    assert users_ratings.groupby("user_id")["time"].is_monotonic_increasing.all()
    assert users_ratings.groupby("user_id")["session_id"].is_monotonic_increasing.all()
    users_ratings = users_ratings.copy()
    users_ratings_with_sufficient_votes = []
    users_ids = users_ratings["user_id"].unique()
    for user_id in users_ids:
        user_ratings = users_ratings[users_ratings["user_id"] == user_id].reset_index(drop = True)
        user_ratings_split = split_single_user(user_ratings, test_size, min_n_posrated_train, min_n_negrated_train, min_n_posrated_val, min_n_negrated_val)
        if user_ratings_split is not None:
            users_ratings_with_sufficient_votes.append(user_ratings_split)
    if not users_ratings_with_sufficient_votes:
        return pd.DataFrame(columns = users_ratings.columns)
    users_ratings = pd.concat(users_ratings_with_sufficient_votes, ignore_index = True)
    return users_ratings

def get_users_ids_with_sufficient_votes(users_ratings: pd.DataFrame) -> np.ndarray:
    users_ids_with_sufficient_votes = users_ratings["user_id"].unique()
    assert np.all(users_ids_with_sufficient_votes[:-1] <= users_ids_with_sufficient_votes[1:])
    print(len(users_ids_with_sufficient_votes), "Users with sufficient Votes.")
    return users_ids_with_sufficient_votes

def process_selected_users(users_ids_selected: np.ndarray, users_ids_with_sufficient_votes: np.ndarray, take_complement: bool) -> np.ndarray:
    assert np.all(users_ids_selected[:-1] <= users_ids_selected[1:])
    assert len(users_ids_selected) == len(set(users_ids_selected))
    assert set(users_ids_selected) <= set(users_ids_with_sufficient_votes)
    n_users_not_selected = len(users_ids_with_sufficient_votes) - len(users_ids_selected)
    if take_complement:
        assert n_users_not_selected > 0
        users_ids_selected = users_ids_with_sufficient_votes[~np.isin(users_ids_with_sufficient_votes, users_ids_selected)]
        assert np.all(users_ids_selected[:-1] <= users_ids_selected[1:])
    print(len(users_ids_selected), "Users selected.")
    return users_ids_selected

def select_users_explicit(users_selection: str, users_ids_with_sufficient_votes: np.ndarray, take_complement: bool) -> np.ndarray:
    users_ids_selected = np.array(users_selection, dtype = np.int64)
    return process_selected_users(users_ids_selected, users_ids_with_sufficient_votes, take_complement)

def select_users_random(users_ids_with_sufficient_votes: np.ndarray, max_users: int, random_state: int, take_complement: bool) -> np.ndarray:
    if max_users is not None and max_users < len(users_ids_with_sufficient_votes):
        series = pd.Series(users_ids_with_sufficient_votes)
        sampled_series = series.sample(n = max_users, random_state = random_state)
        users_ids_selected = sampled_series.values
        users_ids_selected.sort()
    else:
        users_ids_selected = users_ids_with_sufficient_votes
    return process_selected_users(users_ids_selected, users_ids_with_sufficient_votes, take_complement)

def get_users_ratings(users_selection: str, evaluation: Evaluation, test_size: float, max_users: int = None, users_random_state: int = 42, model_random_state: int = 42,
                      stratify: bool = True, min_n_posrated: int = 0, min_n_negrated: int = 0, take_complement: bool = False, users_mapped: bool = False,
                      min_n_posrated_train: int = 0, min_n_negrated_train: int = 0, min_n_posrated_val: int = 0, min_n_negrated_val: int = 0) -> tuple:
    users_ratings = load_users_ratings_from_backup(users_mapped)
    if evaluation in [Evaluation.TRAIN_TEST_SPLIT, Evaluation.CROSS_VALIDATION]:
        users_ratings = filter_users_ratings_with_sufficient_votes(users_ratings, min_n_posrated, min_n_negrated)
    elif evaluation == Evaluation.SESSION_BASED:
        users_ratings = filter_users_ratings_with_sufficient_votes_session_based(users_ratings, test_size, min_n_posrated_train, min_n_negrated_train,
                                                                                 min_n_posrated_val, min_n_negrated_val)
    users_ids_with_sufficient_votes = get_users_ids_with_sufficient_votes(users_ratings)
    
    if users_selection != "random":
        assert max_users is None
        users_ids_selected = select_users_explicit(users_selection, users_ids_with_sufficient_votes, take_complement)
    else:
        users_ids_selected = select_users_random(users_ids_with_sufficient_votes, max_users, users_random_state, take_complement)

    users_ratings = users_ratings[users_ratings["user_id"].isin(users_ids_selected)].reset_index(drop = True)
    assert np.all(users_ratings["user_id"].unique() == users_ids_selected)
    if evaluation == Evaluation.TRAIN_TEST_SPLIT:
        users_ratings = get_train_test_split(users_ratings, test_size, model_random_state, stratify)
    return users_ratings, users_ids_selected.tolist()


if __name__ == "__main__":
    _, users_ids = get_users_ratings(users_selection = "random", evaluation = Evaluation.SESSION_BASED, test_size = 0.0, max_users = 750,
                        users_random_state = 42, model_random_state = 42, stratify = True, min_n_posrated = 20, min_n_negrated = 20,
                        take_complement = False, users_mapped = False, min_n_posrated_train = 16, min_n_negrated_train = 16,
                        min_n_posrated_val = 5, min_n_negrated_val = 5)