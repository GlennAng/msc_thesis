from data_handling import sql_execute
import numpy as np
import pandas as pd
import os
pd.set_option("display.max_rows", 1000)

TRAIN_VAL_SPLIT = 0.8
MIN_NEG_TRAIN, MIN_NEG_VAL = 10, 4
MIN_POS_TRAIN, MIN_POS_VAL = 10, 4


def split_user_data(group : pd.DataFrame) -> pd.DataFrame:
    n = len(group)
    split_point = int(np.ceil(n * 0.8)) 
    group = group.copy()
    group["split"] = ["train"] * split_point + ["val"] * (n - split_point)
    return group

def save_users_ratings() -> None:
    users_ratings_query = """SELECT user_id, paper_id, rating, time FROM users_ratings WHERE rating IN (-1, 1) AND time IS NOT NULL ORDER BY user_id, time"""
    users_ratings = sql_execute(users_ratings_query)
    users_ratings = pd.DataFrame(users_ratings, columns = ["user_id", "paper_id", "rating", "time"])
    users_ratings = users_ratings.groupby("user_id").apply(split_user_data, include_groups = True).reset_index(drop = True)
    users_ratings_train = users_ratings[users_ratings["split"] == "train"]
    users_ratings_train = users_ratings_train.drop(columns = "split", errors = "ignore")
    users_ratings_val = users_ratings[users_ratings["split"] == "val"]
    users_ratings_val = users_ratings_val.drop(columns = "split", errors = "ignore")
    users_ratings_train.to_csv("../data/users_ratings_train.csv", index = False)
    users_ratings_val.to_csv("../data/users_ratings_val.csv", index = False)

def load_users_ratings(folder : str = "../data/") -> tuple:
    users_ratings_train = pd.read_csv(os.path.join(folder, "users_ratings_train.csv"))
    users_ratings_val = pd.read_csv(os.path.join(folder, "users_ratings_val.csv"))
    print(users_ratings_train["user_id"].dtype, users_ratings_val["time"].dtype)

    assert not users_ratings_train.isnull().any().any()
    assert not users_ratings_val.isnull().any().any()
    assert users_ratings_train[["user_id", "time"]].equals(users_ratings_train[["user_id", "time"]].sort_values(["user_id", "time"]))
    assert users_ratings_val[["user_id", "time"]].equals(users_ratings_val[["user_id", "time"]].sort_values(["user_id", "time"]))


    


    print(f"Loaded {len(users_ratings_train)} Training Ratings and {len(users_ratings_val)} Validation Ratings successfully.")
    return users_ratings_train, users_ratings_val

def count_pos_neg_ratings(users_ratings : pd.DataFrame) -> pd.DataFrame:
    users_ratings = users_ratings.copy()
    users_ratings = users_ratings.drop(columns = ["paper_id", "time"], errors = "ignore")
    users_ratings["n_posrated"] = users_ratings.groupby("user_id")["rating"].transform(lambda x: (x == 1).sum())
    users_ratings["n_negrated"] = users_ratings.groupby("user_id")["rating"].transform(lambda x: (x == -1).sum())
    users_ratings["n_rated"] = users_ratings.groupby("user_id")["rating"].transform("count")
    users_ratings = users_ratings.drop(columns = "rating", errors = "ignore").drop_duplicates()
    users_ratings = users_ratings.reset_index(drop = True)
    return users_ratings

def filter_users_ids(users_ratings_train : pd.DataFrame, users_ratings_val : pd.DataFrame, min_n_posrated_train : int = MIN_POS_TRAIN, 
                    min_n_negrated_train : int = MIN_NEG_TRAIN, min_n_posrated_val : int = MIN_POS_VAL, min_n_negrated_val : int = MIN_NEG_VAL) -> pd.Series:
    users_ratings_train_counted = count_pos_neg_ratings(users_ratings_train)
    users_ratings_val_counted = count_pos_neg_ratings(users_ratings_val)
    users_ids_train = users_ratings_train_counted[
        (users_ratings_train_counted["n_posrated"] >= min_n_posrated_train) & 
        (users_ratings_train_counted["n_negrated"] >= min_n_negrated_train)
    ]["user_id"].unique()
    users_ids_val = users_ratings_val_counted[
        (users_ratings_val_counted["n_posrated"] >= min_n_posrated_val) & 
        (users_ratings_val_counted["n_negrated"] >= min_n_negrated_val)
    ]["user_id"].unique()
    users_ids = np.intersect1d(users_ids_train, users_ids_val)
    return pd.DataFrame({"user_id": users_ids})

users_ratings_train, users_ratings_val = load_users_ratings()


def get_users_ids(users_selection : str, max_users : int = None, min_n_posrated : int = 20, min_n_negrated : int = 20, take_complement : bool = False, 
                  random_state : int = None, survey : bool = False) -> pd.DataFrame:
    users_ids_with_sufficient_votes = get_users_ids_with_sufficient_votes(min_n_posrated = min_n_posrated, min_n_negrated = min_n_negrated, sort_ids = False)
    if users_selection not in ["random", "largest_n", "smallest_n"]:
        users_ids_with_sufficient_votes = users_ids_with_sufficient_votes[users_ids_with_sufficient_votes["user_id"].isin(list(users_selection))]
    n_users_with_sufficient_votes = len(users_ids_with_sufficient_votes)
    print(n_users_with_sufficient_votes, "users with sufficient votes.")
    max_users = n_users_with_sufficient_votes if max_users is None else min(max_users, n_users_with_sufficient_votes)
    if max_users >= n_users_with_sufficient_votes:
        assert not take_complement, "Users Selection: take_complement must be False when all users are selected."
    else:
        if take_complement:
            users_ids_with_sufficient_votes_complement = users_ids_with_sufficient_votes.copy()
        if users_selection == "random":
            users_ids_with_sufficient_votes = users_ids_with_sufficient_votes.sort_values(by = "user_id")
            users_ids_with_sufficient_votes = users_ids_with_sufficient_votes.sample(n = max_users, random_state = random_state)
        elif users_selection in ["largest_n", "smallest_n"]:
            smallest_n_bool = (users_selection == "smallest_n")
            users_ids_with_sufficient_votes = users_ids_with_sufficient_votes.sort_values(["n_rated", "n_posrated", "user_id"], 
                                                                              ascending = [smallest_n_bool, smallest_n_bool, False]).head(max_users)
        if take_complement:
            users_ids_with_sufficient_votes = users_ids_with_sufficient_votes_complement[~users_ids_with_sufficient_votes_complement["user_id"].isin(users_ids_with_sufficient_votes["user_id"])]
    print(len(users_ids_with_sufficient_votes), "users selected.")
    return users_ids_with_sufficient_votes.sort_values(by = "user_id")