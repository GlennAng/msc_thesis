from data_handling import sql_execute
import numpy as np
import pandas as pd
import os
pd.set_option("display.max_rows", 1000)
PATH = "/home/scholar/glenn_rp/msc_thesis/data/"

def create_session_column(users_ratings : pd.DataFrame, session_threshold_min : int = 420) -> pd.Series:
    session_threshold = pd.Timedelta(minutes = session_threshold_min)
    users_ratings = users_ratings.sort_values(by = ["user_id", "time"]).copy()
    users_ratings["time_diff"] = users_ratings.groupby("user_id")["time"].diff()
    users_ratings["session_break"] = (users_ratings["time_diff"] > session_threshold) | users_ratings["time_diff"].isna()
    return users_ratings.groupby("user_id")["session_break"].cumsum()

def save_users_ratings(path : str = os.path.join(PATH, "users_ratings.parquet")) -> None:
    users_ratings_query = """SELECT user_id, paper_id, rating, time FROM users_ratings WHERE rating IN (-1, 1) AND time IS NOT NULL ORDER BY user_id, time"""
    users_ratings = sql_execute(users_ratings_query)
    users_ratings = pd.DataFrame(users_ratings, columns = ["user_id", "paper_id", "rating", "time"])
    users_ratings["rating"] = users_ratings["rating"].replace(-1, 0)
    users_ratings["time"] = pd.to_datetime(users_ratings["time"]).dt.floor('s')
    assert not users_ratings.isnull().any().any(), "There are null values in the users_ratings DataFrame."
    users_ratings["session_id"] = create_session_column(users_ratings)
    users_ratings.to_parquet(path, index = False, compression = "gzip")

def load_users_ratings(path : str, relevant_users_ids : list = None) -> pd.DataFrame:
    if type(path) == str:
        users_ratings = pd.read_parquet(path, engine = "pyarrow")
    if relevant_users_ids is not None:
        users_ratings = users_ratings[users_ratings["user_id"].isin(relevant_users_ids)]
    assert not users_ratings.isnull().any().any()
    assert users_ratings["time"].dtype == "datetime64[ns]"
    assert users_ratings.groupby("user_id")["time"].is_monotonic_increasing.all()
    return users_ratings.copy()

def create_split_column_random(users_ratings : pd.DataFrame, train_split : float, random_state : int, stratified : bool) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split
    users_ratings["split"] = np.nan
    unique_users_ids = users_ratings["user_id"].unique()
    for user_id in unique_users_ids:
        user_mask = users_ratings["user_id"] == user_id
        if user_mask.sum() <= 1:
            users_ratings.loc[user_mask, "split"] = "train"
        else:
            papers_ids, papers_ratings = users_ratings[user_mask]["paper_id"].to_numpy(), users_ratings[user_mask]["rating"].to_numpy()
            train_papers_ids, test_papers_ids, _, _ = train_test_split(papers_ids, papers_ratings, train_size = train_split, random_state = random_state, 
                                                                    stratify = papers_ratings if stratified else None)
            train_mask = users_ratings["paper_id"].isin(train_papers_ids)
            users_ratings.loc[user_mask & train_mask, "split"] = "train" 
            users_ratings.loc[user_mask & ~train_mask, "split"] = "test"
    return users_ratings
        
# split users_ratings based on sessions because otherwise we are asking whether the system can predict papers from the same session whereas Scholar Inbox only predicts once per day
def load_users_ratings_split(split_by : str, train_split : float = None, random_state : int = None, stratified : bool = None, 
                             relevant_users_ids : list = None, path = os.path.join(PATH, "users_ratings.parquet")) -> pd.DataFrame:
    assert split_by in ["sessions", "time", "random", None]
    users_ratings = load_users_ratings(path, relevant_users_ids)
    if split_by == None:
        return users_ratings
    assert train_split is not None
    if split_by == "random":
        assert random_state is not None and stratified is not None
        users_ratings = create_split_column_random(users_ratings, train_split, random_state, stratified)
    return users_ratings

users_ratings = load_users_ratings_split(split_by = "random", random_state = 42, train_split = 0.8, stratified = True)
print((users_ratings["split"] == "train").sum(), "train ratings")
print((users_ratings["split"] == "test").sum(), "test ratings")
print(len(users_ratings))
