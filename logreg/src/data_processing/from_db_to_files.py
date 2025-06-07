from data_handling import sql_execute
from paths import PATHS
from pathlib import PosixPath
import numpy as np, pandas as pd, pickle, os, random
pd.set_option("display.max_rows", 1000)

def save_users_mapping(path : PosixPath = PATHS["users_mapping_path"], random_state : int = 42) -> None:
    users_ids_query = "SELECT DISTINCT user_id FROM users_ratings WHERE rating IN (-1, 1) AND time IS NOT NULL ORDER BY user_id"
    users_ids = [row[0] for row in sql_execute(users_ids_query)]
    random.seed(random_state)
    indices = list(range(len(users_ids)))
    random.shuffle(indices)
    users_mapping = {user_id: idx for user_id, idx in zip(users_ids, indices)}
    with open(path, "wb") as f:
        pickle.dump(users_mapping, f)

def load_users_mapping(path : PosixPath = PATHS["users_mapping_path"]) -> dict:
    with open(path, "rb") as f:
        users_mapping = pickle.load(f)
    values = list(users_mapping.values())
    assert sorted(values) == list(range(len(users_mapping)))
    return users_mapping

def create_session_column(users_ratings : pd.DataFrame, session_threshold_min : int = 420) -> pd.Series:
    session_threshold = pd.Timedelta(minutes = session_threshold_min)
    users_ratings = users_ratings.sort_values(by = ["user_id", "time"]).copy()
    users_ratings["time_diff"] = users_ratings.groupby("user_id")["time"].diff()
    users_ratings["session_break"] = (users_ratings["time_diff"] > session_threshold) | users_ratings["time_diff"].isna()
    return users_ratings.groupby("user_id")["session_break"].cumsum() - 1

def save_users_ratings(path : PosixPath = PATHS["users_ratings_path"], users_mapping : dict = PATHS["users_mapping_path"]) -> None:
    if type(users_mapping) == PosixPath:
        users_mapping = load_users_mapping(users_mapping)
    users_ratings_query = """SELECT user_id, paper_id, rating, time FROM users_ratings WHERE rating IN (-1, 1) AND time IS NOT NULL"""
    users_ratings = sql_execute(users_ratings_query)
    users_ratings = pd.DataFrame(users_ratings, columns = ["user_id", "paper_id", "rating", "time"])
    users_ratings["user_id"] = users_ratings["user_id"].map(users_mapping).astype("int64")
    users_ratings["rating"] = users_ratings["rating"].replace(-1, 0)
    users_ratings["time"] = pd.to_datetime(users_ratings["time"]).dt.floor('s')
    assert not users_ratings.isnull().any().any(), "There are null values in the users_ratings DataFrame."
    users_ratings = users_ratings.sort_values(by = ["user_id", "time"]).reset_index(drop = True)
    users_ratings["session_id"] = create_session_column(users_ratings)
    users_ratings.to_parquet(path, index = False, compression = "gzip")

def load_users_ratings(path : PosixPath = PATHS["users_ratings_path"], relevant_users_ids : list = None, relevant_columns : list = None) -> pd.DataFrame:
    if type(path) == PosixPath:
        users_ratings = pd.read_parquet(path, engine = "pyarrow")
    else:
        users_ratings = path.copy()
    if relevant_users_ids is not None:
        users_ratings = users_ratings[users_ratings["user_id"].isin(relevant_users_ids)]
    assert not users_ratings.isnull().any().any()
    assert users_ratings["time"].dtype == "datetime64[ns]"
    assert users_ratings.groupby("user_id")["time"].is_monotonic_increasing.all()
    assert users_ratings.groupby("user_id")["session_id"].is_monotonic_increasing.all()
    if relevant_columns is not None:
        users_ratings = users_ratings[relevant_columns]
    return users_ratings

def convert_papers_categories_parquet(papers_categories : pd.DataFrame, papers_ids : pd.Series, path : PosixPath = PATHS["papers_categories_path"]) -> None:
    assert papers_ids.is_monotonic_increasing
    papers_categories = papers_categories[["paper_id", "l1", "l2", "l3"]].copy()
    papers_categories = papers_categories.astype({"paper_id" : "int64", "l1": "string", "l2": "string", "l3": "string"})
    papers_categories = papers_categories[papers_categories["paper_id"].isin(papers_ids)]
    missing_papers_ids = papers_ids[~papers_ids.isin(papers_categories["paper_id"])]
    if len(missing_papers_ids) > 0:
        missing_papers_categories = pd.DataFrame({"paper_id": missing_papers_ids}).reindex(columns = ["paper_id", "l1", "l2", "l3"])
        missing_papers_categories = missing_papers_categories.astype(papers_categories.dtypes.to_dict())
        papers_categories = pd.concat([papers_categories, missing_papers_categories], ignore_index = True)
    papers_categories = papers_categories.sort_values(by = "paper_id").reset_index(drop = True)
    papers_categories.to_parquet(path, index = False, compression = "gzip")

def load_papers_categories(path : PosixPath = PATHS["papers_categories_path"], relevant_papers_ids : list = None, relevant_columns : list = None) -> pd.DataFrame:
    if type(path) == PosixPath:
        papers_categories = pd.read_parquet(path, engine = "pyarrow")
    else:
        papers_categories = path.copy()
    if relevant_papers_ids is not None:
        papers_categories = papers_categories[papers_categories["paper_id"].isin(relevant_papers_ids)]
    assert papers_categories["paper_id"].is_monotonic_increasing
    if relevant_columns is not None:
        papers_categories = papers_categories[relevant_columns]
    return papers_categories

def save_papers(path : PosixPath = PATHS["papers_path"]) -> None:
    papers_query = """SELECT paper_id, title, abstract, authors FROM papers ORDER BY paper_id"""
    papers = sql_execute(papers_query)
    papers = pd.DataFrame(papers, columns = ["paper_id", "title", "abstract", "authors"])
    ratings_papers_ids = [row[0] for row in sql_execute(
        "SELECT DISTINCT paper_id FROM users_ratings WHERE rating in (-1, 1) AND time IS NOT NULL ORDER BY paper_id")]
    assert set(ratings_papers_ids) <= set(papers["paper_id"])
    papers["in_ratings"] = papers["paper_id"].isin(ratings_papers_ids)
    cache_papers_ids = [row[0] for row in sql_execute("SELECT DISTINCT paper_id FROM cache_papers ORDER BY paper_id")]
    assert set(cache_papers_ids) <= set(papers["paper_id"])
    papers["in_cache"] = papers["paper_id"].isin(cache_papers_ids)
    papers = papers.astype({"title": "string", "abstract": "string", "authors": "string"})
    papers.to_parquet(path, index = False, compression = "gzip")

def load_papers(path : PosixPath = PATHS["papers_path"], relevant_papers_ids : list = None, relevant_columns : list = None, 
                include_papers_categories : bool = False, papers_categories_path : PosixPath = PATHS["papers_categories_path"]) -> pd.DataFrame:
    if type(path) == PosixPath:
        papers = pd.read_parquet(path, engine = "pyarrow")
    else:
        papers = path.copy()
    if relevant_papers_ids is not None:
        papers = papers[papers["paper_id"].isin(relevant_papers_ids)]
    assert not papers.isnull().any().any()
    assert papers["paper_id"].is_monotonic_increasing
    if include_papers_categories:
        papers_categories = load_papers_categories(papers_categories_path, relevant_papers_ids)
        assert (papers_categories["paper_id"] == papers["paper_id"]).all()
        papers = papers.merge(papers_categories, on = "paper_id", how = "left")
    if relevant_columns is not None:
        papers = papers[relevant_columns]
    return papers