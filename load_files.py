import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[0]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_paths_to_sys()

import pickle, os, random
import numpy as np, pandas as pd

def load_users_mapping(path: Path) -> dict:
    with open(path, "rb") as f:
        users_mapping = pickle.load(f)
    keys = list(users_mapping.keys())
    assert keys == sorted(keys)
    values = list(users_mapping.values())
    assert sorted(values) == list(range(len(users_mapping)))
    return users_mapping

def load_users_ratings(path: Path = ProjectPaths.data_users_ratings_path(), relevant_users_ids: list = None, relevant_columns: list = None) -> pd.DataFrame:
    users_ratings = pd.read_parquet(path, engine = "pyarrow")
    assert users_ratings["user_id"].is_monotonic_increasing
    if relevant_users_ids is not None:
        users_ratings = users_ratings[users_ratings["user_id"].isin(relevant_users_ids)]
    assert not users_ratings.isnull().any().any()
    assert users_ratings["time"].dtype == "datetime64[ns]"
    assert users_ratings.groupby("user_id")["time"].is_monotonic_increasing.all()
    assert users_ratings.groupby("user_id")["session_id"].is_monotonic_increasing.all()
    if relevant_columns is not None:
        existing_columns = [col for col in relevant_columns if col in users_ratings.columns]
        if existing_columns:
            users_ratings = users_ratings[existing_columns]
        users_ratings = users_ratings[relevant_columns]
    return users_ratings

def load_papers_texts(path: Path = ProjectPaths.data_papers_texts_path(), relevant_papers_ids: list = None, relevant_columns: list = None) -> pd.DataFrame:
    papers_texts = pd.read_parquet(path, engine = "pyarrow")
    assert papers_texts["paper_id"].is_monotonic_increasing
    if relevant_papers_ids is not None:
        papers_texts = papers_texts[papers_texts["paper_id"].isin(relevant_papers_ids)]
    assert not papers_texts.isnull().any().any()
    if relevant_columns is not None:
        existing_columns = [col for col in relevant_columns if col in papers_texts.columns]
        if existing_columns:
            papers_texts = papers_texts[existing_columns]
    return papers_texts

def load_papers(path: Path = ProjectPaths.data_papers_path(), relevant_papers_ids: list = None, relevant_columns: list = None) -> pd.DataFrame:
    papers = pd.read_parquet(path, engine = "pyarrow")
    assert papers["paper_id"].is_monotonic_increasing
    if relevant_papers_ids is not None:
        papers = papers[papers["paper_id"].isin(relevant_papers_ids)]
    assert not papers[["paper_id", "in_ratings", "in_cache"]].isnull().any().any()
    if relevant_columns is not None:
        existing_columns = [col for col in relevant_columns if col in papers.columns]
        if existing_columns:
            papers = papers[existing_columns]
    return papers

def load_finetuning_users(path: Path = ProjectPaths.data_finetuning_users_path(), selection: str = "all") -> dict:
    assert selection in ["all", "train", "val", "test"]
    if not path.exists():
        raise FileNotFoundError(f"Finetuning users file not found at {path}. Run 'get_users_ratings.py' to create it.")
    with open(path, "rb") as f:
        finetuning_users = pickle.load(f)
    assert isinstance(finetuning_users, dict)
    assert set(finetuning_users.keys()) == {"train", "val", "test"}
    assert all(isinstance(users, list) for users in finetuning_users.values())
    assert all(users == sorted(users) for users in finetuning_users.values())
    assert all(len(users) == len(set(users)) for users in finetuning_users.values())
    assert set(finetuning_users["train"]) & set(finetuning_users["val"]) & set(finetuning_users["test"]) == set()
    if selection == "all":
        return finetuning_users
    else:
        return finetuning_users[selection]

if __name__ == "__main__":

    papers_texts = load_papers_texts()
    papers = load_papers()
    assert (papers["paper_id"] == papers_texts["paper_id"]).all()

    users_ratings = load_users_ratings()
    unique_users_ids = users_ratings["user_id"].unique()
    assert list(unique_users_ids) == list(range(len(unique_users_ids)))

    users_mapping_path = ProjectPaths.data_users_mapping_path()
    if users_mapping_path.exists():
        users_mapping = load_users_mapping(users_mapping_path)
    users_ratings_before_mapping_path = ProjectPaths.data_users_ratings_before_mapping_path()
    if users_ratings_before_mapping_path.exists():
        users_ratings_before_mapping = load_users_ratings(users_ratings_before_mapping_path)
        unique_users_ids_before_mapping = users_ratings_before_mapping["user_id"].unique()
        assert len(unique_users_ids_before_mapping) == len(unique_users_ids)

    finetuning_users_path = ProjectPaths.data_finetuning_users_path()
    if finetuning_users_path.exists():
        finetuning_users = load_finetuning_users(finetuning_users_path)