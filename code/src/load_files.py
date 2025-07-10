import pickle
from pathlib import Path

import pandas as pd

from .project_paths import ProjectPaths

FINETUNING_MODEL = "gte_large_256"
FINETUNING_MODEL_HF = "Alibaba-NLP/gte-large-en-v1.5"


def load_users_mapping(path: Path) -> dict:
    with open(path, "rb") as f:
        users_mapping = pickle.load(f)
    keys = list(users_mapping.keys())
    assert keys == sorted(keys)
    values = list(users_mapping.values())
    assert sorted(values) == list(range(len(users_mapping)))
    return users_mapping


def load_users_ratings(
    path: Path = ProjectPaths.data_users_ratings_path(),
    relevant_users_ids: list = None,
    relevant_columns: list = None,
) -> pd.DataFrame:
    users_ratings = pd.read_parquet(path, engine="pyarrow")
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


def load_papers_texts(
    path: Path = ProjectPaths.data_papers_texts_path(),
    relevant_papers_ids: list = None,
    relevant_columns: list = None,
) -> pd.DataFrame:
    papers_texts = pd.read_parquet(path, engine="pyarrow")
    assert papers_texts["paper_id"].is_monotonic_increasing
    if relevant_papers_ids is not None:
        papers_texts = papers_texts[papers_texts["paper_id"].isin(relevant_papers_ids)].reset_index(
            drop=True
        )
    assert not papers_texts.isnull().any().any()
    if relevant_columns is not None:
        existing_columns = [col for col in relevant_columns if col in papers_texts.columns]
        if existing_columns:
            papers_texts = papers_texts[existing_columns]
    return papers_texts


def load_papers(
    path: Path = ProjectPaths.data_papers_path(),
    relevant_papers_ids: list = None,
    relevant_columns: list = None,
) -> pd.DataFrame:
    papers = pd.read_parquet(path, engine="pyarrow")
    assert papers["paper_id"].is_monotonic_increasing
    if relevant_papers_ids is not None:
        papers = papers[papers["paper_id"].isin(relevant_papers_ids)].reset_index(drop=True)
    assert not papers[["paper_id", "in_ratings", "in_cache"]].isnull().any().any()
    if relevant_columns is not None:
        existing_columns = [col for col in relevant_columns if col in papers.columns]
        if existing_columns:
            papers = papers[existing_columns]
    return papers


def load_relevant_papers_ids(
    path: Path = ProjectPaths.data_relevant_papers_ids_path(),
) -> list:
    with open(path, "rb") as f:
        relevant_papers_ids = pickle.load(f)
    assert isinstance(relevant_papers_ids, list)
    assert len(relevant_papers_ids) == len(set(relevant_papers_ids))
    assert relevant_papers_ids == sorted(relevant_papers_ids)
    return relevant_papers_ids


def load_finetuning_users(
    path: Path = ProjectPaths.data_finetuning_users_path(), selection: str = "all"
) -> dict:
    assert selection in ["all", "train", "val", "test"]
    if not path.exists():
        raise FileNotFoundError(
            f"Finetuning users file not found at {path}. Run 'get_users_ratings.py' to create it."
        )
    with open(path, "rb") as f:
        finetuning_users = pickle.load(f)
    assert isinstance(finetuning_users, dict)
    assert set(finetuning_users.keys()) == {"train", "val", "test"}
    assert all(isinstance(users, list) for users in finetuning_users.values())
    assert all(users == sorted(users) for users in finetuning_users.values())
    assert all(len(users) == len(set(users)) for users in finetuning_users.values())
    assert (
        set(finetuning_users["train"])
        & set(finetuning_users["val"])
        & set(finetuning_users["test"])
        == set()
    )
    if selection == "all":
        return finetuning_users
    else:
        return finetuning_users[selection]


def load_users_significant_categories(
    path: Path = ProjectPaths.data_users_significant_categories_path(),
    relevant_users_ids: list = None,
    relevant_columns: list = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Users significant categories file not found at {path}. Run 'get_users_ratings.py' to create it."
        )
    users_significant_categories = pd.read_parquet(path, engine="pyarrow")
    assert users_significant_categories["user_id"].is_monotonic_increasing
    assert (users_significant_categories["proportion"] >= 0.1).all()
    assert users_significant_categories.groupby("user_id").size().max() <= 4
    if relevant_users_ids is not None:
        users_significant_categories = users_significant_categories[
            users_significant_categories["user_id"].isin(relevant_users_ids)
        ].reset_index(drop=True)
    assert not users_significant_categories.isnull().any().any()
    if relevant_columns is not None:
        existing_columns = [
            col for col in relevant_columns if col in users_significant_categories.columns
        ]
        if existing_columns:
            users_significant_categories = users_significant_categories[existing_columns]
    
    return users_significant_categories


if __name__ == "__main__":
    papers_texts = load_papers_texts()
    papers = load_papers()
    assert (papers["paper_id"] == papers_texts["paper_id"]).all()
    relevant_papers_ids = load_relevant_papers_ids()
    assert set(relevant_papers_ids) <= set(papers["paper_id"])

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

    finetuning_users = load_finetuning_users()
    users_significant_categories = load_users_significant_categories()
