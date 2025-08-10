import argparse
import functools
import os
import pickle
import random
from pathlib import Path

import pandas as pd
import sqlalchemy
from sqlalchemy import bindparam, create_engine

from ..src.project_paths import ProjectPaths

pd.set_option("display.max_rows", 1000)


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Database Login Parameters")
    parser.add_argument("--os_env_dict", action="store_true")
    parser.add_argument("--scholar_inbox_dict", action="store_true")
    parser.add_argument("--db_name", type=str)
    parser.add_argument("--db_user", type=str)
    parser.add_argument("--db_password", type=str)
    parser.add_argument("--db_host", type=str)
    parser.add_argument("--db_port", type=str)
    parser.add_argument("--papers_categories_old_file", type=Path)
    args_dict = vars(parser.parse_args())
    if not isinstance(args_dict["papers_categories_old_file"], Path):
        args_dict["papers_categories_old_file"] = Path(
            args_dict["papers_categories_old_file"]
        ).resolve()
    return set_login_parameters(args_dict), args_dict["papers_categories_old_file"]


def set_login_parameters(args_dict: dict) -> None:
    assert not (args_dict["os_env_dict"] and args_dict["scholar_inbox_dict"])
    if args_dict["os_env_dict"] or args_dict["scholar_inbox_dict"]:
        assert not any(
            [
                args_dict["db_name"],
                args_dict["db_user"],
                args_dict["db_password"],
                args_dict["db_host"],
                args_dict["db_port"],
            ]
        )
    login_parameters = {}
    if args_dict["os_env_dict"]:
        login_parameters = {
            "db_name": "backup_2025_02_23",
            "db_user": os.getenv("DB_USER"),
            "db_password": os.getenv("DB_PASSWORD", "scholar"),
            "db_host": os.getenv("DB_HOST", "localhost"),
            "db_port": os.getenv("DB_PORT", "5432"),
        }
    elif args_dict["scholar_inbox_dict"]:
        login_parameters = {
            "db_name": "maindb",
            "db_user": "scholar",
            "db_password": "scholar",
            "db_host": "localhost",
            "db_port": "5432",
        }
    else:
        login_parameters = {
            "db_name": args_dict["db_name"],
            "db_user": args_dict["db_user"],
            "db_password": args_dict["db_password"],
            "db_host": args_dict["db_host"],
            "db_port": args_dict["db_port"],
        }
    return login_parameters


def create_sql_connection():
    """
    Creates a new sql connection
    """
    sql_connection = global_sql_engine.connect()
    return sql_connection


def with_sql_connection():
    """
    Wrapper to make sure db connection objects are created and terminated appropriately
    :param func: Function
    :return:
    """

    # https://lemonfold.io/posts/2022/dbc/typed_decorator/
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            connection_needs_to_be_closed = False
            sql_connection = create_sql_connection()
            connection_needs_to_be_closed = True
            if "sql_connection" not in kwargs:
                result = func(*args, sql_connection=sql_connection, **kwargs)
            else:
                result = func(*args, **kwargs)

            if connection_needs_to_be_closed:
                sql_connection.close()
            return result

        return wrapper

    return decorator


def bind_list_params(query, **kwargs):
    query = sqlalchemy.text(query)

    params = {}
    for key, value in kwargs.items():
        params[key] = value
        if isinstance(value, list):
            query = query.bindparams(bindparam(key, expanding=True))
    return query, params


@with_sql_connection()
def sql_execute(query, sql_connection, **kwargs):
    """
    Executes an SQL statement on the gmailgooglescholar database.
    :param query: string
    :return:
    """
    query, params = bind_list_params(query, **kwargs)
    result_proxy = sql_connection.execute(query, params)
    if result_proxy.returns_rows:
        res = result_proxy.fetchall()
        result_proxy.close()
    else:
        res = None
    return res


def save_papers_texts(path: Path) -> pd.DataFrame:
    papers_texts_query = (
        """SELECT paper_id, title, abstract, authors FROM papers ORDER BY paper_id"""
    )
    papers_texts = sql_execute(papers_texts_query)
    papers_texts = pd.DataFrame(papers_texts, columns=["paper_id", "title", "abstract", "authors"])
    papers_texts = papers_texts.astype(
        {
            "paper_id": "int64",
            "title": "string",
            "abstract": "string",
            "authors": "string",
        }
    )
    papers_texts.to_parquet(path, index=False, compression="gzip")
    return papers_texts


def get_papers_categories(
    papers_categories_old: pd.DataFrame, papers_ids: pd.Series
) -> pd.DataFrame:
    assert papers_ids.is_monotonic_increasing
    papers_categories = papers_categories_old[["paper_id", "l1", "l2", "l3"]].copy()
    papers_categories = papers_categories.astype(
        {"paper_id": "int64", "l1": "string", "l2": "string", "l3": "string"}
    )
    papers_categories = papers_categories[papers_categories["paper_id"].isin(papers_ids)]
    missing_papers_ids = papers_ids[~papers_ids.isin(papers_categories["paper_id"])]
    if len(missing_papers_ids) > 0:
        missing_papers_categories = pd.DataFrame({"paper_id": missing_papers_ids}).reindex(
            columns=["paper_id", "l1", "l2", "l3"]
        )
        missing_papers_categories = missing_papers_categories.astype(
            papers_categories.dtypes.to_dict()
        )
        papers_categories = pd.concat(
            [papers_categories, missing_papers_categories], ignore_index=True
        )
    papers_categories = papers_categories.sort_values(by="paper_id").reset_index(drop=True)
    return papers_categories


def save_papers(path: Path, papers_categories_old: pd.DataFrame) -> pd.DataFrame:
    papers_query = """SELECT paper_id FROM papers ORDER BY paper_id"""
    papers = sql_execute(papers_query)
    papers = pd.DataFrame(papers, columns=["paper_id"], dtype="int64")
    ratings_papers_ids = [
        row[0]
        for row in sql_execute(
            """
            SELECT DISTINCT paper_id FROM users_ratings WHERE rating in (-1, 1)
            AND time IS NOT NULL ORDER BY paper_id
            """
        )
    ]
    assert set(ratings_papers_ids) <= set(papers["paper_id"])
    papers["in_ratings"] = papers["paper_id"].isin(ratings_papers_ids)
    cache_papers_ids = [
        row[0]
        for row in sql_execute("SELECT DISTINCT paper_id FROM cache_papers ORDER BY paper_id")
    ]
    assert set(cache_papers_ids) <= set(papers["paper_id"])
    papers["in_cache"] = papers["paper_id"].isin(cache_papers_ids)
    papers_categories = get_papers_categories(papers_categories_old, papers["paper_id"])
    assert (papers["paper_id"] == papers_categories["paper_id"]).all()
    papers = papers.merge(papers_categories, on="paper_id", how="left")
    papers.to_parquet(path, index=False, compression="gzip")
    return papers


def save_users_mapping(
    path: Path, users_ratings_before_mapping: pd.DataFrame, random_state: int = 42
) -> dict:
    users_ids = sorted(users_ratings_before_mapping["user_id"].unique().tolist())
    random.seed(random_state)
    indices = list(range(len(users_ids)))
    random.shuffle(indices)
    users_mapping = {user_id: idx for user_id, idx in zip(users_ids, indices)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(users_mapping, f)
    return users_mapping


def create_session_column(
    users_ratings: pd.DataFrame, session_threshold_min: int = 420
) -> pd.Series:
    session_threshold = pd.Timedelta(minutes=session_threshold_min)
    users_ratings = users_ratings.sort_values(by=["user_id", "time"]).copy()
    users_ratings["time_diff"] = users_ratings.groupby("user_id")["time"].diff()
    users_ratings["session_break"] = (
        users_ratings["time_diff"] > session_threshold
    ) | users_ratings["time_diff"].isna()
    return users_ratings.groupby("user_id")["session_break"].cumsum() - 1


def create_n_negrated_still_to_come_column(users_ratings: pd.DataFrame) -> pd.Series:
    users_ratings = users_ratings.sort_values(by=["user_id", "time"]).copy()
    neg_indicator = (users_ratings["rating"] == 0).astype(int)
    users_ratings = users_ratings.assign(neg_indicator=neg_indicator)
    
    def calculate_causal_counts(group):
        session_counts = group.groupby("session_id")["neg_indicator"].sum()
        causal_counts = session_counts[::-1].cumsum()[::-1]
        return group["session_id"].map(causal_counts).fillna(0)
    
    return users_ratings.groupby("user_id", group_keys=False).apply(calculate_causal_counts)


def save_users_ratings(
    path: Path, users_mapping: dict = None, papers: pd.DataFrame = None
) -> pd.DataFrame:
    users_ratings_query = """
        SELECT user_id, paper_id, rating, time FROM users_ratings 
        WHERE rating IN (-1, 1) AND time IS NOT NULL
        """
    users_ratings = sql_execute(users_ratings_query)
    users_ratings = pd.DataFrame(users_ratings, columns=["user_id", "paper_id", "rating", "time"])
    users_ratings["rating"] = users_ratings["rating"].replace(-1, 0)
    users_ratings["time"] = pd.to_datetime(users_ratings["time"]).dt.floor("ms")
    if papers is not None:
        users_ratings_with_papers = users_ratings.merge(
            papers[["paper_id", "l1", "l2"]], on="paper_id", how="left"
        )
        users_ratings_filtered = users_ratings_with_papers[
            users_ratings_with_papers["l1"].notna() & users_ratings_with_papers["l2"].notna()
        ]
        users_ratings = users_ratings_filtered.drop(columns=["l1", "l2"])
    if users_mapping is not None:
        users_ratings["user_id"] = users_ratings["user_id"].map(users_mapping).astype("int64")
        unique_users_ids = users_ratings["user_id"].unique()
        assert set(unique_users_ids) == set(range(len(users_mapping)))
    assert not users_ratings.isnull().any().any()
    users_ratings = users_ratings.sort_values(by=["user_id", "time"]).reset_index(drop=True)
    users_ratings["session_id"] = create_session_column(users_ratings)
    users_ratings["n_negrated_still_to_come"] = create_n_negrated_still_to_come_column(users_ratings)
    path.parent.mkdir(parents=True, exist_ok=True)
    users_ratings.to_parquet(path, index=False, compression="gzip")
    return users_ratings


def get_users_distributions(users_ratings: pd.DataFrame, papers: pd.DataFrame) -> pd.DataFrame:
    l1_categories = sorted([cat for cat in papers["l1"].unique() if pd.notna(cat)])
    users_ratings = users_ratings.copy()
    users_ratings = users_ratings[users_ratings["rating"] > 0]
    users_ratings_merged = users_ratings.merge(
        papers[["paper_id", "l1"]],
        on="paper_id",
        how="left",
    )
    assert len(users_ratings_merged) == len(users_ratings), "Not all paper_ids have L1 labels."
    users_distributions = (
        users_ratings_merged.groupby(["user_id", "l1"])
        .size()
        .reset_index(name="count")
        .pivot(index="user_id", columns="l1", values="count")
        .fillna(0)
    )
    for category in l1_categories:
        if category not in users_distributions.columns:
            users_distributions[category] = 0
    users_distributions = users_distributions[l1_categories]
    users_distributions = users_distributions.div(users_distributions.sum(axis=1), axis=0)
    return users_distributions.reset_index()


def save_significant_categories_for_all_users(
    path: Path,
    users_distributions: pd.DataFrame,
    min_percentage: float = 0.1,
    top_n: int = 4,
) -> pd.DataFrame:
    results = []
    for _, row in users_distributions.iterrows():
        user_id = int(row["user_id"])
        categories = row.drop("user_id")
        significant_categories = categories[categories >= min_percentage]
        significant_categories = significant_categories.sort_values(ascending=False)
        for rank, (category, proportion) in enumerate(significant_categories.items(), 1):
            results.append(
                {"user_id": user_id, "rank": rank, "category": category, "proportion": proportion}
            )
    results_df = pd.DataFrame(results)
    results_df = results_df[results_df["rank"] <= top_n].reset_index(drop=True)
    results_df.to_parquet(path, index=False, compression="gzip")
    return results_df


login_parameters, papers_categories_old_file = parse_args()
db_name, db_user, db_password, db_host, db_port = login_parameters.values()
sql_connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
global_sql_engine = create_engine(
    sql_connection_string, pool_size=20, pool_recycle=3600, pool_pre_ping=True
)

if __name__ == "__main__":
    data_path = ProjectPaths.data_path()
    data_path.mkdir(parents=True, exist_ok=True)

    save_papers_texts(ProjectPaths.data_papers_texts_path())

    papers_categories_old = pd.read_parquet(papers_categories_old_file, engine="pyarrow")
    papers = save_papers(ProjectPaths.data_papers_path(), papers_categories_old)

    users_ratings = save_users_ratings(
        ProjectPaths.data_users_ratings_path(), papers=papers
    )

    users_distributions = get_users_distributions(users_ratings, papers)
    users_significant_categories = save_significant_categories_for_all_users(
        ProjectPaths.data_users_significant_categories_path(), users_distributions
    )
