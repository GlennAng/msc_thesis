import argparse, functools, json, pickle, os, random, sqlalchemy
import numpy as np, pandas as pd
pd.set_option("display.max_rows", 1000)
from pathlib import Path
from sqlalchemy import bindparam, create_engine

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description = "Database Login Parameters")
    parser.add_argument("--os_env_dict", action = "store_true")
    parser.add_argument("--scholar_inbox_dict", action = "store_true")
    parser.add_argument("--db_name", type = str)
    parser.add_argument("--db_user", type = str)
    parser.add_argument("--db_password", type = str)
    parser.add_argument("--db_host", type = str)
    parser.add_argument("--db_port", type = str)
    parser.add_argument("--papers_categories_old_file", type = Path)
    args_dict = vars(parser.parse_args())
    if args_dict["papers_categories_old_file"] is not None:
        if not isinstance(args_dict["papers_categories_old_file"], Path):
            args_dict["papers_categories_old_file"] = Path(args_dict["papers_categories_old_file"]).resolve()
    return set_login_parameters(args_dict), args_dict["papers_categories_old_file"]

def set_login_parameters(args_dict: dict) -> None:
    assert not (args_dict["os_env_dict"] and args_dict["scholar_inbox_dict"])
    if args_dict["os_env_dict"] or args_dict["scholar_inbox_dict"]:
        assert not any([args_dict["db_name"], args_dict["db_user"], args_dict["db_password"], args_dict["db_host"], args_dict["db_port"]])
    login_parameters = {}
    if args_dict["os_env_dict"]:
        login_parameters = {
            "db_name": "backup_2025_02_23",
            "db_user": os.getenv("DB_USER"),
            "db_password": os.getenv("DB_PASSWORD", "scholar"),
            "db_host": os.getenv("DB_HOST", "localhost"),
            "db_port": os.getenv("DB_PORT", "5432")
        }
    elif args_dict["scholar_inbox_dict"]:
        login_parameters = {
            "db_name": "maindb",
            "db_user": "scholar",
            "db_password": "scholar",
            "db_host": "localhost",
            "db_port": "5432"
        }
    else:
        login_parameters = {
            "db_name": args_dict["db_name"],
            "db_user": args_dict["db_user"],
            "db_password": args_dict["db_password"],
            "db_host": args_dict["db_host"],
            "db_port": args_dict["db_port"]
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
            if not 'sql_connection' in kwargs:
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

def get_db_backup_date() -> str:
    query = '''SELECT MAX(time) FROM users_ratings;'''
    backup_date = str(sql_execute(query)[0][0])
    return backup_date.split(" ")[0]

def save_users_mapping(path : Path, random_state : int = 42) -> dict:
    users_ids_query = "SELECT DISTINCT user_id FROM users_ratings WHERE rating IN (-1, 1) AND time IS NOT NULL ORDER BY user_id"
    users_ids = [row[0] for row in sql_execute(users_ids_query)]
    random.seed(random_state)
    indices = list(range(len(users_ids)))
    random.shuffle(indices)
    users_mapping = {user_id: idx for user_id, idx in zip(users_ids, indices)}
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "wb") as f:
        pickle.dump(users_mapping, f)
    return users_mapping

def create_session_column(users_ratings : pd.DataFrame, session_threshold_min : int = 420) -> pd.Series:
    session_threshold = pd.Timedelta(minutes = session_threshold_min)
    users_ratings = users_ratings.sort_values(by = ["user_id", "time"]).copy()
    users_ratings["time_diff"] = users_ratings.groupby("user_id")["time"].diff()
    users_ratings["session_break"] = (users_ratings["time_diff"] > session_threshold) | users_ratings["time_diff"].isna()
    return users_ratings.groupby("user_id")["session_break"].cumsum() - 1

def save_users_ratings(path : Path, users_mapping : dict = None) -> pd.DataFrame:
    users_ratings_query = """SELECT user_id, paper_id, rating, time FROM users_ratings WHERE rating IN (-1, 1) AND time IS NOT NULL"""
    users_ratings = sql_execute(users_ratings_query)
    users_ratings = pd.DataFrame(users_ratings, columns = ["user_id", "paper_id", "rating", "time"])
    users_ratings["rating"] = users_ratings["rating"].replace(-1, 0)
    users_ratings["time"] = pd.to_datetime(users_ratings["time"]).dt.floor('ms')
    if users_mapping is not None:
        users_ratings["user_id"] = users_ratings["user_id"].map(users_mapping).astype("int64")
        unique_users_ids = users_ratings["user_id"].unique()
        assert set(unique_users_ids) == set(range(len(users_mapping)))
    assert not users_ratings.isnull().any().any()
    users_ratings = users_ratings.sort_values(by = ["user_id", "time"]).reset_index(drop = True)
    users_ratings["session_id"] = create_session_column(users_ratings)
    path.parent.mkdir(parents = True, exist_ok = True)
    users_ratings.to_parquet(path, index = False, compression = "gzip")
    return users_ratings

def save_papers_texts(path: Path) -> pd.DataFrame:
    papers_texts_query = """SELECT paper_id, title, abstract, authors FROM papers ORDER BY paper_id"""
    papers_texts = sql_execute(papers_texts_query)
    papers_texts = pd.DataFrame(papers_texts, columns = ["paper_id", "title", "abstract", "authors"])
    papers_texts = papers_texts.astype({"paper_id": "int64", "title": "string", "abstract": "string", "authors": "string"})
    papers_texts.to_parquet(path, index = False, compression = "gzip")
    return papers_texts

def get_papers_categories(papers_categories_old: pd.DataFrame, papers_ids: pd.Series) -> pd.DataFrame:
    assert papers_ids.is_monotonic_increasing
    papers_categories = papers_categories_old[["paper_id", "l1", "l2", "l3"]].copy()
    papers_categories = papers_categories.astype({"paper_id": "int64", "l1": "string", "l2": "string", "l3": "string"})
    papers_categories = papers_categories[papers_categories["paper_id"].isin(papers_ids)]
    missing_papers_ids = papers_ids[~papers_ids.isin(papers_categories["paper_id"])]
    if len(missing_papers_ids) > 0:
        missing_papers_categories = pd.DataFrame({"paper_id": missing_papers_ids}).reindex(columns = ["paper_id", "l1", "l2", "l3"])
        missing_papers_categories = missing_papers_categories.astype(papers_categories.dtypes.to_dict())
        papers_categories = pd.concat([papers_categories, missing_papers_categories], ignore_index = True)
    papers_categories = papers_categories.sort_values(by = "paper_id").reset_index(drop = True)
    return papers_categories

def save_papers(path : Path, papers_categories_old: pd.DataFrame = None) -> pd.DataFrame:
    papers_query = """SELECT paper_id FROM papers ORDER BY paper_id"""
    papers = sql_execute(papers_query)
    papers = pd.DataFrame(papers, columns = ["paper_id"], dtype = "int64")
    ratings_papers_ids = [row[0] for row in sql_execute(
        "SELECT DISTINCT paper_id FROM users_ratings WHERE rating in (-1, 1) AND time IS NOT NULL ORDER BY paper_id")]
    assert set(ratings_papers_ids) <= set(papers["paper_id"])
    papers["in_ratings"] = papers["paper_id"].isin(ratings_papers_ids)
    cache_papers_ids = [row[0] for row in sql_execute("SELECT DISTINCT paper_id FROM cache_papers ORDER BY paper_id")]
    assert set(cache_papers_ids) <= set(papers["paper_id"])
    papers["in_cache"] = papers["paper_id"].isin(cache_papers_ids)
    if papers_categories_old is not None:
        papers_categories = get_papers_categories(papers_categories_old, papers["paper_id"])
        assert (papers["paper_id"] == papers_categories["paper_id"]).all()
        papers = papers.merge(papers_categories, on = "paper_id", how = "left")
    papers.to_parquet(path, index = False, compression = "gzip")
    return papers

login_parameters, papers_categories_old_file = parse_args()
db_name, db_user, db_password, db_host, db_port = login_parameters.values()
sql_connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
global_sql_engine = create_engine(sql_connection_string, pool_size = 20, pool_recycle = 3600, pool_pre_ping = True)

if __name__ == "__main__":
    db_backup_date = get_db_backup_date()
    data_path = Path(__file__).resolve().parent / "data"
    data_db_backup_path = data_path / db_backup_date
    data_db_backup_path.mkdir(parents = True, exist_ok = True)
    db_backup_date_dict = {"db_backup_date": db_backup_date}
    with open(data_path/ "db_backup_date.json", "w") as f:
        json.dump(db_backup_date_dict, f, indent = 4)

    users_mapping_path = data_db_backup_path / "users_mapping.pkl"
    users_mapping = save_users_mapping(users_mapping_path)
    users_ratings_path = data_db_backup_path / "users_ratings.parquet"
    save_users_ratings(users_ratings_path)
    users_ratings_mapped_path = data_db_backup_path / "users_ratings_mapped.parquet"
    save_users_ratings(users_ratings_mapped_path, users_mapping)

    papers_texts_path = data_db_backup_path / "papers_texts.parquet"
    save_papers_texts(papers_texts_path)

    if papers_categories_old_file is not None:
        papers_categories_old = pd.read_parquet(papers_categories_old_file, engine = "pyarrow")
    else:
        papers_categories_old = None
    papers_path = data_db_backup_path / "papers.parquet"
    papers = save_papers(papers_path, papers_categories_old)