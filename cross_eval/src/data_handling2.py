from sqlalchemy import create_engine, bindparam
import functools
import pandas as pd
import os
import random
import sqlalchemy

DB_NAME = os.getenv("DB_NAME") if os.getenv("DB_NAME") is not None else "maindb"
DB_USER = os.getenv("DB_USER") if os.getenv("DB_USER") is not None else "scholar"
DB_PASSWORD = os.getenv("DB_PASSWORD") if os.getenv("DB_PASSWORD") is not None else "scholar"
DB_HOST = os.getenv("DB_HOST") if os.getenv("DB_HOST") is not None else "localhost"
DB_PORT = os.getenv("DB_PORT") if os.getenv("DB_PORT") is not None else "5432"
SQL_CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
global_sql_engine = create_engine(SQL_CONNECTION_STRING, pool_size = 20, pool_recycle = 3600, pool_pre_ping = True)

def create_sql_connection():
    '''
    Creates a new sql connection
    '''
    sql_connection = global_sql_engine.connect()
    return sql_connection

def with_sql_connection():
    '''
    Wrapper to make sure db connection objects are created and terminated appropriately
    :param func: Function
    :return:
    '''
    # https://lemonfold.io/posts/2022/dbc/typed_decorator/
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            connection_needs_to_be_closed = False
            sql_connection = create_sql_connection()
            connection_needs_to_be_closed = True
            if not "sql_connection" in kwargs:
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
    '''
    Executes an SQL statement on the gmailgooglescholar database.
    :param query: string
    :return:
    '''
    query, params = bind_list_params(query, **kwargs)
    result_proxy = sql_connection.execute(query, params)
    if result_proxy.returns_rows:
        res = result_proxy.fetchall()
        result_proxy.close()
    else:
        res = None
    return res

def get_db_name() -> str:
    return DB_NAME

def get_db_backup_date() -> str:
    query = """SELECT MAX(time) FROM users_ratings;"""
    backup_date = str(sql_execute(query)[0][0])
    return backup_date.split(" ")[0]

def get_titles_and_abstracts(papers_ids : list = None) -> str:
    query = f"""
            SELECT paper_id, title, abstract FROM papers
            {f'WHERE paper_id IN ({", ".join([str(x) for x in papers_ids])})' if papers_ids else ''}
            ORDER BY paper_id;
            """
    papers = sql_execute(query)
    return sorted(papers, key = lambda x: x[0])

def get_users_ids_with_sufficient_votes(min_n_posrated : int = 0, min_n_negrated : int = 0) -> pd.DataFrame:
    query = """
            WITH users_ratings_n AS (
                SELECT  user_id,
                        SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) AS n_posrated,
                        SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS n_negrated
                    FROM users_ratings
                    GROUP BY user_id)
            SELECT user_id, n_posrated, n_negrated
            FROM users_ratings_n
            WHERE n_posrated >= :min_n_posrated AND n_negrated >= :min_n_negrated
            ORDER BY user_id;
            """
    tuple_list = sql_execute(query, min_n_posrated = min_n_posrated, min_n_negrated = min_n_negrated)
    return pd.DataFrame(tuple_list, columns = ["user_id", "n_posrated", "n_negrated"])

def get_cache_papers_ids(random_state : int = None, cache_size : int = 5000) -> list:
    rated_papers_query = """SELECT DISTINCT paper_id FROM users_ratings WHERE rating IN (-1, 1);"""
    rated_papers_ids = set([t[0] for t in sql_execute(rated_papers_query)])
    base_papers_query = """SELECT DISTINCT paper_id FROM base_papers;"""
    base_papers_ids = set([t[0] for t in sql_execute(base_papers_query)])
    cache_papers_query = """SELECT paper_id FROM cache_papers;"""
    cache_papers_ids = set([t[0] for t in sql_execute(cache_papers_query)])
    filtered_cache_papers_ids = sorted(list(cache_papers_ids - rated_papers_ids.union(base_papers_ids)))
    n_filtered_cache_papers = len(filtered_cache_papers_ids)
    if n_filtered_cache_papers < cache_size:
        raise ValueError(f"Required cache size ({cache_size}) is greater than the number of filtered cache papers ({n_filtered_cache_papers}).")
    rng = random.Random(random_state)
    filtered_cache_papers_ids = rng.sample(filtered_cache_papers_ids, cache_size)
    return sorted(filtered_cache_papers_ids)
    
def get_rated_papers_ids_for_user(user_id : int, positive : bool) -> list:
    query = f"""
            SELECT paper_id FROM users_ratings 
            WHERE user_id = {user_id}
            AND rating = {1 if positive else -1}
            ORDER BY paper_id;
            """
    return [t[0] for t in sql_execute(query)]