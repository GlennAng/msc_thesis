import functools
import os
import pandas as pd
import random
import sqlalchemy
from enum import Enum, auto
from sqlalchemy import create_engine, bindparam

class Paper_Removal(Enum):
    NONE = auto()
    OLDEST = auto()
    RANDOM = auto()
    NEWEST = auto()

def get_paper_removal_from_arg(paper_removal_arg : str) -> Paper_Removal:
    valid_paper_removal_args = [paper_removal.name.lower() for paper_removal in Paper_Removal]
    if paper_removal_arg.lower() not in valid_paper_removal_args:
        raise ValueError(f"Invalid argument {paper_removal_arg} 'paper_removal'. Possible values: {valid_paper_removal_args}.")
    return Paper_Removal[paper_removal_arg.upper()]

DB_NAME = "postgres"
DB_USER = os.getenv('DB_USER') if os.getenv('DB_USER') is not None else "scholar"
DB_PASSWORD = os.getenv('DB_PASSWORD') if os.getenv('DB_PASSWORD') is not None else "scholar"
DB_HOST = os.getenv('DB_HOST') if os.getenv('DB_HOST') is not None else "localhost"
DB_PORT = os.getenv('DB_PORT') if os.getenv('DB_PORT') is not None else "5432"
SQL_CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
global_sql_engine = create_engine(SQL_CONNECTION_STRING, pool_size = 20, pool_recycle = 3600, pool_pre_ping = True)


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

def get_users_ids_with_sufficient_votes(min_n_posrated : int, min_n_negrated : int, sort_ids : bool = False) -> pd.DataFrame:
    query = f"""
    WITH users_ratings_n AS (
        SELECT  user_id,
                SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) AS n_posrated,
                SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS n_negrated
            FROM users_ratings
            GROUP BY user_id)
    SELECT user_id, n_posrated, n_negrated, n_posrated + n_negrated AS n_rated
    FROM users_ratings_n
    WHERE n_posrated >= :min_n_posrated AND n_negrated >= :min_n_negrated
    {"ORDER BY user_id" if sort_ids else ""}
    """
    tuple_list = sql_execute(query, min_n_posrated = min_n_posrated, min_n_negrated = min_n_negrated)
    return pd.DataFrame(tuple_list, columns = ["user_id", "n_posrated", "n_negrated", "n_rated"])

def get_user_id_from_sha_key(sha_key : str) -> int:
    query = """SELECT user_id FROM users WHERE sha_key = :sha_key"""
    tuple_list = sql_execute(query, sha_key = sha_key)
    return tuple_list[0][0] if len(tuple_list) > 0 else None

def get_user_id_from_sha_key(sha_key : str) -> int:
    query = '''
    SELECT user_id FROM users WHERE sha_key = :sha_key
    '''
    tuple_list = sql_execute(query, sha_key = sha_key)
    return tuple_list[0][0] if len(tuple_list) > 0 else None

def get_users_ids_from_sha_keys(sha_keys : list) -> dict:
    users_dict = {}
    for sha_key in sha_keys:
        users_dict[sha_key] = get_user_id_from_sha_key(sha_key)
    return users_dict

def get_rated_papers_ids_for_user(user_id : int, rating : int, paper_removal : Paper_Removal = None, remaining_percentage : float = None, random_state : int = None) -> list:
    query = f"""
    SELECT paper_id FROM users_ratings 
    WHERE user_id = {user_id} 
    AND rating = {rating}
    {"ORDER BY time" if paper_removal in [Paper_Removal.OLDEST, Paper_Removal.NEWEST] else ""};
    """
    result = [t[0] for t in sql_execute(query)]
    if paper_removal == Paper_Removal.OLDEST:
        result = result[:int(len(result) * remaining_percentage)]
    elif paper_removal == Paper_Removal.NEWEST:
        result = result[-int(len(result) * remaining_percentage):]
    elif paper_removal == Paper_Removal.RANDOM:
        result = sorted(result)
        rng = random.Random(random_state)
        result = rng.sample(result, int(len(result) * remaining_percentage))
    return sorted(result)

def get_base_papers_ids_for_user(user_id : int, paper_removal : Paper_Removal = None,  remaining_percentage : float = None, random_state : int = None) -> list:
    query = f"""
    SELECT paper_id FROM base_papers
    WHERE user_id = {user_id}
    AND paper_id NOT IN (
        SELECT paper_id FROM users_ratings
        WHERE user_id = {user_id}
        AND rating IN (-1, 1)
    )
    {"ORDER BY time" if paper_removal in [Paper_Removal.OLDEST, Paper_Removal.NEWEST] else ""};
    """
    result = [t[0] for t in sql_execute(query)]
    if paper_removal == Paper_Removal.OLDEST:
        result = result[:int(len(result) * remaining_percentage)]
    elif paper_removal == Paper_Removal.NEWEST:
        result = result[-int(len(result) * remaining_percentage):]
    elif paper_removal == Paper_Removal.RANDOM:
        result = sorted(result)
        rng = random.Random(random_state)
        result = rng.sample(result, int(len(result) * remaining_percentage))
    return sorted(result)

def get_voting_weight_for_user(user_id : int) -> float:
    query = '''
    SELECT voting_weight FROM users 
    WHERE user_id = :user_id;
    '''
    return sql_execute(query, user_id = user_id)[0][0]

def get_global_cache_papers_ids(max_cache : int = None, random_state : int = None, draw_cache_from_users_ratings : bool = False) -> list:
    if draw_cache_from_users_ratings:
        query = '''SELECT paper_id FROM users_ratings WHERE rating IN (-1, 1);'''
    else:
        query = '''SELECT paper_id FROM cache_papers;'''
    cache = [t[0] for t in sql_execute(query)]
    n_cache = len(cache)
    max_cache = n_cache if max_cache is None else min(max_cache, n_cache)
    if n_cache < max_cache:
        raise ValueError(f"Required cache size ({max_cache}) is greater than the number of valid cache papers ({n_cache}).")
    elif n_cache > max_cache:
        cache = sorted(cache)
        rng = random.Random(random_state)
        cache = rng.sample(cache, max_cache)
    return sorted(cache)

def get_cache_papers_ids_for_user(user_id : int, max_cache : int = None, random_state : int = None, draw_cache_from_users_ratings : bool = False) -> list:
    if draw_cache_from_users_ratings:
        query = """
                SELECT DISTINCT paper_id FROM users_ratings
                WHERE rating IN (-1, 1) 
                AND user_id != :user_id
                AND paper_id NOT IN (
                    SELECT paper_id FROM base_papers
                    WHERE user_id = :user_id)
                """
    else:
        query = """
                SELECT paper_id FROM cache_papers
                WHERE paper_id NOT IN (
                    SELECT paper_id FROM users_ratings
                    WHERE user_id = :user_id)
                AND paper_id NOT IN (
                    SELECT paper_id FROM base_papers
                    WHERE user_id = :user_id);
                """
    cache = [t[0] for t in sql_execute(query, user_id = user_id)]
    n_cache = len(cache)
    max_cache = n_cache if max_cache is None else min(max_cache, n_cache)
    if n_cache < max_cache:
        raise ValueError(f"Required cache size ({max_cache}) is greater than the number of valid cache papers ({n_cache}) for User ({user_id}).")
    elif n_cache > max_cache:
        cache = sorted(cache)
        rng = random.Random(random_state)
        cache = rng.sample(cache, max_cache)
    return sorted(cache)

def get_negative_samples_ids_for_user(n_negative_samples : int, random_state : int, excluded_papers : list = None) -> list:
    if excluded_papers:
        excluded_papers_str = f"({', '.join([str(x) for x in excluded_papers])})"
        query = f"""
                SELECT paper_id FROM papers
                WHERE digest_date IS NOT NULL
                AND paper_id NOT IN {excluded_papers_str};
                """
    else:
        query = """
                SELECT paper_id FROM papers
                WHERE digest_date IS NOT NULL;
                """
    digest_papers = [t[0] for t in sql_execute(query)]
    n_digest_papers = len(digest_papers)
    if n_digest_papers < n_negative_samples:
        raise ValueError(f"Required negative samples ({n_negative_samples}) is greater than the number of valid digest papers ({n_digest_papers}).")
    elif n_digest_papers > n_negative_samples:
        digest_papers = sorted(digest_papers)
        rng = random.Random(random_state)
        digest_papers = rng.sample(digest_papers, n_negative_samples)
    return sorted(digest_papers)

def get_title_and_abstract(paper_id : int) -> str:
    query = '''
    SELECT title, abstract FROM papers WHERE paper_id = :paper_id;
    '''
    return sql_execute(query, paper_id = paper_id)[0]

def get_titles_and_abstracts(papers_ids : list = None) -> str:
    query = f"""
            SELECT paper_id, title, abstract FROM papers
            {f'WHERE paper_id IN ({", ".join([str(x) for x in papers_ids])})' if papers_ids else ''}
            ORDER BY paper_id;
            """
    papers = sql_execute(query)
    return sorted(papers, key = lambda x: x[0])

def get_users_survey_ratings() -> pd.DataFrame:
    query = """SELECT user_id, rating 
                FROM survey_answers 
                WHERE rating IS NOT NULL
                ORDER BY user_id;"""
    tuple_list = sql_execute(query)
    return pd.DataFrame(tuple_list, columns = ["user_id", "survey_rating"])

def get_db_name() -> str:
    return DB_NAME

def get_db_backup_date() -> str:
    query = '''SELECT MAX(time) FROM users_ratings;'''
    backup_date = str(sql_execute(query)[0][0])
    return backup_date.split(" ")[0]