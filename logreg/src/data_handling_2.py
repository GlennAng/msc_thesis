import functools
import os
import pandas as pd
import pickle
import random
import sqlalchemy
import tqdm
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

DB_NAME = "backup_2025_02_23"
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

def get_negative_samples_ids_arxiv(n_negative_samples : int, random_state : int) -> list:
    arxiv_ratios = {"cs": 0.0, "math": 0.25, "cond-mat": 0.20, "hep": 0.20, "astro-ph": 0.15, "physics": 0.12, "eess": 0.0, "stat": 0.0, "nucl": 0.03, "q-bio": 0.02, "nlin": 0.01, "q-fin": 0.01, "econ": 0.01}
    samples_per_category = {category: int(n_negative_samples * ratio) for category, ratio in arxiv_ratios.items()}
    negative_samples_ids = []
    rng = random.Random(random_state)
    exclude_query = """
    SELECT paper_id FROM users_ratings UNION SELECT paper_id FROM base_papers UNION SELECT paper_id FROM cache_papers"""
    papers_to_exclude = set([t[0] for t in sql_execute(exclude_query)])
    for category in list(arxiv_ratios.keys()):
        n_samples_category = samples_per_category[category]
        if n_samples_category == 0:
            continue
        query = f"SELECT paper_id FROM papers WHERE arxiv_category LIKE '{category}%'"
        papers = sorted([t[0] for t in sql_execute(query) if t[0] not in papers_to_exclude])
        negative_samples_ids += rng.sample(papers, n_samples_category)
    return sorted(negative_samples_ids)

def get_negative_samples_ids_for_user(n_negative_samples : int, random_state : int, excluded_papers : list = None) -> list:
    if excluded_papers:
        excluded_papers_str = f"({', '.join([str(x) for x in excluded_papers])})"
        query = f"""
                SELECT paper_id FROM papers
                WHERE arxiv_category = 'hep-ph'
                AND paper_id NOT IN {excluded_papers_str};
                """
    else:
        query = """
                SELECT paper_id FROM papers
                WHERE arxiv_category = 'hep-ph';
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

def get_titles_and_abstracts(papers_ids : list = None, include_arxiv_categories : bool = False) -> list:
    query = f"""
            SELECT paper_id, title, abstract {', arxiv_category' if include_arxiv_categories else ''} FROM papers
            {f'WHERE paper_id IN ({", ".join([str(x) for x in papers_ids])})' if papers_ids else ''}
            ORDER BY paper_id;
            """
    papers = sql_execute(query)
    papers = sorted(papers, key = lambda x: x[0])
    if include_arxiv_categories:
        from arxiv import arxiv_categories
        papers = [(paper_id, title, abstract, arxiv_categories[category.lower() if category else category]) for paper_id, title, abstract, category in papers]
    return papers

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

def convert_paper_category_original(category_str : str) -> str:
    transformed_category_str = category_str.lower()
    transformed_category_str = transformed_category_str.replace(" ", "_")
    return transformed_category_str

def convert_paper_category(category_str : str) -> str:
    transformed_category_str = convert_paper_category_original(category_str)
    if transformed_category_str in ["engineering", "electrical_engineering"]:
        transformed_category_str = "engineering"
    elif transformed_category_str in ["physics", "astronomy", "materials_science"]:
        transformed_category_str = "physics"
    elif transformed_category_str in ["biology", "chemistry", "earth_science", "geography"]:
        transformed_category_str = "natural_science"
    elif transformed_category_str in ["linguistics", "sociology", "psychology", "economics", "philosophy"]:
        transformed_category_str = "social_science"
    return transformed_category_str

def turn_parquet_to_dict(conversion_function : callable, parquet_df : pd.DataFrame = "../data/tsne_with_meta_full_for_plot_sorted.parquet") -> dict:
    papers_query = """SELECT paper_id FROM papers"""
    papers_ids = [t[0] for t in sql_execute(papers_query)]
    if type(parquet_df) == str:
        parquet_df = pd.read_parquet(parquet_df, engine = "pyarrow")
    papers_ids_to_categories = {paper_id : None for paper_id in papers_ids}
    for row in tqdm.tqdm(parquet_df.iterrows(), total = len(parquet_df), desc = "Loading papers ids to categories"):
        paper_id = row[1].paper_id
        if paper_id in papers_ids_to_categories:
            papers_ids_to_categories[paper_id] = conversion_function(row[1].l1)
    assert len(papers_ids_to_categories) == len(papers_ids)
    return papers_ids_to_categories

def save_papers_ids_to_categories(conversion_function : callable, file_path : str = "../data/papers_ids_to_categories.pkl", papers_ids_to_categories : dict = None) -> None:
    if papers_ids_to_categories is None:
        papers_ids_to_categories = turn_parquet_to_dict(conversion_function)
    with open(file_path, "wb") as file:
        pickle.dump(papers_ids_to_categories, file)
    
def load_papers_ids_to_categories(file_path : str = "../data/papers_ids_to_categories.pkl") -> dict:
    with open(file_path, "rb") as file:
        papers_ids_to_categories = pickle.load(file)
    return papers_ids_to_categories

def get_papers_categories_dataset_distribution(papers_ids_to_categories : dict = "../data/papers_ids_to_categories.pkl") -> tuple:
    if type(papers_ids_to_categories) == str:
        papers_ids_to_categories = load_papers_ids_to_categories(papers_ids_to_categories)
    print(len(papers_ids_to_categories))
    unique_categories = set(papers_ids_to_categories.values())
    categories_counts = {category: 0 for category in unique_categories}
    n_total = 0
    for key, value in papers_ids_to_categories.items():
        if value in categories_counts:
            categories_counts[value] += 1
            n_total += 1
        else:
            print(f"Unknown category: {value}.")
    categories_counts = {category: count / n_total for category, count in categories_counts.items()}
    sorted_categories = sorted(categories_counts.items(), key = lambda x: x[1], reverse = True)
    print(f"Total papers: {n_total}.")
    for category, count in sorted_categories:
        print(f"{category}: {count:.2%} ({int(count * n_total)})")
    return sorted_categories, n_total

def get_papers_categories_ratings_distribution(papers_ids_to_categories : dict = "../data/papers_ids_to_categories.pkl") -> tuple:
    if type(papers_ids_to_categories) == str:
        papers_ids_to_categories = load_papers_ids_to_categories(papers_ids_to_categories)
    unique_categories = set(papers_ids_to_categories.values())
    categories_counts = {category: 0 for category in unique_categories}
    n_total = 0
    query = """SELECT paper_id, COUNT(*) as count FROM users_ratings GROUP BY paper_id"""   
    result = sql_execute(query)
    for row in result:
        paper_id = row[0]
        if paper_id in papers_ids_to_categories:
            categories_counts[papers_ids_to_categories[paper_id]] += row[1]
            n_total += row[1]
        else:
            print(f"Unknown paper id: {paper_id}.")
    categories_counts = {category: count / n_total for category, count in categories_counts.items()}
    sorted_categories = sorted(categories_counts.items(), key = lambda x: x[1], reverse = True)
    print(f"Total papers: {n_total}.")
    for category, count in sorted_categories:
        print(f"{category}: {count:.2%} ({int(count * n_total)})")
    return sorted_categories, n_total

def get_cache_papers_categories_dataset_distribution(papers_ids_to_categories : dict = "../data/papers_ids_to_categories.pkl") -> tuple:
    if type(papers_ids_to_categories) == str:
        papers_ids_to_categories = load_papers_ids_to_categories(papers_ids_to_categories)
    cache_papers_ids = get_global_cache_papers_ids(max_cache = 5000, random_state = 42)
    cache_papers_ids_to_categories = {paper_id: papers_ids_to_categories[paper_id] for paper_id in cache_papers_ids}
    get_papers_categories_dataset_distribution(cache_papers_ids_to_categories)

def get_negative_samples_ids(n_negative_samples : int, random_state : int, papers_ids_to_categories : dict = "../data/papers_ids_to_categories.pkl") -> list:
    if type(papers_ids_to_categories) == str:
        papers_ids_to_categories = load_papers_ids_to_categories(papers_ids_to_categories)
    categories_ratios = {"physics" : 0.3, "medicine" : 0.25, "natural_science" : 0.2, "social_science" : 0.15, "engineering" : 0.1}
    samples_per_category = {category : int(n_negative_samples * ratio) for category, ratio in categories_ratios.items()}
    negative_samples_ids = []
    rng = random.Random(random_state)
    exclude_query = """SELECT paper_id FROM users_ratings UNION SELECT paper_id FROM base_papers UNION SELECT paper_id FROM cache_papers"""
    papers_to_exclude = set([t[0] for t in sql_execute(exclude_query)])
    for category in list(categories_ratios.keys()):
        n_samples_category = samples_per_category[category]
        if n_samples_category == 0:
            continue
        potential_papers = sorted([paper_id for paper_id, paper_category in papers_ids_to_categories.items() if paper_category == category and paper_id not in papers_to_exclude])
        negative_samples_ids += rng.sample(potential_papers, n_samples_category)
    return sorted(negative_samples_ids)