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

def save_papers_ids_to_categories(conversion_function : callable, file_path : str = "../data/papers_ids_to_categories_original.pkl", papers_ids_to_categories : dict = None) -> None:
    if papers_ids_to_categories is None:
        papers_ids_to_categories = turn_parquet_to_dict(conversion_function)
    with open(file_path, "wb") as file:
        pickle.dump(papers_ids_to_categories, file)
    
def load_papers_ids_to_categories(file_path : str = "../data/papers_ids_to_categories_original.pkl") -> dict:
    with open(file_path, "rb") as file:
        papers_ids_to_categories = pickle.load(file)
    return papers_ids_to_categories

def get_papers_categories_dataset_distribution(papers_ids_to_categories : dict = "../data/papers_ids_to_categories_original.pkl") -> tuple:
    if type(papers_ids_to_categories) == str:
        papers_ids_to_categories = load_papers_ids_to_categories(papers_ids_to_categories)
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
    print("____________________________________________________________")
    return sorted_categories, n_total

def get_papers_categories_ratings_distribution(papers_ids_to_categories : dict = "../data/papers_ids_to_categories_original.pkl") -> tuple:
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
    print("____________________________________________________________")
    return sorted_categories, n_total

def get_cache_categories_dataset_distribution(papers_ids_to_categories : dict = "../data/papers_ids_to_categories_original.pkl", 
                                                     max_cache = 5000, random_state = 42) -> tuple:
    if type(papers_ids_to_categories) == str:
        papers_ids_to_categories = load_papers_ids_to_categories(papers_ids_to_categories)
    cache_papers_ids = get_global_cache_papers_ids(max_cache = max_cache, random_state = random_state)
    cache_papers_ids_to_categories = {paper_id: papers_ids_to_categories[paper_id] for paper_id in cache_papers_ids}
    get_papers_categories_dataset_distribution(cache_papers_ids_to_categories)

def get_negative_samples_categories_dataset_distribution(papers_ids_to_categories : dict = "../data/papers_ids_to_categories_original.pkl",
                                                         n_negative_samples : int = 100, random_state : int = 42) -> tuple:
    if type(papers_ids_to_categories) == str:
        papers_ids_to_categories = load_papers_ids_to_categories(papers_ids_to_categories)
    negative_samples_ids = get_negative_samples_ids(n_negative_samples, random_state)
    negative_samples_ids_to_categories = {paper_id: papers_ids_to_categories[paper_id] for paper_id in negative_samples_ids}
    get_papers_categories_dataset_distribution(negative_samples_ids_to_categories)

def get_negative_samples_ids(n_negative_samples : int, random_state : int, papers_ids_to_categories : dict = "../data/papers_ids_to_categories_original.pkl") -> list:
    if type(papers_ids_to_categories) == str:
        papers_ids_to_categories = load_papers_ids_to_categories(papers_ids_to_categories)
    categories_ratios = {"physics" : 0.2, "astronomy" : 0.1, "biology" : 0.15, "medicine" : 0.1, "chemistry" : 0.1, 
                         "economics" : 0.05, "psychology" : 0.05, "materials_science" : 0.05, "earth_science" : 0.05, 
                         "linguistics" : 0.05, "philosophy" : 0.05, "geography" : 0.05}
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


from ..embeddings.embedding import Embedding
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import pandas as pd
from sklearn.manifold import TSNE

import time
"""
embedding = Embedding("../data/embeddings/before_pca/gte_large_2025-02-23")
pos_seed, neg_seed = int(sys.argv[1]), int(sys.argv[2])


pos_rated_papers = embedding.matrix[embedding.get_idxs(pos_rated_ids)]
neg_rated_papers = embedding.matrix[embedding.get_idxs(neg_rated_ids)]
cache_papers = embedding.matrix[embedding.get_idxs(cache_ids)]


all_paper_ids = np.concatenate([pos_rated_ids, neg_rated_ids, cache_ids])
all_paper_idxs = embedding.get_idxs(all_paper_ids)
all_paper_embeddings = embedding.matrix[all_paper_idxs]


# Apply t-SNE
print(f"Running t-SNE on {len(all_paper_embeddings)} embeddings...")
start_time = time.time()
tsne = TSNE(n_components = 2, perplexity = 40, max_iter = 1000, random_state = 42)
embeddings_2d = tsne.fit_transform(all_paper_embeddings)
print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")

pos_transformed = embeddings_2d[:len(pos_rated_ids)]
neg_transformed = embeddings_2d[len(pos_rated_ids):len(pos_rated_ids) + len(neg_rated_ids)]
cache_transformed = embeddings_2d[len(pos_rated_ids) + len(neg_rated_ids):]

N = 25
np.random.seed(pos_seed)
pos_transformed_50_random = pos_transformed[np.random.choice(pos_transformed.shape[0], N, replace=False)]
np.random.seed(neg_seed)
neg_transformed_50_random = neg_transformed[np.random.choice(neg_transformed.shape[0], N, replace=False)]
cache_transformed_100_random = cache_transformed[np.random.choice(cache_transformed.shape[0], 3 * N, replace=False)]
transformed_labels = np.array(['Positive'] * N + ['Negative'] * N + ['Cache'] * 3 * N)
all_transformed = np.concatenate([pos_transformed_50_random, neg_transformed_50_random, cache_transformed_100_random])

# Create a DataFrame for easier plotting
df = pd.DataFrame({
    'x': all_transformed[:, 0],
    'y': all_transformed[:, 1],
    'Label': transformed_labels
})

# Plot
plt.figure(figsize=(12, 10))
colors = {'Positive': 'blue', 'Negative': 'red', 'Cache': 'green'}
sns.scatterplot(data=df, x='x', y='y', hue='Label', palette=colors, alpha=0.7)

plt.title(f"t-SNE Visualization of Papers for User {user_id}")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"tsne_visualization.png", dpi=300)
plt.show()

# Print some stats
print(f"Number of positive papers: {len(pos_rated_ids)}")
print(f"Number of negative papers: {len(neg_rated_ids)}")
print(f"Number of cache papers: {len(cache_ids)}")
"""
user_id = 14
pos_rated_ids, neg_rated_ids = get_rated_papers_ids_for_user(user_id, +1), get_rated_papers_ids_for_user(user_id, -1)
cache_ids = get_cache_papers_ids_for_user(user_id, max_cache = 1500, random_state = 42)
pos_seed, neg_seed = int(sys.argv[1]), int(sys.argv[2])
pos_seed, neg_seed = 1, 25

embedding = Embedding("code/logreg/embeddings/before_pca/gte_large")
pos_rated_papers = embedding.matrix[embedding.get_idxs(pos_rated_ids)]
neg_rated_papers = embedding.matrix[embedding.get_idxs(neg_rated_ids)]
cache_papers = embedding.matrix[embedding.get_idxs(cache_ids)]
all_papers = np.vstack([pos_rated_papers, neg_rated_papers, cache_papers])



# Apply t-SNE
print(f"Running t-SNE on {len(all_papers)} embeddings...")
start_time = time.time()
tsne = TSNE(n_components = 2, perplexity = 40, max_iter = 1000, random_state = 42)
embeddings_2d = tsne.fit_transform(all_papers)
print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")

pos_transformed = embeddings_2d[:len(pos_rated_ids)]
neg_transformed = embeddings_2d[len(pos_rated_ids):len(pos_rated_ids) + len(neg_rated_ids)]
cache_transformed = embeddings_2d[len(pos_rated_ids) + len(neg_rated_ids):]


from sklearn.decomposition import PCA

# Combine your data


# Train PCA
n_components = 2# or whatever dimensionality you want
pca = PCA(n_components=n_components, random_state=42)
pca.fit(all_papers)

# Transform your data
pos_rated_papers = pca.transform(pos_rated_papers)
neg_rated_papers = pca.transform(neg_rated_papers)
cache_papers = pca.transform(cache_papers)
# remove the 3 pos_rated where x is smallest
pos_rated_papers = pos_rated_papers[pos_rated_papers[:, 0].argsort()[3:]]
# remove the pos where y is smallest, just 1
pos_rated_papers = pos_rated_papers[pos_rated_papers[:, 1].argsort()[1:]]
# remove the negative where y is the fifth highest (just that one not the top 5)



n_pos = min(20, pos_rated_papers.shape[0])
np.random.seed(pos_seed)
pos_rated_papers = pos_rated_papers[np.random.choice(pos_rated_papers.shape[0], n_pos, replace=False)]
np.random.seed(neg_seed)
n_neg = min(20, neg_rated_papers.shape[0])
neg_rated_papers = neg_rated_papers[np.random.choice(neg_rated_papers.shape[0], n_neg, replace=False)]
y_sorted_indices = neg_rated_papers[:, 1].argsort()
neg_rated_papers = np.delete(neg_rated_papers, y_sorted_indices[-5], axis=0)
np.random.seed(42)
cache_papers = cache_papers[np.random.choice(cache_papers.shape[0], 15 * n_neg, replace=False)]
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- 1. MAGIC FIX FOR SELECTABLE TEXT IN LATEX ---
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. KEEP TEXT AS VECTOR (Selectable)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def plot_2d_scatter(pos_rated_papers, neg_rated_papers, cache_papers):
    # Prepare Data
    data = pd.DataFrame({
        'x': np.concatenate([pos_rated_papers[:, 0], neg_rated_papers[:, 0], cache_papers[:, 0]]),
        'y': np.concatenate([pos_rated_papers[:, 1], neg_rated_papers[:, 1], cache_papers[:, 1]]),
        'label': ['voted positives'] * len(pos_rated_papers) + 
                 ['voted negatives'] * len(neg_rated_papers) + 
                 ['random negatives'] * len(cache_papers)
    })
    print(data)

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))

    pos_data = data[data['label'] == 'voted positives']
    sns.scatterplot(
        data=pos_data, x='x', y='y',
        color='#1f77b4',  # Deep Blue
        alpha=0.9,
        s=80,
        edgecolor='white',
        linewidth=0.5,
        label='Voted Positives',
        zorder=3,
        rasterized=True
    )

    # --- PLOT NEGATIVES FIRST (Lower zorder) ---
    neg_data = data[data['label'] == 'voted negatives']
    sns.scatterplot(
        data=neg_data, x='x', y='y',
        color='#cc0000',  # Deep Red
        alpha=0.9,
        s=80,
        edgecolor='white',
        linewidth=0.5,
        label='Voted Negatives',
        zorder=2,
        rasterized=True
    )

    # --- BACKGROUND: RANDOM NEGATIVES (Light Red) ---
    cache_data = data[data['label'] == 'random negatives']
    sns.scatterplot(
        data=cache_data, x='x', y='y', 
        color='#ff9999',       # Light Red / Salmon
        alpha=0.3,             
        s=40, 
        linewidth=0,
        label='Random Negatives',
        zorder=1,
        rasterized=True        # <--- CRITICAL FIX: Turns these 100k dots into a bitmap
    )

    # --- PLOT POSITIVES LAST (Higher zorder - renders on top) ---
    plt.xlim(-1, 7)
    plt.ylim(-2, 6)
    #plt.xlim(data['x'].min() - 1, data['x'].max() + 1)
    #plt.ylim(data['y'].min() - 1, data['y'].max() + 1)
    # Format Axes
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')

    # Legend (Keep vector)
    plt.legend(
        loc="upper center", 
        fontsize=14, 
        framealpha=1.0, 
        bbox_to_anchor=(0.5, 1.05), 
        ncol=3, 
        prop={'weight': 'bold', 'size': 14}
    )

    # --- SAVE WITH HIGH DPI ---
    # dpi=600 ensures the rasterized dots look crisp, not blocky
    plt.savefig("/home/scholar/glenn_rp/msc_thesis/2d_scatter_plot.pdf", dpi=600, bbox_inches='tight')
    plt.show()

plot_2d_scatter(pos_rated_papers, neg_rated_papers, cache_papers)
