from data_handling import get_db_backup_date, sql_execute
from random import shuffle
from scipy.sparse import save_npz
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import numpy as np
import os
import pickle
import sys

def load_vectorizer(embedding_folder : str) -> TfidfVectorizer:
    vectorizer_path = f"{embedding_folder}/vectorizer.pkl"
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer

def print_tfidf_vector(vectorizer, paper_embedding) -> str:
    feature_names = vectorizer.get_feature_names_out()
    non_zero_indices = np.nonzero(paper_embedding)[0]
    non_zero_values = paper_embedding[non_zero_indices]
    s = ""
    for index, value in zip(non_zero_indices, non_zero_values):
        s += f"{feature_names[index]}: {value}\n"
    return s

def train_vectorizer(train_corpus, max_features = 10000, max_df = 1.0, min_df = 1) -> TfidfVectorizer:
    """
    Encodes all files in 'text_paths' according to Text frequency times inverse document frequency (TFIDF).
    The ordering of the paths is preserved in the design matrix.
    :param train_corpus:
    :param corpus:
    :return: Encoded design matrix
    """
    # I took this parameterisation from Mr. Karpathy's Arxiv Sanity Preserver code at: https://github.com/karpathy/arxiv-sanity-preserver/blob/master/analyze.py
    stop_words = list(text.ENGLISH_STOP_WORDS.union(['et al', 'et', 'al']))
    v = TfidfVectorizer(input='content',
                        encoding='utf-8', decode_error='replace', strip_accents='unicode',
                        lowercase=True, analyzer='word', stop_words=stop_words,
                        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
                        ngram_range=(1, 2), max_features=max_features,
                        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                        max_df=max_df, min_df=min_df)
    v.fit(tqdm(train_corpus))
    return v

def _generate_corpus(ids, max_items = 10000):
    """
    Generator to read one text file at a time.
    :param paths:
    :return:
    """
    c = 0
    for id in ids:
        if c >= max_items:
            return
        query = "select title || '. ' || abstract from papers where paper_id = :id"
        txt = sql_execute(query, id=id)[0][0]
        c += 1
        yield txt

def load_cache_paper_ids_to_idx():
    paper_ids_idx = sql_execute("select paper_id, idx from cache_papers order by idx")
    unrated_ids_to_idx = {}
    for i, (paper_id, idx) in enumerate(paper_ids_idx):
        unrated_ids_to_idx[paper_id] = i
    return unrated_ids_to_idx

def retrain_vectorizer_celery(max_features : int = 5000) -> TfidfVectorizer:
    """
    Retrains the TFIDF vectorizer and replaces the TFIDF embeddings.
    """
    cache_paper_ids_to_idx = load_cache_paper_ids_to_idx()
    paper_ids = list(cache_paper_ids_to_idx.keys())
    shuffle(paper_ids)
    train_corpus = _generate_corpus(paper_ids, max_items=len(paper_ids))
    v = train_vectorizer(train_corpus=train_corpus, max_features = max_features, max_df = 1.0, min_df = 1)
    return v

def train_vectorizer_for_user(paper_ids : list, max_features : int = 10000) -> TfidfVectorizer:
    train_corpus = _generate_corpus(paper_ids, max_items = len(paper_ids))
    v = train_vectorizer(train_corpus = train_corpus, max_features = max_features, max_df = 1.0, min_df = 1)
    return v

def get_mean_embedding(v : TfidfVectorizer, paper_ids : list) -> np.array:
    X = v.transform(_generate_corpus(paper_ids, max_items = len(paper_ids)))
    return np.mean(X, axis = 0)

def embedding_to_word_scores(embedding : np.ndarray, feature_names) -> dict:
    return {word: score for word, score in zip(feature_names, embedding.A1)}

def recompute_embeddings_celery(v : TfidfVectorizer):
    query = "select paper_id from papers where abstract is not NULL;"
    paper_ids = [id[0] for id in sql_execute(query)]
    corpus = _generate_corpus(paper_ids, max_items=len(paper_ids))

    X = v.transform(tqdm(corpus, total=len(paper_ids), desc="Transforming corpus"))
    papers_ids_to_idx = {}
    for idx, id in enumerate(paper_ids):
        papers_ids_to_idx[id] = idx
    save_npz(f"{folder}/abs_X.npz", X, compressed = False)
    with open(f"{folder}/abs_paper_ids_to_idx.pkl", "wb") as f:
        pickle.dump(papers_ids_to_idx, f)
    with open(f"{folder}/vectorizer.pkl", "wb") as f:
        pickle.dump(v, f)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: python main.py <max_features_in_k> <embedding_folder>")
    max_features_in_k = int(sys.argv[1])
    embedding_folder = sys.argv[2]
    embedding_folder = embedding_folder if embedding_folder[-1] != "/" else embedding_folder[:-1]
    db_backup_date = get_db_backup_date()
    folder = f"{embedding_folder}/tfidf_{max_features_in_k}k_{db_backup_date}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        sys.exit(f"Folder {folder} already exists. Exiting.")
    v = retrain_vectorizer_celery(max_features = max_features_in_k * 1000)
    recompute_embeddings_celery(v)