import os
import pickle
import sys
from pathlib import Path
from random import shuffle

import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from ....src.load_files import load_papers, load_papers_texts


def load_vectorizer(embedding_folder: Path) -> TfidfVectorizer:
    if not isinstance(embedding_folder, Path):
        embedding_folder = Path(embedding_folder).resolve()
    vectorizer_path = embedding_folder / "vectorizer.pkl"
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


def train_vectorizer(train_corpus, max_features=10000, max_df=1.0, min_df=1) -> TfidfVectorizer:
    """
    Encodes all files in 'text_paths' according to Text frequency times inverse document frequency (TFIDF).
    The ordering of the paths is preserved in the design matrix.
    :param train_corpus:
    :param corpus:
    :return: Encoded design matrix
    """
    # I took this parameterisation from Mr. Karpathy's Arxiv Sanity Preserver code at: https://github.com/karpathy/arxiv-sanity-preserver/blob/master/analyze.py
    stop_words = list(text.ENGLISH_STOP_WORDS.union(["et al", "et", "al"]))
    v = TfidfVectorizer(
        input="content",
        encoding="utf-8",
        decode_error="replace",
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        stop_words=stop_words,
        token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b",
        ngram_range=(1, 2),
        max_features=max_features,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        max_df=max_df,
        min_df=min_df,
    )
    v.fit(tqdm(train_corpus))
    return v


def _generate_corpus(ids, max_items=10000, require_abstract: bool = False):
    """
    Generator to read one text file at a time.
    :param paths:
    :return:
    """
    papers_texts = load_papers_texts(relevant_columns=["paper_id", "title", "abstract"])
    if require_abstract:
        papers_texts = papers_texts[papers_texts["abstract"].notnull()]
    papers_texts = papers_texts.set_index("paper_id")
    c = 0
    for id in ids:
        if c >= max_items:
            return
        row = papers_texts.loc[id]
        combined_text = row["title"] + ". " + row["abstract"]
        c += 1
        yield combined_text


def load_cache_paper_ids_to_idx():
    papers = load_papers(relevant_columns=["paper_id", "in_cache"])
    cache_papers_ids = papers[papers["in_cache"]]["paper_id"].tolist()
    unrated_ids_to_idx = {cache_paper_id: i for i, cache_paper_id in enumerate(cache_papers_ids)}
    return unrated_ids_to_idx


def retrain_vectorizer_celery(max_features: int = 5000) -> TfidfVectorizer:
    """
    Retrains the TFIDF vectorizer and replaces the TFIDF embeddings.
    """
    cache_paper_ids_to_idx = load_cache_paper_ids_to_idx()
    paper_ids = list(cache_paper_ids_to_idx.keys())
    shuffle(paper_ids)
    train_corpus = _generate_corpus(paper_ids, max_items=len(paper_ids))
    v = train_vectorizer(train_corpus=train_corpus, max_features=max_features, max_df=1.0, min_df=1)
    return v


def train_vectorizer_for_user(paper_ids: list, max_features: int = 10000) -> TfidfVectorizer:
    train_corpus = _generate_corpus(paper_ids, max_items=len(paper_ids))
    v = train_vectorizer(train_corpus=train_corpus, max_features=max_features, max_df=1.0, min_df=1)
    return v


def get_mean_embedding(v: TfidfVectorizer, paper_ids: list) -> np.array:
    X = v.transform(_generate_corpus(paper_ids, max_items=len(paper_ids)))
    return np.mean(X, axis=0)


def embedding_to_word_scores(embedding: np.ndarray, feature_names) -> dict:
    return {word: score for word, score in zip(feature_names, embedding.A1)}


def recompute_embeddings_celery(v: TfidfVectorizer):
    papers_texts = load_papers_texts(relevant_columns=["paper_id"])
    paper_ids = papers_texts["paper_id"].tolist()
    corpus = _generate_corpus(paper_ids, max_items=len(paper_ids))

    X = v.transform(tqdm(corpus, total=len(paper_ids), desc="Transforming corpus"))
    papers_ids_to_idx = {}
    for idx, id in enumerate(paper_ids):
        papers_ids_to_idx[id] = idx
    save_npz(folder / "abs_X.npz", X, compressed=False)
    with open(folder / "abs_paper_ids_to_idx.pkl", "wb") as f:
        pickle.dump(papers_ids_to_idx, f)
    with open(folder / "vectorizer.pkl", "wb") as f:
        pickle.dump(v, f)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: python main.py <max_features_in_k> <embedding_folder>")
    max_features_in_k = int(sys.argv[1])
    embedding_folder = Path(sys.argv[2]).resolve()
    folder = embedding_folder / f"tfidf_{max_features_in_k}k_"
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        sys.exit(f"Folder {folder} already exists. Exiting.")
    papers_texts = load_papers_texts()
    v = retrain_vectorizer_celery(max_features=max_features_in_k * 1000)
    recompute_embeddings_celery(v)
