import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[3]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_src_paths_to_sys()

import numpy as np, random
from enum import Enum, auto

from data_handling import *
from embedding import Embedding
LABEL_DTYPE = np.int64

def load_base_for_user(embedding : Embedding, user_id : int) -> tuple:
    base_ids = get_base_papers_ids_for_user(user_id)
    base_idxs = embedding.get_idxs(base_ids)
    base_n = len(base_idxs)
    y_base = np.ones(base_n, dtype = LABEL_DTYPE)
    return base_ids, base_idxs, base_n, y_base

def load_zerorated_for_user(embedding : Embedding, user_id : int) -> tuple:
    zerorated_ids = get_rated_papers_ids_for_user(user_id, 0)
    zerorated_idxs = embedding.get_idxs(zerorated_ids)
    zerorated_n = len(zerorated_idxs)
    y_zerorated = np.zeros(zerorated_n, dtype = LABEL_DTYPE)
    return zerorated_ids, zerorated_idxs, zerorated_n, y_zerorated

def load_global_cache(embedding : Embedding, max_cache : int = None, random_state : int = None) -> tuple:
    global_cache_ids = get_global_cache_papers_ids(max_cache, random_state)
    global_cache_idxs = embedding.get_idxs(global_cache_ids)
    global_cache_n = len(global_cache_idxs)
    y_global_cache = np.zeros(global_cache_n, dtype = LABEL_DTYPE)
    return global_cache_ids, global_cache_idxs, global_cache_n, y_global_cache

def load_filtered_cache_for_user(embedding : Embedding, user_id : int, max_cache : int, random_state : int, pos_n : int, negrated_n : int) -> tuple:
    user_filtered_cache_ids = get_cache_papers_ids_for_user(user_id, max_cache, random_state)
    user_filtered_cache_idxs = embedding.get_idxs(user_filtered_cache_ids)
    user_filtered_cache_n = len(user_filtered_cache_idxs)
    y_user_filtered_cache = np.zeros(user_filtered_cache_n, dtype = LABEL_DTYPE)
    return user_filtered_cache_ids, user_filtered_cache_idxs, user_filtered_cache_n, y_user_filtered_cache

def load_negative_samples_embeddings(embedding : Embedding, n_negative_samples : int, random_state : int, papers_to_exclude : list = None) -> tuple:
    negative_samples_ids = get_negative_samples_ids(n_negative_samples, random_state, papers_to_exclude = papers_to_exclude)
    return negative_samples_ids, embedding.matrix[embedding.get_idxs(negative_samples_ids)]

def load_negrated_ranking_ids_for_user(negrated_ids : list, random_state : int) -> list:
    min_n_negrated = min(4, len(negrated_ids))
    negrated_ids = sorted(negrated_ids)
    random.seed(random_state)
    random.shuffle(negrated_ids)
    negrated_ids = negrated_ids[:min_n_negrated]
    return sorted(negrated_ids)
   
def load_training_data_for_user(embedding : Embedding, include_base : bool, include_zerorated : bool, include_cache : bool, train_rated_idxs : np.ndarray, y_train_rated : np.ndarray, 
                                base_idxs : np.ndarray, y_base : np.ndarray, zerorated_idxs : np.ndarray, y_zerorated : np.ndarray, cache_idxs : np.ndarray, y_cache : np.ndarray) -> tuple:
    X_idxs, y = train_rated_idxs, y_train_rated
    if include_base:
        X_idxs = np.concatenate((X_idxs, base_idxs))
        y = np.concatenate((y, y_base))
    if include_zerorated:
        X_idxs = np.concatenate((X_idxs, zerorated_idxs))
        y = np.concatenate((y, y_zerorated))
    if include_cache:
        X_idxs = np.concatenate((X_idxs, cache_idxs))
        y = np.concatenate((y, y_cache))
    X_train = embedding.matrix[X_idxs]
    return X_train, y