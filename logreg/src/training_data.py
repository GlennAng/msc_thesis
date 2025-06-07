from data_handling import get_base_papers_ids_for_user, get_rated_papers_ids_for_user
from data_handling import get_global_cache_papers_ids, get_cache_papers_ids_for_user, get_negative_samples_ids
from embedding import Embedding
from enum import Enum, auto
import numpy as np
import random

LABEL_DTYPE = np.int32

class Cache_Type(Enum):
    GLOBAL = auto()
    USER_FILTERED = auto()
    USER_FILTERED_FILLUP = auto()
    USER_FILTERED_BALANCE = auto()

def get_cache_type_from_arg(cache_type_arg : str) -> Cache_Type:
    valid_cache_type_args = [cache_type.name.lower() for cache_type in Cache_Type]
    if cache_type_arg.lower() not in valid_cache_type_args:
        raise ValueError(f"Invalid argument {cache_type_arg} 'cache_type'. Possible values: {valid_cache_type_args}.")
    return Cache_Type[cache_type_arg.upper()]

def load_base_for_user(embedding : Embedding, user_id : int, paper_removal = None, remaining_percentage : float = None, random_state : int = None) -> tuple:
    base_ids = get_base_papers_ids_for_user(user_id, paper_removal, remaining_percentage, random_state)
    base_idxs = embedding.get_idxs(base_ids)
    base_n = len(base_idxs)
    y_base = np.ones(base_n, dtype = LABEL_DTYPE)
    return base_ids, base_idxs, base_n, y_base

def load_zerorated_for_user(embedding : Embedding, user_id : int, paper_removal = None, remaining_percentage : float = None, random_state : int = None) -> tuple:
    zerorated_ids = get_rated_papers_ids_for_user(user_id, 0, paper_removal, remaining_percentage, random_state)
    zerorated_idxs = embedding.get_idxs(zerorated_ids)
    zerorated_n = len(zerorated_idxs)
    y_zerorated = np.zeros(zerorated_n, dtype = LABEL_DTYPE)
    return zerorated_ids, zerorated_idxs, zerorated_n, y_zerorated

def load_global_cache(embedding : Embedding, max_cache : int = None, random_state : int = None, draw_cache_from_users_ratings : bool = False) -> tuple:
    global_cache_ids = get_global_cache_papers_ids(max_cache, random_state, draw_cache_from_users_ratings)
    global_cache_idxs = embedding.get_idxs(global_cache_ids)
    global_cache_n = len(global_cache_idxs)
    y_global_cache = np.zeros(global_cache_n, dtype = LABEL_DTYPE)
    return global_cache_ids, global_cache_idxs, global_cache_n, y_global_cache

def load_filtered_cache_for_user(embedding : Embedding, cache_type : Cache_Type, user_id : int, max_cache : int, random_state : int, 
                                 pos_n : int, negrated_n : int, target_ratio : float = None, draw_cache_from_users_ratings : bool = False) -> tuple:
    if cache_type == Cache_Type.USER_FILTERED_FILLUP:
        max_cache = max(0, max_cache - negrated_n)
    elif cache_type == Cache_Type.USER_FILTERED_BALANCE:
        pos_neg_ratio = pos_n / (pos_n + negrated_n)
        max_cache = max(0, round(pos_n / target_ratio) - pos_n - negrated_n)
    user_filtered_cache_ids = get_cache_papers_ids_for_user(user_id, max_cache, random_state, draw_cache_from_users_ratings)
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