from data_handling import get_base_papers_ids_for_user
from embedding import Embedding
from enum import Enum, auto
import numpy as np

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

def load_global_cache(embedding : Embedding, max_cache : int = None, random_state : int = None) -> tuple:
    global_cache_idxs = embedding.get_global_cache_idxs(max_cache, random_state)
    global_cache_n = len(global_cache_idxs)
    y_global_cache = np.zeros(global_cache_n, dtype = LABEL_DTYPE)
    return global_cache_idxs, global_cache_n, y_global_cache

def load_filtered_cache_for_user(embedding : Embedding, cache_type : Cache_Type, user_id : int, max_cache : int, random_state : int, 
                                 pos_n : int, negrated_n : int, target_ratio : float = None) -> tuple:
    if cache_type == Cache_Type.USER_FILTERED_FILLUP:
        max_cache = max(0, max_cache - negrated_n)
    elif cache_type == Cache_Type.USER_FILTERED_BALANCE:
        pos_neg_ratio = pos_n / (pos_n + negrated_n)
        max_cache = max(0, round(pos_n / target_ratio) - pos_n - negrated_n)
    user_filtered_cache_idxs = embedding.get_cache_idxs_for_user(user_id, max_cache, random_state)
    user_filtered_cache_n = len(user_filtered_cache_idxs)
    y_user_filtered_cache = np.zeros(user_filtered_cache_n, dtype = LABEL_DTYPE)
    return user_filtered_cache_idxs, user_filtered_cache_n, y_user_filtered_cache

def load_training_data_for_user(embedding : Embedding, include_base : bool, include_cache : bool, train_rated_idxs : np.ndarray, y_train_rated : np.ndarray, 
                                base_idxs : np.ndarray, y_base : np.ndarray, cache_idxs : np.ndarray, y_cache : np.ndarray) -> tuple:
    if not include_base and not include_cache:
        X_train = embedding.matrix[train_rated_idxs]
        y_train = y_train_rated
    elif include_base and not include_cache:
        X_train = embedding.matrix[np.concatenate((train_rated_idxs, base_idxs))]
        y_train = np.concatenate((y_train_rated, y_base))
    elif not include_base and include_cache:
        X_train = embedding.matrix[np.concatenate((train_rated_idxs, cache_idxs))]
        y_train = np.concatenate((y_train_rated, y_cache))
    else:
        X_train = embedding.matrix[np.concatenate((train_rated_idxs, base_idxs, cache_idxs))]
        y_train = np.concatenate((y_train_rated, y_base, y_cache))
    return X_train, y_train