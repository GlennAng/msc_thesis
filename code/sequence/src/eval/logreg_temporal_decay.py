from enum import Enum, auto

import numpy as np


class TemporalDecay(Enum):
    NONE = auto()
    LINEAR = auto()
    EXPONENTIAL = auto()


def get_temporal_decay_from_arg(temporal_decay_arg: str) -> TemporalDecay:
    valid_temporal_decay_args = [decay.name.lower() for decay in TemporalDecay]
    if temporal_decay_arg.lower() not in valid_temporal_decay_args:
        raise ValueError(
            f"Invalid argument {temporal_decay_arg} 'temporal_decay'. Possible values: {valid_temporal_decay_args}."
        )
    return TemporalDecay[temporal_decay_arg.upper()]


class TemporalDecayNormalization(Enum):
    JOINTLY = auto()
    SEPARATELY = auto()


def get_temporal_decay_normalization_from_arg(
    temporal_decay_normalization_arg: str,
) -> TemporalDecayNormalization:
    valid_args = [norm.name.lower() for norm in TemporalDecayNormalization]
    if temporal_decay_normalization_arg.lower() not in valid_args:
        raise ValueError(
            f"Invalid argument {temporal_decay_normalization_arg} 'temporal_decay_normalization'. Possible values: {valid_args}."
        )
    return TemporalDecayNormalization[temporal_decay_normalization_arg.upper()]


def compute_temporal_decay_exponential(time_diffs: np.ndarray, decay_param: float) -> np.ndarray:
    return np.exp(-decay_param * time_diffs)


def get_sample_weights_temporal_decay_normalization_jointly(
    pos_decays: np.ndarray,
    neg_decays: np.ndarray,
    n_cache: int,
    weights_neg_scale: float,
    weights_cache_v: float,
) -> tuple:
    sum_all_decays = np.sum(pos_decays) + np.sum(neg_decays)
    pos_decays = pos_decays / sum_all_decays if pos_decays.shape[0] > 0 else pos_decays
    neg_decays = neg_decays / sum_all_decays if neg_decays.shape[0] > 0 else neg_decays
    pos_sum, neg_sum = np.sum(pos_decays),np.sum(neg_decays)
    train_negrated_n = neg_decays.shape[0]
    neg_denominator = weights_cache_v * train_negrated_n + (1.0 - weights_cache_v) * n_cache
    assert neg_denominator > 0
    neg_weight = weights_neg_scale * weights_cache_v / neg_denominator
    neg_decays = train_negrated_n * neg_weight * neg_decays
    cache_weight = weights_neg_scale * neg_sum * (1.0 - weights_cache_v) / neg_denominator
    correction = (weights_neg_scale + 1.0) / (pos_sum + weights_neg_scale * neg_sum)
    pos_decays = correction * pos_decays
    neg_decays = correction * neg_decays
    cache_weight = correction * cache_weight
    return pos_decays, neg_decays, cache_weight


def get_sample_weights_temporal_decay_normalization_separately(
    pos_decays: np.ndarray,
    neg_decays: np.ndarray,
    n_cache: int,
    weights_neg_scale: float,
    weights_cache_v: float,
) -> tuple:
    pos_decays = pos_decays / np.sum(pos_decays) if pos_decays.shape[0] > 0 else pos_decays
    neg_decays = neg_decays / np.sum(neg_decays) if neg_decays.shape[0] > 0 else neg_decays
    train_negrated_n = neg_decays.shape[0]
    neg_denominator = weights_cache_v * train_negrated_n + (1.0 - weights_cache_v) * n_cache
    assert neg_denominator > 0
    neg_weight = weights_neg_scale * weights_cache_v / neg_denominator
    neg_decays = train_negrated_n * neg_weight * neg_decays
    cache_weight = weights_neg_scale * (1.0 - weights_cache_v) / neg_denominator
    return pos_decays, neg_decays, cache_weight


def get_sample_weights_temporal_decay_by_normalization(
    temporal_decay_normalization: TemporalDecayNormalization,
    pos_decays: np.ndarray,
    neg_decays: np.ndarray,
    n_cache: int,
    weights_neg_scale: float,
    weights_cache_v: float,
) -> tuple:
    if temporal_decay_normalization == TemporalDecayNormalization.JOINTLY:
        return get_sample_weights_temporal_decay_normalization_jointly(
            pos_decays=pos_decays,
            neg_decays=neg_decays,
            n_cache=n_cache,
            weights_neg_scale=weights_neg_scale,
            weights_cache_v=weights_cache_v,
        )
    elif temporal_decay_normalization == TemporalDecayNormalization.SEPARATELY:
        return get_sample_weights_temporal_decay_normalization_separately(
            pos_decays=pos_decays,
            neg_decays=neg_decays,
            n_cache=n_cache,
            weights_neg_scale=weights_neg_scale,
            weights_cache_v=weights_cache_v,
        )
    else:
        raise ValueError(f"Invalid temporal_decay_normalization {temporal_decay_normalization}.")


def get_sample_weights_temporal_decay(
    user_train_set_ratings: np.ndarray,
    user_train_set_time_diffs: np.ndarray,
    n_cache: int,
    weights_neg_scale: float,
    weights_cache_v: float,
    temporal_decay: TemporalDecay,
    temporal_decay_normalization: TemporalDecayNormalization,
    temporal_decay_param: float,
) -> np.ndarray:
    n_total = user_train_set_ratings.shape[0] + n_cache
    sample_weights = np.empty(n_total, dtype=np.float64)
    pos_idxs = np.where(user_train_set_ratings > 0)[0]
    neg_idxs = np.where(user_train_set_ratings == 0)[0]
    assert pos_idxs.shape[0] + neg_idxs.shape[0] == user_train_set_ratings.shape[0]
    cache_idxs = np.arange(user_train_set_ratings.shape[0], n_total)

    if temporal_decay == TemporalDecay.EXPONENTIAL:
        decays = compute_temporal_decay_exponential(
            time_diffs=user_train_set_time_diffs, decay_param=temporal_decay_param
        )
    assert np.all(decays >= 0) and np.all(decays[:-1] <= decays[1:])
    pos_decays, neg_decays = decays[pos_idxs], decays[neg_idxs]

    pos_decays, neg_decays, cache_weight = get_sample_weights_temporal_decay_by_normalization(
        temporal_decay_normalization=temporal_decay_normalization,
        pos_decays=pos_decays,
        neg_decays=neg_decays,
        n_cache=n_cache,
        weights_neg_scale=weights_neg_scale,
        weights_cache_v=weights_cache_v,
    )
    sample_weights[pos_idxs] = pos_decays
    sample_weights[neg_idxs] = neg_decays
    sample_weights[cache_idxs] = cache_weight
    sample_weights = sample_weights * n_total
    return sample_weights
