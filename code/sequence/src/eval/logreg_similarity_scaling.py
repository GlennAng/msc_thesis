from enum import Enum, auto

import numpy as np

from ....finetuning.src.finetuning_compare_embeddings import compute_sims
from ....logreg.src.training.weights_handler import Weights_Handler


class SimilarityScaling(Enum):
    NONE = auto()
    SOFTMAX = auto()
    CLIP = auto()
    LINEAR = auto()
    INVERSE = auto()
    INVERSE_GAUSSIAN = auto()
    INVERSE_PERCENTILE = auto()
    INVERSE_THRESHOLD = auto()


def get_similarity_scaling_from_arg(arg: str) -> SimilarityScaling:
    arg_lower = arg.lower()
    if arg_lower == "none":
        return SimilarityScaling.NONE
    elif arg_lower == "softmax":
        return SimilarityScaling.SOFTMAX
    elif arg_lower == "clip":
        return SimilarityScaling.CLIP
    elif arg_lower == "linear":
        return SimilarityScaling.LINEAR
    elif arg_lower == "inverse":
        return SimilarityScaling.INVERSE
    elif arg_lower == "inverse_gaussian":
        return SimilarityScaling.INVERSE_GAUSSIAN
    elif arg_lower == "inverse_percentile":
        return SimilarityScaling.INVERSE_PERCENTILE
    elif arg_lower == "inverse_threshold":
        return SimilarityScaling.INVERSE_THRESHOLD
    else:
        raise ValueError(f"Unknown similarity scaling: {arg}")


class SimilarityScalingAggregation(Enum):
    MEAN = auto()
    MAX = auto()


def get_similarity_scaling_aggregation_from_arg(arg: str) -> SimilarityScalingAggregation:
    arg_lower = arg.lower()
    if arg_lower == "mean":
        return SimilarityScalingAggregation.MEAN
    elif arg_lower == "max":
        return SimilarityScalingAggregation.MAX
    else:
        raise ValueError(f"Unknown similarity scaling aggregation: {arg}")


def get_neg_sim_scores(
    user_train_set_embeddings: np.ndarray,
    pos_idxs: np.ndarray,
    neg_idxs: np.ndarray,
    agg_method: SimilarityScalingAggregation,
) -> np.ndarray:
    user_train_set_embeddings = user_train_set_embeddings[:, :256].copy()
    pos_embeddings = user_train_set_embeddings[pos_idxs]
    neg_embeddings = user_train_set_embeddings[neg_idxs]
    neg_pos_sims = compute_sims(neg_embeddings, pos_embeddings, agg=False)
    if agg_method == SimilarityScalingAggregation.MEAN:
        neg_sim_scores = np.mean(neg_pos_sims, axis=1)
    elif agg_method == SimilarityScalingAggregation.MAX:
        neg_sim_scores = np.max(neg_pos_sims, axis=1)
    else:
        raise ValueError(f"Unknown agg_method: {agg_method}")
    return neg_sim_scores


def normalize_neg_sim_scores(
    neg_sim_scores: np.ndarray, similarity_scaling: SimilarityScaling, param: float = None
) -> np.ndarray:
    if similarity_scaling == SimilarityScaling.CLIP:
        clipped = np.maximum(neg_sim_scores, 0)
        total = np.sum(clipped)
        if total == 0:
            normalized_scores = np.ones_like(clipped) / len(clipped)
        else:
            normalized_scores = clipped / total
    elif similarity_scaling == SimilarityScaling.LINEAR:
        shifted = neg_sim_scores + 1
        normalized_scores = shifted / np.sum(shifted)
    elif similarity_scaling == SimilarityScaling.SOFTMAX:
        tau = param
        exp_scores = np.exp(neg_sim_scores / tau)
        normalized_scores = exp_scores / np.sum(exp_scores)
    elif similarity_scaling == SimilarityScaling.INVERSE:
        inverted = 1 - neg_sim_scores
        inverted = np.maximum(inverted, 0)
        total = np.sum(inverted)
        if total == 0:
            normalized_scores = np.ones_like(inverted) / len(inverted)
        else:
            normalized_scores = inverted / total
    elif similarity_scaling == SimilarityScaling.INVERSE_GAUSSIAN:
        center = param
        width = 0.15
        weights = np.exp(-((neg_sim_scores - center) ** 2) / (2 * width ** 2))
        normalized_scores = weights / np.sum(weights)
    elif similarity_scaling == SimilarityScaling.INVERSE_PERCENTILE:
        exclude = param
        lower = np.percentile(neg_sim_scores, exclude)
        upper = np.percentile(neg_sim_scores, 100 - exclude)
        mask = (neg_sim_scores >= lower) & (neg_sim_scores <= upper)
        weights = np.zeros_like(neg_sim_scores)
        weights[mask] = 1.0
        normalized_scores = weights / np.sum(weights)
    elif similarity_scaling == SimilarityScaling.INVERSE_THRESHOLD:
        lower = 0.2
        mask = neg_sim_scores >= lower
        n_mask = np.sum(mask)
        if n_mask == 0:
            normalized_scores = np.ones_like(neg_sim_scores) / len(neg_sim_scores)
        else:
            normalized_scores = np.zeros_like(neg_sim_scores)
            normalized_scores[mask] = 1.0 / n_mask
    elif similarity_scaling == SimilarityScaling.NONE:
        normalized_scores = neg_sim_scores / np.sum(neg_sim_scores)
    else:
        raise ValueError(f"Unknown similarity scaling: {similarity_scaling}")
    return normalized_scores * len(neg_sim_scores)


def get_sample_weights_similarity_scaling(
    user_train_set_embeddings: np.ndarray,
    user_train_set_ratings: np.ndarray,
    n_cache: int,
    hyperparameters: dict,
    hyperparameters_combination: tuple,
    similarity_scaling: SimilarityScaling,
    agg_method: SimilarityScalingAggregation,
    param: float = None,
) -> np.ndarray:
    n_total = user_train_set_ratings.shape[0] + n_cache
    sample_weights = np.empty(n_total, dtype=np.float64)
    pos_idxs = np.where(user_train_set_ratings == 1)[0]
    neg_idxs = np.where(user_train_set_ratings == 0)[0]
    assert pos_idxs.shape[0] + neg_idxs.shape[0] == user_train_set_ratings.shape[0]
    cache_idxs = np.arange(user_train_set_ratings.shape[0], n_total)
    train_posrated_n = pos_idxs.shape[0]
    train_negrated_n = neg_idxs.shape[0]

    neg_sim_scores = get_neg_sim_scores(
        user_train_set_embeddings=user_train_set_embeddings,
        pos_idxs=pos_idxs,
        neg_idxs=neg_idxs,
        agg_method=agg_method,
    )
    neg_weights_normalized = normalize_neg_sim_scores(
        neg_sim_scores=neg_sim_scores,
        similarity_scaling=similarity_scaling,
        param=param,
    )
    wh = Weights_Handler(config={"weights": "global:cache_v"})
    w_p, w_n, _, _, w_c = wh.load_weights_for_user(
        hyperparameters=hyperparameters,
        hyperparameters_combination=hyperparameters_combination,
        train_posrated_n=train_posrated_n,
        train_negrated_n=train_negrated_n,
        cache_n=n_cache,
    )
    sample_weights[pos_idxs] = w_p
    sample_weights[neg_idxs] = w_n * neg_weights_normalized
    sample_weights[cache_idxs] = w_c
    return sample_weights
