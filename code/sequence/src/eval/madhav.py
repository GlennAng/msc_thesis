import numpy as np


def get_weights_cache(
    correction: int,
    neg_scale: float,
    cache_v: float,
    n_negrated: int,
    n_cache: int,
) -> tuple:
    cache_denom = cache_v * n_negrated + (1.0 - cache_v) * n_cache
    assert cache_denom > 0
    w_cache = correction * neg_scale * (1.0 - cache_v) / cache_denom
    return w_cache, cache_denom


def get_weights_cluster_pos(
    correction: int,
    n_posrated: int,
    n_pos_cluster_in: int,
    cluster_alpha: float,
) -> tuple:
    n_pos_cluster_out = n_posrated - n_pos_cluster_in
    assert n_pos_cluster_out >= 0
    denom = cluster_alpha * n_pos_cluster_in + (1.0 - cluster_alpha) * n_pos_cluster_out
    assert denom > 0
    w_pos_in_cluster = correction * cluster_alpha / denom
    w_pos_out_cluster = correction * (1.0 - cluster_alpha) / denom
    desired_ratio = cluster_alpha / denom
    return w_pos_in_cluster, w_pos_out_cluster, desired_ratio


def get_weights_cluster_neg_same_ratio(
    neg_scale: float,
    cache_v: float,
    n_posrated: int,
    n_negrated: int,
    n_cache: int,
    n_pos_cluster_in: int,
    n_neg_cluster_in: int,
    cluster_alpha: float,
) -> tuple:
    correction = n_posrated + n_negrated + n_cache
    w_pos_in_cluster, w_pos_out_cluster, _ = get_weights_cluster_pos(
        correction=correction,
        n_posrated=n_posrated,
        n_pos_cluster_in=n_pos_cluster_in,
        cluster_alpha=cluster_alpha,
    )
    pos_sum_in = w_pos_in_cluster * n_pos_cluster_in
    pos_sum_out = w_pos_out_cluster * (n_posrated - n_pos_cluster_in)
    desired_ratio = pos_sum_in / (pos_sum_in + pos_sum_out)
    w_cache, cache_denom = get_weights_cache(
        correction=correction,
        neg_scale=neg_scale,
        cache_v=cache_v,
        n_negrated=n_negrated,
        n_cache=n_cache,
    )
    n_neg_cluster_out = n_negrated - n_neg_cluster_in
    cache_term = (n_negrated * cache_v) / cache_denom
    if n_neg_cluster_in == 0:
        w_neg_in_cluster = 0.0
        w_neg_out_cluster = correction * neg_scale * cache_v / cache_denom
    elif n_neg_cluster_out == 0:
        w_neg_out_cluster = 0.0
        w_neg_in_cluster = correction * neg_scale * cache_v / cache_denom
    else:
        w_neg_in_cluster = correction * neg_scale * desired_ratio * cache_term / n_neg_cluster_in
        w_neg_out_cluster = (
            correction * neg_scale * (1.0 - desired_ratio) * cache_term / n_neg_cluster_out
        )
    return w_pos_in_cluster, w_neg_in_cluster, w_cache, w_pos_out_cluster, w_neg_out_cluster


def get_weights_cluster_neg_none(
    neg_scale: float,
    cache_v: float,
    n_posrated: int,
    n_negrated: int,
    n_cache: int,
    n_pos_cluster_in: int,
    cluster_alpha: float,
) -> tuple:
    correction = n_posrated + n_negrated + n_cache
    w_cache, cache_denom = get_weights_cache(
        correction=correction,
        neg_scale=neg_scale,
        cache_v=cache_v,
        n_negrated=n_negrated,
        n_cache=n_cache,
    )
    w_pos_in_cluster, w_pos_out_cluster, _ = get_weights_cluster_pos(
        correction=correction,
        n_posrated=n_posrated,
        n_pos_cluster_in=n_pos_cluster_in,
        cluster_alpha=cluster_alpha,
    )
    w_neg_in_cluster = correction * neg_scale * cache_v / cache_denom
    return w_pos_in_cluster, w_neg_in_cluster, w_cache, w_pos_out_cluster, w_neg_in_cluster


if __name__ == "__main__":
    # n_posrated is the number of positives for that user
    # => n_posrated = n_pos_cluster_in + n_pos_cluster_out
    # n_negrated is the number of negatives for that user
    # => n_negrated = n_neg_cluster_in + n_neg_cluster_out
    # correction is n_posrated + n_negrated + n_cache
    # cache_v is 0.9
    # cluster_alpha is 0.8
    # neg_scale is a value between 3.0 and 5.0 (not sure which one you use currently)
    # Example:
    n_posrated = 20
    n_negrated = 20
    n_cache = 5000
    n_pos_cluster_in = 16
    n_neg_cluster_in = 10
    cluster_alpha = 0.8
    cache_v = 0.9
    neg_scale = 4.0

    sample_weights = np.zeros(n_posrated + n_negrated + n_cache)
    w_pos_in_cluster, w_neg_in_cluster, w_cache, w_pos_out_cluster, w_neg_out_cluster = (
        get_weights_cluster_neg_none(
            neg_scale=neg_scale,
            cache_v=cache_v,
            n_posrated=n_posrated,
            n_negrated=n_negrated,
            n_cache=n_cache,
            n_pos_cluster_in=n_pos_cluster_in,
            cluster_alpha=cluster_alpha,
        )
    )
    # assuming the sample weights are stacked as: [pos_in_cluster, pos_out_cluster, neg_in_cluster, neg_out_cluster, cache]
    sample_weights[0:n_pos_cluster_in] = w_pos_in_cluster
    sample_weights[n_pos_cluster_in:n_posrated] = w_pos_out_cluster
    sample_weights[n_posrated : n_posrated + n_neg_cluster_in] = w_neg_in_cluster
    sample_weights[n_posrated + n_neg_cluster_in : n_posrated + n_negrated] = w_neg_out_cluster
    sample_weights[n_posrated + n_negrated :] = w_cache
    n_total = n_posrated + n_negrated + n_cache
    total_weights_goal = n_total * (1.0 + neg_scale)
    assert np.isclose(
        np.sum(sample_weights), total_weights_goal
    ), f"{np.sum(sample_weights)} vs {total_weights_goal}"
