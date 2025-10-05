import logging

import torch
from torch_geometric.utils import to_dense_batch

from ....finetuning.src.finetuning_main import log_string


def print_stds_between_tokens(
    vector_hist: torch.Tensor,
    mask_hist: torch.Tensor,
    transformer_out: torch.Tensor,
    logger: logging.Logger,
) -> None:
    largest_number_true_per_row = mask_hist.sum(dim=1).max().item()
    stds_before = vector_hist[0, :largest_number_true_per_row, :].std(dim=0)
    stds_after = transformer_out[0, :largest_number_true_per_row, :].std(dim=0)
    min_std_before = stds_before.min().item()
    max_std_before = stds_before.max().item()
    mean_std_before = stds_before.mean().item()
    s = f"Feature STDs before Transformer: Min {min_std_before:.6f}, Max {max_std_before:.6f}, Mean {mean_std_before:.6f}"
    min_std_after = stds_after.min().item()
    max_std_after = stds_after.max().item()
    mean_std_after = stds_after.mean().item()
    s += f"\nFeature STDs after Transformer: Min {min_std_after:.6f}, Max {max_std_after:.6f}, Mean {mean_std_after:.6f}"
    log_string(logger, s)


def get_batch_pos_neg(batch: dict, include_negatives: bool, required_keys: list) -> tuple:
    if include_negatives:
        pos_mask, neg_mask = batch["y_hist"].bool(), ~batch["y_hist"].bool()
        pos_batch = {k: v[pos_mask] for k, v in batch.items() if k in required_keys}
        neg_batch = {k: v[neg_mask] for k, v in batch.items() if k in required_keys}
        return pos_batch, neg_batch, neg_mask.any()
    else:
        assert batch["y_hist"].all()
        return batch, None, False


def get_hist_neg(batch_neg: dict) -> tuple:
    idxs_neg = batch_neg["batch_hist"].unique().sort()[0]
    vector_hist_neg, mask_hist_neg = to_dense_batch(batch_neg["x_hist"], batch_neg["batch_hist"])
    vector_hist_neg = vector_hist_neg[idxs_neg]
    mask_hist_neg = mask_hist_neg[idxs_neg]
    return vector_hist_neg, mask_hist_neg, idxs_neg


def get_exponential_temporal_decay(
    days_diffs_hist: torch.Tensor, batch_hist: torch.Tensor, decay_factor: float
) -> torch.Tensor:
    days_diffs_vector_hist, _ = to_dense_batch(days_diffs_hist, batch_hist)
    return torch.exp(-decay_factor * days_diffs_vector_hist)
