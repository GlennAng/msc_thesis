import logging

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from .attention import AdditiveAttention
from .users_encoder import UsersEncoder
from .users_encoder_utils import (
    get_batch_pos_neg,
    get_hist_neg,
    print_stds_between_tokens,
)


class NRMSUsersEncoder(UsersEncoder):
    def __init__(
        self,
        include_negatives: bool,
        num_heads: int,
        query_dim: int,
        embed_dim: int = 356,
    ):
        super().__init__()
        self.include_negatives = include_negatives
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.embed_dim = embed_dim
        self.transformer_input_dim = embed_dim
        self.projection_input_dim = embed_dim
        self.output_dim = embed_dim

        self.multihead_attention_pos = nn.MultiheadAttention(
            embed_dim=self.transformer_input_dim, num_heads=self.num_heads, batch_first=True
        )
        self.additive_attention_pos = AdditiveAttention(
            input_dim=self.transformer_input_dim, query_dim=self.query_dim
        )
        if self.include_negatives:
            self.multihead_attention_neg = nn.MultiheadAttention(
                embed_dim=self.transformer_input_dim, num_heads=self.num_heads, batch_first=True
            )
            self.additive_attention_neg = AdditiveAttention(
                input_dim=self.transformer_input_dim, query_dim=self.query_dim
            )
            self.projection_input_dim = self.transformer_input_dim * 2
        if self.projection_input_dim != self.output_dim:
            self.projection = nn.Linear(self.projection_input_dim, self.output_dim)

    def _encode_user(self, batch: dict, logger: logging.Logger = None) -> torch.Tensor:
        batch_pos, batch_neg, neg_mask_any = get_batch_pos_neg(
            batch, self.include_negatives, self._get_required_batch_keys()
        )
        vector_hist_pos, mask_hist_pos = to_dense_batch(
            batch_pos["x_hist"], batch_pos["batch_hist"]
        )
        multi_attn_out_pos, _ = self.multihead_attention_pos(
            query=vector_hist_pos,
            key=vector_hist_pos,
            value=vector_hist_pos,
            key_padding_mask=~mask_hist_pos,
        )
        vector_pos = self.additive_attention_pos(
            input_vector=multi_attn_out_pos, mask_hist=mask_hist_pos, logger=logger
        )
        if self.include_negatives:
            vector_neg = torch.zeros_like(vector_pos)
            if neg_mask_any:
                vector_hist_neg, mask_hist_neg, idxs_neg = get_hist_neg(batch_neg)
                multi_attn_out_neg, _ = self.multihead_attention_neg(
                    query=vector_hist_neg,
                    key=vector_hist_neg,
                    value=vector_hist_neg,
                    key_padding_mask=~mask_hist_neg,
                )
                vector_neg_partial = self.additive_attention_neg(
                    input_vector=multi_attn_out_neg, mask_hist=mask_hist_neg, logger=logger
                )
                vector_neg[idxs_neg] = vector_neg_partial
            vector = self.projection(torch.cat([vector_pos, vector_neg], dim=1))
        else:
            vector = vector_pos
        if not hasattr(self, "counter"):
            self.counter = 0
        self.counter += 1
        if logger is not None and self.counter % 2500 == 0:
            print_stds_between_tokens(vector_hist_pos, mask_hist_pos, multi_attn_out_pos, logger)
        return vector

    def _get_required_batch_keys(self) -> list:
        return ["x_hist", "batch_hist", "y_hist", "days_diffs_hist"]

    def get_config(self) -> dict:
        config = {
            "users_encoder_type": "NRMSUsersEncoder",
            "include_negatives": self.include_negatives,
            "num_heads": self.num_heads,
            "query_dim": self.query_dim,
            "embed_dim": self.embed_dim,
        }
        return config
