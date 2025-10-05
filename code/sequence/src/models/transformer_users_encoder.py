import logging

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from .attention import AdditiveAttention
from .users_encoder import UsersEncoder
from .users_encoder_utils import (
    get_batch_pos_neg,
    get_exponential_temporal_decay,
    get_hist_neg,
    print_stds_between_tokens,
)


class TransformerUsersEncoder(UsersEncoder):
    def __init__(
        self,
        include_negatives: bool,
        num_layers: int,
        num_heads: int,
        query_dim: int,
        embed_dim: int = 356,
        feedforward_factor: int = 4,
        dropout: float = 0.1,
        temporal_args: dict = {},
    ):
        super().__init__()
        self.include_negatives = include_negatives
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.embed_dim = embed_dim
        self.feedforward_factor = feedforward_factor
        self.feedforward_dim = feedforward_factor * embed_dim
        self.dropout = dropout
        self.temporal_args = temporal_args
        assert self.embed_dim % self.num_heads == 0

        if "exponential_decay_factor" in self.temporal_args:
            self.exponential_decay_factor = self.temporal_args["exponential_decay_factor"]

        self.transformer_pos = self._create_transformer()
        self.attention_pos = AdditiveAttention(input_dim=self.embed_dim, query_dim=self.query_dim)
        if self.include_negatives:
            self.transformer_neg = self._create_transformer()
            self.attention_neg = AdditiveAttention(
                input_dim=self.embed_dim, query_dim=self.query_dim
            )
            self.projection = nn.Linear(self.embed_dim * 2, self.embed_dim)

    def _create_transformer(self) -> nn.TransformerEncoder:
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.feedforward_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(
            transformer_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.embed_dim),
            enable_nested_tensor=False,
        )

    def _encode_user(self, batch: dict, logger: logging.Logger = None) -> torch.Tensor:
        batch_pos, batch_neg, neg_mask_any = get_batch_pos_neg(
            batch, self.include_negatives, self._get_required_batch_keys()
        )
        vector_hist_pos, mask_hist_pos = to_dense_batch(
            batch_pos["x_hist"], batch_pos["batch_hist"]
        )
        transformer_out_pos = self.transformer_pos(
            vector_hist_pos, src_key_padding_mask=~mask_hist_pos
        )
        if hasattr(self, "exponential_decay_factor"):
            decay_bias_pos = get_exponential_temporal_decay(
                batch_pos["days_diffs_hist"], batch_pos["batch_hist"], self.exponential_decay_factor
            )
        else:
            decay_bias_pos = None
        vector_pos = self.attention_pos(transformer_out_pos, mask_hist_pos, decay_bias_pos, logger)
        if self.include_negatives:
            vector_neg = torch.zeros_like(vector_pos)
            if neg_mask_any:
                vector_hist_neg, mask_hist_neg, idxs_neg = get_hist_neg(batch_neg)
                transformer_out_neg = self.transformer_neg(
                    vector_hist_neg, src_key_padding_mask=~mask_hist_neg
                )
                if hasattr(self, "exponential_decay_factor"):
                    decay_bias_neg = get_exponential_temporal_decay(
                        days_diffs_hist=batch_neg["days_diffs_hist"],
                        batch_hist=batch_neg["batch_hist"],
                        decay_factor=self.exponential_decay_factor,
                    )
                else:
                    decay_bias_neg = None
                vector_neg_partial = self.attention_neg(
                    transformer_out_neg, mask_hist_neg, decay_bias_neg, logger
                )
                vector_neg[idxs_neg] = vector_neg_partial
            vector = self.projection(torch.cat([vector_pos, vector_neg], dim=1))
        else:
            vector = vector_pos
        if not hasattr(self, "counter"):
            self.counter = 0
        self.counter += 1
        if logger is not None and self.counter % 2500 == 0:
            print_stds_between_tokens(vector_hist_pos, mask_hist_pos, transformer_out_pos, logger)
        return vector

    def _get_required_batch_keys(self) -> list:
        return ["x_hist", "batch_hist", "y_hist", "days_diffs_hist"]

    def get_config(self) -> dict:
        config = {
            "users_encoder_type": "TransformerUsersEncoder",
            "include_negatives": self.include_negatives,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "query_dim": self.query_dim,
            "embed_dim": self.embed_dim,
            "feedforward_factor": self.feedforward_factor,
            "temporal_args": self.temporal_args,
        }
        return config
