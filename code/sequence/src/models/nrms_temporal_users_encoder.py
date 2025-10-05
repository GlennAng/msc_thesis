import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from .attention import AdditiveAttention
from .users_encoder import UsersEncoder


class NRMSTemporalUsersEncoder(UsersEncoder):
    def __init__(
        self,
        num_heads: int,
        query_dim: int,
        embed_dim: int = 356,
        include_negatives: bool = False,
        exponential_decay_factor: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.embed_dim = embed_dim
        self.transformer_input_dim = embed_dim
        self.projection_input_dim = embed_dim
        self.output_dim = embed_dim
        self.include_negatives = include_negatives
        self.exponential_decay_factor = exponential_decay_factor

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

    def _get_required_batch_keys(self) -> list:
        return ["x_hist", "batch_hist", "y_hist", "days_diffs_hist"]


    def _encode_user(self, batch: dict) -> torch.Tensor:
        
        if self.include_negatives:
            pos_mask, neg_mask = batch["y_hist"].bool(), ~batch["y_hist"].bool()
            pos_batch = {k: v[pos_mask] for k, v in batch.items() if k in self._get_required_batch_keys()}
            neg_batch = {k: v[neg_mask] for k, v in batch.items() if k in self._get_required_batch_keys()}
        else:
            pos_batch = batch
        
        pos_vector_agg, pos_mask_hist = to_dense_batch(pos_batch["x_hist"], pos_batch["batch_hist"])
        user_vector, _ = self.multihead_attention_pos(
            query=pos_vector_agg,
            key=pos_vector_agg,
            value=pos_vector_agg,
            key_padding_mask=~pos_mask_hist,
        )
        # pos_vector_agg might be [9, 74, 356] for 9 candidates, 74 history size, 356 embedding dim
        # pos_mask_hist might be [9, 74] with True/False values (False at the ended positions)
        # user vector again [9, 74, 356]

        if self.exponential_decay_factor > 0.0:
            pos_days_diffs_agg, _ = to_dense_batch(pos_batch["days_diffs_hist"], pos_batch["batch_hist"])
            decay_bias = torch.exp(-self.exponential_decay_factor * pos_days_diffs_agg)
        else:
            decay_bias = None
        user_vector = self.additive_attention_pos(user_vector, pos_mask_hist, decay_bias=decay_bias)

        if self.include_negatives:
            num_users = user_vector.shape[0]
            device = user_vector.device
            if neg_mask.any():
                neg_user_ids_original = neg_batch["batch_hist"].unique().sort()[0]
                neg_vector_agg, neg_mask_hist = to_dense_batch(neg_batch["x_hist"], neg_batch["batch_hist"])
                neg_vector_agg = neg_vector_agg[neg_user_ids_original]
                neg_mask_hist = neg_mask_hist[neg_user_ids_original]
                
                
                neg_user_vector_partial, _ = self.multihead_attention_neg(
                    query=neg_vector_agg,
                    key=neg_vector_agg,
                    value=neg_vector_agg,
                    key_padding_mask=~neg_mask_hist,
                )
                if self.exponential_decay_factor > 0.0:
                    neg_days_diffs_agg, _ = to_dense_batch(neg_batch["days_diffs_hist"], neg_batch["batch_hist"])
                    neg_days_diffs_agg = neg_days_diffs_agg[neg_user_ids_original]
                    neg_decay_bias = torch.exp(-self.exponential_decay_factor * neg_days_diffs_agg)
                else:
                    neg_decay_bias = None
                neg_user_vector_partial = self.additive_attention_neg(neg_user_vector_partial, neg_mask_hist, decay_bias=neg_decay_bias)
                neg_user_vector = torch.zeros(num_users, self.transformer_input_dim, device=device)
                neg_user_vector[neg_user_ids_original] = neg_user_vector_partial
            else:
                neg_user_vector = torch.zeros(num_users, self.transformer_input_dim, device=device)
            user_vector = torch.cat([user_vector, neg_user_vector], dim=-1)
        if hasattr(self, "projection"):
            user_vector = self.projection(user_vector)
        return user_vector

    def get_config(self) -> dict:
        config = {
            "users_encoder_type": "NRMSTemporalUsersEncoder",
            "num_heads": self.num_heads,
            "query_dim": self.query_dim,
            "embed_dim": self.embed_dim,
            "include_negatives": self.include_negatives,
            "exponential_decay_factor": self.exponential_decay_factor,
        }
        return config