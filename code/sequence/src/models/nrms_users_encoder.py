import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from .attention import AdditiveAttention
from .users_encoder import UsersEncoder


class NRMSUsersEncoder(UsersEncoder):
    def __init__(self, num_heads: int, query_dim: int, embed_dim: int = 356):
        super().__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.embed_dim = embed_dim
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True
        )
        self.additive_attention = AdditiveAttention(input_dim=self.embed_dim, query_dim=self.query_dim)

    def _get_required_batch_keys(self) -> list:
        return ["x_hist", "batch_hist"]

    def _encode_user(self, batch: dict) -> torch.Tensor:
        hist_vector_agg, mask_hist = to_dense_batch(batch["x_hist"], batch["batch_hist"])
        user_vector, _ = self.multihead_attention(
            query=hist_vector_agg,
            key=hist_vector_agg,
            value=hist_vector_agg,
            key_padding_mask=~mask_hist,
        )
        user_vector = self.additive_attention(user_vector)
        return user_vector
    
    def get_config(self) -> dict:
        return {
            "users_encoder_type": "NRMSUsersEncoder",
            "num_heads": self.num_heads,
            "query_dim": self.query_dim,
            "embed_dim": self.embed_dim,
        }
