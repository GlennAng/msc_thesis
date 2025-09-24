import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from typing import Optional

from .attention import AdditiveAttention
from .users_encoder import UsersEncoder


class MultiLayerNRMSUsersEncoder(UsersEncoder):
    def __init__(
        self, 
        num_heads: int = 4,
        query_dim: int = 356,
        embed_dim: int = 356,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        feedforward_dim: Optional[int] = None,
        activation: str = "relu"
    ):
        """
        Enhanced multi-layer NRMS Users Encoder with additional architectural improvements.
        
        Args:
            num_heads: Number of attention heads for multihead attention
            query_dim: Dimension of query for additive attention
            embed_dim: Embedding dimension
            num_layers: Number of transformer-style layers
            dropout: Dropout probability
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
            feedforward_dim: Dimension of feedforward network (defaults to 4*embed_dim)
            activation: Activation function for feedforward network
        """
        super().__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        if feedforward_dim is None:
            feedforward_dim = 4 * embed_dim
        self.feedforward_dim = feedforward_dim
        
        # Activation function mapping
        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "swish": nn.SiLU()
        }
        self.activation = activation_map.get(activation.lower(), nn.ReLU())
        
        # Input projection (optional, if needed to match embed_dim)
        self.input_projection = nn.Linear(embed_dim, embed_dim)
        
        # Multi-layer transformer blocks
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                feedforward_dim=self.feedforward_dim,
                dropout=self.dropout,
                use_residual=self.use_residual,
                use_layer_norm=self.use_layer_norm,
                activation=self.activation
            ) for _ in range(self.num_layers)
        ])
        
        # Final additive attention for user representation
        self.additive_attention = AdditiveAttention(
            input_dim=self.embed_dim, 
            query_dim=self.query_dim
        )
        
        # Output projection and normalization
        if self.use_layer_norm:
            self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.dropout_layer = nn.Dropout(self.dropout)

    def _get_required_batch_keys(self) -> list:
        return ["x_hist", "batch_hist"]

    def _encode_user(self, batch: dict) -> torch.Tensor:
        # Convert to dense batch format
        hist_vector_agg, mask_hist = to_dense_batch(batch["x_hist"], batch["batch_hist"])
        
        # Input projection
        user_vector = self.input_projection(hist_vector_agg)
        user_vector = self.dropout_layer(user_vector)
        
        # Pass through multiple transformer layers
        for layer in self.transformer_layers:
            user_vector = layer(user_vector, key_padding_mask=~mask_hist)
        
        # Final layer normalization
        if self.use_layer_norm:
            user_vector = self.final_layer_norm(user_vector)
        
        # Apply additive attention to get final user representation
        user_representation = self.additive_attention(user_vector)
        
        return user_representation

    def get_config(self) -> dict:
        return {
            "users_encoder_type": "MultiLayerNRMSUsersEncoder",
            "num_heads": self.num_heads,
            "query_dim": self.query_dim,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_residual": self.use_residual,
            "use_layer_norm": self.use_layer_norm,
            "feedforward_dim": self.feedforward_dim,
            "activation": str(self.activation.__class__.__name__)
        }


class TransformerBlock(nn.Module):
    """
    Single transformer block with multi-head attention and feedforward network.
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        feedforward_dim: int,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        activation: nn.Module = nn.ReLU()
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        if self.use_layer_norm:
            self.ln1 = nn.LayerNorm(embed_dim)
            self.ln2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention block
        if self.use_layer_norm:
            x_norm = self.ln1(x)
        else:
            x_norm = x
            
        attn_output, _ = self.self_attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=key_padding_mask
        )
        
        # Residual connection
        if self.use_residual:
            x = x + self.dropout(attn_output)
        else:
            x = self.dropout(attn_output)
        
        # Feedforward block
        if self.use_layer_norm:
            x_norm = self.ln2(x)
        else:
            x_norm = x
            
        ff_output = self.feedforward(x_norm)
        
        # Residual connection
        if self.use_residual:
            x = x + ff_output
        else:
            x = ff_output
            
        return x


class AdvancedNRMSUsersEncoder(MultiLayerNRMSUsersEncoder):
    """
    Advanced version with adaptive attention mechanisms.
    """
    def __init__(
        self,
        num_heads: int,
        query_dim: int,
        embed_dim: int = 356,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_adaptive_attention: bool = False,
        **kwargs
    ):
        super().__init__(
            num_heads=num_heads,
            query_dim=query_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
        
        self.use_adaptive_attention = use_adaptive_attention
        
        # Adaptive attention mechanism
        if self.use_adaptive_attention:
            self.adaptive_gate = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, 1),
                nn.Sigmoid()
            )

    def _encode_user(self, batch: dict) -> torch.Tensor:
        # Convert to dense batch format
        hist_vector_agg, mask_hist = to_dense_batch(batch["x_hist"], batch["batch_hist"])
        
        # Input projection
        user_vector = self.input_projection(hist_vector_agg)
        user_vector = self.dropout_layer(user_vector)
        
        # Pass through multiple transformer layers
        for layer in self.transformer_layers:
            user_vector = layer(user_vector, key_padding_mask=~mask_hist)
        
        # Final layer normalization
        if self.use_layer_norm:
            user_vector = self.final_layer_norm(user_vector)
        
        # Adaptive attention weighting (optional)
        if self.use_adaptive_attention:
            attention_weights = self.adaptive_gate(user_vector)
            user_vector = user_vector * attention_weights
        
        # Apply additive attention to get final user representation
        user_representation = self.additive_attention(user_vector)
        
        return user_representation