
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from .users_encoder import UsersEncoder

class GRUUsersEncoder(UsersEncoder):
    def __init__(self, embed_dim: int = 356, hidden_dim: int = 356, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True, # Important: input format is (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0 # Dropout only between layers
        )

    def _get_required_batch_keys(self) -> list:
        return ["x_hist", "batch_hist"]

    def _encode_user(self, batch: dict) -> torch.Tensor:
        # Convert batched graph data to a dense tensor for RNN processing
        # hist_vector_agg shape: (batch_size, max_seq_len, embed_dim)
        # mask_hist shape: (batch_size, max_seq_len)
        hist_vector_agg, mask_hist = to_dense_batch(batch["x_hist"], batch["batch_hist"])
        
        # Get the actual sequence lengths for each user in the batch
        # This is crucial for handling variable-length sequences
        seq_lengths = mask_hist.sum(dim=1).cpu() # Must be on CPU for pack_padded_sequence

        # Pack the sequences to ignore padding during GRU computation
        packed_input = nn.utils.rnn.pack_padded_sequence(
            hist_vector_agg, seq_lengths, batch_first=True, enforce_sorted=False
        )
        
        # The GRU outputs packed_output and the final hidden state
        # We only need the final hidden state
        # h_n shape: (num_layers, batch_size, hidden_dim)
        _, h_n = self.gru(packed_input)
        
        # The final user vector is the hidden state of the last layer
        # Squeeze to remove the 'num_layers' dimension if it's 1
        # user_vector shape: (batch_size, hidden_dim)
        user_vector = h_n.squeeze(0) if self.num_layers == 1 else h_n[-1]
        
        return user_vector
    
    def get_config(self) -> dict:
        return {
            "users_encoder_type": "GRUUsersEncoder",
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }