import torch
from torch_geometric.utils import to_dense_batch

from .users_encoder import UsersEncoder


class MeanPoolingUsersEncoder(UsersEncoder):
    def __init__(self):
        super(MeanPoolingUsersEncoder, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _get_required_batch_keys(self) -> list:
        return ["x_hist", "batch_hist"]

    def _encode_user(self, batch: dict) -> torch.Tensor:
        hist_vector_agg, mask_hist = to_dense_batch(batch["x_hist"], batch["batch_hist"])
        user_embeddings = hist_vector_agg.sum(dim=1) / mask_hist.sum(dim=1, keepdim=True)
        return user_embeddings

    def get_config(self) -> dict:
        return {
            "users_encoder_type": "MeanPoolingUsersEncoder",
        }

    def to_device(self, device: torch.device) -> None:
        pass

    def get_device(self) -> torch.device:
        return self.device
