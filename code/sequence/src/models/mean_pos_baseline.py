import argparse

import pandas as pd
import torch
import torch.nn as nn

from .paper_encoder import load_paper_encoder_initial, PaperEncoder
from .click_predictor import DotProduct

class MeanPosEncoder(nn.Module):
    def __init__(self):
        super(MeanPosEncoder, self).__init__()

    def forward(
        self, hist_pos_paper_vector: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            hist_pos_paper_vector: [batch_size, history_length, embed_dim]
            mask: [batch_size, history_length] (1 for real items, 0 for padding)
        """
        if mask is not None:
            masked_vectors = hist_pos_paper_vector * mask.unsqueeze(-1)
            summed = masked_vectors.sum(dim=1)
            counts = mask.sum(dim=1, keepdim=True).float()
            counts = counts.clamp(min=1.0)
            mean_vector = summed / counts
            return mean_vector
        else:
            return hist_pos_paper_vector.mean(dim=1)
        

class MeanPosRecommender(nn.Module):
    def __init__(self):
        super(MeanPosRecommender, self).__init__()
        self.paper_encoder = PaperEncoder()
        self.mean_pos_encoder = MeanPosEncoder()
        self.click_predictor = DotProduct()

    def forward(self, hist_pos_paper_vector: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        mean_vector = self.mean_pos_encoder(hist_pos_paper_vector, mask)
        return self.click_predictor(mean_vector)


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Mean Positives Baseline")
    parser.add_argument("--users_selection", type=str, default="all")
    parser.add_argument("--soft_constraint_max_n_train_sessions", type=int, default=None)
    parser.add_argument("--hard_constraint_min_n_train_posrated", type=int, default=10)
    return vars(parser.parse_args())


def match_both_datasets(
    seq_users_ratings: pd.DataFrame, users_ratings_filtering: pd.DataFrame
) -> pd.DataFrame:
    users_ratings_filtering = users_ratings_filtering[users_ratings_filtering["split"] == "val"]
    users_ratings_filtering = users_ratings_filtering[users_ratings_filtering["rating"] == 1]
    assert len(users_ratings_filtering) <= len(seq_users_ratings)
    matching_keys = set(
        users_ratings_filtering[["user_id", "paper_id", "time"]].itertuples(index=False, name=None)
    )
    mask = seq_users_ratings[["user_id", "paper_id", "time"]].apply(
        lambda row: tuple(row) in matching_keys, axis=1
    )
    seq_users_ratings = seq_users_ratings[mask].copy()
    assert len(users_ratings_filtering) == len(seq_users_ratings)
    return seq_users_ratings


def get_seq_users_ratings(users_selection: str) -> pd.DataFrame:
    from ....logreg.src.training.users_ratings import (
        UsersRatingsSelection,
        load_users_ratings_from_selection,
    )
    from ....src.load_files import load_users_ratings
    from ....src.project_paths import ProjectPaths

    users_ratings_filtering = load_users_ratings_from_selection(
        users_ratings_selection=UsersRatingsSelection.SESSION_BASED_FILTERING
    )
    if users_selection == "all":
        users_ids = users_ratings_filtering["user_id"].unique().tolist()
    seq_users_ratings = load_users_ratings(
        path=ProjectPaths.sequence_data_users_ratings_path(), relevant_users_ids=users_ids
    )
    seq_users_ratings = match_both_datasets(seq_users_ratings, users_ratings_filtering)
    assert seq_users_ratings["user_id"].is_monotonic_increasing
    assert seq_users_ratings["negrated_ids_yet_to_come"].apply(len).min() >= 4
    return seq_users_ratings


def process_seq_users_ratings_history(
    users_ratings: pd.DataFrame, min_n_train_posrated: int, max_n_train_sessions: int = None
) -> pd.DataFrame:
    users_ratings = users_ratings.copy()
    users_ratings["history_posrated_ids"] = None
    for idx, row in users_ratings.iterrows():
        previous_pos_ids = row["history_posrated_ids_per_session"]
        n_previous_pos_ids = [len(session_ids) for session_ids in previous_pos_ids]
        n_total = sum(n_previous_pos_ids)
        assert n_total >= min_n_train_posrated
        min_session_id = 0
        if max_n_train_sessions is not None:
            min_session_id = len(previous_pos_ids) - max_n_train_sessions
        sum_min_session_id = sum(n_previous_pos_ids[min_session_id:])
        while sum_min_session_id < min_n_train_posrated:
            min_session_id -= 1
            sum_min_session_id += n_previous_pos_ids[min_session_id]
        cat_posrated_ids = [
            paper_id
            for session_ids in previous_pos_ids[min_session_id:]
            for paper_id in session_ids
        ]
        users_ratings.at[idx, "history_posrated_ids"] = cat_posrated_ids
    assert users_ratings["history_posrated_ids"].apply(len).min() >= min_n_train_posrated
    return users_ratings


if __name__ == "__main__":
    args_dict = parse_args()
    seq_users_ratings = get_seq_users_ratings(args_dict["users_selection"])

    users_ratings = process_seq_users_ratings_history(
        users_ratings=seq_users_ratings,
        min_n_train_posrated=args_dict["hard_constraint_min_n_train_posrated"],
        max_n_train_sessions=args_dict["soft_constraint_max_n_train_sessions"],
    )
    max_n_history = users_ratings["history_posrated_ids"].apply(len).max() + 5

    device = "cuda" if torch.cuda.is_available() else "cpu"

    paper_encoder = load_paper_encoder_initial(
        device=device,
        l2_scale=0.0,
    )
    paper_encoder.eval()
    paper_encoder.unfreeze_transformer_model_layers(n_unfreeze_layers=0)


