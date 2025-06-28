import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import load_npz


class Embedding:
    def __init__(self, embedding_folder: Path, embedding_float_precision: int = None) -> None:
        if not isinstance(embedding_folder, Path):
            embedding_folder = Path(embedding_folder).resolve()
        self.embedding_folder = embedding_folder
        self.name = embedding_folder.stem
        self.papers_ids_to_idxs = self.load_papers_ids_to_idxs_dict(embedding_folder)
        self.papers_idxs_to_ids = np.zeros(len(self.papers_ids_to_idxs), dtype=np.int64)
        for paper_id, idx in self.papers_ids_to_idxs.items():
            self.papers_idxs_to_ids[idx] = paper_id
        self.papers_idxs_dtype = self.get_papers_idxs_dtype()
        self.matrix = self.load_embedding_matrix(embedding_folder, embedding_float_precision)

    def load_papers_ids_to_idxs_dict(self, embedding_folder: Path) -> dict:
        file_path = embedding_folder / "abs_paper_ids_to_idx.pkl"
        with open(file_path, "rb") as file:
            papers_ids_to_idxs = pickle.load(file)
        if isinstance(papers_ids_to_idxs, dict):
            return papers_ids_to_idxs
        elif isinstance(papers_ids_to_idxs, list):
            if len(papers_ids_to_idxs) != len(set(papers_ids_to_idxs)):
                raise ValueError(f"Duplicate Paper_IDs in '{file_path}'.")
            return {paper_id: i for i, paper_id in enumerate(papers_ids_to_idxs)}

    def get_papers_idxs_dtype(self) -> np.dtype:
        N = len(self.papers_ids_to_idxs)
        if N <= np.iinfo(np.uint16).max:
            return np.uint16
        elif N <= np.iinfo(np.uint32).max:
            return np.uint32
        else:
            return np.uint64

    def load_embedding_matrix(
        self, embedding_folder: Path, embedding_float_precision: int = None
    ) -> np.ndarray:
        file_name = "abs_X"
        if os.path.exists(embedding_folder / f"{file_name}.npz"):
            embedding_matrix = load_npz(embedding_folder / f"{file_name}.npz")
        elif os.path.exists(embedding_folder / f"{file_name}.npy"):
            embedding_matrix = np.load(embedding_folder / f"{file_name}.npy")
        if embedding_matrix.shape[0] != len(self.papers_ids_to_idxs):
            raise ValueError(
                f"Number of Embeddings ({embedding_matrix.shape[0]}) does not match Number of Paper_IDs ({len(self.papers_ids_to_idxs)})."
            )
        if embedding_float_precision is not None:
            if embedding_float_precision == 16:
                embedding_matrix = (
                    embedding_matrix.astype(np.float16)
                    if embedding_matrix.dtype != np.float16
                    else embedding_matrix
                )
            elif embedding_float_precision == 32:
                embedding_matrix = (
                    embedding_matrix.astype(np.float32)
                    if embedding_matrix.dtype != np.float32
                    else embedding_matrix
                )
            else:
                raise ValueError(
                    f"Unsupported Float Precision '{embedding_float_precision}'. Supported values: [16, 32]."
                )
        self.is_sparse = sparse.isspmatrix(embedding_matrix)
        self.n_dimensions = embedding_matrix.shape[1]
        return embedding_matrix

    def get_idxs(self, papers_ids: list) -> np.ndarray:
        idxs = np.array(
            [self.papers_ids_to_idxs[pid] for pid in papers_ids if pid in self.papers_ids_to_idxs],
            dtype=self.papers_idxs_dtype,
        )
        if len(idxs) != len(papers_ids):
            print(
                f"Warning: {len(papers_ids)} Papers_IDs turned into {len(idxs)} Papers_IDxs during get_idxs."
            )
        return idxs

    def get_papers_ids(self, idxs: np.ndarray) -> list:
        return self.papers_idxs_to_ids[idxs].tolist()


def compute_cosine_similarities(
    embedding_folder: Path, users_ids: list, predictions_folder: Path
) -> None:
    from ....src.load_files import load_users_ratings

    if not isinstance(embedding_folder, Path):
        embedding_folder = Path(embedding_folder).resolve()
    if not isinstance(predictions_folder, Path):
        predictions_folder = Path(predictions_folder).resolve()
    users_ratings = load_users_ratings()
    embedding = Embedding(embedding_folder)
    for user_id in users_ids:
        user_ratings = users_ratings[users_ratings["user_id"] == user_id].reset_index(drop=True)
        posrated_ids = user_ratings[user_ratings["rating"] == 1]["paper_id"].tolist()
        negrated_ids = user_ratings[user_ratings["rating"] == 0]["paper_id"].tolist()
        user_predictions = json.load(
            open(predictions_folder / f"user_{user_id}" / "user_predictions.json", "r")
        )
        negative_samples_ids = user_predictions["negative_samples_ids"]
        rated_ids = posrated_ids + negrated_ids
        rated_matrix = embedding.matrix[embedding.get_idxs(rated_ids)]
        negative_samples_matrix = embedding.matrix[embedding.get_idxs(negative_samples_ids)]
        if sparse.isspmatrix(rated_matrix):
            rated_matrix = rated_matrix.toarray()
        if sparse.isspmatrix(negative_samples_matrix):
            negative_samples_matrix = negative_samples_matrix.toarray()
        rated_matrix = rated_matrix / np.linalg.norm(rated_matrix, axis=1, keepdims=True)
        negative_samples_matrix = negative_samples_matrix / np.linalg.norm(
            negative_samples_matrix, axis=1, keepdims=True
        )
        rated_cosine_similarities = np.dot(rated_matrix, rated_matrix.T)
        negative_samples_cosine_similarities = np.dot(negative_samples_matrix, rated_matrix.T)
        user_folder = embedding_folder / "cosine_similarities" / f"user_{user_id}"
        if not os.path.exists(user_folder):
            os.makedirs(user_folder, exist_ok=True)
        np.save(user_folder / "rated_cosine_similarities.npy", rated_cosine_similarities)
        np.save(
            user_folder / "negative_samples_cosine_similarities.npy",
            negative_samples_cosine_similarities,
        )
        with open(user_folder / "posrated_ids.pkl", "wb") as f:
            pickle.dump(posrated_ids, f)
        with open(user_folder / "negrated_ids.pkl", "wb") as f:
            pickle.dump(negrated_ids, f)
        with open(user_folder / "negative_samples_ids.pkl", "wb") as f:
            pickle.dump(negative_samples_ids, f)
