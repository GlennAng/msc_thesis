from scipy import sparse
from scipy.sparse import load_npz
import numpy as np
import os
import pandas as pd
import pickle

class Embedding:
    def __init__(self, embedding_folder : str, embedding_float_precision : int = None) -> None:
        self.embedding_folder = embedding_folder if embedding_folder[-1] != '/' else embedding_folder[:-1]
        self.name = embedding_folder.split('/')[-1]
        self.papers_ids_to_idxs = self.load_papers_ids_to_idxs_dict(embedding_folder)
        self.papers_idxs_to_ids = np.zeros(len(self.papers_ids_to_idxs), dtype = np.int64)
        for paper_id, idx in self.papers_ids_to_idxs.items():
            self.papers_idxs_to_ids[idx] = paper_id
        self.papers_idxs_dtype = self.get_papers_idxs_dtype()
        self.matrix = self.load_embedding_matrix(embedding_folder, embedding_float_precision)

    def load_papers_ids_to_idxs_dict(self, embedding_folder : str) -> dict:
        file_path = embedding_folder + "/abs_paper_ids_to_idx.pkl"
        with open(file_path, 'rb') as file:
            papers_ids_to_idxs = pickle.load(file)
        if type(papers_ids_to_idxs) == dict:
            return papers_ids_to_idxs
        elif type(papers_ids_to_idxs) == list:
            if len(papers_ids_to_idxs) != len(set(papers_ids_to_idxs)):
                raise ValueError(f"Duplicate Paper_IDs in '{file_path}'.")
            return {paper_id : i for i, paper_id in enumerate(papers_ids_to_idxs)}
        
    def get_papers_idxs_dtype(self) -> np.dtype:
        N = len(self.papers_ids_to_idxs)
        if N <= np.iinfo(np.uint16).max:
            return np.uint16
        elif N <= np.iinfo(np.uint32).max:
            return np.uint32          
        else:
            return np.uint64
    
    def load_embedding_matrix(self, embedding_folder : str, embedding_float_precision : int = None) -> np.ndarray:
        file_path = embedding_folder + "/abs_X"
        if os.path.exists(file_path + ".npz"):
            file_path += ".npz"
            embedding_matrix = load_npz(file_path)
        elif os.path.exists(file_path + ".npy"):
            file_path += ".npy"
            embedding_matrix = np.load(file_path)
        if embedding_matrix.shape[0] != len(self.papers_ids_to_idxs):
            raise ValueError(f"Number of Embeddings ({embedding_matrix.shape[0]}) does not match Number of Paper_IDs ({len(self.papers_ids_to_idxs)}) in '{file_path}'.")
        if embedding_float_precision is not None:
            if embedding_float_precision == 16:
                embedding_matrix = embedding_matrix.astype(np.float16) if embedding_matrix.dtype != np.float16 else embedding_matrix
            elif embedding_float_precision == 32:
                embedding_matrix = embedding_matrix.astype(np.float32) if embedding_matrix.dtype != np.float32 else embedding_matrix
            else:
                raise ValueError(f"Unsupported Float Precision '{embedding_float_precision}'. Supported values: [16, 32].")
        self.is_sparse = sparse.isspmatrix(embedding_matrix)
        self.n_dimensions = embedding_matrix.shape[1]
        return embedding_matrix

    def get_idxs(self, papers_ids : list) -> np.ndarray:
        idxs = np.array([self.papers_ids_to_idxs[pid] for pid in papers_ids if pid in self.papers_ids_to_idxs], dtype = self.papers_idxs_dtype)
        if len(idxs) != len(papers_ids):
            print(f"Warning: {len(papers_ids)} Papers_IDs turned into {len(idxs)} Papers_IDXs during get_idxs.")
        return idxs
    
    def get_papers_ids(self, idxs : np.ndarray) -> list:
        return self.papers_idxs_to_ids[idxs].tolist()
    
    def compute_cosine_similarities(self, idxs : np.ndarray) -> np.ndarray:
        embeddings = self.matrix[idxs]
        if sparse.isspmatrix(embeddings):
            embeddings = embeddings.toarray()
        normalized = embeddings / np.linalg.norm(embeddings, axis = 1, keepdims = True)
        return np.dot(normalized, normalized.T)