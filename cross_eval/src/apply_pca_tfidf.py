from embedding import Embedding
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sp
import os
import sys
import time

def check_row_norms(matrix):
    """Check if all rows have unit norm or zero norm."""
    if sp.issparse(matrix):
        norms = np.sqrt(matrix.multiply(matrix).sum(axis = 1)).A1
    else:
        norms = np.linalg.norm(matrix, axis = 1)
    is_valid = np.isclose(norms, 1.0) | np.isclose(norms, 0.0)
    return np.all(is_valid)

def l2_normalize_rows(matrix):
    """L2 normalize each row of the matrix."""
    return normalize(matrix, norm = 'l2', axis = 1, copy = True)

if __name__ == "__main__":
    EMBEDDINGS_INPUT_FOLDER = "/home/scholar/glenn_rp/msc_thesis/data/embeddings/tfidf/tfidf_10k_2025-02-23"
    PCA_DIM = 1024

    embeddings_base_folder = EMBEDDINGS_INPUT_FOLDER.split("/embeddings")[0] + "/embeddings"
    embeddings_name = EMBEDDINGS_INPUT_FOLDER.split("/")[-1]
    embeddings_matrix = Embedding(EMBEDDINGS_INPUT_FOLDER, 32).matrix
    embeddings_output_folder = embeddings_base_folder + f"/tfidf/{embeddings_name}_{PCA_DIM}"
    if not sp.issparse(embeddings_matrix):
        raise ValueError("The embeddings matrix is not sparse.")
    print(f"Loaded embeddings matrix of shape {embeddings_matrix.shape} for dimension reduction to {PCA_DIM} dimensions.")

    start_time = time.time()
    print("Performing TruncatedSVD (equivalent to PCA for sparse matrices)...")
    svd = TruncatedSVD(n_components = PCA_DIM, random_state = 42)
    embeddings_matrix = svd.fit_transform(embeddings_matrix)
    print(f"SVD successful. Took {time.time() - start_time:.2f} seconds. New Shape: {embeddings_matrix.shape}.")

    print("Performing L2-normalization...")
    start_time = time.time()
    embeddings_matrix = l2_normalize_rows(embeddings_matrix)
    if not check_row_norms(embeddings_matrix):
        raise ValueError("The L2-normalization failed.")
    print(f"L2 normalization successful. Took {time.time() - start_time:.2f} seconds. New Shape: {embeddings_matrix.shape}.")

    os.makedirs(embeddings_output_folder, exist_ok = True)
    np.save(f"{embeddings_output_folder}/abs_X.npy", embeddings_matrix)
    os.system(f"cp {EMBEDDINGS_INPUT_FOLDER + '/abs_paper_ids_to_idx.pkl'} {embeddings_output_folder + '/abs_paper_ids_to_idx.pkl'}")
