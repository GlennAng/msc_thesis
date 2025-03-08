from embedding import Embedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import sys
import time

def check_row_norms(matrix : np.ndarray) -> bool:
    norms = np.linalg.norm(matrix, axis = 1)
    is_valid = np.isclose(norms, 1.0) | np.isclose(norms, 0.0)
    return np.all(is_valid)

def l2_normalize_rows(matrix : np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis = 1)
    mask = norms > 0
    return np.divide(matrix, norms[:, np.newaxis], where = mask[:, np.newaxis])

if __name__ == "__main__":
    EMBEDDINGS_INPUT_FOLDER = "/home/scholar/glenn_rp/msc_thesis/data/embeddings/before_pca/specter2_2025-02-23"
    PCA_DIM = 768

    embeddings_base_folder = EMBEDDINGS_INPUT_FOLDER.split("/embeddings")[0] + "/embeddings"
    embeddings_name = EMBEDDINGS_INPUT_FOLDER.split("/")[-1]
    embeddings_matrix = Embedding(EMBEDDINGS_INPUT_FOLDER, 32).matrix
    apply_pca = (PCA_DIM < embeddings_matrix.shape[1])
    embeddings_output_folder = embeddings_base_folder + f"/after_pca/{embeddings_name}_{PCA_DIM}"
    print(f"Loaded embeddings matrix of shape {embeddings_matrix.shape} for PCA down to {PCA_DIM} dimensions.")

    print("Performing Z-score normalization..")
    start_time = time.time()
    scaler = StandardScaler()
    embeddings_matrix = scaler.fit_transform(embeddings_matrix)
    scaler_params = np.vstack((scaler.mean_, scaler.scale_)) # Shape: (2, n_features)
    mean_check = np.allclose(np.mean(embeddings_matrix, axis = 0), 0, atol = 1e-2)
    std_check = np.allclose(np.std(embeddings_matrix, axis = 0), 1, atol = 1e-2)
    if not mean_check or not std_check:
        raise ValueError("The z-score normalization failed.")
    print(f"Z-score normalization successful. Took {time.time() - start_time:.2f} seconds. New Shape: {embeddings_matrix.shape}.")

    if apply_pca:
        print("Performing PCA..")
        start_time = time.time()
        pca = PCA(n_components = PCA_DIM, random_state = 42)
        embeddings_matrix = pca.fit_transform(embeddings_matrix)
        pca_components = pca.components_
        print(f"PCA successful. Took {time.time() - start_time:.2f} seconds. New Shape: {embeddings_matrix.shape}.")
    else:
        print("Skipping PCA..")

    print("Performing L2-normalization..")
    start_time = time.time()
    embeddings_matrix = l2_normalize_rows(embeddings_matrix)
    if not check_row_norms(embeddings_matrix):
        raise ValueError("The L2-normalization failed.")
    print(f"L2 normalization successful. Took {time.time() - start_time:.2f} seconds. New Shape: {embeddings_matrix.shape}.")

    os.makedirs(embeddings_output_folder, exist_ok = True)
    np.save(f"{embeddings_output_folder}/scaler_params.npy", scaler_params)
    if apply_pca:
        np.save(f"{embeddings_output_folder}/pca_components.npy", pca_components)
    np.save(f"{embeddings_output_folder}/abs_X.npy", embeddings_matrix)
    os.system(f"cp {EMBEDDINGS_INPUT_FOLDER + '/abs_paper_ids_to_idx.pkl'} {embeddings_output_folder + '/abs_paper_ids_to_idx.pkl'}")