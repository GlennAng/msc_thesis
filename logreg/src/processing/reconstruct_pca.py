EMBEDDINGS_INPUT_FOLDER = "/home/scholar/glenn_rp/msc_thesis/data/embeddings/before_pca/gte_large_2024-10-23"
SCALER_PARAMS_PATH = "/home/scholar/glenn_rp/msc_thesis/data/embeddings/after_pca/gte_large_2024-10-23_256_test/scaler_params.npy"
PCA_COMPONENTS_PATH = "/home/scholar/glenn_rp/msc_thesis/data/embeddings/after_pca/gte_large_2024-10-23_256_test/pca_components.npy"

from apply_pca import check_row_norms, l2_normalize_rows
from embedding import Embedding
import numpy as np
import os
import time

embeddings_base_folder = EMBEDDINGS_INPUT_FOLDER.split("/embeddings")[0] + "/embeddings"
embeddings_name = EMBEDDINGS_INPUT_FOLDER.split("/")[-1]
embeddings_matrix = Embedding(EMBEDDINGS_INPUT_FOLDER, 32).matrix
print(f"Loaded embeddings matrix of shape {embeddings_matrix.shape} for PCA.")
scaler_params = np.load(SCALER_PARAMS_PATH)
pca_components = np.load(PCA_COMPONENTS_PATH)

start_time = time.time()
embeddings_matrix = (embeddings_matrix - scaler_params[0]) / scaler_params[1]
mean_check = np.allclose(np.mean(embeddings_matrix, axis = 0), 0, atol = 1e-2)
std_check = np.allclose(np.std(embeddings_matrix, axis = 0), 1, atol = 1e-2)
if not mean_check or not std_check:
    raise ValueError("The z-score normalization failed.")
print(f"Z-score normalization successful. Took {time.time() - start_time:.2f} seconds. New Shape: {embeddings_matrix.shape}.")

start_time = time.time()
embeddings_matrix = np.dot(embeddings_matrix, pca_components.T)
print(f"PCA successful. Took {time.time() - start_time:.2f} seconds. New Shape: {embeddings_matrix.shape}.")

start_time = time.time()
embeddings_matrix = l2_normalize_rows(embeddings_matrix)
if not check_row_norms(embeddings_matrix):
    raise ValueError("The L2-normalization failed.")
print(f"L2 normalization successful. Took {time.time() - start_time:.2f} seconds. New Shape: {embeddings_matrix.shape}.")

embeddings_output_folder = embeddings_base_folder + f"/after_pca/{embeddings_name}_{embeddings_matrix.shape[1]}" 
os.makedirs(embeddings_output_folder, exist_ok = True)
np.save(f"{embeddings_output_folder}/abs_X.npy", embeddings_matrix)
os.system(f"cp {EMBEDDINGS_INPUT_FOLDER + '/abs_paper_ids_to_idx.pkl'} {embeddings_output_folder + '/abs_paper_ids_to_idx.pkl'}")
os.system(f"cp {SCALER_PARAMS_PATH} {embeddings_output_folder}/scaler_params.npy")
os.system(f"cp {PCA_COMPONENTS_PATH} {embeddings_output_folder}/pca_components.npy")