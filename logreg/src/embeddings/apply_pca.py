import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[3]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_paths_to_sys()

import argparse, os, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from embedding import Embedding
from load_files import load_papers
papers = load_papers(relevant_columns = ["paper_id", "in_cache"])
cache_papers_ids = papers[papers["in_cache"]]["paper_id"].tolist()

def check_row_norms(matrix: np.ndarray) -> bool:
    norms = np.linalg.norm(matrix, axis = 1)
    is_valid = np.isclose(norms, 1.0) | np.isclose(norms, 0.0)
    return np.all(is_valid)

def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis = 1)
    mask = norms > 0
    return np.divide(matrix, norms[:, np.newaxis], where = mask[:, np.newaxis])

APPLY_ZSCORE = True
APPLY_L2NORM = True

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_input_folder", type = str, required = True)
    parser.add_argument("--pca_dim", type = int, default = 256)
    return parser.parse_args()

if __name__ == "__main__":
    parser = parse_args()
    EMBEDDINGS_INPUT_FOLDER = Path(parser.embeddings_input_folder).resolve()
    PCA_DIM = parser.pca_dim

    embedding = Embedding(EMBEDDINGS_INPUT_FOLDER, 32)
    embeddings_matrix = embedding.matrix
    cache_papers_idxs = embedding.get_idxs(cache_papers_ids)
    cache_embeddings_matrix = embeddings_matrix[cache_papers_idxs, :]
    apply_pca = (PCA_DIM < embeddings_matrix.shape[1])
    embeddings_output_folder = ProjectPaths.logreg_embeddings_path() / "after_pca" / f"{EMBEDDINGS_INPUT_FOLDER.stem}_{PCA_DIM}"
    print(f"Loaded embeddings matrix of shape {embeddings_matrix.shape} for PCA down to {PCA_DIM} dimensions.")
    print(f"Output folder for embeddings: {embeddings_output_folder}")

    if APPLY_ZSCORE:
        print("Performing Z-score normalization..")
        start_time = time.time()
        scaler = StandardScaler()
        scaler.fit(cache_embeddings_matrix)
        embeddings_matrix = scaler.transform(embeddings_matrix)
        scaler_params = np.vstack((scaler.mean_, scaler.scale_)) # Shape: (2, n_features)
        normalized_cache = embeddings_matrix[cache_papers_idxs, :]
        mean_check = np.allclose(np.mean(normalized_cache, axis = 0), 0, atol = 1e-2)
        std_check = np.allclose(np.std(normalized_cache, axis = 0), 1, atol = 1e-2)
        if not mean_check or not std_check:
            raise ValueError("The Z-score normalization failed.")
        print(f"Z-score normalization successful. Took {time.time() - start_time:.2f} seconds. New Shape: {embeddings_matrix.shape}.")

    if apply_pca:
        print("Performing PCA..")
        start_time = time.time()
        pca = PCA(n_components = PCA_DIM, random_state = 42)
        pca.fit(cache_embeddings_matrix)
        embeddings_matrix = pca.transform(embeddings_matrix)
        pca_components = pca.components_
        pca_mean = pca.mean_
        print(f"PCA successful. Took {time.time() - start_time:.2f} seconds. New Shape: {embeddings_matrix.shape}.")
    else:
        print("Skipping PCA..")

    if APPLY_L2NORM:
        print("Performing L2-normalization..")
        start_time = time.time()
        embeddings_matrix = l2_normalize_rows(embeddings_matrix)
        if not check_row_norms(embeddings_matrix):
            raise ValueError("The L2-normalization failed.")
        print(f"L2 normalization successful. Took {time.time() - start_time:.2f} seconds. New Shape: {embeddings_matrix.shape}.")

    os.makedirs(embeddings_output_folder, exist_ok = True)
    if APPLY_ZSCORE:
        np.save(embeddings_output_folder / "scaler_params.npy", scaler_params)
    if apply_pca:
        np.save(embeddings_output_folder / "pca_components.npy", pca.components_)
        np.save(embeddings_output_folder / "pca_mean.npy", pca.mean_)

    np.save(f"{embeddings_output_folder}/abs_X.npy", embeddings_matrix)
    os.system(f"cp {EMBEDDINGS_INPUT_FOLDER / 'abs_paper_ids_to_idx.pkl'} {embeddings_output_folder / 'abs_paper_ids_to_idx.pkl'}")