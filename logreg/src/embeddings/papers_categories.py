import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[3]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_paths_to_sys()

import argparse, pickle
import numpy as np

from embedding import *
from load_files import load_papers
from papers_categories_dicts import PapersCategories

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_input_folder", type = str, required = True)
    parser.add_argument("--dim", type = int, choices = [50, 100, 200, 300], default = 100)
    return parser.parse_args()

def load_glove_embeddings(dim: int) -> dict:
    from tqdm import tqdm
    glove_path = ProjectPaths.logreg_embeddings_path() / "glove" / f"glove.6B.{dim}d.txt"
    embeddings = {}
    try:
        with open(glove_path, 'r', encoding = 'utf-8') as f:
            for line in tqdm(f, desc = "Loading GloVe embeddings"):
                parts = line.split()
                word = parts[0]
                vector = np.array(parts[1:], dtype = np.float32)
                embeddings[word] = vector
    except FileNotFoundError:
        pass
    return embeddings

def normalize_embedding(glove_embedding: np.ndarray, normalization: str = "l2_unit") -> np.ndarray:
    if normalization == "none":
        return glove_embedding
    elif normalization == "l2_unit":
        return glove_embedding / np.linalg.norm(glove_embedding)
    elif normalization == "l2_proportional":
        proportionality = glove_embedding.shape[0] / (glove_embedding.shape[0] + embedding.matrix.shape[1])
        return glove_embedding / np.linalg.norm(glove_embedding) * proportionality
    elif normalization == "l2_05":
        return glove_embedding / np.linalg.norm(glove_embedding) * 0.5

def get_glove_categories_embeddings(categories_to_glove: dict, dim: int = 100, normalization: str = "l2_unit") -> dict:
    glove_embeddings = load_glove_embeddings(dim)
    glove_categories_embeddings = {None : np.zeros(dim, dtype = np.float32)}
    for category, words in categories_to_glove.items():
        glove_category_embedding = np.zeros(dim, dtype = np.float32)
        for word, weight in words.items():
            glove_category_embedding += weight * glove_embeddings[word]
        glove_categories_embeddings[category] = normalize_embedding(glove_category_embedding, normalization)
    return glove_categories_embeddings

def get_papers_ids_to_categories(papers_ids_to_categories_original: dict, original_categories_to_categories: dict) -> dict:
    papers_ids_to_categories = {}
    for paper_id, paper_category_original in papers_ids_to_categories_original.items():
        paper_category = original_categories_to_categories[paper_category_original]
        papers_ids_to_categories[paper_id] = paper_category
    return papers_ids_to_categories

def attach_papers_categories(embeddings: np.ndarray, papers_ids_to_idxs: dict, papers_categories: PapersCategories, 
                             dim: int = 100, normalization: str = "l2_unit") -> np.ndarray:
    n_papers = embeddings.shape[0]
    papers = load_papers(ProjectPaths.data_papers_path(), relevant_columns = ["paper_id", "l1"])
    papers_ids_to_categories_original = papers.set_index("paper_id")["l1"].to_dict()
    papers_ids_to_categories_original = {paper_id: category for paper_id, category in papers_ids_to_categories_original.items() if paper_id in papers_ids_to_idxs}
    papers_ids_to_categories = get_papers_ids_to_categories(papers_ids_to_categories_original, papers_categories.original_categories_to_categories)
    glove_categories_embeddings = get_glove_categories_embeddings(papers_categories.categories_to_glove, dim, normalization)
    glove_matrix = np.zeros((n_papers, dim), dtype = embeddings.dtype)
    for paper_id, paper_category in papers_ids_to_categories.items():
        glove_matrix[papers_ids_to_idxs[paper_id], :] = glove_categories_embeddings[paper_category]
    return np.concatenate((embeddings, glove_matrix), axis = 1)
  
if __name__ == "__main__":
    args = parse_args()
    embedding_path = Path(args.embeddings_input_folder).resolve().stem
    embedding_path = ProjectPaths.logreg_embeddings_path() / "after_pca" / embedding_path
    dim, normalization = args.dim, "l2_unit"
    embedding = Embedding(embedding_path)
    from papers_categories_dicts import PAPERS_CATEGORIES
    papers_categories = PAPERS_CATEGORIES
    glove_matrix = attach_papers_categories(embedding.matrix, embedding.papers_ids_to_idxs, papers_categories, dim, normalization)
    matrix_path = embedding_path.parent / f"{embedding_path.name}_categories_{normalization}_{dim}"
    os.makedirs(matrix_path, exist_ok = True)
    np.save(matrix_path / "abs_X.npy", glove_matrix)
    os.system(f"cp {embedding_path / 'abs_paper_ids_to_idx.pkl'} {matrix_path / 'abs_paper_ids_to_idx.pkl'}")