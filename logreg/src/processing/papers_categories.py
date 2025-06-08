import numpy as np, pickle, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[3]))
from project_paths import ProjectPaths
sys.path.append(str(ProjectPaths.logreg_data_embeddings_path()))
sys.path.append(str(ProjectPaths.logreg_src_embeddings_path()))
sys.path.append(str(ProjectPaths.logreg_src_processing_path()))

from embedding import *
from papers_categories_dicts import *

embedding_path = ProjectPaths.logreg_data_embeddings_path() / "after_pca" / "gte_large_2025-02-23_256"

def load_glove_embeddings(dim : int) -> dict:
    from tqdm import tqdm
    glove_path = ProjectPaths.logreg_data_embeddings_path() / "glove" / f"glove.6B.{dim}d.txt"
    embeddings = {}
    with open(glove_path, 'r', encoding = 'utf-8') as f:
        for line in tqdm(f, desc = "Loading GloVe embeddings"):
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype = np.float32)
            embeddings[word] = vector
    return embeddings

def normalize_embedding(glove_embedding : np.ndarray, normalization : str = "l2_unit") -> np.ndarray:
    if normalization == "none":
        return glove_embedding
    elif normalization == "l2_unit":
        return glove_embedding / np.linalg.norm(glove_embedding)
    elif normalization == "l2_proportional":
        proportionality = glove_embedding.shape[0] / (glove_embedding.shape[0] + embedding.matrix.shape[1])
        return glove_embedding / np.linalg.norm(glove_embedding) * proportionality
    elif normalization == "l2_05":
        return glove_embedding / np.linalg.norm(glove_embedding) * 0.5

def get_glove_categories_embeddings(categories_to_glove : dict, dim : int = 100, normalization : str = "l2_unit") -> dict:
    glove_embeddings = load_glove_embeddings(dim)
    glove_categories_embeddings = {None : np.zeros(dim, dtype = np.float32)}
    for category, words in categories_to_glove.items():
        glove_category_embedding = np.zeros(dim, dtype = np.float32)
        for word, weight in words.items():
            glove_category_embedding += weight * glove_embeddings[word]
        glove_categories_embeddings[category] = normalize_embedding(glove_category_embedding, normalization)
    return glove_categories_embeddings

def get_papers_ids_to_categories(papers_ids_to_categories_original : dict, original_categories_to_categories : dict) -> dict:
    papers_ids_to_categories = {}
    for paper_id, paper_category_original in papers_ids_to_categories_original.items():
        paper_category = original_categories_to_categories[paper_category_original]
        papers_ids_to_categories[paper_id] = paper_category
    return papers_ids_to_categories

def attach_papers_categories(embeddings : np.ndarray, papers_ids_to_idxs : dict, papers_categories : PapersCategories = None, 
                             dim : int = 100, normalization : str = "l2_unit", save_matrix : bool = False) -> np.ndarray:
    n_papers = embeddings.shape[0]
    if papers_categories is None:
        papers_categories = PAPERS_CATEGORIES
    with open("../data/papers_ids_to_categories_original.pkl", "rb") as file:
        papers_ids_to_categories_original = pickle.load(file)
    papers_ids_to_categories_original = {paper_id : category for paper_id, category in papers_ids_to_categories_original.items() if paper_id in papers_ids_to_idxs}
    papers_ids_to_categories = get_papers_ids_to_categories(papers_ids_to_categories_original, papers_categories.original_categories_to_categories)
    glove_categories_embeddings = get_glove_categories_embeddings(papers_categories.categories_to_glove, dim, normalization)
    glove_matrix = np.zeros((n_papers, dim), dtype = embeddings.dtype)
    for paper_id, paper_category in papers_ids_to_categories.items():
        glove_matrix[papers_ids_to_idxs[paper_id], :] = glove_categories_embeddings[paper_category]
    glove_matrix = np.concatenate((embeddings, glove_matrix), axis = 1)
    if save_matrix:
        os.makedirs(embedding_path + f"_categories_{dim}_{normalization}", exist_ok = True)
        os.system(f"cp {embedding_path}/abs_paper_ids_to_idx.pkl {embedding_path}_categories_{dim}_{normalization}/abs_paper_ids_to_idx.pkl")
        np.save(f"{embedding_path}_categories_{dim}_{normalization}/abs_X.npy", glove_matrix)
    return glove_matrix
  
if __name__ == "__main__":
    embedding = Embedding(embedding_path)
    #attach_papers_categories(embedding.matrix, embedding.papers_ids_to_idxs, save_matrix = True)