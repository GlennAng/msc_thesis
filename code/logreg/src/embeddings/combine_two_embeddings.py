import pickle
import sys
from pathlib import Path

import numpy as np

from .embedding import Embedding
from .filter_relevant_embeddings import filter_relevant_embeddings
from ....src.load_files import load_relevant_papers_ids

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python combine_two_embeddings.py <path_to_first_embedding> <path_to_second_embedding>"
        )
        sys.exit(1)

    first_embedding_path = Path(sys.argv[1]).resolve()
    second_embedding_path = Path(sys.argv[2]).resolve()

    first_embedding = Embedding(first_embedding_path)
    second_embedding = Embedding(second_embedding_path)

    first_papers = list(first_embedding.papers_ids_to_idxs.keys())
    second_papers = list(second_embedding.papers_ids_to_idxs.keys())
    combined_papers = sorted(list(set(first_papers + second_papers)))
    combined_matrix = np.zeros((len(combined_papers), first_embedding.matrix.shape[1]))
    combined_papers_to_idxs = {}

    for idx, paper_id in enumerate(combined_papers):
        combined_papers_to_idxs[paper_id] = idx
        if paper_id in first_embedding.papers_ids_to_idxs:
            combined_matrix[idx, :] = first_embedding.matrix[
                first_embedding.papers_ids_to_idxs[paper_id], :
            ]
        if paper_id in second_embedding.papers_ids_to_idxs:
            combined_matrix[idx, :] = second_embedding.matrix[
                second_embedding.papers_ids_to_idxs[paper_id], :
            ]
    matrix_path = first_embedding_path / "abs_X.npy"
    papers_ids_to_idxs_path = first_embedding_path / "abs_paper_ids_to_idx.pkl"

    np.save(matrix_path, combined_matrix)
    with open(papers_ids_to_idxs_path, "wb") as f:
        pickle.dump(combined_papers_to_idxs, f)

    embedding = Embedding(first_embedding_path)
    filter_relevant_embeddings(
        embedding=embedding,
        papers_ids=load_relevant_papers_ids(),
        dir_path=first_embedding_path,
    )
