import os
import pickle

import numpy as np
from pathlib import Path

from ....src.load_files import TEST_RANDOM_STATES, VAL_RANDOM_STATE
from ....src.project_paths import ProjectPaths
from .embedding import Embedding
from .find_relevant_papers import save_relevant_papers


def filter_relevant_embeddings(
    embedding: Embedding, 
    papers_ids: list,
    dir_path: Path,
) -> None:
    assert papers_ids == sorted(papers_ids)
    assert len(papers_ids) == len(set(papers_ids))
    papers_idxs = embedding.get_idxs(papers_ids)
    embedding_filtered_matrix = embedding.matrix[papers_idxs]
    papers_ids_to_idxs = {paper_id: idx for idx, paper_id in enumerate(papers_ids)}
    os.makedirs(dir_path, exist_ok=True)
    np.save(dir_path / "abs_X.npy", embedding_filtered_matrix)
    with open(dir_path / "abs_paper_ids_to_idx.pkl", "wb") as f:
        pickle.dump(papers_ids_to_idxs, f)

if __name__ == "__main__":
    embedding_name = "gte_large_256"
    embedding_attachment = "categories_l2_unit_100"
    embedding_full = f"{embedding_name}_{embedding_attachment}"
    embedding = Embedding(ProjectPaths.logreg_embeddings_path() / "after_pca" / embedding_full)

    selections = ["val", "test", "session_based"]
    for selection in selections:
        papers_ids = save_relevant_papers(
            seeds = TEST_RANDOM_STATES + [VAL_RANDOM_STATE],
            users_selection=selection, 
            save=False,
        )
    dir_path = (
        ProjectPaths.logreg_embeddings_path()
        / "after_pca"
        / f"{embedding_name}_{selection}_{embedding_attachment}"
    )
    filter_relevant_embeddings(embedding, papers_ids, dir_path)
