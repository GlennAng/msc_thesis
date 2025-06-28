import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np


def sort_by_paper_ids(X: np.ndarray, X_papers_ids: list) -> tuple:
    """
    Sorts X and X_paper_ids by paper_ids.
    Returns the sorted X and X_paper_ids.
    """
    zipped = list(zip(X, X_papers_ids))
    zipped.sort(key=lambda x: x[1])
    X_sorted, X_papers_ids_sorted = zip(*zipped)
    X_sorted = np.array(X_sorted)
    X_papers_ids_sorted = list(X_papers_ids_sorted)
    assert X_papers_ids_sorted == sorted(
        X_papers_ids_sorted
    ), "X_papers_ids_sorted is not sorted correctly"
    return X_sorted, X_papers_ids_sorted


def save_X_as_npy_with_paper_ids_as_pkl(folder: Path, X: np.ndarray, X_papers_ids: list):
    """
    Saves the main matrix X and the corresponding paper_ids.
    """
    assert X.shape[0] == len(
        X_papers_ids
    ), f"Number of papers and number of embeddings do not match {X.shape[0]} =! {len(X_papers_ids)}"
    assert len(set(X_papers_ids)) == len(X_papers_ids), "There are duplicate paper_ids."
    embeddings_file = folder / "abs_X.npy"
    ids_file = folder / "abs_paper_ids_to_idx.pkl"
    np.save(embeddings_file, X)
    ids_dict = {paper_id: i for i, paper_id in enumerate(X_papers_ids)}
    with open(ids_file, "wb") as f:
        pickle.dump(ids_dict, f)
    os.system(f"rm -r {folder / 'batches'}")


def concatenate_all_batches_to_main_matrix_and_save(folder: Path):
    print("Start concatenating all batches to main matrix...")
    tic = time.time()
    batches_names = os.listdir(folder / "batches")
    batches_names = [batch_name for batch_name in batches_names if batch_name.startswith("batch")]
    batches, batches_ids = [], []
    for batch_name in batches_names:
        embeddings_file = folder / "batches" / batch_name / "abs_X.npy"
        ids_file = folder / "batches" / batch_name / "abs_ids.pkl"
        with open(embeddings_file, "rb") as f:
            batch_papers_embeddings_np = np.load(f)
        batches.append(batch_papers_embeddings_np)
        with open(ids_file, "rb") as f:
            batch_papers_ids = pickle.load(f)
            batches_ids.extend(batch_papers_ids)
    print(f"Read all batches, start vstacking... took {round(time.time() - tic, 5)} Seconds.")
    X = np.vstack(batches)
    assert X.shape[0] == len(
        batches_ids
    ), f"Number of papers and number of embeddings do not match {X.shape[0]} =! {len(batches_ids)}"
    print(f"Concatenated all batches to main matrix. Number of papers: {X.shape[0]}")
    if batches_ids != sorted(batches_ids):
        print("The paper_ids are not sorted. Sorting paper_ids and X")
    X, batches_ids = sort_by_paper_ids(X, batches_ids)
    save_X_as_npy_with_paper_ids_as_pkl(folder, X, batches_ids)
    print(f"Concatenation took: {round(time.time() - tic, 5)}")


embeddings_folder = sys.argv[1]
embeddings_folder = Path(embeddings_folder).resolve()
concatenate_all_batches_to_main_matrix_and_save(embeddings_folder)
