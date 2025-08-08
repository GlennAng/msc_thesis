import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ...logreg.src.embeddings.embedding import Embedding
from ...src.load_files import load_papers
from ...src.project_paths import ProjectPaths


def compute_sims(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> float:
    if embeddings_a.shape[0] == 0 or embeddings_b.shape[0] == 0:
        return 0.0
    sims = cosine_similarity(embeddings_a, embeddings_b)
    return np.mean(sims)


def compute_sims_same_set(embeddings: np.ndarray) -> float:
    if embeddings.shape[0] == 0:
        return 0.0
    sims = cosine_similarity(embeddings, embeddings)
    return np.mean(sims[np.triu_indices_from(sims, k=1)])


"""
def get_categories_word_embeddings(papers: pd.DataFrame, embeddings: Embedding, categories: ) -> dict:
    categories_word_embeddings = {}
    for category in categories_l1:
        category_papers_ids = papers[papers["l1"] == category]["paper_id"].tolist()
        category_papers_idxs = embeddings.get_idxs(category_papers_ids)
        category_embeddings = embeddings.matrix[category_papers_idxs, -100:].mean(axis=0)
        categories_word_embeddings[category] = category_embeddings
    return categories_word_embeddings
"""


def get_lengths_of_word_embeddings(categories_word_embeddings: dict) -> dict:
    lengths = {}
    for category, embedding in categories_word_embeddings.items():
        lengths[category] = np.linalg.norm(embedding)
    return lengths


def get_sims_between_word_embeddings(
    reference_category: str, categories_word_embeddings: dict
) -> dict:
    reference_embedding = categories_word_embeddings[reference_category].reshape(1, -1)
    sims = {}
    for category, embedding in categories_word_embeddings.items():
        embedding = embedding.reshape(1, -1)
        sim = compute_sims(reference_embedding, embedding)
        sims[category] = sim
    return sims


def get_sims_between_categories_same_set(embeddings: dict, papers: pd.DataFrame) -> dict:
    sims = {}
    for category in categories_l1:
        category_papers_ids = papers[papers["l1"] == category]["paper_id"].tolist()
        if len(category_papers_ids) > 10000:
            category_papers_ids = category_papers_ids[:10000]
        category_papers_idxs = embeddings.get_idxs(category_papers_ids)
        category_embeddings = embeddings.matrix[category_papers_idxs]
        sim = compute_sims_same_set(category_embeddings)
        sims[category] = sim
    return sims


def get_l2_sims(embeddings: Embedding, papers: pd.DataFrame) -> dict:
    l2_sims = {}
    for category in categories_l1:
        categories_l2_sims = []
        categories_l2 = papers[papers["l1"] == category]["l2"].unique().tolist()
        for l2_category in categories_l2:
            category_papers_ids = papers[
                (papers["l1"] == category) & (papers["l2"] == l2_category)
            ]["paper_id"].tolist()
            if len(category_papers_ids) > 10000:
                category_papers_ids = category_papers_ids[:10000]
            if len(category_papers_ids) < 2:
                continue
            category_papers_idxs = embeddings.get_idxs(category_papers_ids)
            category_embeddings = embeddings.matrix[category_papers_idxs]
            sim = compute_sims_same_set(category_embeddings)
            categories_l2_sims.append(sim)
        l2_sims[category] = np.mean(categories_l2_sims) if categories_l2_sims else 0.0
    return l2_sims


def get_sims_between_categories(
    reference_category: str, embeddings: Embedding, papers: pd.DataFrame
) -> dict:
    reference_papers_ids = papers[papers["l1"] == reference_category]["paper_id"].tolist()
    if len(reference_papers_ids) > 10000:
        reference_papers_ids = reference_papers_ids[:10000]
    reference_papers_idxs = embeddings.get_idxs(reference_papers_ids)
    reference_embeddings = embeddings.matrix[reference_papers_idxs]
    sims = {}
    for category in tqdm(categories_l1):
        category_papers_ids = papers[papers["l1"] == category]["paper_id"].tolist()
        if len(category_papers_ids) > 10000:
            category_papers_ids = category_papers_ids[:10000]
        category_papers_idxs = embeddings.get_idxs(category_papers_ids)
        category_embeddings = embeddings.matrix[category_papers_idxs]
        sim = compute_sims(reference_embeddings, category_embeddings)
        sims[category] = sim
    return sims


def print_scores(title: str, scores_before: dict, scores_after: dict) -> None:
    print("\n----------------------")
    print(title)
    for categories in scores_before:
        score_before = scores_before[categories]
        score_after = scores_after[categories]
        print(
            f"{categories}: {score_before:.4f} -> {score_after:.4f} ({score_after - score_before:.4f})"
        )


if __name__ == "__main__":
    np.random.seed(42)

    if len(sys.argv) <= 1:
        print("Usage: python finetuning_compare_embeddings.py <path_to_finetuning_embedding>")
        sys.exit(1)
    embeddings_before = Embedding(ProjectPaths.finetuning_data_model_path() / "embeddings")
    embeddings_after = Embedding(sys.argv[1])

    papers_ids = list(embeddings_before.papers_ids_to_idxs.keys())
    assert set(papers_ids) <= set(embeddings_after.papers_ids_to_idxs.keys())
    papers = load_papers(
        relevant_papers_ids=papers_ids,
        relevant_columns=["paper_id", "l1", "l2"],
    )
    categories_l1 = papers["l1"].unique().tolist()

    categories_word_embeddings_before = get_categories_word_embeddings(embeddings_before)
    categories_word_embeddings_after = get_categories_word_embeddings(embeddings_after)

    sims_between_word_embeddings_cs_before = get_sims_between_word_embeddings(
        "Computer Science", categories_word_embeddings_before
    )
    sims_between_word_embeddings_cs_after = get_sims_between_word_embeddings(
        "Computer Science", categories_word_embeddings_after
    )
    print_scores(
        "Word Embeddings Similarity (CS)",
        sims_between_word_embeddings_cs_before,
        sims_between_word_embeddings_cs_after,
    )

    sims_between_word_embeddings_physics_before = get_sims_between_word_embeddings(
        "Physics", categories_word_embeddings_before
    )
    sims_between_word_embeddings_physics_after = get_sims_between_word_embeddings(
        "Physics", categories_word_embeddings_after
    )
    print_scores(
        "Word Embeddings Similarity (Physics)",
        sims_between_word_embeddings_physics_before,
        sims_between_word_embeddings_physics_after,
    )

    sims_between_categories_same_set_before = get_sims_between_categories_same_set(
        embeddings_before, papers
    )
    sims_between_categories_same_set_after = get_sims_between_categories_same_set(
        embeddings_after, papers
    )
    print_scores(
        "Categories Similarity (Same Set)",
        sims_between_categories_same_set_before,
        sims_between_categories_same_set_after,
    )

    sims_l2_before = get_l2_sims(embeddings_before, papers)
    sims_l2_after = get_l2_sims(embeddings_after, papers)
    print_scores("L2 Similarity", sims_l2_before, sims_l2_after)

    sims_between_categories_cs_before = get_sims_between_categories(
        "Computer Science", embeddings_before, papers
    )
    sims_between_categories_cs_after = get_sims_between_categories(
        "Computer Science", embeddings_after, papers
    )
    print_scores(
        "Categories Similarity (CS)",
        sims_between_categories_cs_before,
        sims_between_categories_cs_after,
    )

    sims_between_categories_physics_before = get_sims_between_categories(
        "Physics", embeddings_before, papers
    )
    sims_between_categories_physics_after = get_sims_between_categories(
        "Physics", embeddings_after, papers
    )
    print_scores(
        "Categories Similarity (Physics)",
        sims_between_categories_physics_before,
        sims_between_categories_physics_after,
    )
