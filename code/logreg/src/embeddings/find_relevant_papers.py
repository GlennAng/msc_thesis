from ....sequence.src.eval.compute_users_embeddings_logreg import (
    LOGREG_CACHE_TYPE,
    LOGREG_N_CACHE,
    LOGREG_N_VAL_NEGATIVE_SAMPLES,
)
from ....src.load_files import load_papers
from ..training.training_data import get_cache_papers_ids, get_categories_samples_ids
from ..training.users_ratings import (
    UsersRatingsSelection,
    load_users_ratings_from_selection,
)


def find_relevant_papers_ids(
    users_ratings_selection: UsersRatingsSelection, seeds: list, include_old_cache: bool = True
) -> list:
    users_ratings = load_users_ratings_from_selection(users_ratings_selection)
    papers = load_papers(relevant_columns=["paper_id", "in_ratings", "in_cache", "l1"])
    relevant_papers_ids = set(users_ratings["paper_id"].values)

    for random_state in seeds:
        val_negative_samples_ids = get_categories_samples_ids(
            papers=papers,
            n_categories_samples=LOGREG_N_VAL_NEGATIVE_SAMPLES,
            random_state=random_state,
            papers_ids_to_exclude=papers[papers["in_ratings"]]["paper_id"].tolist(),
        )[1]
        relevant_papers_ids.update(val_negative_samples_ids)
        cache_papers_ids = get_cache_papers_ids(
            cache_type=LOGREG_CACHE_TYPE,
            papers=papers,
            n_cache=LOGREG_N_CACHE,
            random_state=random_state,
        )
        relevant_papers_ids.update(cache_papers_ids)

    if include_old_cache:
        old_cache_papers_ids = papers[papers["in_cache"]]["paper_id"].tolist()
        assert old_cache_papers_ids == sorted(old_cache_papers_ids)
        relevant_papers_ids.update(old_cache_papers_ids)

    relevant_papers_ids = sorted(list(relevant_papers_ids))
    print(len(relevant_papers_ids), "Final number of relevant papers IDs in List.")
    return relevant_papers_ids
