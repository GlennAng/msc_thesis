import pickle

from tqdm import tqdm

from ....src.load_files import (
    TEST_RANDOM_STATES,
    load_finetuning_users_ids,
    load_papers,
    load_users_ratings,
)
from ....src.project_paths import ProjectPaths
from ..training.training_data import get_cache_papers_ids, get_categories_samples_ids


def save_relevant_papers(seeds: list = TEST_RANDOM_STATES) -> list:
    users_ids = load_finetuning_users_ids(selection="test")
    print(len(users_ids), "Users loaded.")
    users_ratings = load_users_ratings(relevant_users_ids=users_ids)
    assert len(users_ratings["user_id"].unique()) == len(users_ids), "Users IDs do not match."
    papers = load_papers(relevant_columns=["paper_id", "in_ratings", "in_cache", "l1"])
    relevant_papers_ids = set(users_ratings["paper_id"].values)
    print(len(relevant_papers_ids), "Relevant papers loaded.")

    for random_state in tqdm(seeds):
        val_negative_samples_ids = get_categories_samples_ids(
            papers=papers,
            n_categories_samples=100,
            random_state=random_state,
            papers_ids_to_exclude=papers[papers["in_ratings"]]["paper_id"].tolist(),
        )[1]
        print(f"Seed {random_state}: {len(val_negative_samples_ids)} negative samples collected.")
        relevant_papers_ids.update(val_negative_samples_ids)
        cache_papers_ids = get_cache_papers_ids(
            cache_type="categories_cache",
            papers=papers,
            n_cache=5000,
            random_state=random_state,
        )
        print(f"Seed {random_state}: {len(cache_papers_ids)} cache papers collected.")
        relevant_papers_ids.update(cache_papers_ids)
    print(len(relevant_papers_ids), "Total relevant papers IDs collected after negative sampling.")

    old_cache_papers_ids = papers[papers["in_cache"]]["paper_id"].tolist()
    assert old_cache_papers_ids == sorted(old_cache_papers_ids)
    relevant_papers_ids.update(old_cache_papers_ids)

    relevant_papers_ids = sorted(list(relevant_papers_ids))
    print(len(relevant_papers_ids), "Final number of relevant papers IDs in List.")
    path = ProjectPaths.data_relevant_papers_ids_path()
    with open(path, "wb") as f:
        pickle.dump(relevant_papers_ids, f)
    return relevant_papers_ids


if __name__ == "__main__":
    save_relevant_papers()
