import pandas as pd

from ..training.training_data import (
    get_cache_papers_ids_old_cache,
    get_cache_papers_ids_random_cache,
)


def count_to_str(count: int) -> str:
    return f"{count:,}"


def get_categories_distribution(
    papers_ids_to_categories: dict = None, print_results: bool = True
) -> tuple:
    unique_categories = set(papers_ids_to_categories.values())
    categories_counts = {category: 0 for category in unique_categories}
    n_total = 0
    for _, value in papers_ids_to_categories.items():
        categories_counts[value] += 1
        n_total += 1
    categories_counts = {category: count / n_total for category, count in categories_counts.items()}
    if print_results:
        categories_counts_copy = categories_counts.copy()
        categories_counts_copy["Total"] = 1.0
        categories_counts_copy = sorted(
            categories_counts_copy.items(), key=lambda x: x[1], reverse=True
        )
        for category, count in categories_counts_copy:
            print(f"{category}: {count:.2%} ({count_to_str(int(count * n_total))})")
        print("____________________________________________________________")
    return categories_counts, n_total


def get_categories_distribution_database(
    papers: pd.DataFrame, level: str = "l1", print_results: bool = True
) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    return get_categories_distribution(papers_ids_to_categories, print_results)


def get_categories_distribution_ratings(
    papers: pd.DataFrame, level: str = "l1", print_results: bool = True
) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    papers = papers[papers["in_ratings"]]
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    return get_categories_distribution(papers_ids_to_categories, print_results)


def get_categories_distribution_old_cache(
    papers: pd.DataFrame, level: str = "l1", print_results: bool = True
) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    cache_papers_ids = get_cache_papers_ids_old_cache(
        papers=papers,
        n_cache=5000,
        random_state=42,
        papers_ids_to_exclude=None,
    )
    papers = papers[papers["paper_id"].isin(cache_papers_ids)]
    if papers.empty:
        raise ValueError("No papers found in the cache.")
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    return get_categories_distribution(papers_ids_to_categories, print_results)


def get_categories_distribution_random_cache(
    papers: pd.DataFrame, level: str = "l1", print_results: bool = True
) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    cache_papers_ids = get_cache_papers_ids_random_cache(
        papers=papers,
        n_cache=5000,
        random_state=42,
        papers_ids_to_exclude=None,
    )
    papers = papers[papers["paper_id"].isin(cache_papers_ids)]
    if papers.empty:
        raise ValueError("No papers found in the cache.")
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    return get_categories_distribution(papers_ids_to_categories, print_results)


def get_categories_distribution_ratings_specific_users(
    papers: pd.DataFrame,
    users_ids: list,
    users_ratings: pd.DataFrame,
    level: str = "l1",
    print_results: bool = True,
) -> tuple:
    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    users_ratings = users_ratings[users_ratings["user_id"].isin(users_ids)].reset_index(drop=True)
    users_ratings = users_ratings[users_ratings["rating"] == 1]
    papers_users_ratings_ids = papers[papers["paper_id"].isin(users_ratings["paper_id"])]
    papers = papers[papers["paper_id"].isin(papers_users_ratings_ids["paper_id"])]
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    return get_categories_distribution(papers_ids_to_categories, print_results)


def get_categories_distribution_relevant_papers(
    papers: pd.DataFrame,
    level: str = "l1",
    print_results: bool = True,
) -> tuple:
    from ....src.load_files import load_relevant_papers_ids

    if level not in ["l1", "l2", "l3"]:
        raise ValueError(f"Invalid level '{level}'. Must be one of 'l1', 'l2', or 'l3'.")
    relevant_papers_ids = load_relevant_papers_ids()
    papers = papers[papers["paper_id"].isin(relevant_papers_ids)]
    papers_ids_to_categories = papers.set_index("paper_id")[level].to_dict()
    return get_categories_distribution(papers_ids_to_categories, print_results)


if __name__ == "__main__":

    from ....src.load_files import (
        load_papers,
        load_session_based_users_ids,
        load_users_ratings,
        load_users_significant_categories,
    )

    users_ids = load_session_based_users_ids()
    users_ids_non_cs = load_session_based_users_ids(
        select_non_cs_users_only=True,
    )
    users_ids_cs = sorted(list(set(users_ids) - set(users_ids_non_cs)))
    papers = load_papers()
    users_ratings = load_users_ratings(relevant_users_ids=users_ids)
    users_significant_categories = load_users_significant_categories(relevant_users_ids=users_ids)

    print("Categories Distribution for Database:")
    get_categories_distribution_database(papers=papers, level="l1", print_results=True)
    print("\nCategories Distribution for Ratings:")
    get_categories_distribution_ratings(papers=papers, level="l1", print_results=True)
    print("\nCategories Distribution for CS Users:")
    get_categories_distribution_ratings_specific_users(
        papers=papers, users_ids=users_ids_cs, users_ratings=users_ratings
    )
    print("\nCategories Distribution for Non-CS Users:")
    get_categories_distribution_ratings_specific_users(
        papers=papers, users_ids=users_ids_non_cs, users_ratings=users_ratings
    )
    print("\nCategories Distribution for Old Cache:")
    get_categories_distribution_old_cache(papers=papers, level="l1", print_results=True)
    print("\nCategories Distribution for Relevant Papers:")
    get_categories_distribution_relevant_papers(papers=papers, level="l1", print_results=True)
