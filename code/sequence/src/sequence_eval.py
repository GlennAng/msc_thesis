import argparse
from pathlib import Path

from ...logreg.src.embeddings.embedding import Embedding
from ...src.load_files import load_papers, load_users_significant_categories
from .sequence_data import load_users_embeddings_dict
from .sequence_load_users_ratings import sequence_load_users_ratings


class SequenceEvaluator:

    def __init__(self, users_embeddings_dict: dict, args_dict : dict) -> None:
        self.users_embeddings = users_embeddings_dict["users_embeddings"]
        self.embedding = Embedding(Path(users_embeddings_dict["embedding_path"]).resolve())
        users_ratings_tuple = sequence_load_users_ratings(
            selection=users_embeddings_dict["users_selection"]
        )
        self.users_ratings, self.users_ids, self.users_negrated_ranking = users_ratings_tuple
        assert self.users_ids == list(self.users_embeddings.keys())

        papers = load_papers(relevant_columns=["paper_id", "in_ratings", "l1"])
        users_significant_categories = load_users_significant_categories(
            relevant_users_ids=self.users_ids
        )


def args_dict_assertions(args_dict: dict) -> None:
    assert args_dict["users_embeddings_dict_path"].exists()


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Sequence Evaluation")
    parser.add_argument("--users_embeddings_dict_path", type=str, required=True)
    parser.add_argument("--random_state", type=int, default=42)

    args_dict = vars(parser.parse_args())
    args_dict["users_embeddings_dict_path"] = Path(
        args_dict["users_embeddings_dict_path"]
    ).resolve()
    args_dict_assertions(args_dict)
    return args_dict


if __name__ == "__main__":
    args_dict = parse_args()
    users_embeddings_dict = load_users_embeddings_dict(path=args_dict["users_embeddings_dict_path"])

    eval = SequenceEvaluator(users_embeddings_dict=users_embeddings_dict, args_dict=args_dict)


"""

from ...src.project_paths import ProjectPaths
from .load_users_embeddings import load_users_embeddings_dict, check_users_embeddings_dict
from .sequence_get_users_ratings import sequence_get_users_ratings

from ...finetuning.src.finetuning_val import FINETUNING_RANKING_METRICS


class SequenceEvaluator:

    def __init__(self, users_embeddings_dict: dict) -> None:
        check_users_embeddings_dict(users_embeddings_dict)
        self.users_embeddings = users_embeddings_dict["users_embeddings"]
        self.users_selection = users_embeddings_dict["users_selection"]
        self.embedding_path = Path(users_embeddings_dict["embedding_path"]).resolve()

        self.users_ids = list(self.users_embeddings.keys())
        papers = load_papers(relevant_columns = ["paper_id", "in_ratings", "l1"])
        users_significant_categories = load_users_significant_categories(relevant_users_ids=self.users_ids)
        users_val_negative_samples = get_users_val_negative_samples(self.users_ids, papers, users_significant_categories)

def get_users_val_negative_samples(users_ids: list, papers: pd.DataFrame, users_significant_categories: pd.DataFrame) -> dict:
    val_negative_samples_ids = get_categories_samples_ids(
        papers=papers,
        n_categories_samples=100,
        random_state=42,
        papers_ids_to_exclude=papers[papers["in_ratings"]].index.tolist(),
    )
    users_val_negative_samples = {}
    for user_id in users_ids:
        user_significant_categories = users_significant_categories[users_significant_categories["user_id"] == user_id]
        user_categories_ratios = get_user_categories_ratios(
            categories_to_exclude=user_significant_categories
        )
        user_val_negative_samples = get_user_val_negative_samples(
            val_negative_samples_ids=val_negative_samples_ids
            n_negative_samples=100,
            random_state=42,
            user_categories_ratios=user_categories_ratios,
            embedding=self.embedding,
        )
        users_val_negative_samples_ids[user_id] = user_val_negative_samples
    return users_val_negative_samples_ids

if __name__ == "__main__":
    users_ratings = sequence_get_users_ratings(selection="finetuning_test")[0]
    users_embeddings_dict = load_users_embeddings_dict(
        path=ProjectPaths.sequence_data_users_embeddings_path() / "finetuning_test_mean.pkl"
    )
    check_users_embeddings_dict(users_embeddings_dict, users_ratings)
    print(FINETUNING_RANKING_METRICS)


"""
