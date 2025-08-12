import pickle

from ...logreg.src.training.algorithm import Evaluation
from ...logreg.src.training.get_users_ratings import (
    MIN_N_NEGRATED_TRAIN,
    MIN_N_NEGRATED_VAL_USERS_SELECTION,
    MIN_N_POSRATED_TRAIN,
    MIN_N_POSRATED_VAL,
    TRAIN_SIZE,
    get_users_ratings,
)
from ...src.project_paths import ProjectPaths


def sequence_get_users_ratings(selection: str) -> tuple:
    if selection not in ["finetuning_val", "finetuning_test", "session_based"]:
        raise ValueError(f"Invalid selection: {selection}")
    if selection == "finetuning_val":
        path = ProjectPaths.data_sequence_session_based_ratings_val_users_path()
    elif selection == "finetuning_test":
        path = ProjectPaths.data_sequence_session_based_ratings_test_users_path()
    elif selection == "session_based":
        path = ProjectPaths.data_sequence_session_based_ratings_session_based_users_path()
    with open(path, "rb") as f:
        users_ratings_dict = pickle.load(f)
    users_ratings = users_ratings_dict["users_ratings"]
    users_ids = users_ratings_dict["users_ids"]
    users_negrated_ranking = users_ratings_dict["users_negrated_ranking"]
    return users_ratings, users_ids, users_negrated_ranking


def sequence_save_users_ratings(selection: str) -> None:
    if selection not in ["finetuning_val", "finetuning_test", "session_based"]:
        raise ValueError(f"Invalid selection: {selection}")
    if selection == "finetuning_val":
        path = ProjectPaths.data_sequence_session_based_ratings_val_users_path()
    elif selection == "finetuning_test":
        path = ProjectPaths.data_sequence_session_based_ratings_test_users_path()
    elif selection == "session_based":
        path = ProjectPaths.data_sequence_session_based_ratings_session_based_users_path()
    users_ratings, users_ids, users_negrated_ranking = get_users_ratings(
        users_selection=selection,
        evaluation=Evaluation.SESSION_BASED,
        train_size=TRAIN_SIZE,
        min_n_posrated_train=MIN_N_POSRATED_TRAIN,
        min_n_posrated_val=MIN_N_POSRATED_VAL,
        min_n_negrated_train=MIN_N_NEGRATED_TRAIN,
        min_n_negrated_val=MIN_N_NEGRATED_VAL_USERS_SELECTION,
        filter_for_negrated_ranking=True,
    )
    users_ratings_dict = {
        "users_ratings": users_ratings,
        "users_ids": users_ids,
        "users_negrated_ranking": users_negrated_ranking,
    }
    with open(path, "wb") as f:
        pickle.dump(users_ratings_dict, f)
    print(f"Saved users ratings for {selection} to {path}. {len(users_ids)} users saved.")


if __name__ == "__main__":
    selections = ["finetuning_val", "finetuning_test", "session_based"]
    for selection in selections:
        sequence_save_users_ratings(selection)
