import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ...logreg.src.embeddings.papers_categories import get_glove_categories_embeddings
from ...logreg.src.embeddings.papers_categories_dicts import PAPERS_CATEGORIES
from ...logreg.src.training.algorithm import Evaluation
from ...logreg.src.training.get_users_ratings import get_users_ratings
from ...logreg.src.training.training_data import (
    get_filtered_cache_papers_ids_for_user,
    get_negative_samples_ids,
)
from ...src.load_files import (
    load_finetuning_users,
    load_papers,
    load_papers_texts,
    load_users_ratings,
)
from ...src.project_paths import ProjectPaths

VAL_RANDOM_STATE = 42
TEST_RANDOM_STATES = [1, 25, 75, 100, 150]


def load_users_coefs_ids_to_idxs() -> dict:
    users_coefs_ids_to_idxs_path = (
        ProjectPaths.finetuning_data_model_state_dicts_path() / "users_coefs_ids_to_idxs.pkl"
    )
    if not users_coefs_ids_to_idxs_path.exists():
        raise FileNotFoundError(
            f"Users coefs IDs to indices mapping file not found: {users_coefs_ids_to_idxs_path}."
        )
    with open(users_coefs_ids_to_idxs_path, "rb") as f:
        users_coefs_ids_to_idxs = pickle.load(f)
    assert list(users_coefs_ids_to_idxs.values()) == list(range(len(users_coefs_ids_to_idxs)))
    return users_coefs_ids_to_idxs


def load_val_users_embeddings_idxs(
    val_users_ids: list = None, users_coefs_ids_to_idxs: dict = None
) -> torch.Tensor:
    if val_users_ids is None:
        val_users_ids = load_finetuning_users(selection="val")
    if users_coefs_ids_to_idxs is None:
        users_coefs_ids_to_idxs = load_users_coefs_ids_to_idxs()
    val_users_embeddings_idxs = [users_coefs_ids_to_idxs[user_id] for user_id in val_users_ids]
    assert len(val_users_embeddings_idxs) == len(
        set(val_users_embeddings_idxs)
    ) and val_users_embeddings_idxs == sorted(val_users_embeddings_idxs)
    return torch.tensor(val_users_embeddings_idxs, dtype=torch.int64)


def save_users_embeddings_tensor(
    users_coefs: np.ndarray = None, users_coefs_ids_to_idxs: dict = None
) -> None:
    users_embeddings_path = ProjectPaths.finetuning_data_model_state_dicts_users_embeddings_path()
    if users_embeddings_path.exists():
        print("Users embeddings tensor already exists - skipping saving.")
        return
    if users_coefs is None:
        users_coefs = np.load(
            ProjectPaths.finetuning_data_model_state_dicts_path() / "users_coefs.npy"
        )
    if users_coefs_ids_to_idxs is None:
        users_coefs_ids_to_idxs = load_users_coefs_ids_to_idxs()
    num_embeddings, embedding_dim = users_coefs.shape
    users_embeddings = torch.nn.Embedding(num_embeddings, embedding_dim, dtype=torch.float32)
    for user_id, idx in users_coefs_ids_to_idxs.items():
        assert idx == users_coefs_ids_to_idxs[user_id]
        with torch.no_grad():
            users_embeddings.weight.data[idx] = torch.from_numpy(users_coefs[idx]).to(torch.float32)
    torch.save(users_embeddings.state_dict(), users_embeddings_path)
    print(f"Users embeddings tensor saved to {users_embeddings_path}.")


def save_projection_tensor(pca_components: np.ndarray = None, pca_mean: np.ndarray = None) -> None:
    projection_tensor_path = ProjectPaths.finetuning_data_model_state_dicts_projection_path()
    if projection_tensor_path.exists():
        print("Projection tensor already exists - skipping saving.")
        return
    model_name = ProjectPaths.finetuning_data_model_path().stem
    pca_path = ProjectPaths.logreg_embeddings_path() / "after_pca" / model_name
    pca_components = (
        np.load(pca_path / "pca_components.npy") if pca_components is None else pca_components
    )
    pca_mean = np.load(pca_path / "pca_mean.npy") if pca_mean is None else pca_mean
    bias = -(pca_mean @ pca_components.T)
    pca_components = torch.from_numpy(pca_components).to(torch.float32)
    pca_bias = torch.from_numpy(bias).to(torch.float32)
    projection = torch.nn.Linear(
        pca_components.shape[1], pca_components.shape[0], bias=True, dtype=torch.float32
    )
    with torch.no_grad():
        projection.weight.copy_(pca_components)
        projection.bias.copy_(pca_bias)
    torch.save(projection.state_dict(), projection_tensor_path)
    print(f"Projection tensor saved to {projection_tensor_path}.")


def save_categories_embeddings_tensor(glove_categories_embeddings: dict = None) -> None:
    categories_embeddings_l1_path = (
        ProjectPaths.finetuning_data_model_state_dicts_categories_embeddings_l1_path()
    )

    categories_to_idxs_l1_path = (
        ProjectPaths.finetuning_data_model_state_dicts_path() / "categories_to_idxs_l1.pkl"
    )
    categories_to_idxs_l2_path = (
        ProjectPaths.finetuning_data_model_state_dicts_path() / "categories_to_idxs_l2.pkl"
    )
    paths_existing = [
        categories_embeddings_l1_path.exists(),
        categories_to_idxs_l1_path.exists(),
        categories_to_idxs_l2_path.exists(),
    ]
    if all(paths_existing):
        print("Categories embeddings tensors and indices dicts already exist - skipping saving.")
        return
    if any(paths_existing):
        raise FileExistsError(
            "One or more but not all of the files already exist: "
            f"{categories_embeddings_l1_path}, {categories_to_idxs_l1_path}, "
            f"{categories_to_idxs_l2_path}."
        )
    papers = load_papers()
    categories_l1 = sorted(list(papers["l1"].dropna().unique()))
    categories_l2 = sorted(list(papers["l2"].dropna().unique()))
    assert papers["l1"].isna().sum() == papers["l2"].isna().sum() and papers["l1"].isna().sum() > 0
    categories_l1, categories_l2 = [None] + categories_l1, [None] + categories_l2
    if glove_categories_embeddings is None:
        glove_categories_embeddings = get_glove_categories_embeddings(
            PAPERS_CATEGORIES.categories_to_glove
        )
    assert isinstance(glove_categories_embeddings, dict) and set(
        glove_categories_embeddings.keys()
    ) == set(categories_l1)
    categories_to_idxs_l1 = {category: idx for idx, category in enumerate(categories_l1)}
    categories_to_idxs_l2 = {category: idx for idx, category in enumerate(categories_l2)}
    categories_embeddings_l1 = torch.nn.Embedding(
        len(categories_l1),
        len(glove_categories_embeddings[categories_l1[0]]),
        dtype=torch.float32,
    )
    for idx, category in enumerate(categories_l1):
        with torch.no_grad():
            categories_embeddings_l1.weight.data[idx] = torch.from_numpy(
                glove_categories_embeddings[category]
            ).to(torch.float32)
    torch.save(categories_embeddings_l1.state_dict(), categories_embeddings_l1_path)
    print(f"Categories embeddings tensor for L1 saved to {categories_embeddings_l1_path}.")
    with open(categories_to_idxs_l1_path, "wb") as f:
        pickle.dump(categories_to_idxs_l1, f)
    print(f"Categories to indices mapping for L1 saved to {categories_to_idxs_l1_path}.")
    with open(categories_to_idxs_l2_path, "wb") as f:
        pickle.dump(categories_to_idxs_l2, f)
    print(f"Categories to indices mapping for L2 saved to {categories_to_idxs_l2_path}.")


def load_categories_to_idxs(level: str = "l1") -> dict:
    if level not in ["l1", "l2"]:
        raise ValueError(f"Invalid level: {level}. Choose from ['l1', 'l2'].")
    path = ProjectPaths.finetuning_data_model_state_dicts_path() / f"categories_to_idxs_{level}.pkl"
    with open(path, "rb") as f:
        categories_to_idxs = pickle.load(f)
    assert list(categories_to_idxs.values()) == list(range(len(categories_to_idxs)))
    categories = list(categories_to_idxs.keys())
    assert categories[0] is None and categories[1:] == sorted(categories[1:])
    return categories_to_idxs


def save_negative_samples_ids_for_seeds(
    n_negative_samples: int = 100, random_states: list = None
) -> None:
    path = ProjectPaths.finetuning_data_model_datasets_negative_samples_for_seeds_path()
    if path.exists():
        print("Negative samples IDs for seeds already exist - skipping saving.")
        return
    if random_states is None:
        random_states = [VAL_RANDOM_STATE] + TEST_RANDOM_STATES
    random_states = sorted(random_states)
    papers = load_papers(relevant_columns=["paper_id", "in_ratings", "in_cache", "l1"])
    negative_samples_ids_for_seeds = {}
    for random_state in random_states:
        negative_samples_ids = get_negative_samples_ids(
            papers,
            n_negative_samples,
            random_state,
            exclude_in_ratings=True,
            exclude_in_cache=True,
        )
        assert len(negative_samples_ids) == n_negative_samples and len(negative_samples_ids) == len(
            set(negative_samples_ids)
        )
        negative_samples_ids_for_seeds[random_state] = negative_samples_ids
    with open(path, "wb") as f:
        pickle.dump(negative_samples_ids_for_seeds, f)
    print(f"Negative samples IDs for seeds saved to {path}.")


def load_negative_samples_ids_for_seeds() -> dict:
    with open(
        ProjectPaths.finetuning_data_model_datasets_negative_samples_for_seeds_path(),
        "rb",
    ) as f:
        negative_samples_ids_for_seeds = pickle.load(f)
    return negative_samples_ids_for_seeds


def finetuning_tokenize_papers(
    papers_ids: list, tokenizer: AutoTokenizer, max_sequence_length: int
) -> tuple:
    assert isinstance(papers_ids, list), "papers_ids should be a list of paper IDs."
    assert len(papers_ids) == len(set(papers_ids)) and papers_ids == sorted(papers_ids)
    papers_texts = load_papers_texts(
        relevant_papers_ids=papers_ids,
        relevant_columns=["paper_id", "title", "abstract"],
    )
    papers_texts = papers_texts[["paper_id", "title", "abstract"]].values.tolist()
    papers_ids, papers_titles, papers_abstracts = zip(*papers_texts)
    papers_ids, papers_titles, papers_abstracts = (
        list(papers_ids),
        list(papers_titles),
        list(papers_abstracts),
    )
    assert papers_ids == sorted(papers_ids) and len(papers_ids) == len(set(papers_ids))
    papers = [
        f"{title} {tokenizer.sep_token} {abstract}"
        for title, abstract in zip(papers_titles, papers_abstracts)
    ]
    input_ids_list, attention_mask_list = [], []
    for i in tqdm(range(len(papers)), desc="Tokenizing Papers", unit="Paper"):
        encoding = tokenizer(
            papers[i],
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids_list.append(encoding["input_ids"].squeeze(0))
        attention_mask_list.append(encoding["attention_mask"].squeeze(0))
    papers_dict = {
        "paper_id": torch.tensor(papers_ids),
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
    }
    papers_ids_to_idxs = {paper_id: idx for idx, paper_id in enumerate(papers_ids)}
    return papers_dict, papers_ids_to_idxs


def save_val_negative_samples(
    tokenizer: AutoTokenizer,
    max_sequence_length: int,
    n_negative_samples: int = 100,
    random_state: int = VAL_RANDOM_STATE,
) -> None:
    tensor_path = ProjectPaths.finetuning_data_model_datasets_val_negative_samples_path()
    if tensor_path.exists():
        print("Validation negative samples already exist - skipping saving.")
        return
    papers = load_papers(relevant_columns=["paper_id", "in_ratings", "in_cache", "l1"])
    negative_samples_ids = get_negative_samples_ids(
        papers,
        n_negative_samples,
        random_state,
        exclude_in_ratings=True,
        exclude_in_cache=True,
    )
    negative_samples_ids = sorted(list(negative_samples_ids))
    negative_samples = finetuning_tokenize_papers(
        negative_samples_ids, tokenizer, max_sequence_length
    )[0]
    torch.save(negative_samples, tensor_path)
    print(f"Validation negative samples tensor saved to {tensor_path}.")


def finetuning_get_cache_papers_ids_for_user(
    users_ratings: pd.DataFrame,
    user_id: int,
    cache_ids: list,
    max_cache: int,
    random_state: int,
) -> set:
    user_ratings = users_ratings[users_ratings["user_id"] == user_id]
    posrated_ids = user_ratings[user_ratings["rating"] == 1]["paper_id"].tolist()
    negrated_ids = user_ratings[user_ratings["rating"] == 0]["paper_id"].tolist()
    rated_ids = posrated_ids + negrated_ids
    user_cache_ids = get_filtered_cache_papers_ids_for_user(
        cache_ids, rated_ids, max_cache, random_state
    )
    return set(user_cache_ids)


def save_test_papers(
    tokenizer: AutoTokenizer,
    max_sequence_length: int,
    n_cache: int = 5000,
    n_cache_attached: int = 5000,
    n_negative_samples: int = 100,
    val_random_state: int = VAL_RANDOM_STATE,
    test_random_states: list = TEST_RANDOM_STATES,
) -> None:
    assert val_random_state not in test_random_states
    assert len(test_random_states) == len(set(test_random_states)) and test_random_states == sorted(
        test_random_states
    )
    tensor_path = ProjectPaths.finetuning_data_model_datasets_test_papers_path()
    if tensor_path.exists():
        print("Test papers already exist - skipping saving.")
        return
    papers = load_papers(relevant_columns=["paper_id", "in_ratings", "in_cache", "l1"])
    cache_ids = papers[papers["in_cache"]]["paper_id"].tolist()
    assert len(cache_ids) == len(set(cache_ids)) and cache_ids == sorted(cache_ids)
    full_random_states = sorted(set([val_random_state] + test_random_states))
    negative_samples_ids_for_seeds = load_negative_samples_ids_for_seeds()
    assert list(negative_samples_ids_for_seeds.keys()) == full_random_states
    papers_ids = set()
    for random_state in full_random_states:
        negative_samples_ids = negative_samples_ids_for_seeds[random_state]
        assert negative_samples_ids == get_negative_samples_ids(
            papers,
            n_negative_samples,
            random_state,
            exclude_in_ratings=True,
            exclude_in_cache=True,
        )
        cache_attached_papers_ids = get_negative_samples_ids(
            papers,
            n_cache_attached,
            random_state,
            papers_to_exclude=negative_samples_ids,
        )
        papers_ids.update(negative_samples_ids + cache_attached_papers_ids)
    val_users_ids = load_finetuning_users(selection="val")
    test_users_ids = load_finetuning_users(selection="test")
    users_ratings = load_users_ratings(
        relevant_columns=["user_id", "paper_id", "rating"],
        relevant_users_ids=val_users_ids + test_users_ids,
    )
    papers_ids.update(users_ratings["paper_id"].unique())
    for val_user_id in tqdm(
        val_users_ids, desc="Getting Cache Papers for Val Users", unit="Val User"
    ):
        papers_ids.update(
            finetuning_get_cache_papers_ids_for_user(
                users_ratings, val_user_id, cache_ids, n_cache, val_random_state
            )
        )
    for test_user_id in tqdm(
        test_users_ids, desc="Getting Cache Papers for Test Users", unit="Test User"
    ):
        for random_state in test_random_states:
            papers_ids.update(
                finetuning_get_cache_papers_ids_for_user(
                    users_ratings, test_user_id, cache_ids, n_cache, random_state
                )
            )
    papers_ids = sorted(list(papers_ids))
    papers_tokenized = finetuning_tokenize_papers(papers_ids, tokenizer, max_sequence_length)[0]
    torch.save(papers_tokenized, tensor_path)
    print(f"Test papers tensor saved to {tensor_path}.")


def save_train_val_users_papers(
    papers_type: str, tokenizer: AutoTokenizer, max_sequence_length: int
) -> None:
    papers_types = ["train_users", "val_users"]
    if papers_type not in papers_types:
        raise ValueError(f"Invalid papers type: {papers_type}. Choose from {papers_types}.")
    tensor_path = ProjectPaths.finetuning_data_model_datasets_path() / f"{papers_type}_papers.pt"
    papers_ids_to_idxs_path = (
        ProjectPaths.finetuning_data_model_datasets_path() / f"{papers_type}_papers_ids_to_idxs.pkl"
    )
    if tensor_path.exists() and papers_ids_to_idxs_path.exists():
        print(f"{papers_type.capitalize()} papers already exist - skipping saving.")
        return
    if tensor_path.exists() or papers_ids_to_idxs_path.exists():
        raise FileExistsError(
            f"One or more but not all of the files already exist: {tensor_path}, {papers_ids_to_idxs_path}."
        )
    selection = "train" if papers_type == "train_users" else "val"
    users_ids = load_finetuning_users(selection=selection)
    users_ratings = load_users_ratings(
        relevant_columns=["user_id", "paper_id"], relevant_users_ids=users_ids
    )
    papers_ids = sorted(list(users_ratings["paper_id"].unique()))
    papers_dict, papers_ids_to_idxs = finetuning_tokenize_papers(
        papers_ids, tokenizer, max_sequence_length
    )
    torch.save(papers_dict, tensor_path)
    print(f"{papers_type.capitalize()} papers tensor saved to {tensor_path}.")
    with open(papers_ids_to_idxs_path, "wb") as f:
        pickle.dump(papers_ids_to_idxs, f)
    print(
        f"{papers_type.capitalize()} papers IDs to indices mapping saved to {papers_ids_to_idxs_path}."
    )


def save_finetuning_papers_type(
    papers_type: str, tokenizer: AutoTokenizer = None, max_sequence_length: int = 512
) -> None:
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(ProjectPaths.finetuning_data_model_hf())
    if papers_type == "val_negative_samples":
        save_val_negative_samples(tokenizer, max_sequence_length)
    elif papers_type == "test":
        save_test_papers(tokenizer, max_sequence_length)
    elif papers_type in ["train_users", "val_users"]:
        save_train_val_users_papers(papers_type, tokenizer, max_sequence_length)
    else:
        raise ValueError(
            f"Invalid papers type: {papers_type}. "
            "Choose from ['val_negative_samples', 'test', 'train_users', 'val_users']."
        )


def save_finetuning_papers(tokenizer: AutoTokenizer = None, max_sequence_length: int = 512) -> None:
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(ProjectPaths.finetuning_data_model_hf())
    papers_types = ["val_negative_samples", "test", "train_users", "val_users"]
    for papers_type in papers_types:
        save_finetuning_papers_type(papers_type, tokenizer, max_sequence_length)


def load_finetuning_papers(
    papers_type: str, attach_l1: bool = True, attach_l2: bool = True
) -> dict:
    papers_types = ["val_negative_samples", "test", "train_users", "val_users"]
    if papers_type not in papers_types:
        raise ValueError(f"Invalid papers type: {papers_type}. Choose from {papers_types}.")
    if papers_type == "val_negative_samples":
        tensor_path = ProjectPaths.finetuning_data_model_datasets_val_negative_samples_path()
    elif papers_type == "test":
        tensor_path = ProjectPaths.finetuning_data_model_datasets_test_papers_path()
    else:
        tensor_path = (
            ProjectPaths.finetuning_data_model_datasets_path() / f"{papers_type}_papers.pt"
        )
    papers_tensor = torch.load(tensor_path, weights_only=True)
    if attach_l1 or attach_l2:
        papers = load_papers(
            relevant_columns=["paper_id", "l1", "l2"],
            relevant_papers_ids=papers_tensor["paper_id"].tolist(),
        )
        assert papers_tensor["paper_id"].tolist() == papers["paper_id"].tolist()
        if attach_l1:
            categories_to_idxs_l1 = load_categories_to_idxs("l1")
            categories_l1 = papers["l1"].tolist()
            categories_l1 = [
                category if category in categories_to_idxs_l1 else None
                for category in categories_l1
            ]
            papers_tensor["l1"] = torch.tensor(
                [categories_to_idxs_l1[category] for category in categories_l1]
            )
        if attach_l2:
            categories_to_idxs_l2 = load_categories_to_idxs("l2")
            categories_l2 = papers["l2"].tolist()
            categories_l2 = [
                category if category in categories_to_idxs_l2 else None
                for category in categories_l2
            ]
            papers_tensor["l2"] = torch.tensor(
                [categories_to_idxs_l2[category] for category in categories_l2]
            )
    return papers_tensor


def load_finetuning_papers_ids_to_idxs(papers_type: str) -> dict:
    papers_types = ["train_users", "val_users"]
    if papers_type not in papers_types:
        raise ValueError(f"Invalid papers type: {papers_type}. Choose from {papers_types}.")
    papers_ids_to_idxs_path = (
        ProjectPaths.finetuning_data_model_datasets_path() / f"{papers_type}_papers_ids_to_idxs.pkl"
    )
    with open(papers_ids_to_idxs_path, "rb") as f:
        papers_ids_to_idxs = pickle.load(f)
    return papers_ids_to_idxs


def setup_finetuning_papers_dataset(dataset_type: str, users_type: str) -> tuple:
    dataset_types = ["train", "val"]
    if dataset_type not in dataset_types:
        raise ValueError(f"Invalid dataset type: {dataset_type}. Choose from {dataset_types}.")
    if dataset_type == "train" and users_type not in ["train_users", "val_users"]:
        raise ValueError(
            f"Invalid users type: {users_type}. Choose from ['train_users', 'val_users'] for "
            f"train dataset."
        )
    if dataset_type == "val" and users_type != "val_users":
        raise ValueError(
            f"Invalid users type: {users_type}. Choose 'val_users' for validation dataset."
        )
    users_ids = (
        load_finetuning_users(selection="train")
        if users_type == "train_users"
        else load_finetuning_users(selection="val")
    )
    papers = load_finetuning_papers(users_type, attach_l1=True, attach_l2=True)
    papers_ids_to_idxs = load_finetuning_papers_ids_to_idxs(users_type)
    if users_type == "train_users":
        users_ratings = load_users_ratings(
            relevant_columns=["user_id", "paper_id", "rating", "time"],
            relevant_users_ids=users_ids,
        )
    else:
        users_ratings = get_users_ratings(
            users_selection=users_ids,
            evaluation=Evaluation.SESSION_BASED,
            train_size=1.0,
            min_n_posrated_train=16,
            min_n_negrated_train=16,
            min_n_posrated_val=4,
            min_n_negrated_val=4,
        )[0]
        if dataset_type == "train":
            users_ratings = users_ratings[users_ratings["split"] == "train"]
        elif dataset_type == "val":
            users_ratings = users_ratings[users_ratings["split"] == "val"]
    return users_ratings, papers, papers_ids_to_idxs


def fill_finetuning_papers_dataset(
    users_ratings: pd.DataFrame,
    papers: pd.DataFrame,
    papers_ids_to_idxs: dict,
    users_coefs_ids_to_idxs: dict,
) -> dict:
    unique_users_ids = users_ratings["user_id"].unique().tolist()
    assert len(unique_users_ids) == len(set(unique_users_ids)) and unique_users_ids == sorted(
        unique_users_ids
    )
    user_idx_list, paper_id_list, rating_list, input_ids_list, attention_mask_list = (
        [],
        [],
        [],
        [],
        [],
    )
    category_l1_list, category_l2_list, time_list = [], [], []
    for user_id in unique_users_ids:
        user_ratings = users_ratings[users_ratings["user_id"] == user_id]
        n_ratings = len(user_ratings)
        assert user_ratings["time"].is_monotonic_increasing
        user_idx_list.extend([users_coefs_ids_to_idxs[user_id]] * n_ratings)
        assert user_ratings["paper_id"].nunique() == n_ratings
        paper_to_time = dict(zip(user_ratings["paper_id"], user_ratings["time"]))
        paper_to_rating = dict(zip(user_ratings["paper_id"], user_ratings["rating"]))
        posrated_ids = user_ratings[user_ratings["rating"] == 1]["paper_id"].tolist()
        negrated_ids = user_ratings[user_ratings["rating"] == 0]["paper_id"].tolist()
        rated_ids = posrated_ids + negrated_ids
        assert len(rated_ids) == n_ratings
        paper_id_list.extend(rated_ids)
        for paper_id in rated_ids:
            rating_list.append(paper_to_rating[paper_id])
            time_list.append(paper_to_time[paper_id])
            paper_idx = papers_ids_to_idxs[paper_id]
            input_ids_list.append(papers["input_ids"][paper_idx])
            attention_mask_list.append(papers["attention_mask"][paper_idx])
            category_l1_list.append(papers["l1"][paper_idx])
            category_l2_list.append(papers["l2"][paper_idx])
    assert (
        len(users_ratings)
        == len(user_idx_list)
        == len(paper_id_list)
        == len(input_ids_list)
        == len(attention_mask_list)
    )
    assert len(users_ratings) == len(category_l1_list) == len(category_l2_list) == len(time_list)
    return {
        "user_idx": user_idx_list,
        "paper_id": paper_id_list,
        "rating": rating_list,
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "category_l1": category_l1_list,
        "category_l2": category_l2_list,
        "time": time_list,
    }


def get_finetuning_papers_dataset(
    dataset_type: str, users_type: str, users_coefs_ids_to_idxs: dict
) -> dict:
    users_ratings, papers, papers_ids_to_idxs = setup_finetuning_papers_dataset(
        dataset_type, users_type
    )
    dataset = fill_finetuning_papers_dataset(
        users_ratings, papers, papers_ids_to_idxs, users_coefs_ids_to_idxs
    )
    user_idx_set = set(dataset["user_idx"])
    if users_type == "train_users":
        assert sorted(list(user_idx_set)) == list(range(len(user_idx_set)))
    else:
        user_idx_min = min(user_idx_set)
        assert sorted(list(user_idx_set)) == list(
            range(user_idx_min, user_idx_min + len(user_idx_set))
        )
    return dataset


def save_finetuning_dataset(dataset_type: str) -> None:
    if dataset_type not in ["train", "val"]:
        raise ValueError(f"Invalid dataset type: {dataset_type}. Choose from ['train', 'val'].")
    if dataset_type == "train":
        tensor_path = ProjectPaths.finetuning_data_model_datasets_train_dataset_path()
    elif dataset_type == "val":
        tensor_path = ProjectPaths.finetuning_data_model_datasets_val_dataset_path()
    if tensor_path.exists():
        print(f"{dataset_type.capitalize()} dataset already exists - skipping saving.")
        return
    users_coefs_ids_to_idxs = load_users_coefs_ids_to_idxs()
    if dataset_type == "train":
        train_papers_train_users = get_finetuning_papers_dataset(
            "train", "train_users", users_coefs_ids_to_idxs
        )
        train_papers_val_users = get_finetuning_papers_dataset(
            "train", "val_users", users_coefs_ids_to_idxs
        )
        dataset = {
            key: train_papers_train_users[key] + train_papers_val_users[key]
            for key in train_papers_train_users
        }
    elif dataset_type == "val":
        dataset = get_finetuning_papers_dataset("val", "val_users", users_coefs_ids_to_idxs)
    time_values = []
    for t in dataset["time"]:
        if hasattr(t, "timestamp"):
            time_values.append(int(t.timestamp()))
        else:
            time_values.append(int(t))
    dataset = {
        "user_idx": torch.tensor(dataset["user_idx"], dtype=torch.int64),
        "paper_id": torch.tensor(dataset["paper_id"], dtype=torch.int64),
        "rating": torch.tensor(dataset["rating"], dtype=torch.int64),
        "input_ids": torch.stack(dataset["input_ids"]),
        "attention_mask": torch.stack(dataset["attention_mask"]),
        "category_l1": torch.tensor(dataset["category_l1"], dtype=torch.int64),
        "category_l2": torch.tensor(dataset["category_l2"], dtype=torch.int64),
        "time": torch.tensor(time_values, dtype=torch.int64),
    }
    torch.save(dataset, tensor_path)
    print(f"{dataset_type.capitalize()} dataset saved to {tensor_path}.")


def save_finetuning_datasets() -> None:
    dataset_types = ["train", "val"]
    for dataset_type in dataset_types:
        save_finetuning_dataset(dataset_type)


def load_finetuning_dataset(dataset_type: str, check: bool = False) -> dict:
    if dataset_type not in ["train", "val"]:
        raise ValueError(f"Invalid dataset type: {dataset_type}. Choose from ['train', 'val'].")
    if dataset_type == "train":
        tensor_path = ProjectPaths.finetuning_data_model_datasets_train_dataset_path()
    elif dataset_type == "val":
        tensor_path = ProjectPaths.finetuning_data_model_datasets_val_dataset_path()
    if not tensor_path.exists():
        raise FileNotFoundError(
            f"{dataset_type.capitalize()} dataset file not found: {tensor_path}."
        )
    dataset = torch.load(tensor_path, weights_only=True)
    if check:
        users_coefs_ids_to_idxs = load_users_coefs_ids_to_idxs()
        users_coefs_idxs_to_ids = {idx: user_id for user_id, idx in users_coefs_ids_to_idxs.items()}
        finetuning_users = load_finetuning_users()
        unique_users_idxs = torch.unique(dataset["user_idx"]).tolist()
        unique_users_ids = [users_coefs_idxs_to_ids[idx] for idx in unique_users_idxs]
        if dataset_type == "val":
            assert unique_users_ids == finetuning_users["val"]
        elif dataset_type == "train":
            assert unique_users_ids == (finetuning_users["train"] + finetuning_users["val"])
    return dataset


def save_finetuning_model(model_path: str = ProjectPaths.finetuning_data_model_hf()) -> None:
    tensor_path = ProjectPaths.finetuning_data_model_state_dicts_path() / "transformer_model"
    if tensor_path.exists():
        print("Transformer model already exists - skipping saving.")
        return
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, unpad_inputs=True, torch_dtype="auto"
    )
    model.save_pretrained(tensor_path)


def test_loading() -> None:
    val_users_embeddings_idxs = load_val_users_embeddings_idxs()
    print(f"Loaded {len(val_users_embeddings_idxs)} validation users embeddings indices.")
    categories_to_idxs_l1 = load_categories_to_idxs("l1")
    print(f"Loaded {len(categories_to_idxs_l1)} categories for L1.")
    categories_to_idxs_l2 = load_categories_to_idxs("l2")
    print(f"Loaded {len(categories_to_idxs_l2)} categories for L2.")
    test_papers = load_finetuning_papers("test", attach_l1=True, attach_l2=True)
    print(f"Loaded {len(test_papers['paper_id'])} test papers with L1 and L2 categories.")
    train_dataset = load_finetuning_dataset("train", check=True)
    print(f"Loaded train dataset with {len(train_dataset['user_idx'])} entries.")
    val_dataset = load_finetuning_dataset("val", check=True)
    print(f"Loaded validation dataset with {len(val_dataset['user_idx'])} entries.")
    unique_train_users = torch.unique(train_dataset["user_idx"]).tolist()
    unique_val_users = torch.unique(val_dataset["user_idx"]).tolist()
    for train_user in unique_train_users:
        time_tensor_pos = train_dataset["time"][
            (train_dataset["user_idx"] == train_user) & (train_dataset["rating"] == 1)
        ]
        time_tensor_neg = train_dataset["time"][
            (train_dataset["user_idx"] == train_user) & (train_dataset["rating"] == 0)
        ]
        assert torch.all(torch.diff(time_tensor_pos) >= 0) and torch.all(
            torch.diff(time_tensor_neg) >= 0
        )
    for val_user in unique_val_users:
        time_tensor_pos = val_dataset["time"][
            (val_dataset["user_idx"] == val_user) & (val_dataset["rating"] == 1)
        ]
        time_tensor_neg = val_dataset["time"][
            (val_dataset["user_idx"] == val_user) & (val_dataset["rating"] == 0)
        ]
        assert torch.all(torch.diff(time_tensor_pos) >= 0) and torch.all(
            torch.diff(time_tensor_neg) >= 0
        )


if __name__ == "__main__":
    os.makedirs(ProjectPaths.finetuning_data_model_path(), exist_ok=True)
    os.makedirs(ProjectPaths.finetuning_data_model_state_dicts_path(), exist_ok=True)
    os.makedirs(ProjectPaths.finetuning_data_model_datasets_path(), exist_ok=True)
    save_users_embeddings_tensor()
    save_projection_tensor()
    save_categories_embeddings_tensor()
    save_negative_samples_ids_for_seeds()
    save_finetuning_papers()
    save_finetuning_datasets()
    save_finetuning_model()
    test_loading()
