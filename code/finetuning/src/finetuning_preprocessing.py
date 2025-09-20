import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ...logreg.src.embeddings.papers_categories import get_glove_categories_embeddings
from ...logreg.src.embeddings.papers_categories_dicts import PAPERS_CATEGORIES
from ...logreg.src.training.training_data import (
    get_cache_papers_ids,
    get_categories_ratios_for_validation,
    get_categories_samples_ids,
    get_user_categories_ratios,
    get_user_categories_samples_ids,
)
from ...logreg.src.training.users_ratings import (
    UsersRatingsSelection,
    load_users_ratings_from_selection,
)
from ...src.load_files import (
    TEST_RANDOM_STATES,
    VAL_RANDOM_STATE,
    load_finetuning_users_ids,
    load_papers,
    load_papers_texts,
    load_sequence_users_ids,
    load_users_ratings,
    load_users_significant_categories,
)
from ...src.project_paths import ProjectPaths

N_VAL_NEGATIVE_SAMPLES = 100
N_TRAIN_NEGATIVE_SAMPLES_PER_CATEGORY_MAX = 150000


def save_transformer_model(model_path: Path) -> None:
    if model_path.exists():
        print("Transformer model already exists - skipping saving.")
        return
    transformer_model = AutoModel.from_pretrained(
        ProjectPaths.finetuning_data_model_hf(),
        trust_remote_code=True,
        unpad_inputs=True,
        torch_dtype="auto",
    )
    transformer_model.save_pretrained(model_path)
    print(f"Transformer model saved to {model_path}.")


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
    val_users_ids: list = None, users_coefs_ids_to_idxs: dict = None, no_seq_eval: bool = False
) -> torch.Tensor:
    if val_users_ids is None:
        val_users_ids = load_finetuning_users_ids(selection="val", no_seq_eval=no_seq_eval)
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


def save_projection_tensor(
    projection_tensor_path=ProjectPaths.finetuning_data_model_state_dicts_projection_path(),
    pca_components: np.ndarray = None,
    pca_mean: np.ndarray = None,
) -> None:
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


def save_categories_embeddings_tensor(
    categories_embeddings_l1_path: Path = ProjectPaths.finetuning_data_model_state_dicts_categories_embeddings_l1_path(),
    categories_to_idxs_l1_path: Path = ProjectPaths.finetuning_data_model_state_dicts_path()
    / "categories_to_idxs_l1.pkl",
    categories_to_idxs_l2_path: Path = ProjectPaths.finetuning_data_model_state_dicts_path()
    / "categories_to_idxs_l2.pkl",
    glove_categories_embeddings: dict = None,
) -> None:
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


def get_eval_papers_ids(
    users_ratings: pd.DataFrame, random_states: list, include_cache: bool = True
) -> list:
    papers = load_papers(relevant_columns=["paper_id", "in_ratings", "l1"])
    eval_papers_ids = set()
    eval_papers_ids.update(users_ratings["paper_id"].unique().tolist())
    for random_state in tqdm(random_states):
        val_negative_samples_ids = get_categories_samples_ids(
            papers=papers,
            n_categories_samples=N_VAL_NEGATIVE_SAMPLES,
            random_state=random_state,
            papers_ids_to_exclude=papers[papers["in_ratings"]]["paper_id"].tolist(),
        )[1]
        eval_papers_ids.update(val_negative_samples_ids)
        if include_cache:
            cache_papers_ids = get_cache_papers_ids(
                cache_type="categories_cache",
                papers=papers,
                n_cache=5000,
                random_state=random_state,
            )
            eval_papers_ids.update(cache_papers_ids)
    return sorted(list(eval_papers_ids))


def get_eval_papers_ids_val_users(mode: str) -> list:
    random_states = [VAL_RANDOM_STATE]
    if mode == "finetuning":
        val_users_ids = load_finetuning_users_ids(selection="val")
        include_cache = True
    elif mode == "sequence":
        val_users_ids = load_sequence_users_ids(selection="val")
        include_cache = False
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose from ['finetuning', 'sequence'].")
    users_ratings = load_users_ratings(relevant_users_ids=val_users_ids)
    return get_eval_papers_ids(users_ratings, random_states, include_cache)


def get_eval_papers_ids_test_users(mode: str) -> list:
    random_states = sorted(TEST_RANDOM_STATES)
    if mode == "finetuning":
        test_users_ids = load_finetuning_users_ids(selection="test")
        include_cache = True
    elif mode == "sequence":
        test_users_ids = load_sequence_users_ids(selection="test")
        include_cache = False
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose from ['finetuning', 'sequence'].")
    users_ratings = load_users_ratings(relevant_users_ids=test_users_ids)
    return get_eval_papers_ids(users_ratings, random_states, include_cache)


def save_eval_papers_tokenized_val_test_users(
    selection: str, tokenizer: AutoTokenizer, max_sequence_length: int, mode: str = "finetuning"
) -> None:
    if mode == "finetuning":
        path = ProjectPaths.finetuning_data_model_datasets_path()
    elif mode == "sequence":
        path = ProjectPaths.sequence_data_model_datasets_path()
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose from ['finetuning', 'sequence'].")

    selections = ["val_users", "test_users"]
    if selection not in selections:
        raise ValueError(f"Invalid selection: {selection}. Choose from {selections}.")
    tensor_path = path / f"eval_papers_tokenized_{selection}.pt"
    if tensor_path.exists():
        print(f"{selection.capitalize()} eval papers tokenized already exist - skipping saving.")
        return
    if selection == "val_users":
        eval_papers_ids = get_eval_papers_ids_val_users(mode)
    elif selection == "test_users":
        eval_papers_ids = get_eval_papers_ids_test_users(mode)
    eval_papers_tokenized = finetuning_tokenize_papers(
        papers_ids=eval_papers_ids,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
    )[0]
    torch.save(eval_papers_tokenized, tensor_path)
    print(f"{selection.capitalize()} eval papers tokenized saved to {tensor_path}.")


def save_eval_papers_tokenized(
    tokenizer: AutoTokenizer = None,
    max_sequence_length: int = 512,
    mode: str = "finetuning",
) -> None:
    assert mode in ["finetuning", "sequence"]
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(ProjectPaths.finetuning_data_model_hf())
    save_eval_papers_tokenized_val_test_users("val_users", tokenizer, max_sequence_length, mode)
    save_eval_papers_tokenized_val_test_users("test_users", tokenizer, max_sequence_length, mode)


def get_negative_samples_matrix_val(users_ids: list, val_negative_samples_ids: dict) -> np.ndarray:
    assert len(users_ids) == len(set(users_ids)) and users_ids == sorted(users_ids)
    users_significant_categories = load_users_significant_categories(
        relevant_users_ids=users_ids,
    )
    negative_samples_matrix_val = np.zeros((len(users_ids), N_VAL_NEGATIVE_SAMPLES), dtype=np.int64)
    for i, user_id in enumerate(users_ids):
        user_significant_categories = users_significant_categories[
            users_significant_categories["user_id"] == user_id
        ]["category"].tolist()
        user_categories_ratios = get_user_categories_ratios(
            categories_to_exclude=user_significant_categories
        )
        user_val_negative_samples_ids = get_user_categories_samples_ids(
            categories_samples_ids=val_negative_samples_ids,
            n_categories_samples=N_VAL_NEGATIVE_SAMPLES,
            random_state=VAL_RANDOM_STATE,
            sort_samples=True,
            user_categories_ratios=user_categories_ratios,
        )
        negative_samples_matrix_val[i, :] = np.array(user_val_negative_samples_ids)
    return negative_samples_matrix_val


def save_negative_samples_val(
    tensor_path: Path = None,
    matrix_path: Path = None,
    users_ids: list = None,
    tokenizer: AutoTokenizer = None,
    max_sequence_length: int = 512,
    no_seq_eval: bool = False,
) -> None:
    if tensor_path is None:
        tensor_path = (
            ProjectPaths.finetuning_data_model_datasets_negative_samples_tokenized_val_path()
        )
    if matrix_path is None:
        matrix_path = ProjectPaths.finetuning_data_model_datasets_negative_samples_matrix_val_path()
        if no_seq_eval:
            matrix_path = (
                ProjectPaths.finetuning_data_model_datasets_path()
                / "negative_samples_matrix_val_no_seq_eval.pt"
            )
    if tensor_path.exists() and matrix_path.exists():
        print("Validation negative samples tensor and matrix already exist - skipping saving.")
        return
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(ProjectPaths.finetuning_data_model_hf())
    papers = load_papers(relevant_columns=["paper_id", "in_ratings", "l1"])
    if users_ids is None:
        users_ids = load_finetuning_users_ids(selection="val", no_seq_eval=no_seq_eval)
    negative_samples_ids, negative_samples_ids_flattened = get_categories_samples_ids(
        papers=papers,
        n_categories_samples=N_VAL_NEGATIVE_SAMPLES,
        random_state=VAL_RANDOM_STATE,
        papers_ids_to_exclude=papers[papers["in_ratings"]]["paper_id"].tolist(),
    )
    negative_samples_matrix = get_negative_samples_matrix_val(users_ids, negative_samples_ids)
    negative_samples, negative_samples_ids_to_idxs = finetuning_tokenize_papers(
        papers_ids=negative_samples_ids_flattened,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
    )
    matrix_map = np.vectorize(negative_samples_ids_to_idxs.get)
    negative_samples_matrix = matrix_map(negative_samples_matrix)
    negative_samples_matrix = torch.from_numpy(negative_samples_matrix).to(torch.int64)
    if not no_seq_eval:
        torch.save(negative_samples, tensor_path)
        print(f"Validation negative samples tensor saved to {tensor_path}.")
    torch.save(negative_samples_matrix, matrix_path)
    print(f"Validation negative samples matrix tensor saved to {matrix_path}.")


def load_negative_samples_matrix_val(
    matrix_path: Path = None, no_seq_eval: bool = False
) -> torch.Tensor:
    if matrix_path is None:
        matrix_path = ProjectPaths.finetuning_data_model_datasets_negative_samples_matrix_val_path()
        if no_seq_eval:
            matrix_path = (
                ProjectPaths.finetuning_data_model_datasets_path()
                / "negative_samples_matrix_val_no_seq_eval.pt"
            )
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Validation negative samples matrix tensor not found: {matrix_path}."
        )
    negative_samples_matrix = torch.load(matrix_path, weights_only=True)
    assert negative_samples_matrix.dtype == torch.int64
    return negative_samples_matrix


def save_rated_papers_tokenized_train_val_users(
    selection: str, tokenizer: AutoTokenizer, max_sequence_length: int
) -> None:
    selections = ["train_users", "val_users"]
    if selection not in selections:
        raise ValueError(f"Invalid selection: {selection}. Choose from {selections}.")
    tensor_path = (
        ProjectPaths.finetuning_data_model_datasets_path()
        / f"rated_papers_tokenized_{selection}.pt"
    )
    papers_ids_to_idxs_path = (
        ProjectPaths.finetuning_data_model_datasets_path()
        / f"rated_papers_ids_to_idxs_{selection}.pkl"
    )
    if tensor_path.exists() and papers_ids_to_idxs_path.exists():
        print(f"{selection.capitalize()} papers already exist - skipping saving.")
        return
    if tensor_path.exists() or papers_ids_to_idxs_path.exists():
        raise FileExistsError(
            f"One or more but not all of the files already exist: {tensor_path}, {papers_ids_to_idxs_path}."
        )
    selection = "train" if selection == "train_users" else "val"
    users_ids = load_finetuning_users_ids(selection=selection)
    users_ratings = load_users_ratings(
        relevant_columns=["user_id", "paper_id"], relevant_users_ids=users_ids
    )
    papers_ids = sorted(list(users_ratings["paper_id"].unique()))
    papers_dict, papers_ids_to_idxs = finetuning_tokenize_papers(
        papers_ids, tokenizer, max_sequence_length
    )
    torch.save(papers_dict, tensor_path)
    print(f"{selection.capitalize()} papers tensor saved to {tensor_path}.")
    with open(papers_ids_to_idxs_path, "wb") as f:
        pickle.dump(papers_ids_to_idxs, f)
    print(
        f"{selection.capitalize()} papers IDs to indices mapping saved to {papers_ids_to_idxs_path}."
    )


def save_rated_papers_tokenized(
    tokenizer: AutoTokenizer = None,
    max_sequence_length: int = 512,
) -> None:
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(ProjectPaths.finetuning_data_model_hf())
    save_rated_papers_tokenized_train_val_users("train_users", tokenizer, max_sequence_length)
    save_rated_papers_tokenized_train_val_users("val_users", tokenizer, max_sequence_length)


def load_rated_papers_ids_to_idxs(selection: str) -> dict:
    selections = ["train_users", "val_users"]
    if selection not in selections:
        raise ValueError(f"Invalid selection: {selection}. Choose from {selections}.")
    papers_ids_to_idxs_path = (
        ProjectPaths.finetuning_data_model_datasets_path()
        / f"rated_papers_ids_to_idxs_{selection}.pkl"
    )
    if not papers_ids_to_idxs_path.exists():
        raise FileNotFoundError(
            f"Rated papers IDs to indices mapping file not found: {papers_ids_to_idxs_path}."
        )
    with open(papers_ids_to_idxs_path, "rb") as f:
        papers_ids_to_idxs = pickle.load(f)
    assert list(papers_ids_to_idxs.keys()) == sorted(list(papers_ids_to_idxs.keys()))
    assert list(papers_ids_to_idxs.values()) == list(range(len(papers_ids_to_idxs)))
    return papers_ids_to_idxs


def attach_categories_to_papers_tensor(
    papers_tensor: dict,
    attach_l1: bool = True,
    attach_l2: bool = True,
    papers: pd.DataFrame = None,
    categories_to_idxs_l1: dict = None,
    categories_to_idxs_l2: dict = None,
) -> dict:
    if attach_l1 or attach_l2:
        if papers is None:
            papers = load_papers(
                relevant_columns=["paper_id", "l1", "l2"],
                relevant_papers_ids=papers_tensor["paper_id"].tolist(),
            )
        else:
            papers = papers[papers["paper_id"].isin(papers_tensor["paper_id"].tolist())]
        assert papers_tensor["paper_id"].tolist() == papers["paper_id"].tolist()
        if attach_l1:
            if categories_to_idxs_l1 is None:
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
            if categories_to_idxs_l2 is None:
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


def load_finetuning_papers_tokenized(
    papers_type: str,
    attach_l1: bool = True,
    attach_l2: bool = True,
    papers: pd.DataFrame = None,
    categories_to_idxs_l1: dict = None,
    categories_to_idxs_l2: dict = None,
) -> dict:
    papers_types = [
        "eval_val_users",
        "eval_test_users",
        "negative_samples_val",
        "rated_train_users",
        "rated_val_users",
    ]
    if papers_type not in papers_types:
        raise ValueError(f"Invalid papers type: {papers_type}. Choose from {papers_types}.")
    if papers_type == "eval_val_users":
        tensor_path = (
            ProjectPaths.finetuning_data_model_datasets_eval_papers_tokenized_val_users_path()
        )
    elif papers_type == "eval_test_users":
        tensor_path = (
            ProjectPaths.finetuning_data_model_datasets_eval_papers_tokenized_test_users_path()
        )
    elif papers_type == "negative_samples_val":
        tensor_path = (
            ProjectPaths.finetuning_data_model_datasets_negative_samples_tokenized_val_path()
        )
    elif papers_type in ["rated_train_users", "rated_val_users"]:
        selection = papers_type.replace("rated_", "")
        tensor_path = (
            ProjectPaths.finetuning_data_model_datasets_path()
            / f"rated_papers_tokenized_{selection}.pt"
        )
    else:
        tensor_path = (
            ProjectPaths.finetuning_data_model_datasets_path()
            / f"{papers_type}_papers_tokenized.pt"
        )
    papers_tensor = torch.load(tensor_path, weights_only=True)
    papers_tensor = attach_categories_to_papers_tensor(
        papers_tensor,
        attach_l1=attach_l1,
        attach_l2=attach_l2,
        papers=papers,
        categories_to_idxs_l1=categories_to_idxs_l1,
        categories_to_idxs_l2=categories_to_idxs_l2,
    )
    assert papers_tensor["paper_id"].tolist() == sorted(papers_tensor["paper_id"].tolist())
    assert len(papers_tensor["paper_id"].tolist()) == len(set(papers_tensor["paper_id"].tolist()))
    return papers_tensor


def get_negative_samples_ids_per_category_dict_train(
    finetuning: bool = True,
    n_train_negative_samples_per_category_max: int = N_TRAIN_NEGATIVE_SAMPLES_PER_CATEGORY_MAX,
    selection_random_state: int = None,
) -> dict:
    papers = load_papers(relevant_columns=["paper_id", "in_ratings", "l1"])
    ratings_papers_ids = papers[papers["in_ratings"]]["paper_id"].tolist()
    if finetuning:
        val_users_ids = load_finetuning_users_ids(selection="val")
        test_users_ids = load_finetuning_users_ids(selection="test")
    else:
        val_users_ids = load_sequence_users_ids(selection="val")
        test_users_ids = load_sequence_users_ids(selection="test")
    eval_users_ids = sorted(list(set(val_users_ids + test_users_ids)))
    users_ratings = load_users_ratings(relevant_users_ids=eval_users_ids)
    papers_ids_to_exclude = set(users_ratings["paper_id"].unique().tolist())
    for random_state in [VAL_RANDOM_STATE] + TEST_RANDOM_STATES:
        val_negative_samples_ids = get_categories_samples_ids(
            papers=papers,
            n_categories_samples=N_VAL_NEGATIVE_SAMPLES,
            random_state=random_state,
            papers_ids_to_exclude=ratings_papers_ids,
        )[1]
        papers_ids_to_exclude.update(val_negative_samples_ids)
    papers = papers[papers["l1"].notna()]
    papers = papers[~papers["paper_id"].isin(papers_ids_to_exclude)]
    papers_ids_per_category_dict = {}
    categories = list(get_categories_ratios_for_validation().keys())
    for category in categories:
        category_papers = papers[papers["l1"] == category]
        if n_train_negative_samples_per_category_max is not None:
            if len(category_papers) > n_train_negative_samples_per_category_max:
                category_papers = category_papers.sample(
                    n=n_train_negative_samples_per_category_max,
                    random_state=selection_random_state,
                    replace=False,
                )
        category_papers_ids = sorted(list(category_papers["paper_id"].unique()))
        assert len(category_papers_ids) == len(set(category_papers_ids))
        assert category_papers_ids == sorted(category_papers_ids)
        if n_train_negative_samples_per_category_max is not None:
            assert len(category_papers_ids) <= n_train_negative_samples_per_category_max
        papers_ids_per_category_dict[category] = category_papers_ids
    return papers_ids_per_category_dict


def save_negative_samples_tokenized_train(
    selection_random_state: int = 42,
    tokenizer: AutoTokenizer = None,
    max_sequence_length: int = 512,
) -> None:
    tensor_path = (
        ProjectPaths.finetuning_data_model_datasets_negative_samples_tokenized_train_path()
    )
    if tensor_path.exists():
        print("Negative samples tokenized for train already exist - skipping saving.")
        return
    papers_ids_per_category_dict = get_negative_samples_ids_per_category_dict_train(
        selection_random_state=selection_random_state
    )
    papers_tokenized_per_category_dict = {}
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(ProjectPaths.finetuning_data_model_hf())
    for category, papers_ids in papers_ids_per_category_dict.items():
        papers_tensor = finetuning_tokenize_papers(
            papers_ids=papers_ids,
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
        )[0]
        papers_tokenized_per_category_dict[category] = papers_tensor
        assert papers_tensor["paper_id"].tolist() == papers_ids
    with open(tensor_path, "wb") as f:
        torch.save(papers_tokenized_per_category_dict, f)
    print(f"Negative samples tokenized for train saved to {tensor_path}.")


def shuffle_negative_samples_tokenized_train(
    negative_samples_tokenized_train: dict,
    random_state: int,
) -> dict:
    assert isinstance(negative_samples_tokenized_train, dict)
    for category, papers_tensor in negative_samples_tokenized_train.items():
        permuted_indices = torch.randperm(
            len(papers_tensor["paper_id"]),
            generator=torch.Generator().manual_seed(random_state),
        )
        for key in papers_tensor:
            papers_tensor[key] = papers_tensor[key][permuted_indices]
        negative_samples_tokenized_train[category] = papers_tensor
    return negative_samples_tokenized_train


def load_negative_samples_tokenized_train(
    attach_l1: bool = True,
    attach_l2: bool = True,
    papers: pd.DataFrame = None,
    categories_to_idxs_l1: dict = None,
    categories_to_idxs_l2: dict = None,
    shuffle_papers: bool = False,
    random_state: int = None,
) -> dict:
    negative_samples_tokenized_train = torch.load(
        ProjectPaths.finetuning_data_model_datasets_negative_samples_tokenized_train_path(),
        weights_only=True,
    )
    if attach_l1 and categories_to_idxs_l1 is None:
        categories_to_idxs_l1 = load_categories_to_idxs("l1")
    if attach_l2 and categories_to_idxs_l2 is None:
        categories_to_idxs_l2 = load_categories_to_idxs("l2")
    papers = load_papers()
    if attach_l1 or attach_l2:
        for category, papers_tensor in negative_samples_tokenized_train.items():
            papers_tensor = attach_categories_to_papers_tensor(
                papers_tensor,
                attach_l1=attach_l1,
                attach_l2=attach_l2,
                papers=papers.copy(),
                categories_to_idxs_l1=categories_to_idxs_l1,
                categories_to_idxs_l2=categories_to_idxs_l2,
            )
            negative_samples_tokenized_train[category] = papers_tensor
    if shuffle_papers:
        assert random_state is not None
        negative_samples_tokenized_train = shuffle_negative_samples_tokenized_train(
            negative_samples_tokenized_train, random_state=random_state
        )
    return negative_samples_tokenized_train


def gather_finetuning_dataset(dataset_type: str, users_type: str) -> tuple:
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
        load_finetuning_users_ids(selection="train")
        if users_type == "train_users"
        else load_finetuning_users_ids(selection="val")
    )
    papers_tokenized = load_finetuning_papers_tokenized(
        papers_type=f"rated_{users_type}",
        attach_l1=True,
        attach_l2=True,
    )
    papers_ids_to_idxs = load_rated_papers_ids_to_idxs(users_type)
    if users_type == "train_users":
        users_ratings = load_users_ratings(
            relevant_columns=["user_id", "paper_id", "rating", "time"],
            relevant_users_ids=users_ids,
        )
    else:
        users_ratings = load_users_ratings_from_selection(
            users_ratings_selection=UsersRatingsSelection.SESSION_BASED_NO_FILTERING,
            relevant_users_ids="finetuning_val",
        )
        if dataset_type == "train":
            users_ratings = users_ratings[users_ratings["split"] == "train"]
        elif dataset_type == "val":
            users_ratings = users_ratings[users_ratings["split"] == "val"]
    return users_ratings, papers_tokenized, papers_ids_to_idxs


def get_finetuning_dataset_dict_from_gathering(
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


def get_finetuning_dataset_dict(
    dataset_type: str, users_type: str, users_coefs_ids_to_idxs: dict
) -> dict:
    users_ratings, papers, papers_ids_to_idxs = gather_finetuning_dataset(dataset_type, users_type)
    dataset = get_finetuning_dataset_dict_from_gathering(
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
        tensor_path = ProjectPaths.finetuning_data_model_datasets_dataset_train_path()
    elif dataset_type == "val":
        tensor_path = ProjectPaths.finetuning_data_model_datasets_dataset_val_path()
    if tensor_path.exists():
        print(f"{dataset_type.capitalize()} dataset already exists - skipping saving.")
        return
    users_coefs_ids_to_idxs = load_users_coefs_ids_to_idxs()
    if dataset_type == "train":
        train_papers_train_users = get_finetuning_dataset_dict(
            "train", "train_users", users_coefs_ids_to_idxs
        )
        train_papers_val_users = get_finetuning_dataset_dict(
            "train", "val_users", users_coefs_ids_to_idxs
        )
        dataset = {
            key: train_papers_train_users[key] + train_papers_val_users[key]
            for key in train_papers_train_users
        }
    elif dataset_type == "val":
        dataset = get_finetuning_dataset_dict("val", "val_users", users_coefs_ids_to_idxs)
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


def load_finetuning_dataset(
    dataset_type: str, check: bool = False, no_seq_eval: bool = False
) -> dict:
    if dataset_type not in ["train", "val"]:
        raise ValueError(f"Invalid dataset type: {dataset_type}. Choose from ['train', 'val'].")
    if dataset_type == "train":
        tensor_path = ProjectPaths.finetuning_data_model_datasets_dataset_train_path()
    elif dataset_type == "val":
        tensor_path = ProjectPaths.finetuning_data_model_datasets_dataset_val_path()
    if not tensor_path.exists():
        raise FileNotFoundError(
            f"{dataset_type.capitalize()} dataset file not found: {tensor_path}."
        )
    dataset = torch.load(tensor_path, weights_only=True)
    if no_seq_eval:
        users_ids = load_finetuning_users_ids(no_seq_eval=True)
        if dataset_type == "train":
            users_ids = users_ids["train"] + users_ids["val"]
        elif dataset_type == "val":
            users_ids = users_ids["val"]
        users_coefs_ids_to_idxs = load_users_coefs_ids_to_idxs()
        users_idxs_to_keep = [users_coefs_ids_to_idxs[user_id] for user_id in users_ids]
        mask = torch.isin(dataset["user_idx"], torch.tensor(users_idxs_to_keep))
        dataset = {key: value[mask] for key, value in dataset.items()}
    if check:
        users_coefs_ids_to_idxs = load_users_coefs_ids_to_idxs()
        users_coefs_idxs_to_ids = {idx: user_id for user_id, idx in users_coefs_ids_to_idxs.items()}
        finetuning_users_ids = load_finetuning_users_ids(no_seq_eval=no_seq_eval)
        unique_users_idxs = torch.unique(dataset["user_idx"]).tolist()
        unique_users_ids = [users_coefs_idxs_to_ids[idx] for idx in unique_users_idxs]
        if dataset_type == "val":
            assert unique_users_ids == finetuning_users_ids["val"]
        elif dataset_type == "train":
            assert unique_users_ids == finetuning_users_ids["train"] + finetuning_users_ids["val"]
    return dataset


def test_loading() -> None:
    val_users_embeddings_idxs = load_val_users_embeddings_idxs()
    print(f"Loaded {len(val_users_embeddings_idxs)} validation users embeddings indices.")
    categories_to_idxs_l1 = load_categories_to_idxs("l1")
    print(f"Loaded {len(categories_to_idxs_l1)} categories for L1.")
    categories_to_idxs_l2 = load_categories_to_idxs("l2")
    print(f"Loaded {len(categories_to_idxs_l2)} categories for L2.")

    eval_papers_tokenized_val_users = load_finetuning_papers_tokenized("eval_val_users")
    n_eval_papers_val_users = len(eval_papers_tokenized_val_users["paper_id"])
    print(f"Loaded eval papers tokenized for val users with {n_eval_papers_val_users} entries.")
    eval_papers_tokenized_test_users = load_finetuning_papers_tokenized("eval_test_users")
    n_eval_papers_test_users = len(eval_papers_tokenized_test_users["paper_id"])
    print(f"Loaded eval papers tokenized for test users with {n_eval_papers_test_users} entries.")

    negative_samples_val = load_finetuning_papers_tokenized("negative_samples_val")
    n_negative_samples_val = len(negative_samples_val["paper_id"])
    negative_samples_matrix_val = load_negative_samples_matrix_val()
    assert negative_samples_matrix_val.shape[0] == len(load_finetuning_users_ids("val"))
    print(f"Loaded negative samples tokenized for val with {n_negative_samples_val} entries.")

    negative_samples_train = load_negative_samples_tokenized_train(
        shuffle_papers=True,
        random_state=42,
    )
    for category, papers_tensor in negative_samples_train.items():
        n_negative_samples_train = len(papers_tensor["paper_id"])
        assert n_negative_samples_train <= N_TRAIN_NEGATIVE_SAMPLES_PER_CATEGORY_MAX
        print(
            f"Loaded negative samples tokenized for train category '{category}'"
            f" with {n_negative_samples_train} entries."
        )
        assert papers_tensor["paper_id"].tolist() != sorted(papers_tensor["paper_id"].tolist())

    rated_papers_tokenized_train_users = load_finetuning_papers_tokenized("rated_train_users")
    n_rated_papers_train_users = len(rated_papers_tokenized_train_users["paper_id"])
    rated_papers_ids_to_idxs_train_users = load_rated_papers_ids_to_idxs("train_users")
    assert len(rated_papers_ids_to_idxs_train_users) == n_rated_papers_train_users
    print(
        f"Loaded rated papers tokenized for train users with {n_rated_papers_train_users} entries."
    )
    rated_papers_tokenized_val_users = load_finetuning_papers_tokenized("rated_val_users")
    n_rated_papers_val_users = len(rated_papers_tokenized_val_users["paper_id"])
    rated_papers_ids_to_idxs_val_users = load_rated_papers_ids_to_idxs("val_users")
    assert len(rated_papers_ids_to_idxs_val_users) == n_rated_papers_val_users
    print(f"Loaded rated papers tokenized for val users with {n_rated_papers_val_users} entries.")

    dataset_train = load_finetuning_dataset("train", check=True)
    print(f"Loaded train dataset with {len(dataset_train['user_idx'])} entries.")
    dataset_val = load_finetuning_dataset("val", check=True)
    print(f"Loaded validation dataset with {len(dataset_val['user_idx'])} entries.")

    unique_train_users = torch.unique(dataset_train["user_idx"]).tolist()
    unique_val_users = torch.unique(dataset_val["user_idx"]).tolist()
    for train_user in unique_train_users:
        time_tensor_pos = dataset_train["time"][
            (dataset_train["user_idx"] == train_user) & (dataset_train["rating"] == 1)
        ]
        time_tensor_neg = dataset_train["time"][
            (dataset_train["user_idx"] == train_user) & (dataset_train["rating"] == 0)
        ]
        assert torch.all(torch.diff(time_tensor_pos) >= 0) and torch.all(
            torch.diff(time_tensor_neg) >= 0
        )
    for val_user in unique_val_users:
        time_tensor_pos = dataset_val["time"][
            (dataset_val["user_idx"] == val_user) & (dataset_val["rating"] == 1)
        ]
        time_tensor_neg = dataset_val["time"][
            (dataset_val["user_idx"] == val_user) & (dataset_val["rating"] == 0)
        ]
        assert torch.all(torch.diff(time_tensor_pos) >= 0) and torch.all(
            torch.diff(time_tensor_neg) >= 0
        )


if __name__ == "__main__":
    os.makedirs(ProjectPaths.finetuning_data_model_path(), exist_ok=True)
    os.makedirs(ProjectPaths.finetuning_data_model_state_dicts_path(), exist_ok=True)
    os.makedirs(ProjectPaths.finetuning_data_model_datasets_path(), exist_ok=True)
    save_transformer_model(
        model_path=ProjectPaths.finetuning_data_model_path() / "state_dicts" / "transformer_model"
    )
    save_users_embeddings_tensor()
    save_projection_tensor()
    save_categories_embeddings_tensor()
    save_eval_papers_tokenized()
    save_negative_samples_val()
    save_negative_samples_val(no_seq_eval=True)
    save_rated_papers_tokenized()
    save_finetuning_datasets()
    save_negative_samples_tokenized_train()
    load_negative_samples_tokenized_train()
    test_loading()
