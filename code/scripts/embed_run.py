import argparse
import shutil
import subprocess
import sys

from ..src.project_paths import ProjectPaths

MODEL_CHOICES = [
    "gte-base-en-v1.5",
    "gte-large-en-v1.5",
    "specter2_base",
    "Qwen3-Embedding-0.6B",
    "Qwen3-Embedding-4B",
    "Qwen3-Embedding-8B",
    "gte_qwen1p5_7B_instruct",
    "gte_Qwen2_7B_instruct",
    "F2LLM-0.6B",
    "F2LLM-1.7B",
    "F2LLM-4B",
]
EMBEDDINGS_FOLDER = ProjectPaths.logreg_embeddings_path() / "before_pca"
MAX_SEQUENCE_LENGTH = 512
PCA_DIM = 256


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_CHOICES)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--all_papers", action="store_true", default=False)
    parser.add_argument("--existing_embedding", type=str, required=False)
    args_dict = vars(parser.parse_args())
    return args_dict


args_dict = parse_args()
model_name, batch_size = args_dict["model_name"], args_dict["batch_size"]
if model_name == "gte-base-en-v1.5":
    model_abbreviation = "gte_base"
    model_path = f"Alibaba-NLP/{model_name}"
    if batch_size is None:
        batch_size = 100
elif model_name == "gte-large-en-v1.5":
    model_abbreviation = "gte_large"
    model_path = f"Alibaba-NLP/{model_name}"
    if batch_size is None:
        batch_size = 100
elif model_name == "specter2_base":
    model_abbreviation = "specter2"
    model_path = f"allenai/{model_name}"
    if batch_size is None:
        batch_size = 1000
elif model_name == "Qwen3-Embedding-0.6B":
    model_abbreviation = "qwen3_0p6B"
    model_path = f"Qwen/{model_name}"
    if batch_size is None:
        batch_size = 100
elif model_name == "Qwen3-Embedding-4B":
    model_abbreviation = "qwen3_4B"
    model_path = f"Qwen/{model_name}"
    if batch_size is None:
        batch_size = 50
elif model_name == "Qwen3-Embedding-8B":
    model_abbreviation = "qwen3_8B"
    model_path = f"Qwen/{model_name}"
    if batch_size is None:
        batch_size = 25
elif model_name == "gte_qwen1p5_7B_instruct":
    model_abbreviation = "gte_qwen1p5_7B_instruct"
    model_path = "Alibaba-NLP/gte-qwen1p5-7B-instruct"
    if batch_size is None:
        batch_size = 10
elif model_name == "gte_Qwen2_7B_instruct":
    model_abbreviation = "gte_Qwen2_7B_instruct"
    model_path = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    if batch_size is None:
        batch_size = 20
elif model_name == "F2LLM-0.6B":
    model_abbreviation = "F2LLM_0p6B"
    model_path = "codefuse-ai/F2LLM-0.6B"
    if batch_size is None:
        batch_size = 100
elif model_name == "F2LLM-1.7B":
    model_abbreviation = "F2LLM_1p7B"
    model_path = "codefuse-ai/F2LLM-1.7B"
    if batch_size is None:
        batch_size = 50
elif model_name == "F2LLM-4B":
    model_abbreviation = "F2LLM_4B"
    model_path = "codefuse-ai/F2LLM-4B"
    if batch_size is None:
        batch_size = 25


embeddings_folder = EMBEDDINGS_FOLDER / f"{model_abbreviation}"

subprocess.run(
    [
        sys.executable,
        "-m",
        "code.logreg.src.embeddings.compute_embeddings",
        "--model_path",
        model_path,
        "--embeddings_folder",
        str(embeddings_folder),
        "--max_batch_size",
        str(batch_size),
        "--max_sequence_length",
        str(MAX_SEQUENCE_LENGTH),
    ]
    + (
        ["--existing_embedding", args_dict["existing_embedding"]]
        if args_dict["existing_embedding"]
        else []
    ),
    check=True,
)

subprocess.run(
    [
        sys.executable,
        "-m",
        "code.logreg.src.embeddings.merge_embeddings",
        str(embeddings_folder),
    ],
    check=True,
)

if args_dict["existing_embedding"]:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "code.logreg.src.embeddings.combine_two_embeddings",
            str(embeddings_folder),
            str(args_dict["existing_embedding"]),
        ],
        check=True,
    )

subprocess.run(
    [
        sys.executable,
        "-m",
        "code.logreg.src.embeddings.apply_pca",
        "--embeddings_input_folder",
        str(embeddings_folder),
        "--pca_dim",
        str(PCA_DIM),
    ],
    check=True,
)

"""
embeddings_folder_after_pca = (
    ProjectPaths.logreg_embeddings_path() / "after_pca" / f"{model_abbreviation}_{PCA_DIM}"
)

subprocess.run(
    [
        sys.executable,
        "-m",
        "code.logreg.src.embeddings.papers_categories",
        "--embeddings_input_folder",
        str(embeddings_folder_after_pca),
    ],
    check=True,
)

shutil.rmtree(embeddings_folder_after_pca, ignore_errors=True)
"""
