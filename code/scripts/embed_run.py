import argparse
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
]
EMBEDDINGS_FOLDER = ProjectPaths.logreg_embeddings_path() / "before_pca"
MAX_SEQUENCE_LENGTH = 512


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_CHOICES)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--all_papers", action="store_true", default=False)
    args_dict = vars(parser.parse_args())
    return args_dict


args_dict = parse_args()
model_name, batch_size = args_dict["model_name"], args_dict["batch_size"]
if model_name == "gte-base-en-v1.5":
    model_abbreviation = "gte_base"
    model_path = f"Alibaba-NLP/{model_name}"
    if batch_size is None:
        batch_size = 1000
elif model_name == "gte-large-en-v1.5":
    model_abbreviation = "gte_large"
    model_path = f"Alibaba-NLP/{model_name}"
    if batch_size is None:
        batch_size = 500
elif model_name == "specter2_base":
    model_abbreviation = "specter2"
    model_path = f"allenai/{model_name}"
    if batch_size is None:
        batch_size = 1000
elif model_name == "Qwen3-Embedding-0.6B":
    model_abbreviation = "qwen3_06B"
    model_path = f"Qwen/{model_name}"
    if batch_size is None:
        batch_size = 175
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
    + (["--save_scores_tables"] if args_dict["all_papers"] else []),
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
