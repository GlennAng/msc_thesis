import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[3]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_src_paths_to_sys()

import os

MODEL_NAME = "Qwen3-Embedding-8B"
EMBEDDINGS_FOLDER = ProjectPaths.logreg_embeddings_path() / "before_pca"
MAX_BATCH_SIZE = 25
MAX_SEQUENCE_LENGTH = 512

VALID_MODEL_NAMES = ["gte-base-en-v1.5", "gte-large-en-v1.5", "specter2_base", "Qwen3-Embedding-0.6B", "Qwen3-Embedding-8B"]
if MODEL_NAME == "gte-base-en-v1.5":
    model_abbreviation = "gte_base"
    model_path = f"Alibaba-NLP/{MODEL_NAME}"
elif MODEL_NAME == "gte-large-en-v1.5":
    model_abbreviation = "gte_large_x"
    model_path = f"Alibaba-NLP/{MODEL_NAME}"
elif MODEL_NAME == "specter2_base":
    model_abbreviation = "specter2"
    model_path = f"allenai/{MODEL_NAME}"
elif MODEL_NAME == "Qwen3-Embedding-0.6B":
    model_abbreviation = "qwen3_06B"
    model_path = f"Qwen/{MODEL_NAME}"
elif MODEL_NAME == "Qwen3-Embedding-8B":
    model_abbreviation = "qwen3_8B"
    model_path = f"Qwen/{MODEL_NAME}"
db_backup_date = ProjectPaths.data_db_backup_date_path().stem
embeddings_folder = EMBEDDINGS_FOLDER / f"{model_abbreviation}_{db_backup_date}"
os.system(f"python {ProjectPaths.logreg_src_embeddings_path() / 'compute_embeddings.py'} --model_path {model_path} --embeddings_folder {embeddings_folder} --max_batch_size {MAX_BATCH_SIZE} --max_sequence_length {MAX_SEQUENCE_LENGTH}")