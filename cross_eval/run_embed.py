MODEL_NAME = "gte-base-en-v1.5"
EMBEDDINGS_FOLDER = "/home/scholar/glenn_rp/msc_thesis/data/embeddings/before_pca"
MAX_BATCH_SIZE = 500
MAX_SEQUENCE_LENGTH = 1024

from src.data_handling import get_db_backup_date
import os
VALID_MODEL_NAMES = ["gte-base-en-v1.5", "gte-large-en-v1.5"]
if MODEL_NAME not in VALID_MODEL_NAMES:
    raise ValueError(f"Invalid model name. Pick one of {VALID_MODEL_NAMES}.")
model_path = f"Alibaba-NLP/{MODEL_NAME}"
if MODEL_NAME == "gte-base-en-v1.5":
    model_abbreviation = "gte_base"
elif MODEL_NAME == "gte-large-en-v1.5":
    model_abbreviation = "gte_large"
db_backup_date = get_db_backup_date()
embeddings_folder = EMBEDDINGS_FOLDER + f"/{model_abbreviation}_{db_backup_date}"  

os.system(f"python src/compute_embeddings.py --model_path {model_path} --embeddings_folder {embeddings_folder} --max_batch_size {MAX_BATCH_SIZE} --max_sequence_length {MAX_SEQUENCE_LENGTH}")
os.system(f"python src/merge_embeddings.py {embeddings_folder}")