import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from ...sequence.src.models.papers_encoder import load_papers_encoder
from ...src.load_files import load_papers, load_papers_texts
from ...src.project_paths import ProjectPaths
from .finetuning_preprocessing import load_categories_to_idxs

model_path = Path(sys.argv[1]).resolve()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


papers_encoder = load_papers_encoder(
    path=model_path,
    overwrite_l1_scale=None,
    overwrite_use_l2_embeddings=None,
    overwrite_l2_scale=None,
    l2_init_seed=None,
    unfreeze_l1_scale=False,
    unfreeze_l2_scale=False,
    n_unfreeze_layers=0,
    unfreeze_word_embeddings=False,
    unfreeze_from_bottom=False,
    device=device,
).eval()
tokenizer = AutoTokenizer.from_pretrained(ProjectPaths.finetuning_data_model_hf())
papers = load_papers()
papers_texts = load_papers_texts()
papers = papers.merge(papers_texts, on="paper_id", how="left")

path = model_path.parent / "all_embeddings"
os.makedirs(path, exist_ok=True)
batches_path = path / "batches"
os.makedirs(batches_path, exist_ok=True)
batch_size = 512

l1_categories_to_idxs = load_categories_to_idxs("l1")
l2_categories_to_idxs = load_categories_to_idxs("l2")

for i in tqdm(range(0, len(papers), batch_size)):
    batch_papers = papers.iloc[i : i + batch_size]
    batch_texts = batch_papers[["paper_id", "title", "abstract"]].values.tolist()
    papers_ids, papers_titles, papers_abstracts = zip(*batch_texts)
    papers_ids, papers_titles, papers_abstracts = (
        list(papers_ids),
        list(papers_titles),
        list(papers_abstracts),
    )
    assert papers_ids == sorted(papers_ids) and len(papers_ids) == len(set(papers_ids))
    papers_for_tokenization = [
        f"{title} {tokenizer.sep_token} {abstract}"
        for title, abstract in zip(papers_titles, papers_abstracts)
    ]
    input_ids_list, attention_mask_list = [], []
    encoding = tokenizer(
        papers_for_tokenization,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids_tensor = encoding["input_ids"].to(device)
    attention_mask_tensor = encoding["attention_mask"].to(device)
    categories_l1 = batch_papers["l1"].tolist()
    categories_l1 = [
        category if category in l1_categories_to_idxs else None for category in categories_l1
    ]
    l1_tensor = torch.tensor([l1_categories_to_idxs[category] for category in categories_l1])
    categories_l2 = batch_papers["l2"].tolist()
    categories_l2 = [
        category if category in l2_categories_to_idxs else None for category in categories_l2
    ]
    l2_tensor = torch.tensor([l2_categories_to_idxs[category] for category in categories_l2])
    l1_tensor = l1_tensor.to(device)
    l2_tensor = l2_tensor.to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            with torch.inference_mode():
                embeddings = (
                    papers_encoder(
                        input_ids_tensor=input_ids_tensor,
                        attention_mask_tensor=attention_mask_tensor,
                        category_l1_tensor=l1_tensor,
                        category_l2_tensor=l2_tensor,
                        normalize_embeddings=True,
                    )
                    .cpu()
                    .numpy()
                )
    batch_dir = batches_path / f"batch_{i}" / "abs_X.npy"
    os.makedirs(batch_dir.parent, exist_ok=True)
    np.save(batch_dir, embeddings)
    with open(batch_dir.parent / "abs_ids.pkl", "wb") as f:
        pickle.dump(papers_ids, f)


subprocess.run(
    [
        sys.executable,
        "-m",
        "code.logreg.src.embeddings.merge_embeddings",
        str(path),
    ],
    check=True,
)
