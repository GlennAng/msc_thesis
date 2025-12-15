import os
import subprocess
import sys

params = [("code/logreg/embeddings/after_pca/gte_Qwen2_7B_instruct_256", 0.85)]
for param in params:
    cmd = [
        sys.executable,
        "-m",
        "code.scripts.sliding_window_eval",
        "--clustering_approach",
        "k_means_fixed_k",
        "--clustering_k_means_n_clusters",
        "1",
        "--embed_function",
        "clustering",
        "--logreg_clf_C",
        "0.5",
        "--papers_embedding_path",
        param[0],
        "--logreg_weights_neg_scale",
        str(param[1]),
    ]
    
    subprocess.run(cmd, check=True)