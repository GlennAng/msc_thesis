import os
import subprocess
import sys


embedding = "code/finetuning/data/experiments/gte_large_256_2025-12-19-16-46/embeddings"
C_values = [0.5]
neg_scales = [0.85]
for C in C_values:
    for neg_scale in neg_scales:
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
            str(C),
            "--papers_embedding_path",
            embedding,
            "--logreg_weights_neg_scale",
            str(neg_scale),
            "--users_selection",
            "finetuning_test",
        ]
        
        subprocess.run(cmd, check=True)