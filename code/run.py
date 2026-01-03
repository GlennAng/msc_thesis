import os
import subprocess
import sys

embedding = "code/logreg/embeddings/after_pca/gte_large_256"
V_values = [0.9]
C_values = [0.25]
neg_scales = [0.8]
knn_alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 10.0]
for C in C_values:
    for neg_scale in neg_scales:
        for V in V_values:
            #random_states = [1, 25, 75, 100, 150]
            random_states = [42]
            for alpha in knn_alphas:
                cmd = [
                    sys.executable,
                    "-m",
                    "code.scripts.sliding_window_eval",
                    "--clustering_approach",
                    "knn",
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
                    "--logreg_weights_cache_v",
                    str(V),
                    "--users_selection",
                    "finetuning_val",
                    "--clustering_knn_alpha",
                    str(alpha),
                    "--single_random_state",
                ]
                
                subprocess.run(cmd, check=True)