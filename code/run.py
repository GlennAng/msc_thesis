import os
import subprocess
import sys


embedding = "code/finetuning/data/checkpoints/cat_best/embeddings"
V_values = [0.9]
C_values = [0.25]
neg_scales = [0.77]
KNN_ALPHAS = [5.0]
TEMP_DECAY_PARAMS = [0.15]
for C in C_values:
    for neg_scale in neg_scales:
        for V in V_values:
            #random_states = [1, 25, 75, 100, 150]
            random_states = [42]
            for knn_alpha in KNN_ALPHAS:
                for temp_decay_param in TEMP_DECAY_PARAMS:
                    for random_state in random_states:
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
                            "--logreg_weights_cache_v",
                            str(V),
                            "--users_selection",
                            "finetuning_test",
                            "--clustering_selection_min_cluster_size",
                            "0",
                            "--clustering_knn_alpha",
                            str(knn_alpha),
                        ]  
                        subprocess.run(cmd, check=True)