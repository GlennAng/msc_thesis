import subprocess
import sys

users_ids = [119, 6199, 17474]
outputs_folders = [
    "code/sequence/data/regular_before_ft/outputs",
    "code/sequence/data/regular_after_ft/outputs",
]
for folder in outputs_folders:

    cmd = [
        sys.executable,
        "-m",
        "code.logreg.src.visualization.visualize_users",
        "--outputs_folder",
        folder,
        "--users",
        *[str(uid) for uid in users_ids]
    ] 
    subprocess.run(cmd, check=True)

""""
embedding = "code/logreg/embeddings/after_pca/gte_large_256"

#embedding = "code/logreg/embeddings/after_pca/gte_large_256"
V_values = [0.9]
C_values = [0.25]
neg_scales = [0.8]
knn_alphas = [None]
for C in C_values:
    for neg_scale in neg_scales:
        for V in V_values:
            for alpha in knn_alphas:
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
                ]
                
                subprocess.run(cmd, check=True)
"""