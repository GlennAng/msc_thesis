import subprocess
import sys


params = [0.1]
for param in params:
    cmd = [
        sys.executable,
        "-m",
        "code.scripts.sliding_window_eval",
        "--clustering_approach",
        "k_means_fixed_k",
        "--clustering_k_means_n_clusters",
        "7",
        "--single_random_state",
        "--clustering_cluster_alpha",
        "0.8",
        "--clustering_pos_weighting_scheme",
        "relative",
        "--clustering_neg_weighting_scheme",
        "same_ratio",
        "--save_users_predictions",
    ]
    
    subprocess.run(cmd, check=True)