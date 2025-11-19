import subprocess
import sys

params = [1.0]
for param in params:
    cmd = [
        sys.executable,
        "-m",
        "code.scripts.sliding_window_eval",
        "--clustering_approach",
        "k_means_fixed_k",
        "--clustering_k_means_n_clusters",
        "1",
        "--single_random_state",
        "--save_users_predictions",
        "--embed_function",
        "clustering",
        "--clustering_selection_min_cluster_size",
        "10",
        "--old_ratings",
        "--clustering_cluster_alpha",
        str(param),
    ]
    
    subprocess.run(cmd, check=True)