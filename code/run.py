import subprocess
import sys

params = [5]
alpha = [0.6, 0.7, 0.75]
max_n_posrated = [None]

for param in params:
    for a in alpha:
        for max_n in max_n_posrated:
            subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "code.scripts.sliding_window_eval",
                            "--clustering_approach",
                            "k_means_fixed_k",
                            "--clustering_k_means_n_clusters",
                            str(param),
                            "--single_random_state",
                            "--old_ratings",
                            "--save_users_predictions",
                            "--clustering_cluster_alpha",
                            str(a),
                        ],
                        check=True,
                    )