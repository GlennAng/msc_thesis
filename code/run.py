import subprocess
import sys

params = [1, 2, 3, 4, 5, 7, 10]
alpha = [0.8]
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
                            "--save_users_predictions",
                            "--clustering_cluster_alpha",
                            str(a),
                        ],
                        check=True,
                    )