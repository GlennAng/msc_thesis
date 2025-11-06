import subprocess
import sys

params = [10]
max_n_posrated = [None]

for param in params:
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
                    ],
                    check=True,
                )