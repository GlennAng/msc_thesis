import subprocess
import sys

params = [None]
max_n_posrated = [None]
alpha = [0.8]

for param in params:
    for a in alpha:
        for max_n in max_n_posrated:
            subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "code.scripts.sliding_window_eval",
                            "--clustering_approach",
                            "val_split",
                            "--single_random_state",
                            "--clustering_cluster_alpha",
                            str(a),
                            "--clustering_pos_weighting_scheme",
                            "relative",
                            "--clustering_neg_weighting_scheme",
                            "same_ratio",
                        ],
                        check=True,
                    )