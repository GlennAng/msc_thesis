import os
import subprocess
import sys

"""
params = [1, 2, 3, 4, 5, 7, 10]
for param in params:
    cmd = [
        sys.executable,
        "-m",
        "code.scripts.sliding_window_eval",
        "--clustering_approach",
        "k_means_fixed_k",
        "--clustering_k_means_n_clusters",
        str(param),
        "--single_random_state",
        "--save_users_predictions",
        "--embed_function",
        "clustering",
        "--old_ratings"
    ]
    
    subprocess.run(cmd, check=True)
"""
experiments_folder = "code/logreg/experiments/christmas"
all_files = os.listdir(experiments_folder)
for file in all_files:
    config_path = os.path.join(experiments_folder, file)
    cmd = [
        sys.executable,
        "-m",
        "code.scripts.average_seeds",
        "--config_path",
        config_path,
    ]
    subprocess.run(cmd, check=True)