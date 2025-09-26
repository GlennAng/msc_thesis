import subprocess
import sys

params = [100, 200, 300, 500, 750]

for param in params:
    subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "code.scripts.sliding_window_eval",
                    "--embed_function",
                    "logistic_regression",
                    "--single_random_state",
                    "--histories_soft_constraint_max_n_train_days",
                    str(param)
                ],
                check=True,
            )