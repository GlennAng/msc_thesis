import subprocess
import sys

params = [0.001, 0.005, 0.01, 0.05, 0.1]

for param in params:
    subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "code.scripts.sliding_window_eval",
                    "--embed_function",
                    "logistic_regression",
                    "--single_random_state",
                    "--logreg_temporal_decay",
                    "exponential",
                    "--logreg_temporal_decay_param",
                    str(param),
                    "--logreg_temporal_decay_normalization",
                    "jointly",
                ],
                check=True,
            )