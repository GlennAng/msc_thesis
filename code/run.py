import subprocess
import sys

params = [("linear", 1.0), ("clip", 1.0), ("softmax", 0.1), ("softmax", 1.0), ("softmax", 2.0), ("softmax", 0.01), ("none", 1.0)]

for param in params:
    subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "code.scripts.sliding_window_eval",
                    "--embed_function",
                    "logistic_regression",
                    "--logreg_similarity_scaling",
                    param[0],
                    "--logreg_similarity_scaling_param",
                    str(param[1]),
                ],
                check=True,
            )