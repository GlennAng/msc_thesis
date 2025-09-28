import subprocess
import sys

params = ["specter2", "gte_base", "gte_large", "qwen3_0p6B", "qwen3_4B", "qwen3_8B"]

for param in params:
    subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "code.scripts.average_seeds",
                    "--config_path",
                    f"code/logreg/experiments/cross_val/{param}.json"
                ],
                check=True,
            )