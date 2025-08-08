import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from ..src.load_files import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--visualize_users", action="store_true", default=False)
    parser.add_argument("--save_scores_tables", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_files = []
    config_path = Path(args.config_path).resolve()
    if config_path.is_file() and config_path.suffix == ".json":
        config_files.append(config_path)
    else:
        for file in os.listdir(config_path):
            file_path = config_path / file
            if file_path.is_file() and file_path.suffix == ".json":
                config_files.append(file_path)
    for config_file in config_files:
        config_file_name = os.path.basename(config_file)
        start_time = time.time()
        print(f"Running '{config_file_name}' ...")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "code.logreg.src.main",
                str(config_file),
            ],
            check=True,
        )
        outputs_folder = ProjectPaths.logreg_outputs_path() / config_file_name.replace(".json", "")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "code.logreg.src.visualization.visualize_globally",
                "--outputs_folder",
                str(outputs_folder),
                "--score",
                "ndcg_all",
            ]
            + (["--save_scores_tables"] if args.save_scores_tables else []),
            check=True,
        )
        if args.visualize_users:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "code.logreg.src.visualization.visualize_users",
                    str(outputs_folder),
                ],
                check=True,
            )
        minutes_elapsed = (time.time() - start_time) / 60
        print(f"Finished '{config_file_name}' in {minutes_elapsed:.2f} minutes.")
        print("------------------------")
