import argparse
import os
import shutil

from ..src.load_files import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clear finetuning experiments")
    parser.add_argument("--clear_state_dicts", action="store_true", default=False)
    return parser.parse_args()


args = parse_args()
experiments_folder = ProjectPaths.finetuning_data_experiments_path()
for folder in os.listdir(experiments_folder):
    folder_path = experiments_folder / folder
    if os.path.isdir(folder_path):
        if args.clear_state_dicts or ("state_dicts" not in os.listdir(folder_path)):
            shutil.rmtree(folder_path)
            print(f"Removed {folder_path}.")
