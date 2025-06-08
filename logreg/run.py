import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[1]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_src_paths_to_sys()

import argparse, os, time

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type = str, required = True)
    parser.add_argument('--visualize_users_infos', action = 'store_true', default = False)
    parser.add_argument('--visualize_users_papers', action = 'store_true', default = False)
    parser.add_argument('--fold_idx', type = str, default = '0')
    parser.add_argument('--save_scores_tables', action = 'store_true', default = False)
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
        os.system(f"python {ProjectPaths.logreg_src_path() / 'main.py'} {config_file}")
        outputs_folder = ProjectPaths.logreg_outputs_path() / config_file_name.replace(".json", "")
        visualization_path = ProjectPaths.logreg_src_visualization_path()
        os.system(f"python {visualization_path / 'visualize_globally.py'}" + f" --outputs_folder {outputs_folder} --score balanced_accuracy {'--save_scores_tables' if args.save_scores_tables else ''}")
        if args.visualize_users_infos:
            os.system(f"python {visualization_path / 'visualize_users.py'} {outputs_folder}")
        if args.visualize_users_papers:
            os.system(f"python {visualization_path / 'visualize_users_papers.py'} --outputs_folder {outputs_folder} --fold_idx {args.fold_idx} --n_print_interesting_users all")
        minutes_elapsed = (time.time() - start_time) / 60
        print(f"Finished '{config_file_name}' in {minutes_elapsed:.2f} minutes.")
        print("------------------------")