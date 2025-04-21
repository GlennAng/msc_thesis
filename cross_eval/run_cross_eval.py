import argparse
import os
import time

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type = str, required = True)
    parser.add_argument('--continue_from_previous', action = 'store_true', default = False)
    parser.add_argument('--visualize_users_infos', action = 'store_true', default = False)
    parser.add_argument('--visualize_users_papers', action = 'store_true', default = False)
    parser.add_argument('--fold_idx', type = str, default = '0')
    parser.add_argument('--save_hyperparameters_table', action = 'store_true', default = False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config_files = []
    if args.config_path.endswith(".json"):
        config_files.append(args.config_path)
    else:
        config_path = args.config_path if args.config_path[-1] != "/" else args.config_path[:-1]
        for file in os.listdir(config_path):
            if file.endswith(".json"):
                config_files.append(config_path + "/" + file)
    for config_file in config_files:
        start_time = time.time()
        print(f"Running '{config_file}' ...")
        
        os.system("python src/main.py " + config_file + (" continue_from_previous" if args.continue_from_previous else ""))
        outputs_folder = f"outputs/{config_file.split('/')[-1].split('.')[0]}"
        os.system("python src/visualize_globally.py" + f" --outputs_folder {outputs_folder} --score balanced_accuracy {'--save_hyperparameters_table' if args.save_hyperparameters_table else ''}")
        if args.visualize_users_infos:
            os.system("python src/visualize_users_infos.py " + outputs_folder)
        if args.visualize_users_papers:
            os.system(f"python src/visualize_users_papers.py --outputs_folder {outputs_folder} --fold_idx {args.fold_idx} --n_print_interesting_users all")
        minutes_elapsed = (time.time() - start_time) / 60
        print(f"Finished '{config_file}' in {minutes_elapsed:.2f} minutes.")
        print("------------------------")