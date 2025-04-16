import os
EXPERIMENTS_FOLDER = "/home/scholar/glenn_rp/msc_thesis/data/finetuning/experiments"
for folder in os.listdir(EXPERIMENTS_FOLDER):
    folder_path = os.path.join(EXPERIMENTS_FOLDER, folder)
    if os.path.isdir(folder_path):
        if "state_dicts" not in os.listdir(folder_path):
            os.system(f"rm -rf {folder_path}")
            print(f"Removed {folder_path}")