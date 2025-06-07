import os
import sys
EXPERIMENTS_FOLDER = "/home/scholar/glenn_rp/msc_thesis/data/finetuning/experiments"
CLEAR_STATE_DICTS = False
if len(sys.argv) > 1:
    CLEAR_STATE_DICTS = sys.argv[1].lower() == "clear_state_dicts"

for folder in os.listdir(EXPERIMENTS_FOLDER):
    folder_path = os.path.join(EXPERIMENTS_FOLDER, folder)
    if os.path.isdir(folder_path):
        if CLEAR_STATE_DICTS or ("state_dicts" not in os.listdir(folder_path)):
            os.system(f"rm -rf {folder_path}")
            print(f"Removed {folder_path}")