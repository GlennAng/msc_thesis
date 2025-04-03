import os
EXPERIMENTS_FOLDER = "/home/scholar/glenn_rp/msc_thesis/data/finetuning/experiments"
os.system(f"rm -rf {EXPERIMENTS_FOLDER}")
os.system(f"mkdir -p {EXPERIMENTS_FOLDER}")