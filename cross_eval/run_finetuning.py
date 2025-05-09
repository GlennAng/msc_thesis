import os

random_seeds = [100, 200]
for seed in random_seeds:
    os.system(f"python src/finetuning.py --seed {seed} --n_unfreeze_layers 8")