import os

"""

class_balancing = [True, False]
n_samples_per_user = [4, 8, 16]

s = "python src/finetuning.py "
for batch_size in batch_sizes:
    for class_balance in class_balancing:
        for n_samples in n_samples_per_user:
            command = s + f"--batch_size {batch_size} {'--class_balancing' if class_balance else ''} --n_samples_per_user {n_samples}"
            os.system(command)
"""

random_seeds = [42, 100, 200]
for seed in random_seeds:
    os.system(f"python src/finetuning.py --not_pretrained_categories_embeddings --not_categories_attached --seed {seed}")