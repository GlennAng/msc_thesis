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

os.system("python src/finetuning.py --transformer_model_lr 1e-4 --projection_lr 1e-3 --users_embeddings_lr 1e-3")
os.system("python src/finetuning.py --batch_size 64")
os.system("python src/finetuning.py --batch_size 32")
os.system("python src/finetuning.py --class_balancing")