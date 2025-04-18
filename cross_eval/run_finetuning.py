import os
batch_sizes = [128, 256]
class_balancing = [True, False]
n_samples_per_user = [4, 8, 16]

s = "python finetuning.py "
for batch_size in batch_sizes:
    for class_balance in class_balancing:
        for n_samples in n_samples_per_user:
            command = s + f"--batch_size {batch_size} {'class_balancing' if class_balanced else ''} --n_samples_per_user {n_samples}"
            os.system(command)