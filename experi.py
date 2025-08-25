import subprocess
import sys

min_pos = [10]
max_sess = [None]
max_time = [None]

for p in min_pos:
    for s in max_sess:
        for t in max_time:
            print(f"Running experiment with min_pos={p}, max_sess={s}, max_time={t}")
            args = [
                sys.executable,
                "-m",
                "code.scripts.sequence_eval",
                "--embed_function",
                "logreg",
                "--hard_constraint_min_n_train_posrated",
                str(p),
            ]
            if s is not None:
                args += [
                    "--soft_constraint_max_n_train_sessions",
                    str(s),
                ]
            if t is not None:
                args += [
                    "--soft_constraint_max_n_train_days",
                    str(t),
                ]
            subprocess.run(args)
