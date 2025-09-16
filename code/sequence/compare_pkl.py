from pathlib import Path
import pickle

p1 = "code/sequence/data/users_embeddings/neural_sequence_test_pos_0_sess_None_days_None/users_embeddings/s_42/users_embeddings.pkl"
with open(p1, "rb") as f:
    d1 = pickle.load(f)

print(d1)
print()