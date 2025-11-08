from pathlib import Path
import pandas as pd

df = pd.read_csv(Path("code/sequence/data/sliding_window_eval_fixed/fixed_k_5/outputs/users_results.csv"))
df = df[df["user_id"] == 9]
bal_acc = df["val_balanced_accuracy"].tolist()[0]
ndcg = df["val_ndcg_all"].tolist()[0]
mrr = df["val_mrr_all"].tolist()[0]
recall = df["val_recall"].tolist()[0]
specificity = df["val_specificity"].tolist()[0]
train_recall = df["train_recall"].tolist()[0]
train_specificity = df["train_specificity"].tolist()[0]
train_bal_acc = df["train_balanced_accuracy"].tolist()[0]
train_cel = df["train_cel"].tolist()[0]
info_nce = df["val_info_nce_1"].tolist()[0]
print(f"Balanced Accuracy: {bal_acc}")
print(f"NDCG: {ndcg}")
print(f"MRR: {mrr}")
print(f"Train Recall: {train_recall}")
print(f"Train Specificity: {train_specificity}")
print(f"InfoNCE: {info_nce}")
print(f"Train CEL: {train_cel}")
print(f"Train Balanced Accuracy: {train_bal_acc}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")