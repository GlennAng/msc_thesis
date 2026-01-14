import numpy as np
from scipy.stats import spearmanr

# Data from the tables
models = ['GTE-Large', 'Qwen2 7B', 'F2LLM 0.6B', 'F2LLM 1.7B', 
          'F2LLM 4B', 'Qwen3 0.6B', 'Qwen3 4B', 'Qwen3 8B']

# AUC-D from table_results
auc_d = np.array([79.92, 82.54, 80.98, 81.90, 81.96, 80.29, 83.05, 83.58])

# Gap from table_results
gap = np.array([15.18, 21.30, 16.72, 16.89, 18.61, 16.92, 20.73, 21.75])

# MTEB Rank from table_overview (lower is better, so we'll negate for correlation)
# GTE-Large has no rank, excluding it
mteb_rank = np.array([192, 188, 187, 46, 6, 5, 3])  # excluding GTE-Large
auc_d_mteb = np.array([80.98, 81.90, 81.96, 82.54, 80.29, 83.05, 83.58])  # excluding GTE-Large

# ArXiv S2S from table_overview (higher is better)
arxiv_s2s = np.array([64.00, 64.29, 64.50, 64.85, 63.82, 65.16, 65.48])  # excluding GTE-Large

# Compute Spearman correlations
rho_gap, p_gap = spearmanr(auc_d, gap)
rho_mteb, p_mteb = spearmanr(auc_d_mteb, -mteb_rank)  # negative because lower rank is better
rho_arxiv, p_arxiv = spearmanr(auc_d_mteb, arxiv_s2s)

print("Spearman Correlations with AUC-D:")
print(f"Gap:             ρ = {rho_gap:.3f} (p = {p_gap:.4f})")
print(f"MTEB Rank:       ρ = {rho_mteb:.3f} (p = {p_mteb:.4f})")
print(f"ArXiv S2S:       ρ = {rho_arxiv:.3f} (p = {p_arxiv:.4f})")