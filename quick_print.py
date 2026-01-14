import matplotlib.pyplot as plt
import numpy as np

# Data Setup
beta_vals = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0])

auc_d = np.array([79.92, 80.33, 80.6, 80.77, 81.09, 81.14, 81.13, 81.23, 81.45, 81.19, 80.78, 79.89])
recall = np.array([74.51, 75.21, 76.02, 76.72, 77.38, 77.95, 78.51, 78.71, 78.46, 77.92, 74.41, 69.55])

# Normalize to 0-1 scale
auc_d /= 100.0
recall /= 100.0

# Create Plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.rcParams.update({'font.size': 16})

# Add subtle grey background
ax.set_facecolor("#f0f0f0")

# Plot both lines with complementary colors
ax.plot(beta_vals, auc_d, color='#2E86AB', linewidth=3, label='AUC-D', zorder=4)  # Teal blue
ax.plot(beta_vals, recall, color='#A23B72', linewidth=3, label='Recall', zorder=4)  # Magenta

# Mark peaks
auc_peak_idx = np.argmax(auc_d)
recall_peak_idx = np.argmax(recall)

ax.scatter(beta_vals[auc_peak_idx], auc_d[auc_peak_idx], 
           color='#2E86AB', s=150, zorder=5, edgecolors='none', 
           label=rf'AUC-D Peak ($\beta = {beta_vals[auc_peak_idx]}$)')

ax.scatter(beta_vals[recall_peak_idx], recall[recall_peak_idx], 
           color='#A23B72', s=150, zorder=5, edgecolors='none', 
           label=rf'Recall Peak ($\beta = {beta_vals[recall_peak_idx]}$)')

# Axes and Labels with padding
ax.set_xlim(0.48, 1.02)
ax.set_ylim(0.65, 1.0)
ax.set_xlabel(r'$\beta$', fontsize=22)
ax.set_ylabel('Score', fontsize=22)

# Set x ticks
xticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ax.set_xticks(xticks)
ax.set_xticklabels([str(t) for t in xticks], fontsize=18)

# Set y ticks (major) - excluding 0.65
yticks_major = [0.7, 0.8, 0.9, 1.0]
ax.set_yticks(yticks_major)
ax.set_yticklabels([str(t) for t in yticks_major], fontsize=18)

# Set minor y ticks
yticks_minor = [0.65, 0.75, 0.85, 0.95]
ax.set_yticks(yticks_minor, minor=True)
ax.tick_params(axis='y', which='minor', length=0)

# Grid lines
ax.grid(True, which="major", linestyle='solid', linewidth=1.0, 
        color='#333333', alpha=0.2, zorder=1)
ax.grid(True, which="minor", axis='y', linestyle='solid', linewidth=1.0, 
        color='#333333', alpha=0.2, zorder=1)

ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=1, fontsize=15)

# Export
plt.tight_layout()
plt.savefig('auc_d_recall_comparison.pdf', format='pdf')
plt.show()