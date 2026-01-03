import matplotlib.pyplot as plt
import numpy as np

# 1. Data Setup
lambda_vals = np.array([
    0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.5, 0.75, 1.0,
    1.25, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0
])

auc_mean = np.array([
    80.33, 80.40, 80.44, 80.82, 81.06, 80.70, 80.66, 80.54, 80.41, 79.69,
    79.55, 79.33, 78.90, 78.66, 77.80, 75.86, 74.64
])

auc_std = np.array([
    19.64, 19.54, 19.44, 18.98, 19.09, 19.34, 19.61, 19.75, 19.85, 20.27,
    20.35, 20.28, 20.50, 20.53, 20.52, 22.07, 22.31
])

mask = lambda_vals > 0
x = lambda_vals[mask]
y = auc_mean[mask]
error = auc_std[mask]

# 3. Create Plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.rcParams.update({'font.size': 16})

# Add subtle grey background
ax.set_facecolor("#f0f0f0")

# Plot mean line and shaded spread
ax.plot(x, y, color='blue', linewidth=3, label='Mean', zorder=4)

# Peak at 0.15 - now with label
ax.scatter(0.15, 81.06, color='blue', s=150, zorder=5, edgecolors='none',
           label=r'Peak ($\lambda = 0.15$)')

# 4. Axes and Labels
ax.set_xscale('log')
ax.set_ylim(60, 100)
ax.set_xlabel(r'$\lambda$ (log scale)', fontsize=22)
ax.set_ylabel('AUC-D', fontsize=22)

# Set x ticks WITHOUT 0.15
ticks = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks], fontsize=18)

# Set y ticks to only 60, 70, 80, 90, 100 (with labels)
yticks_major = [60, 70, 80, 90, 100]
ax.set_yticks(yticks_major)
ax.set_yticklabels([str(t) for t in yticks_major], fontsize=18)

# Set minor y ticks at 65, 75, 85, 95 (without labels, for grid only)
yticks_minor = [65, 75, 85, 95]
ax.set_yticks(yticks_minor, minor=True)

# Remove minor tick marks (but keep the grid lines)
ax.tick_params(axis='y', which='minor', length=0)

# FULL SOLID GRID LINES - both major and minor for y-axis
ax.grid(True, which="major", linestyle='solid', linewidth=1.0,
        color='#333333', alpha=0.2, zorder=1)
ax.grid(True, which="minor", axis='y', linestyle='solid', linewidth=1.0,
        color='#333333', alpha=0.2, zorder=1)

ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=1)

# 5. Export
plt.tight_layout()
plt.savefig('auc_d_final_fixed_ticks.pdf', format='pdf')
plt.show()