import matplotlib.pyplot as plt
import numpy as np

# Data Setup
layers = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
times = ['46m', '51m', '52m', '54m', '56m', '59m', '62m', '64m', '67m']
auc_d = np.array([79.91, 82.25, 83.19, 83.40, 83.91, 83.99, 84.09, 84.11, 84.07]) / 100.0

# Create Plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.rcParams.update({'font.size': 16})

# Add subtle grey background
ax.set_facecolor("#f0f0f0")

# Plot line with markers
ax.plot(layers, auc_d, color='#2E86AB', linewidth=3, marker='o', 
        markersize=8, label='AUC-D', zorder=4)

# Highlight the chosen point (layer 4) with a larger marker and ring
ax.scatter(4, auc_d[4], color='#2E86AB', s=200, zorder=5, 
           edgecolors='#FF6B35', linewidths=3)

# Add time labels below each point with slight right offset
for i, (layer, score, time) in enumerate(zip(layers, auc_d, times)):
    ax.annotate(time, 
                xy=(layer, score), 
                xytext=(-2, -25),  # slight right offset + below
                textcoords='offset points',
                fontsize=16,
                color='#333333',
                ha='left')

# Axes and Labels
ax.set_xlim(-0.3, 8.5)
ax.set_ylim(0.79, 0.855)
ax.set_xlabel('Layers', fontsize=18)
ax.set_ylabel('AUC-D', fontsize=18)

# Set x ticks
ax.set_xticks(layers)
ax.set_xticklabels([str(l) for l in layers], fontsize=16)

# Set y ticks
yticks = [0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85]
ax.set_yticks(yticks)
ax.set_yticklabels([f'{t:.2f}' for t in yticks], fontsize=16)

# Grid lines
ax.grid(True, which="major", linestyle='solid', linewidth=1.0, 
        color='#333333', alpha=0.2, zorder=1)

# Export
plt.tight_layout()
plt.savefig('unfrozen_layers_tradeoff.pdf', format='pdf', bbox_inches='tight')
plt.show()