import matplotlib.pyplot as plt
import numpy as np

def plot_correlation(
    models: list,
    x_values: list,
    y_values: list,
    x_label: str,
    y_label: str,
    save_path: str,
    title="Model Performance Correlation",
):
    plt.figure(figsize=(12, 8))
    plt.scatter(
        x_values,
        y_values,
        s=100,
        alpha=0.7,
        c="steelblue",
        edgecolors="black",
        linewidth=1,
    )

    # Define positioning for each model
    left_models = ["Qwen3-0.6B", "Qwen3-4B"]
    
    for i, model in enumerate(models):
        if model == "GTE-Base":
            xytext = (-4, 6)  # A chunk left but not as far as the others
            ha = "right"
        elif model in left_models:
            xytext = (-4, 6)
            ha = "right"
        else:  # right_models: GTE-Large, SPECTER2, Qwen3-8B
            xytext = (5, 6)
            ha = "left"
            
        plt.annotate(
            model,
            (x_values[i], y_values[i]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            ha=ha,
        )

    x_range = np.linspace(min(x_values), max(x_values), 100)
    z = np.polyfit(x_values, y_values, 1)
    p = np.poly1d(z)
    plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)

    plt.xlabel(x_label, fontsize=16, fontweight="bold")
    plt.ylabel(y_label, fontsize=16, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.gca().set_facecolor("#f5f5f5")
    plt.grid(True, alpha=0.3, which="major", linestyle="-", linewidth=0.5)

    ax = plt.gca()
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()

    for i in range(len(x_ticks) - 1):
        mid_x = (x_ticks[i] + x_ticks[i + 1]) / 2
        plt.axvline(x=mid_x, color="gray", alpha=0.2, linewidth=0.3)

    for i in range(len(y_ticks) - 1):
        mid_y = (y_ticks[i] + y_ticks[i + 1]) / 2
        plt.axhline(y=mid_y, color="gray", alpha=0.2, linewidth=0.3)

    corr = np.corrcoef(x_values, y_values)[0, 1]
    plt.text(
        0.05,
        0.95,
        f"Correlation: {corr:.3f}",
        transform=plt.gca().transAxes,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()
    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")

models = ["SPECTER2", "GTE-Base", "GTE-Large", "Qwen3-0.6B", "Qwen3-4B", "Qwen3-8B"]
session_based_ndcg = [80.54, 79.80, 80.55, 80.55, 82.88, 83.20]
cross_val_ndcg = [84.80, 84.97, 85.32, 85.37, 86.85, 86.82]
v_measure = [0.5849, 0.5836, 0.6028, 0.5943, 0.6029, 0.6138]

plot_correlation(
    models=models,
    x_values=session_based_ndcg,
    y_values=cross_val_ndcg,
    x_label="Session-based",
    y_label="Cross-validation",
    save_path="session_vs_cross_val_correlation.png",
    title="Correlation between Session-based and Cross-validation nDCG (both on the Scholar Inbox Dataset)",
)

plot_correlation(
    models=models,
    x_values=session_based_ndcg,
    y_values=v_measure,
    x_label="Session-based",
    y_label="V-measure",
    save_path="session_vs_v_measure_correlation.png",
    title="Correlation between Session-based nDCG (Scholar Inbox Dataset) and V-Measure (arXiv Clustering Dataset)",
)