import matplotlib.pyplot as plt
import numpy as np


def plot_correlation(
    models, cross_val_ndcg, session_based_ndcg, title="Model Performance Correlation"
):
    plt.figure(figsize=(10, 8))
    plt.scatter(
        cross_val_ndcg,
        session_based_ndcg,
        s=100,
        alpha=0.7,
        c="steelblue",
        edgecolors="black",
        linewidth=1,
    )

    for i, model in enumerate(models):
        plt.annotate(
            model,
            (cross_val_ndcg[i], session_based_ndcg[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    x_range = np.linspace(min(cross_val_ndcg), max(cross_val_ndcg), 100)
    z = np.polyfit(cross_val_ndcg, session_based_ndcg, 1)
    p = np.poly1d(z)
    plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)

    plt.xlabel("Cross-Validation nDCG", fontsize=12)
    plt.ylabel("Session-Based nDCG", fontsize=12)
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

    corr = np.corrcoef(cross_val_ndcg, session_based_ndcg)[0, 1]
    plt.text(
        0.05,
        0.95,
        f"Correlation: {corr:.3f}",
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()
    plt.savefig("model_performance_correlation.pdf", format="pdf", dpi=300, bbox_inches="tight")


models = ["SPECTER2", "GTE-Base", "GTE-Large", "Qwen3-0.6B", "Qwen3-4B", "Qwen3-8B"]
cross_val_ndcg = [84.19, 84.24, 84.52, 84.66, 86.03, 86.12]
session_based_ndcg = [80.89, 80.25, 80.84, 80.55, 82.70, 83.34]
v_measure = [0.5849, 0.5836, 0.6028, 0.5943, 0.6029, 0.6138]

plot_correlation(models, cross_val_ndcg, session_based_ndcg)
