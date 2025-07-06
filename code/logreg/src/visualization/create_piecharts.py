import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ....src.load_files import load_papers, load_users_ratings


def get_l1_distribution(papers_df: pd.DataFrame) -> pd.Series:
    distribution = papers_df["l1"].value_counts()
    distribution = distribution / distribution.sum()
    return distribution


def plot_piecharts(
    distribution_1: pd.Series, distribution_2: pd.Series, title_1: str, title_2: str
) -> None:
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    categories = distribution_1.index
    distribution_2 = distribution_2.reindex(categories)
    colors = plt.cm.tab20.colors
    radius = 3.0
    wedges1, _ = ax1.pie(distribution_1, labels=None, startangle=140, colors=colors, radius=radius)
    ax2.pie(distribution_2, labels=None, startangle=140, colors=colors, radius=radius)
    ax1.set_title(title_1, fontweight="bold", fontsize=18)
    ax2.set_title(title_2, fontweight="bold", fontsize=18)
    ax1.axis("equal")
    ax2.axis("equal")
    ax1.legend(
        wedges1, categories, loc="upper right", bbox_to_anchor=(2.0, 0.05), ncol=4, fontsize=14
    )  # only need 1 legend
    plt.savefig("piecharts.pdf", dpi=300, bbox_inches="tight")
    plt.show()


def plot_vertical_mirrorchart(
    distribution_1: pd.Series, distribution_2: pd.Series, title: str, title_1: str, title_2: str
) -> None:
    distribution_1 = distribution_1 * 100
    distribution_2 = distribution_2 * 100
    _, ax = plt.subplots(figsize=(12, 8))
    categories = distribution_1.index
    distribution_2 = distribution_2.reindex(categories)
    x_pos = np.arange(len(categories))
    bar_width = 0.35
    bars1 = ax.bar(
        x_pos - bar_width / 2,
        distribution_1.values,
        width=bar_width,
        color="steelblue",
        label=title_1,
    )
    bars2 = ax.bar(
        x_pos + bar_width / 2,
        -distribution_2.values,
        width=bar_width,
        color="firebrick",
        label=title_2,
    )

    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{distribution_1.values[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height - 0.5,
            f"{distribution_2.values[i]:.2f}",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Share (in %)", fontsize=12)
    max_value = 75
    y_ticks = np.linspace(0, max_value, 4)
    ax.set_yticks(list(-y_ticks[1:]) + list(y_ticks))
    ax.set_yticklabels([str(abs(y)) for y in ax.get_yticks()])
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [bars1[0], bars2[0]]  # Use only the first bar of each type
    labels = [title_1, title_2]
    ax.legend(handles, labels, loc="best")
    plt.title(title, fontweight="bold", fontsize=14)
    plt.savefig("mirrorchart.pdf", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


papers = load_papers()
papers = papers[papers["l1"].notna()]
users_ratings = load_users_ratings()
papers_in_ratings = papers[papers["paper_id"].isin(users_ratings["paper_id"])]

papers_distribution = get_l1_distribution(papers)
papers_in_ratings_distribution = get_l1_distribution(papers_in_ratings)

print("Papers distribution:")
print(papers_distribution)
print("\nPapers in ratings distribution:")
print(papers_in_ratings_distribution)

piecharts_title_1 = "Papers from Database Category Distribution"
piecharts_title_2 = "Papers in Ratings Category Distribution"
plot_piecharts(
    papers_distribution, papers_in_ratings_distribution, piecharts_title_1, piecharts_title_2
)

plot_vertical_mirrorchart(
    papers_distribution,
    papers_in_ratings_distribution,
    "Papers Category Distribution Comparison",
    piecharts_title_1,
    piecharts_title_2,
)
