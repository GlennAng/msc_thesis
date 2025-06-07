import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

def load_arxiv_dataset_scholar_inbox() -> tuple:
    # load the two lists containing the paper ids and their arxiv categories (data for Scholar Inbox)
    with open('data/arxiv_ids.pkl', 'rb') as f:
        arxiv_ids = pickle.load(f)
    with open('data/arxiv_categories.pkl', 'rb') as f:
        arxiv_categories = pickle.load(f)
    assert len(arxiv_ids) == len(arxiv_categories), "arxiv_ids and arxiv_categories must have the same length"
    return arxiv_ids, arxiv_categories

def convert_arxiv_category(category : str) -> str:
    """
    Convert the arxiv category by only looking at the main category. For example, from cs.AI to 'Computer Science'
    The full taxonomy can be found on https://arxiv.org/category_taxonomy.
    """
    category = category.lower() # make lower case
    category = category.split(" ")[0] # if there are multiple categories (listed with spaces between them), take the first one
    category = category.split(".")[0] # split the string at the dot so that cs.ai becomes [cs, ai] and take first element which would be cs
    physics_subcategories = ["astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th", "math-ph", "nlin", "nucl-ex", "nucl-th", "physics", "quant-ph", 
                             "chao-dyn", "solv-int", "patt-sol", "adap-org"]
    if category == "cs":
        return "Computer Science"
    elif category == "econ":
        return "Economics"
    elif category == "eess":
        return "Electrical Engineering and Systems Science"
    elif category in ["math", "q-alg", "alg-geom", "funct-an", "dg-ga"]:
        return "Mathematics"
    elif category in physics_subcategories:
        return "Physics"
    elif category == "q-bio":
        return "Quantitative Biology"
    elif category == "q-fin":
        return "Quantitative Finance"
    elif category == "stat":
        return "Statistics"
    else:
        return "Other"

def convert_arxiv_categories(arxiv_categories : list) -> list:
    """
    Convert the arxiv categories to their main category using list comprehension by calling the convert_arxiv_category function.
    """
    return [convert_arxiv_category(category) for category in arxiv_categories]

def create_id_category_dataframe(arxiv_ids : list, arxiv_categories : list) -> pd.DataFrame:
    """
    Create a dataframe with the arxiv ids and their categories.
    """
    df = pd.DataFrame({'arxiv_id': arxiv_ids, 'category': arxiv_categories})
    return df

def get_arxiv_distribution(df : pd.DataFrame) -> pd.Series:
    """
    Get the distribution of the arxiv categories in the dataframe.
    """
    distribution = df['category'].value_counts() # get the counts of each category
    distribution = distribution / distribution.sum() # normalize the distribution
    return distribution

def plot_piecharts(distribution_1: pd.Series, distribution_2: pd.Series, title_1: str, title_2: str) -> None:
    """
    Plot two pie charts side by side for the given distributions.
    """
    # Create a figure with two subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    categories = distribution_1.index
    # Reorder the second distribution to match the first one
    distribution_2 = distribution_2.reindex(categories)
    
    # Create colors for the pie chart
    colors = plt.cm.tab20.colors
    radius = 3.0
    
    # Plot both pie charts with the same colors but not displaying percentages
    wedges1, _ = ax1.pie(distribution_1, labels = None, startangle = 140, colors = colors, radius = radius)
    ax2.pie(distribution_2, labels = None, startangle = 140, colors = colors, radius = radius)
    
    # Set titles
    ax1.set_title(title_1, fontweight ='bold', fontsize = 18)
    ax2.set_title(title_2, fontweight = 'bold', fontsize = 18)
    
    # Set equal aspect ratio to ensure pie chart is circular
    ax1.axis('equal')
    ax2.axis('equal')

    ax1.legend(wedges1, categories, loc = 'upper right', bbox_to_anchor = (2.0, 0.05), ncol = 4, fontsize = 14) # only need 1 legend
    plt.savefig('plots/piecharts.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

def plot_vertical_mirrorchart(distribution_1 : pd.Series, distribution_2 : pd.Series, title : str, title_1 : str, title_2 : str) -> None:
    # multiply by 100 to get percentage
    distribution_1 = distribution_1 * 100
    distribution_2 = distribution_2 * 100

    fig, ax = plt.subplots(figsize = (12, 8))
    categories = distribution_1.index
    # Reorder the second distribution to match the first one
    distribution_2 = distribution_2.reindex(categories)

    # Set positions for the bars
    x_pos = np.arange(len(categories))
    bar_width = 0.35

    # Plot the bars - upward for distribution_1, downward for distribution_2
    bars1 = ax.bar(x_pos - bar_width/2, distribution_1.values, width = bar_width, color = "steelblue", label = title_1)
    bars2 = ax.bar(x_pos + bar_width/2, -distribution_2.values, width = bar_width, color = "firebrick", label = title_2)

    # Add percentage labels on top of the bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{distribution_1.values[i]:.2f}", ha = 'center', va = 'bottom', fontsize = 9, fontweight = 'bold')
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height - 0.5, f"{distribution_2.values[i]:.2f}", ha = 'center', va = 'top', fontsize = 9, fontweight = 'bold')

    # Set x-ticks in the middle with appropriate labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation = 45, ha = 'right')
    # Set y-axis labels
    ax.set_ylabel('Share (in %)', fontsize = 12)
    # Create custom y-ticks that show absolute values
    max_value = 75
    y_ticks = np.linspace(0, max_value, 4)
    ax.set_yticks(list(-y_ticks[1:]) + list(y_ticks))
    ax.set_yticklabels([str(abs(y)) for y in ax.get_yticks()])
    
    # Add a horizontal line at y=0
    ax.axhline(y = 0, color = 'black', linestyle = '-', alpha = 0.3)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    handles = [bars1[0], bars2[0]]  # Use only the first bar of each type
    labels = [title_1, title_2]
    ax.legend(handles, labels, loc = 'best')
        
    # Add title
    plt.title(title, fontweight = 'bold', fontsize = 14)
    
    plt.savefig('plots/mirrorchart.pdf', dpi = 300, bbox_inches = 'tight')
    # Adjust layout
    plt.tight_layout()
    plt.show()

def count_to_str(count: int) -> str:
    """
    Convert a count to a string with commas after every three digits.
    """
    return f"{count:,}"

if __name__ == '__main__':
    from data_handling import get_papers_categories_dataset_distribution, get_papers_categories_ratings_distribution
    papers_categories_dataset_distribution, n1 = get_papers_categories_dataset_distribution(print_results = False)
    papers_categories_dataset_distribution = {key : val for key, val in papers_categories_dataset_distribution}
    papers_categories_dataset_distribution["unknown"] = papers_categories_dataset_distribution.pop(None)
    papers_categories_dataset_distribution = pd.Series(papers_categories_dataset_distribution)
    papers_categories_ratings_distribution, n2 = get_papers_categories_ratings_distribution(print_results = False, count_duplicates = False)
    papers_categories_ratings_distribution = {key : val for key, val in papers_categories_ratings_distribution}
    papers_categories_ratings_distribution["unknown"] = papers_categories_ratings_distribution.pop(None)
    papers_categories_ratings_distribution = pd.Series(papers_categories_ratings_distribution)
    plot_piecharts(papers_categories_ratings_distribution, papers_categories_dataset_distribution, f"Papers Ratings Distribution\nN = {count_to_str(n2)}", f"Papers Dataset Distribution\nN = {count_to_str(n1)}")

    
    