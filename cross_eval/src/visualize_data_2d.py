from embedding import Embedding
from data_handling import get_rated_papers_ids_for_user, get_cache_papers_ids_for_user
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import pandas as pd
from sklearn.manifold import TSNE

import time
"""
embedding = Embedding("../data/embeddings/before_pca/gte_large_2025-02-23")
pos_seed, neg_seed = int(sys.argv[1]), int(sys.argv[2])


pos_rated_papers = embedding.matrix[embedding.get_idxs(pos_rated_ids)]
neg_rated_papers = embedding.matrix[embedding.get_idxs(neg_rated_ids)]
cache_papers = embedding.matrix[embedding.get_idxs(cache_ids)]


all_paper_ids = np.concatenate([pos_rated_ids, neg_rated_ids, cache_ids])
all_paper_idxs = embedding.get_idxs(all_paper_ids)
all_paper_embeddings = embedding.matrix[all_paper_idxs]


# Apply t-SNE
print(f"Running t-SNE on {len(all_paper_embeddings)} embeddings...")
start_time = time.time()
tsne = TSNE(n_components = 2, perplexity = 40, max_iter = 1000, random_state = 42)
embeddings_2d = tsne.fit_transform(all_paper_embeddings)
print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")

pos_transformed = embeddings_2d[:len(pos_rated_ids)]
neg_transformed = embeddings_2d[len(pos_rated_ids):len(pos_rated_ids) + len(neg_rated_ids)]
cache_transformed = embeddings_2d[len(pos_rated_ids) + len(neg_rated_ids):]

N = 25
np.random.seed(pos_seed)
pos_transformed_50_random = pos_transformed[np.random.choice(pos_transformed.shape[0], N, replace=False)]
np.random.seed(neg_seed)
neg_transformed_50_random = neg_transformed[np.random.choice(neg_transformed.shape[0], N, replace=False)]
cache_transformed_100_random = cache_transformed[np.random.choice(cache_transformed.shape[0], 3 * N, replace=False)]
transformed_labels = np.array(['Positive'] * N + ['Negative'] * N + ['Cache'] * 3 * N)
all_transformed = np.concatenate([pos_transformed_50_random, neg_transformed_50_random, cache_transformed_100_random])

# Create a DataFrame for easier plotting
df = pd.DataFrame({
    'x': all_transformed[:, 0],
    'y': all_transformed[:, 1],
    'Label': transformed_labels
})

# Plot
plt.figure(figsize=(12, 10))
colors = {'Positive': 'blue', 'Negative': 'red', 'Cache': 'green'}
sns.scatterplot(data=df, x='x', y='y', hue='Label', palette=colors, alpha=0.7)

plt.title(f"t-SNE Visualization of Papers for User {user_id}")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"tsne_visualization.png", dpi=300)
plt.show()

# Print some stats
print(f"Number of positive papers: {len(pos_rated_ids)}")
print(f"Number of negative papers: {len(neg_rated_ids)}")
print(f"Number of cache papers: {len(cache_ids)}")
"""
user_id = 14
pos_rated_ids, neg_rated_ids = get_rated_papers_ids_for_user(user_id, +1), get_rated_papers_ids_for_user(user_id, -1)
cache_ids = get_cache_papers_ids_for_user(user_id, max_cache = 1500, random_state = 42)
pos_seed, neg_seed = int(sys.argv[1]), int(sys.argv[2])
pos_seed, neg_seed = 1, 25

embedding = Embedding("../data/embeddings/after_pca/gte_large_2025-02-23_256")
pos_rated_papers = embedding.matrix[embedding.get_idxs(pos_rated_ids)]
neg_rated_papers = embedding.matrix[embedding.get_idxs(neg_rated_ids)]
cache_papers = embedding.matrix[embedding.get_idxs(cache_ids)]

N = 20
np.random.seed(pos_seed)
pos_rated_papers = pos_rated_papers[np.random.choice(pos_rated_papers.shape[0], N, replace=False)]
np.random.seed(neg_seed)
neg_rated_papers = neg_rated_papers[np.random.choice(neg_rated_papers.shape[0], N, replace=False)]
np.random.seed(42)
cache_papers = cache_papers[np.random.choice(cache_papers.shape[0], 3 * N, replace=False)]
def plot_2d_scatter(pos_rated_papers, neg_rated_papers, cache_papers):
    # Create a DataFrame for the data
    data = pd.DataFrame({
        'x': np.concatenate([pos_rated_papers[:, 0], neg_rated_papers[:, 0], cache_papers[:, 0]]),
        'y': np.concatenate([pos_rated_papers[:, 1], neg_rated_papers[:, 1], cache_papers[:, 1]]),
        'label': ['P: Positive Votes'] * len(pos_rated_papers) + ['N: Negative Votes'] * len(neg_rated_papers) + ['Cache'] * len(cache_papers)
    })

    pos_seed, neg_seed = int(sys.argv[1]), int(sys.argv[2])

    # Create the scatter plot
    plt.figure(figsize=(10, 8))

    # Plot cache points first with lower alpha
    cache_data = data[data['label'] == 'Cache']
    sns.scatterplot(data=cache_data, x='x', y='y', color='red', alpha=0.25, label='R: Random Negatives', legend=False)

    # Plot negative and positive points with normal alpha
    non_cache_data = data[data['label'] != 'Cache']
    sns.scatterplot(data=non_cache_data, x='x', y='y', hue='label', 
                palette={'P: Positive Votes': 'blue', 'N: Negative Votes': 'red'}, 
                alpha=0.7, legend=False)

    plt.xlim([-0.5, 0.2])  # Replace with your desired x-axis range
    plt.ylim([-0.3, 0.25])  # Replace with your desired y-axis range

    # Add labels and title
    plt.title('2D Scatter Plot of Papers')

    
    #plt.grid(True)
    plt.show()
    plt.savefig("/home/scholar/glenn_rp/msc_thesis/cross_eval/2d_scatter_plot.pdf", dpi=300, bbox_inches='tight')

plot_2d_scatter(pos_rated_papers, neg_rated_papers, cache_papers)
