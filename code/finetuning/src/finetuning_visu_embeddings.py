import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

from ...logreg.src.embeddings.embedding import Embedding
from ...src.load_files import load_papers
from ...src.project_paths import ProjectPaths
from .finetuning_compare_embeddings import compute_sims, compute_sims_same_set


def compute_category_metrics(embeddings, papers_ids_cats, name=""):
    """
    Compute quantitative metrics to detect category collapse
    Uses cosine similarity (higher = more similar)
    """
    print(f"\n{'='*60}")
    print(f"METRICS FOR: {name}")
    print(f"{'='*60}")
    
    # Compute centroids for each category
    centroids = {}
    category_embeddings = {}
    
    for category, paper_ids in papers_ids_cats.items():
        if category.startswith("CS -"):  # Skip subcategories for main analysis
            continue
        paper_idxs = embeddings.get_idxs(paper_ids[:5000])  # Sample for speed
        cat_embs = embeddings.matrix[paper_idxs]
        
        # Normalize embeddings for cosine similarity
        cat_embs_normalized = cat_embs / (np.linalg.norm(cat_embs, axis=1, keepdims=True) + 1e-8)
        category_embeddings[category] = cat_embs_normalized
        
        # Centroid of normalized embeddings
        centroid = cat_embs_normalized.mean(axis=0, keepdims=True)
        centroid_normalized = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids[category] = centroid_normalized
    
    # Compute inter-category centroid similarities (higher = more similar/collapsed)
    print("\nInter-category Centroid Cosine Similarities (lower = better separated):")
    print("-" * 60)
    categories = list(centroids.keys())
    for i, cat1 in enumerate(categories):
        for cat2 in categories[i+1:]:
            # Cosine similarity between centroids
            cos_sim = np.dot(centroids[cat1].flatten(), centroids[cat2].flatten())
            print(f"  {cat1:20s} <-> {cat2:20s}: {cos_sim:.4f}")
    
    # Compute intra-category cohesion (average cosine similarity to centroid)
    print("\nIntra-category Cohesion (avg cosine sim to centroid, higher = tighter):")
    print("-" * 60)
    for category in categories:
        # Cosine similarity of each point to centroid
        cos_sims = np.dot(category_embeddings[category], centroids[category].T).flatten()
        print(f"  {category:20s}: {cos_sims.mean():.4f} ± {cos_sims.std():.4f}")
    
    # Compute separation quality (1 - inter_sim) / (1 - intra_cohesion)
    # Higher ratio = better separated categories
    print("\nSeparation Quality (higher = better separated):")
    print("-" * 60)
    for i, cat1 in enumerate(categories):
        for cat2 in categories[i+1:]:
            # Inter-category similarity (want this LOW)
            inter_sim = np.dot(centroids[cat1].flatten(), centroids[cat2].flatten())
            
            # Intra-category cohesion (average similarity within cluster)
            cohesion1 = np.dot(category_embeddings[cat1], centroids[cat1].T).mean()
            cohesion2 = np.dot(category_embeddings[cat2], centroids[cat2].T).mean()
            avg_cohesion = (cohesion1 + cohesion2) / 2
            
            # Separation quality: how much more similar are items within category vs between
            # = (avg within-category sim) - (between-category sim)
            separation = avg_cohesion - inter_sim
            print(f"  {cat1:20s} <-> {cat2:20s}: {separation:.4f}")
    
    return centroids, category_embeddings


def visualize_embeddings_2d(embeddings, papers_ids_cats, pca_projector=None, method='pca', n_samples=5000, title_suffix=""):
    """
    Visualize paper embeddings in 2D using PCA or t-SNE
    
    Args:
        embeddings: Embedding object containing the paper embeddings
        papers_ids_cats: Dictionary mapping categories to paper IDs
        pca_projector: Pre-fitted PCA projector (for consistency across before/after)
        method: 'pca' or 'tsne'
        n_samples: Number of samples per category to visualize
        title_suffix: Additional text for the title
    """
    # Set up colors for each category
    colors = {
        'Computer Science': '#2E86AB',      # Blue
        'Physics': '#A23B72',               # Purple/Pink
        'Biology': '#F18F01',               # Orange
        'CS - Computer Vision': '#1E5A8E',  # Dark Blue
        'CS - NLP': '#4FA8D5',              # Light Blue
    }
    
    # Collect embeddings and labels for each category
    all_embeddings = []
    all_labels = []
    
    for category, paper_ids in papers_ids_cats.items():
        # Sample papers if there are too many
        if len(paper_ids) > n_samples:
            sampled_ids = np.random.choice(paper_ids, n_samples, replace=False)
        else:
            sampled_ids = paper_ids
        
        # Get embeddings for these papers
        paper_idxs = embeddings.get_idxs(sampled_ids)
        category_embeddings = embeddings.matrix[paper_idxs]
        
        all_embeddings.append(category_embeddings)
        all_labels.extend([category] * len(category_embeddings))
        
        print(f"{category}: {len(category_embeddings)} papers")
    
    # Concatenate all embeddings
    X = np.vstack(all_embeddings)
    print(f"\nTotal papers to visualize: {X.shape[0]}")
    print(f"Embedding dimension: {X.shape[1]}")
    
    # Project to 2D
    print(f"\nProjecting to 2D using {method.upper()}...")
    if method.lower() == 'pca':
        if pca_projector is None:
            raise ValueError("pca_projector must be provided for PCA method")
        X_2d = pca_projector.transform(X)
    elif method.lower() == 'tsne':
        projector = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        X_2d = projector.fit_transform(X)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    
    # Compute 2D metrics
    print(f"\n2D Projection Metrics:")
    print("-" * 60)
    main_categories = ['Computer Science', 'Physics', 'Biology']
    for category in main_categories:
        if category in papers_ids_cats:
            mask = np.array([label == category for label in all_labels])
            cat_points = X_2d[mask]
            centroid_2d = cat_points.mean(axis=0)
            spread_2d = np.linalg.norm(cat_points - centroid_2d, axis=1).mean()
            print(f"  {category:20s} 2D spread: {spread_2d:.4f}")
    
    # Inter-category distances in 2D
    print("\n2D Inter-category Distances:")
    print("-" * 60)
    centroids_2d = {}
    for category in main_categories:
        if category in papers_ids_cats:
            mask = np.array([label == category for label in all_labels])
            centroids_2d[category] = X_2d[mask].mean(axis=0)
    
    for i, cat1 in enumerate(main_categories):
        for cat2 in main_categories[i+1:]:
            if cat1 in centroids_2d and cat2 in centroids_2d:
                dist_2d = np.linalg.norm(centroids_2d[cat1] - centroids_2d[cat2])
                print(f"  {cat1:20s} <-> {cat2:20s}: {dist_2d:.4f}")
    
    # Plot each category separately for better legend
    for category in papers_ids_cats.keys():
        mask = np.array([label == category for label in all_labels])
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   c=colors[category], 
                   label=category, 
                   alpha=0.6, 
                   s=20,
                   edgecolors='none')
    
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.legend(fontsize=11, markerscale=2)
    plt.grid(True, alpha=0.3)
    
    return X_2d


if __name__ == "__main__":
    np.random.seed(42)
    
    if len(sys.argv) <= 1:
        print("Usage: python visualize_embeddings.py <path_to_finetuning_embedding> [method] [n_samples]")
        print("  Visualizes both before (baseline) and after (finetuned) embeddings")
        print("  method: 'pca' (default) or 'tsne'")
        print("  n_samples: number of samples per category (default: 5000)")
        sys.exit(1)
    
    method = sys.argv[2] if len(sys.argv) > 2 else 'pca'
    n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    
    # Load both embeddings
    print("Loading BEFORE embeddings (baseline)...")
    embeddings_before = Embedding(ProjectPaths.logreg_embeddings_path() / "after_pca" / "gte_large_256")
    
    print(f"Loading AFTER embeddings (finetuned) from: {sys.argv[1]}")
    embeddings_after = Embedding(sys.argv[1])
    
    # Load papers
    papers_ids = list(embeddings_after.papers_ids_to_idxs.keys())
    papers = load_papers(
        relevant_papers_ids=papers_ids,
        relevant_columns=["paper_id", "l1", "l2", "in_cache"],
    )
    
    # Define categories to include
    all_categories = ["Computer Science", "Physics", "Biology"]
    
    # Get paper IDs for each category
    papers_ids_cats = {}
    for category in all_categories:
        cat_papers = papers[papers["l1"] == category]["paper_id"].tolist()
        # Filter to only papers in embeddings
        cat_papers = [pid for pid in cat_papers if pid in embeddings_after.papers_ids_to_idxs]
        if len(cat_papers) > 0:
            papers_ids_cats[category] = cat_papers
    
    # Add Computer Vision and NLP subcategories
    cs_cv_papers = papers[
        (papers["l1"] == "Computer Science") & (papers["l2"] == "Computer Vision")
    ]["paper_id"].tolist()
    cs_cv_papers = [pid for pid in cs_cv_papers if pid in embeddings_after.papers_ids_to_idxs]
    if len(cs_cv_papers) > 0:
        papers_ids_cats["CS - Computer Vision"] = cs_cv_papers
    
    cs_nlp_papers = papers[
        (papers["l1"] == "Computer Science") & (papers["l2"] == "Natural Language Processing")
    ]["paper_id"].tolist()
    cs_nlp_papers = [pid for pid in cs_nlp_papers if pid in embeddings_after.papers_ids_to_idxs]
    if len(cs_nlp_papers) > 0:
        papers_ids_cats["CS - NLP"] = cs_nlp_papers
    
    print(f"\nCategories found: {list(papers_ids_cats.keys())}")
    print(f"Sample sizes: {[(cat, len(ids)) for cat, ids in papers_ids_cats.items()]}")
    
    # ========== KEY FIX: Compute high-dimensional metrics FIRST ==========
    print("\n" + "="*80)
    print("STEP 1: ANALYZING HIGH-DIMENSIONAL EMBEDDINGS")
    print("="*80)
    
    centroids_before, cat_embs_before = compute_category_metrics(
        embeddings_before, papers_ids_cats, name="BEFORE (Baseline)"
    )
    
    centroids_after, cat_embs_after = compute_category_metrics(
        embeddings_after, papers_ids_cats, name="AFTER (Fine-tuned)"
    )
    
    # ========== FIT PCA ON ALL MAIN CATEGORIES (not just cache) ==========
    print("\n" + "="*80)
    print("STEP 2: FITTING PCA FOR 2D VISUALIZATION")
    print("="*80)
    
    # Collect balanced sample from all main categories for PCA fitting
    pca_fit_embeddings = []
    pca_sample_size = 3000  # samples per category
    
    for category in ["Computer Science", "Physics", "Biology"]:
        if category in papers_ids_cats:
            paper_ids = papers_ids_cats[category]
            if len(paper_ids) > pca_sample_size:
                sampled_ids = np.random.choice(paper_ids, pca_sample_size, replace=False)
            else:
                sampled_ids = paper_ids
            
            paper_idxs = embeddings_before.get_idxs(sampled_ids)
            pca_fit_embeddings.append(embeddings_before.matrix[paper_idxs])
            print(f"PCA fitting: {category} - {len(sampled_ids)} papers")
    
    pca_fit_data = np.vstack(pca_fit_embeddings)
    print(f"\nFitting PCA on {pca_fit_data.shape[0]} balanced samples from all categories...")
    pca_projector = PCA(n_components=2, random_state=42)
    pca_projector.fit(pca_fit_data)
    print(f"Explained variance ratio: {pca_projector.explained_variance_ratio_}")
    print(f"Total explained variance: {pca_projector.explained_variance_ratio_.sum():.4f}")
    
    # ========== VISUALIZE IN 2D ==========
    print("\n" + "="*80)
    print("STEP 3: 2D VISUALIZATION")
    print("="*80)
    
    # Create side-by-side comparison
    fig = plt.figure(figsize=(20, 8))
    
    # Plot BEFORE
    plt.subplot(1, 2, 1)
    print("\n=== BEFORE (Baseline) ===")
    visualize_embeddings_2d(embeddings_before, papers_ids_cats, pca_projector=pca_projector, 
                           method=method, n_samples=n_samples)
    plt.title(f'BEFORE Fine-tuning ({method.upper()})', fontsize=14, fontweight='bold')
    
    # Plot AFTER
    plt.subplot(1, 2, 2)
    print("\n=== AFTER (Fine-tuned) ===")
    visualize_embeddings_2d(embeddings_after, papers_ids_cats, pca_projector=pca_projector, 
                           method=method, n_samples=n_samples)
    plt.title(f'AFTER Fine-tuning ({method.upper()})', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = f'embeddings_comparison_{method}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_filename}")
    
    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("SUMMARY: CHECKING FOR COLLAPSE")
    print("="*80)
    print("\nTo confirm Bio/Physics collapse, look for:")
    print("  1. HIGH-D: Bio-Physics cosine similarity should be HIGH in BEFORE, LOW in AFTER")
    print("  2. HIGH-D: Intra-category cohesion should increase for well-separated categories")
    print("  3. HIGH-D: Separation quality should be HIGH (within-sim much > between-sim)")
    print("  4. 2D: Bio/Physics should overlap in BEFORE plot, separate in AFTER plot")
    print("="*80)