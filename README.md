# I. Setup Instructions
First, create and activate a conda environment via:
```bash
conda env create -f environment.yml
conda activate glenn_msc_env
```


In order to run the code, you require 6 files that should all be placed in a directionary called `data`:

- `papers.parquet`
- `papers_texts.parquet`
- `relevant_papers_ids.pkl`
- `users_ratings.parquet` 
- `finetuning_users.pkl`
- `users_significant_categories.parquet`

You can check for completeness and correctness by running:
```bash
python -m code.src.load_files
```

If you have access to the internal Scholar Inbox database, you can generate these files automatically. But additionally, you would need a file  `tsne_with_meta_full_for_plot_sorted.parquet` to include information about the paper categories. Given this access, then run:


```bash
python -m code.scripts.from_db_to_files --scholar_inbox_dict --papers_categories_old_file /path/to/tsne_with_meta_full_for_plot_sorted.parquet 
```

```bash
python -m logreg.src.training.users_ratings
```
```bash
python -m logreg.src.embeddings.find_relevant_papers
```

Should the first step fail to work, you may have to enter your database login credentials db_name, db_user, db_password, db_host and db_port as individual arguments (instead of scholar_inbox_dict which chooses the default settings).

# II. Overview of the Files

### 1. Columns in Papers:
- **paper_id:** Integer identifier of the paper in the database.
- **in_ratings:** Boolean indicating whether the paper was rated by any user (whether it appears in the users ratings table).
- **in_cache:** Boolean indicating whether the paper is used in the random cache (for negative sampling).
- **l1:** String of the hierarchy level-1 category (e.g. "Computer Science")
- **l2:** String of the hierarchy level-2 category (e.g. "Machine Learning)
- **l3:** String of the hierarchy level-3 category (e.g. "Deep Learning)

The DataFrame is sorted by paper_id.

### 2. Columns in Papers Texts:
- **paper_id:** Integer identifier of the paper in the database.
- **title:** String of the scientific paper's title.
- **abstract:** String of the scientific paper's abstract.
- **authors:** String of the scientific paper's authors.

The DataFrame is sorted by paper_id.

### 3. Relevant Papers IDs:
A sorted list containing a subset of paper IDs. These papers are sufficient to run our logistic regression experiments (no need to embed and load the entire database).

### 4. Columns in Users Ratings:
- **user_id:** Integer identifier of the user that performed the respective rating.
- **paper_id:** Integer identifier of the paper that was rated by the user.
- **rating:** Either 1 (upvote) or 0 (downvote).
- **time:** Datetime at which the rating was performed.
- **session_id:** Integer identifier of the session during which the rating was performed (if the user voted on two different papers within 7 hours, they are counted towards the same session). The session_id starts at 0 for each user.

The DataFrame is sorted first by user_id and second by time.

### 5. Finetuning Users:
A dictionary with 3 keys, each referencing a disjoint list of sorted user IDs:
- **test:** 500 users for which there exists a session-based split such that there are at least 16 up/down-votes in the training split and at least 4 up/down-votes in the validation set (while further assuring the training split makes up at least 70% in total).
- **val:** Same as test but 500 different users.
- **train:** All users with at least 20 up/down-votes (independent of session-based splits) except those in test and val.

### 6. Columns in Users Significant Categories:
- **user_id:** Integer identifier of the user for which the 4 most popular scientific categories among his upvotes are stated 
(but only if they make up at least 10% of the total upvotes).
- **rank:** Integer from 1 to 4 with 1 indicating the most popular category.
- **category:** String representing the level-1 category name (e.g. "Computer Science").
- **proportion:** Float in [0, 1] indicating how much of the total upvotes belong to that category.

# III. Logistic Regression Evaluation:
The following describes how to perform experiments for evaluating the recommender model via logistic regression.
## 1. Embedding Computation:
In order to run our experiments, you need a directory that includes the two files `abs_X.npy` and `abs_paper_ids_to_idx.pkl`. 
The first is a numpy matrix of shape (n_papers, embedding_dim) containing the text embedding of each paper in the database whereas the second is a dictionary which maps the respective paper IDs to their index in this matrix. Note that for most experiments, it is sufficient to only embed the papers stored in `relevant_papers_ids.pkl`.

You may use your own embeddings or compute them for one of the provided models (details in `code/logreg`). For example, then run the following to compute embeddings only for the relevant papers.
Instead, you could compute embeddings for the entire database by further appending `--all_papers`:
```bash
python -m code.scripts.embed_run --model_name gte-large-en-v1.5 --batch_size 500
```

Afterwards, the embeddings of the original dimensionality will be stored in `code/logreg/embeddings/before_pca/gte_large` and the 
356-dimensional embeddings (after PCA and attaching GloVe category embeddings) in `code/logreg/embeddings/after_pca/gte_large_256_categories_l2_unit_100`.
But note that the latter requires you to have the GloVe word embeddings downloaded into `code/logreg/embeddings/glove/glove.6B.100d.txt`.

## 2. Experiments:
Without needing to make any adjustments, you can run comprehensive experiments as follows:
1) Construct example config-files with recommended settings.
```bash
python -m code.scripts.create_example_configs
```
2) Run the experiments:

```bash
python -m code.scripts.average_seeds --config_path code/logreg/experiments/example_configs/example_config.json
```
This will perform 5 experiments (5 different random seeds) and summarize the results in 
`code/logreg/outputs/example_config_averaged/global_visu_bal.pdf`.

More details on changing the configurations for these experiments are given in `code/logreg`.

# IV. Embedding Model Fine-tuning:
The following describes how to fine-tune your text embedding model in PyTorch.
