# I. Setup Instructions
First, create and activate a conda environment via:
```bash
conda env create -f environment.yml
conda activate glenn_msc_env
```


In order to run the code, you require 5 files that should all be placed in a directionary called `data`:

- `papers.parquet`
- `papers_texts.parquet`
- `relevant_papers_ids.pkl`
- `users_ratings.parquet` 
- `finetuning_users.pkl`

You can check for completeness and correctness by running:
```bash
python -m shared.src.load_files
```

If you have access to the internal Scholar Inbox database, you can generate these files automatically. But additionally, you would need a file  `tsne_with_meta_full_for_plot_sorted.parquet` to include information about the paper categories. Given this access, then run:


```bash
python -m shared.scripts.from_db_to_files --scholar_inbox_dict --papers_categories_old_file /path/to/tsne_with_meta_full_for_plot_sorted.parquet 
```
Should the above fail to work, you may have to enter your database login credentials db_name, db_user, db_password, db_host and db_port as individual arguments (instead of scholar_inbox_dict which chooses the default settings).

```bash
python -m logreg.src.training.get_users_ratings
```
```bash
python -m logreg.src.embeddings.find_relevant_papers
```

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

# III. Overview of the Subdirectories
#### 1. logreg:  Experiments for evaluating the recommender model via logistic regression.
#### 2. finetuning: Experiments for fine-tuning the embedding model in PyTorch.