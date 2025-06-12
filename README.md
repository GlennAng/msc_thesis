# I. Setup Instructions

In order to run the code, you require 3 files that should all be placed in a directionary called `data`:
- `users_ratings.parquet` 
- `papers_texts.parquet`
- `papers.parquet`

If you have access to the internal Scholar Inbox database, you can generate these files automatically. But additionally, you would need a file  `tsne_with_meta_full_for_plot_sorted.parquet` to include information about the paper categories. Given this access, then run:


```bash
python from_db_to_files.py --scholar_inbox_dict --papers_categories_old_file /path/to/tsne_with_meta_full_for_plot_sorted.parquet 
```
Should the above fail to work, you may have to enter your database login credentials db_name, db_user, db_password, db_host and db_port as individual arguments (instead of scholar_inbox_dict which chooses the default settings).

In either case, afterwards check for correctness by running:
```bash
python load_files.py
```
# II. Overview of the Files
### 1. Columns in Users Ratings (865,157 Rows):
- **user_id:** Integer identifier of the user that performed the respective rating.
- **paper_id:** Integer identifier of the paper that was rated by the user.
- **rating:** Either 1 (upvote) or 0 (downvote).
- **time:** Datetime at which the rating was performed.
- **session_id:** Integer identifier of the session during which the rating was performed (if the user voted on two different papers within 7 hours, they are counted towards the same session). The session_id starts at 0 for each user.

The DataFrame is sorted first by user_id and second by time.

### 2. Columns in Papers (3,696,884 Rows):
- **paper_id:** Integer identifier of the paper in the database.
- **in_ratings:** Boolean indicating whether the paper was rated by any user (whether it appears in the users ratings table).
- **in_cache:** Boolean indicating whether the paper is used in the random cache (for negative sampling).
- **l1:** String of the hierarchy level-1 category (e.g. "Computer Science")
- **l2:** String of the hierarchy level-2 category (e.g. "Machine Learning)
- **l3:** String of the hierarchy level-3 category (e.g. "Deep Learning)

The DataFrame is sorted by paper_id.

### 3. Columns in Papers Texts (3,696,884 Rows):
- **paper_id:** Integer identifier of the paper in the database.
- **title:** String of the scientific paper's title.
- **abstract:** String of the scientific paper's abstract.
- **authors:** String of the scientific paper's authors.

The DataFrame is sorted by paper_id.

# III. Overview of the Subdirectories
#### 1. logreg:  Experiments for evaluating the recommender model via logistic regression.
#### 2. finetuning: Experiments for fine-tuning the embedding model in PyTorch.