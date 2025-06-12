# Setup Instructions

In order to generate the necessary directories and files, you require access to the internal Scholar Inbox database and (if you want to include information about the paper categories) a file `tsne_with_meta_full_for_plot_sorted.parquet`.

Given this access, then first run:

```bash
python from_db_to_files.py --papers_categories_old_file tsne_with_meta_full_for_plot_sorted.parquet --scholar_inbox_dict /path/to/scholar_inbox.db
```
