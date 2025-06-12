import json
from pathlib import Path

class ProjectPaths:
    _db_backup_date = None

    @staticmethod
    def get_db_backup_date():
        if ProjectPaths._db_backup_date is None:
            db_backup_date_json = ProjectPaths.data_path() / "db_backup_date.json"
            with open(db_backup_date_json, "r") as f:
                ProjectPaths._db_backup_date = json.load(f)["db_backup_date"]
        return ProjectPaths._db_backup_date

    @staticmethod
    def add_all_paths_to_sys(paths_list : list) -> None:
        import sys
        for path in paths_list:
            path_str = str(path.resolve())
            if path_str not in sys.path:
                sys.path.append(path_str)

    @staticmethod
    def _base_path():
        return Path(__file__).resolve().parent

    @staticmethod
    def data_path():
        return ProjectPaths._base_path() / "data"

    @staticmethod
    def data_db_backup_date_path():
        return ProjectPaths.data_path() / ProjectPaths.get_db_backup_date()

    @staticmethod
    def data_db_backup_date_papers_path():
        return ProjectPaths.data_db_backup_date_path() / "papers.parquet"

    @staticmethod
    def data_db_backup_date_papers_texts_path():
        return ProjectPaths.data_db_backup_date_path() / "papers_texts.parquet"

    @staticmethod
    def data_db_backup_date_users_mapping_path():
        return ProjectPaths.data_db_backup_date_path() / "users_mapping.pkl"

    @staticmethod
    def data_db_backup_date_users_ratings_path():
        return ProjectPaths.data_db_backup_date_path() / "users_ratings.parquet"

    @staticmethod
    def data_db_backup_date_users_ratings_mapped_path():
        return ProjectPaths.data_db_backup_date_path() / "users_ratings_mapped.parquet"

    @staticmethod
    def finetuning_path():
        return ProjectPaths._base_path() / "finetuning"

    def finetuning_data_path():
        return ProjectPaths.finetuning_path() / "data"

    @staticmethod
    def finetuning_src_path():
        return ProjectPaths.finetuning_path() / "src"
    
    @staticmethod
    def logreg_path():
        return ProjectPaths._base_path() / "logreg"

    @staticmethod
    def logreg_embeddings_path():
        return ProjectPaths.logreg_path() / "embeddings"

    @staticmethod
    def logreg_embeddings_relevant_papers_path():
        return ProjectPaths.logreg_embeddings_path() / "relevant_papers.pkl"

    @staticmethod
    def logreg_experiments_path():
        return ProjectPaths.logreg_path() / "experiments"

    @staticmethod
    def logreg_outputs_path():
        return ProjectPaths.logreg_path() / "outputs"

    @staticmethod
    def logreg_src_path():
        return ProjectPaths.logreg_path() / "src"

    @staticmethod
    def logreg_src_embeddings_path():
        return ProjectPaths.logreg_src_path() / "embeddings"

    @staticmethod
    def logreg_src_processing_path():
        return ProjectPaths.logreg_src_path() / "processing"

    @staticmethod
    def logreg_src_training_path():
        return ProjectPaths.logreg_src_path() / "training"

    @staticmethod
    def logreg_src_visualization_path():
        return ProjectPaths.logreg_src_path() / "visualization"

    @staticmethod
    def add_logreg_src_paths_to_sys() -> None:
        paths = [
            ProjectPaths.logreg_src_path(),
            ProjectPaths.logreg_src_embeddings_path(),
            ProjectPaths.logreg_src_processing_path(),
            ProjectPaths.logreg_src_training_path(),
            ProjectPaths.logreg_src_visualization_path()
        ]
        ProjectPaths.add_all_paths_to_sys(paths)