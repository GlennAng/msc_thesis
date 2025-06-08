from pathlib import Path

class ProjectPaths:
    @staticmethod
    def _base_path():
        return Path(__file__).resolve().parent

    @staticmethod
    def add_all_paths_to_sys(paths_list : list) -> None:
        import sys
        for path in paths_list:
            path_str = str(path.resolve())
            if path_str not in sys.path:
                sys.path.append(path_str)

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
    def logreg_data_path():
        return ProjectPaths.logreg_path() / "data"

    @staticmethod
    def logreg_data_embeddings_path():
        return ProjectPaths.logreg_data_path() / "embeddings"

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