from pathlib import Path

FINETUNING_MODEL = "gte_large_256"
FINETUNING_MODEL_HF = "Alibaba-NLP/gte-large-en-v1.5"


class ProjectPaths:

    @staticmethod
    def _base_path():
        return Path(__file__).parents[2].resolve()

    @staticmethod
    def data_path():
        return ProjectPaths._base_path() / "data"

    @staticmethod
    def data_papers_path():
        return ProjectPaths.data_path() / "papers.parquet"

    @staticmethod
    def data_papers_texts_path():
        return ProjectPaths.data_path() / "papers_texts.parquet"

    @staticmethod
    def data_relevant_papers_ids_path():
        return ProjectPaths.data_path() / "relevant_papers_ids.pkl"

    @staticmethod
    def data_users_mapping_path():
        return ProjectPaths.data_path() / "users_mapping.pkl"

    @staticmethod
    def data_users_ratings_path():
        return ProjectPaths.data_path() / "users_ratings.parquet"

    @staticmethod
    def data_users_ratings_before_mapping_path():
        return ProjectPaths.data_path() / "users_ratings_before_mapping.parquet"

    @staticmethod
    def data_finetuning_users_path():
        return ProjectPaths.data_path() / "finetuning_users.pkl"

    @staticmethod
    def finetuning_path():
        return ProjectPaths._base_path() / "code" / "finetuning"

    @staticmethod
    def finetuning_data_path():
        return ProjectPaths.finetuning_path() / "data"

    @staticmethod
    def finetuning_data_model_path():
        return ProjectPaths.finetuning_data_path() / FINETUNING_MODEL

    @staticmethod
    def finetuning_data_model_hf():
        return FINETUNING_MODEL_HF

    @staticmethod
    def finetuning_data_model_state_dicts_path():
        return ProjectPaths.finetuning_data_model_path() / "state_dicts"

    @staticmethod
    def finetuning_data_model_state_dicts_projection_path():
        return ProjectPaths.finetuning_data_model_state_dicts_path() / "projection.pt"

    @staticmethod
    def finetuning_data_model_state_dicts_categories_embeddings_l1_path():
        return ProjectPaths.finetuning_data_model_state_dicts_path() / "categories_embeddings_l1.pt"

    @staticmethod
    def finetuning_data_model_state_dicts_users_embeddings_path():
        return ProjectPaths.finetuning_data_model_state_dicts_path() / "users_embeddings.pt"

    @staticmethod
    def finetuning_data_model_datasets_path():
        return ProjectPaths.finetuning_data_model_path() / "datasets"

    @staticmethod
    def finetuning_data_experiments_path():
        return ProjectPaths.finetuning_data_path() / "experiments"

    @staticmethod
    def finetuning_data_model_datasets_negative_samples_for_seeds_path():
        return ProjectPaths.finetuning_data_model_datasets_path() / "negative_samples_for_seeds.pkl"

    @staticmethod
    def finetuning_data_model_datasets_val_negative_samples_path():
        return ProjectPaths.finetuning_data_model_datasets_path() / "val_negative_samples.pt"

    @staticmethod
    def finetuning_data_model_datasets_test_papers_path():
        return ProjectPaths.finetuning_data_model_datasets_path() / "test_papers.pt"

    @staticmethod
    def finetuning_data_model_datasets_train_dataset_path():
        return ProjectPaths.finetuning_data_model_datasets_path() / "train_dataset.pt"

    @staticmethod
    def finetuning_data_model_datasets_val_dataset_path():
        return ProjectPaths.finetuning_data_model_datasets_path() / "val_dataset.pt"

    @staticmethod
    def logreg_path():
        return ProjectPaths._base_path() / "code" / "logreg"

    @staticmethod
    def logreg_embeddings_path():
        return ProjectPaths.logreg_path() / "embeddings"

    @staticmethod
    def logreg_experiments_path():
        return ProjectPaths.logreg_path() / "experiments"

    @staticmethod
    def logreg_outputs_path():
        return ProjectPaths.logreg_path() / "outputs"
