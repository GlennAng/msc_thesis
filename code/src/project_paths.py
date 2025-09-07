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
    def data_users_significant_categories_path():
        return ProjectPaths.data_path() / "users_significant_categories.parquet"

    @staticmethod
    def data_users_ratings_path():
        return ProjectPaths.data_path() / "users_ratings.parquet"

    @staticmethod
    def data_finetuning_users_ids_path():
        return ProjectPaths.data_path() / "finetuning_users_ids.pkl"
    
    @staticmethod
    def data_finetuning_users_ids_old_path():
        return ProjectPaths.data_path() / "finetuning_users_ids_old.pkl"
    
    @staticmethod
    def data_sequence_users_ids_path():
        return ProjectPaths.data_path() / "sequence_users_ids.pkl"

    @staticmethod
    def data_session_based_no_filtering_ratings_path():
        return ProjectPaths.data_path() / "session_based_no_filtering_ratings.parquet"
    
    @staticmethod
    def data_session_based_no_filtering_ratings_old_path():
        return ProjectPaths.data_path() / "session_based_no_filtering_ratings_old.parquet"

    @staticmethod
    def data_session_based_filtering_ratings_path():
        return ProjectPaths.data_path() / "session_based_filtering_ratings.parquet"

    @staticmethod
    def data_session_based_filtering_ratings_old_path():
        return ProjectPaths.data_path() / "session_based_filtering_ratings_old.parquet"
    

    @staticmethod
    def finetuning_path():
        return ProjectPaths._base_path() / "code" / "finetuning"

    @staticmethod
    def finetuning_data_path():
        return ProjectPaths.finetuning_path() / "data"

    @staticmethod
    def finetuning_data_checkpoints_path():
        return ProjectPaths.finetuning_data_path() / "checkpoints"

    @staticmethod
    def finetuning_data_experiments_path():
        return ProjectPaths.finetuning_data_path() / "experiments"

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
    def finetuning_data_model_datasets_dataset_train_path():
        return ProjectPaths.finetuning_data_model_datasets_path() / "dataset_train.pt"

    @staticmethod
    def finetuning_data_model_datasets_dataset_val_path():
        return ProjectPaths.finetuning_data_model_datasets_path() / "dataset_val.pt"

    @staticmethod
    def finetuning_data_model_datasets_eval_papers_tokenized_val_users_path():
        return (
            ProjectPaths.finetuning_data_model_datasets_path()
            / "eval_papers_tokenized_val_users.pt"
        )

    @staticmethod
    def finetuning_data_model_datasets_eval_papers_tokenized_test_users_path():
        return (
            ProjectPaths.finetuning_data_model_datasets_path()
            / "eval_papers_tokenized_test_users.pt"
        )
    
    @staticmethod
    def finetuning_data_model_datasets_negative_samples_tokenized_train_path():
        return (
            ProjectPaths.finetuning_data_model_datasets_path() / "negative_samples_tokenized_train.pt"
        )

    @staticmethod
    def finetuning_data_model_datasets_negative_samples_tokenized_val_path():
        return (
            ProjectPaths.finetuning_data_model_datasets_path() / "negative_samples_tokenized_val.pt"
        )

    @staticmethod
    def finetuning_data_model_datasets_negative_samples_matrix_val_path():
        return ProjectPaths.finetuning_data_model_datasets_path() / "negative_samples_matrix_val.pt"

    @staticmethod
    def finetuning_data_train_negative_samples_ids_path():
        return ProjectPaths.finetuning_data_path() / "train_negative_samples_ids.pkl"

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
    
    @staticmethod
    def sequence_path():
        return ProjectPaths._base_path() / "code" / "sequence"
    
    @staticmethod
    def sequence_data_path():
        return ProjectPaths.sequence_path() / "data"
    
    @staticmethod
    def sequence_data_users_embeddings_path():
        return ProjectPaths.sequence_data_path() / "users_embeddings"
    
    @staticmethod
    def sequence_data_model_path():
        return ProjectPaths.sequence_data_path() / FINETUNING_MODEL
    
    @staticmethod
    def sequence_data_model_state_dicts_path():
        return ProjectPaths.sequence_data_model_path() / "state_dicts"
    
    @staticmethod
    def sequence_data_model_state_dicts_papers_encoder_path():
        return ProjectPaths.sequence_data_model_state_dicts_path() / "papers_encoder"

    @staticmethod
    def sequence_data_model_datasets_path():
        return ProjectPaths.sequence_data_model_path() / "datasets"

    @staticmethod
    def sequence_data_mind_path():
        return ProjectPaths.sequence_data_path() / "mind"
    
    @staticmethod
    def sequence_data_processed_users_ratings_path():
        return ProjectPaths.sequence_data_path() / "processed_users_ratings.parquet"
