from finetuning_preprocessing import *
from finetuning_data import load_datasets
from finetuning_model import *
from finetuning_evaluation import *

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, train_val_dataset, val_users_embeddings_idxs, negative_samples, test_papers = load_datasets()
    projection_state_dict = "/home/scholar/glenn_rp/msc_thesis/data/finetuning/parameters/gte_base_256_projection.pt"
    users_embeddings_state_dict = "/home/scholar/glenn_rp/msc_thesis/data/finetuning/parameters/gte_base_256_users_embeddings.pt"
    finetuning_model = load_finetuning_model(GTE_BASE_PATH, device, projection_state_dict, users_embeddings_state_dict, val_users_embeddings_idxs)
    print(run_validation(finetuning_model, val_dataset, negative_samples, train_val_dataset))
