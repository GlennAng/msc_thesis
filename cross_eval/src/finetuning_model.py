from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_SAVE_PATH = "../data/models/"

def load_transformer_model(transformer_path : str, device : torch.device) -> AutoModel:
    return AutoModel.from_pretrained(transformer_path, trust_remote_code = True, unpad_inputs = True, torch_dtype = "auto").to(device)

def load_transformer_tokenizer(transformer_path : str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(transformer_path)

def save_transformer_model(transformer_model : AutoModel, model_path : str) -> None:
    transformer_model.save_pretrained(model_path)

def get_transformer_embedding_dim(transformer_model : AutoModel) -> int:
    return transformer_model.config.hidden_size

def load_projection(embedding_dim : int, projection_dim : int, device : torch.device, projection_state_dict : dict = None, seed : int = None) -> nn.Linear:
    if seed is not None:
        torch.manual_seed(seed)
    projection = nn.Linear(embedding_dim, projection_dim).to(device)
    if projection_state_dict is not None:
        projection.load_state_dict(projection_state_dict)
    return projection


class FinetuningModel(nn.Module):
    def __init__(self, transformer_model : AutoModel, projection : nn.Linear) -> None:
        super().__init__()
        self.transformer_model = transformer_model
        self.transformer_model_name = transformer_model.config._name_or_path
        self.projection = projection

    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor) -> torch.Tensor:
        outputs = self.transformer_model(input_ids = input_ids, attention_mask = attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = self.projection(embeddings)
        embeddings = F.normalize(embeddings, p = 2, dim = 1)
        return embeddings

def load_finetuning_model(transformer_path : str, device : torch.device, projection_dim : int = 256, projection_state_dict : dict = None, seed : int = None) -> FinetuningModel:
    transformer_model = load_transformer_model(transformer_path, device)
    embedding_dim = get_transformer_embedding_dim(transformer_model)
    if type(projection_state_dict) == str:
        projection_state_dict = torch.load(projection_state_dict, map_location = device, weights_only = True)
    if projection_state_dict is not None:
        projection_dim = projection_state_dict["weight"].shape[0]
    projection = load_projection(embedding_dim, projection_dim, device, projection_state_dict, seed)
    return FinetuningModel(transformer_model, projection)


"""
def load_projection(embedding_dim : int, projection_dim : int, device : torch.device, seed : int = None,
                    projection_state_dict : dict = None, projection_coefs : np.ndarray = None) -> nn.Linear:
    assert projection_state_dict is None or projection_coefs is None
    if seed is not None:
        torch.manual_seed(seed)
    projection = nn.Linear(embedding_dim, projection_dim).to(device)
    if projection_state_dict is not None:
        projection.load_state_dict(projection_state_dict)
    elif projection_coefs is not None:
        if type(projection_coefs) == str:
            projection_coefs = np.load(projection_coefs)
            projection_coefs = torch.from_numpy(projection_coefs)
        else:
            projection_coefs = torch.tensor(projection_coefs)
        assert projection_coefs.shape[1] == embedding_dim
        assert projection_coefs.shape[0] == projection_dim
        with torch.no_grad():
            projection.weight.copy_(projection_coefs)
            projection.bias = nn.Parameter(torch.zeros_like(projection.bias))
    return projection





def load_finetuning_model()







def load_model(model : nn.Module, model_file : str, device : torch.device) -> None:
    model_state_dict = torch.load(model_file, map_location = device, weights_only = True)
    model.load_state_dict(model_state_dict)

def save_model(model : nn.Module, model_path : str):
    model_path = model_path.rstrip(".pt").rstrip(".pth")
    try:
        model.save_pretrained(model_path)
    except AttributeError:
        model_path += ".pt"
        torch.save(model.state_dict(), model_path)


def load_transformer(transformer_path : str, device : torch.device) -> AutoModel:
    return AutoModel.from_pretrained(transformer_path, trust_remote_code = True, unpad_inputs = True, torch_dtype = "auto").to(device)

def get_transformer_embedding_dim(transformer : AutoModel) -> int:
    return transformer.config.hidden_size

def load_projection(embedding_dim : int, projection_dim : int, device : torch.device, seed : int = None, projection_file : str = None) -> nn.Linear:
    if seed is not None:
        torch.manual_seed(seed)
    projection = nn.Linear(embedding_dim, projection_dim).to(device)
    if projection_file is not None:
        load_model(projection, projection_file, device)
    return projection

def load_projection(config : dict, projection_state_dict : dict = None, projection_coefs_file : str = None) -> torch.nn.Linear:
    


    projection = None
    if config["apply_projection"]:
        assert projection_state_dict is None or projection_coefs_file is None
        embedding_dim, projection_dim = config["embedding_dim"], config["projection_dim"]
        projection = nn.Linear(embedding_dim, projection_dim).to(config["device"])
        if projection_state_dict is not None:
            projection.load_state_dict(projection_state_dict)
        elif projection_coefs_file is not None:
            projection_coefs = torch.load(projection_coefs_file, map_location = config["device"], weights_only = True)
            assert projection_coefs.shape[1] == embedding_dim
            assert projection_coefs.shape[0] == projection_dim
            with torch.no_grad():
                projection.weight.copy_(projection_coefs)
                projection.bias.zero_()
        config["embedding_dim"] = projection_dim
    return projection
"""
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_model = load_transformer_model("Alibaba-NLP/gte-base-en-v1.5", device)
    print(transformer_model.config._name_or_path)


    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gte_base = load_transformer_model(GTE_BASE_PATH, device)
    tokenizer = load_transformer_tokenizer(GTE_BASE_PATH)
    projection = load_projection(768, 256, device, seed = 42, projection_coefs = "../data/embeddings/after_pca/gte_base_2025-02-23_256_no_z/pca_components.npy")
    original_papers_embeddings = load_papers_embeddings("../data/embeddings/after_pca/gte_base_2025-02-23_256_no_z")
    recomputed_papers_embeddings = recompute_papers_embeddings(gte_base, tokenizer, projection = projection, l2_normalization = True)
    print(original_papers_embeddings[1, :15])
    print(recomputed_papers_embeddings[1, :15])
    """

    