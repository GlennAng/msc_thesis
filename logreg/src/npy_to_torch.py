import numpy as np
import pickle
import torch
import torch.nn as nn
from main import get_users_ids

users_coefs_file = "../data/models/gte_base_init/users_coefs.npy"
users_ids_to_idxs_file = "../data/models/gte_base_init/users_coefs_ids_to_idxs.pkl"
with open(users_ids_to_idxs_file, "rb") as f:
    users_ids_to_idxs = pickle.load(f)
pca_components_file = "../data/models/gte_base_init/pca_components.npy"

def users_coefs_to_torch(users_coefs_file : str, users_idxs : list):
    users_coefs_torch_file = users_coefs_file.replace(".npy", ".pt")
    users_coefs = np.load(users_coefs_file)
    users_coefs_tensor = torch.tensor(users_coefs)
    #torch.save(users_coefs_tensor, users_coefs_torch_file)

def pca_components_to_torch(pca_components_file : str):
    pca_components_torch_file = pca_components_file.replace(".npy", ".pt")
    pca_components = np.load(pca_components_file)
    pca_components_tensor = torch.tensor(pca_components)
    torch.save(pca_components_tensor, pca_components_torch_file)


users_ids = get_users_ids(users_selection = "random", max_users = 500, min_n_posrated = 20, min_n_negrated = 20, 
                          random_state = 42, take_complement = True)
print(users_ids_to_idxs)
users_idxs = [users_ids_to_idxs[user_id] for user_id in users_ids]

#users_coefs_to_torch(users_coefs_file)
#pca_components_to_torch(pca_components_file)

