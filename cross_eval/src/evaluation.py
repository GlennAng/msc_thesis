from algorithm import Algorithm, Evaluation, Score, get_score
from algorithm import get_cross_val, get_model
from compute_tfidf import load_vectorizer
from data_handling import get_voting_weight_for_user
from embedding import Embedding
from training_data import load_base_for_user, load_global_cache, load_filtered_cache_for_user, load_training_data_for_user, Cache_Type, LABEL_DTYPE
from weights_handler import Weights_Handler

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

import json
import numpy as np
import os
import pickle

class Evaluator:
    def __init__(self, config : dict, users_ids : list, hyperparameters_combinations : list, wh : Weights_Handler) -> None:
        self.config = config
        self.users_ids = users_ids
        self.hyperparameters, self.hyperparameters_combinations = config["hyperparameters"], hyperparameters_combinations
        self.wh = wh

        self.random_state = config["random_state"]
        self.scores, self.scores_n = config["scores"], len(config["scores"])
        self.users_voting_weights = {user_id : get_voting_weight_for_user(user_id) for user_id in users_ids} if wh.need_voting_weight else None
        self.include_global_cache = self.config["include_cache"] and self.config["cache_type"] == Cache_Type.GLOBAL
        if self.config["evaluation"] == Evaluation.CROSS_VALIDATION:
            self.cross_val = get_cross_val(config["stratified"], config["k_folds"], self.random_state)

    def evaluate_embedding(self, embedding : Embedding) -> None:
        self.embedding = embedding
        if self.config["save_tfidf_coefs"]:
            vectorizer = load_vectorizer(self.config["embedding_folder"])
            feature_names = vectorizer.get_feature_names_out()
            with open(f"{self.config['outputs_dir']}/feature_names.pkl", "wb") as f:
                pickle.dump(feature_names, f)
        if "save_coefs" in self.config and self.config["save_coefs"]:
            if self.config["evaluation"] != Evaluation.TRAIN_TEST_SPLIT:
                raise ValueError("Coefficient saving is only supported with train-test split evaluation.")
        if self.include_global_cache:
            self.global_cache_idxs, self.global_cache_n, self.y_global_cache = load_global_cache(self.embedding, self.config["max_cache"], self.random_state)
        if self.config["n_jobs"] == 1:
            for user_id in self.users_ids:
                self.evaluate_user(user_id)
        else:
            Parallel(n_jobs = self.config["n_jobs"])(delayed(self.evaluate_user)(user_id) for user_id in self.users_ids)

    def evaluate_user(self, user_id : int) -> None:
        user_results_dict, user_predictions_dict = {}, {}
        posrated_idxs = sorted(self.embedding.get_posrated_idxs_for_user(user_id, self.config["rated_paper_removal"], self.config["remaining_percentage"], self.random_state))
        negrated_idxs = sorted(self.embedding.get_negrated_idxs_for_user(user_id, self.config["rated_paper_removal"], self.config["remaining_percentage"], self.random_state))
        posrated_n, negrated_n = len(posrated_idxs), len(negrated_idxs)
        rated_idxs = np.concatenate((posrated_idxs, negrated_idxs))
        rated_labels = np.concatenate((np.ones(len(posrated_idxs), dtype = LABEL_DTYPE), np.zeros(len(negrated_idxs), dtype = LABEL_DTYPE)))
        base_idxs, base_n, y_base = (load_base_for_user(self.embedding, user_id, self.config["base_paper_removal"], self.config["remaining_percentage"], self.random_state) 
                                     if self.config["include_base"] else ([], 0, []))
        user_predictions_dict["base_ids"] = self.embedding.get_papers_ids(base_idxs)
        cache_idxs, cache_n, y_cache = self.set_cache_for_user(user_id, posrated_n, negrated_n, base_n)
        user_info = self.store_user_info(user_id, posrated_n, negrated_n, base_n, cache_n)

        if self.config["evaluation"] == Evaluation.TRAIN_TEST_SPLIT:
            train_rated_idxs, val_rated_idxs, y_train_rated, y_val = train_test_split(rated_idxs, rated_labels, test_size = self.config["test_size"], random_state = self.random_state,
                                                                                      stratify = rated_labels if self.config["stratified"] else None)
            X_train_rated, X_val, X_train, y_train = self.load_data_for_user(train_rated_idxs, val_rated_idxs, y_train_rated, base_idxs, y_base, cache_idxs, y_cache)
            user_data_statistics = self.get_data_statistics_for_user(y_train_rated, base_n, cache_n)
            user_results, user_predictions, user_coefs = self.train_model_for_user(X_train, y_train, X_train_rated, y_train_rated, X_val, y_val, user_data_statistics, user_info["voting_weight"])
            user_results_dict[0] = user_results
            if self.config["save_users_predictions"]:
                user_predictions_dict[0] = {**self.get_papers_ids(train_rated_idxs, y_train_rated, val_rated_idxs, y_val), 
                                            **{"train_predictions": user_predictions["train_predictions"], "val_predictions": user_predictions["val_predictions"], 
                                               "tfidf_coefs": user_predictions["tfidf_coefs"]}}

        elif self.config["evaluation"] == Evaluation.CROSS_VALIDATION:
            split = self.cross_val.split(X = rated_idxs, y = rated_labels)
            for fold_idx, (fold_train_idxs, fold_val_idxs) in enumerate(split):
                train_rated_idxs, val_rated_idxs = rated_idxs[fold_train_idxs], rated_idxs[fold_val_idxs]
                y_train_rated, y_val = rated_labels[fold_train_idxs], rated_labels[fold_val_idxs]
                X_train_rated, X_val, X_train, y_train = self.load_data_for_user(train_rated_idxs, val_rated_idxs, y_train_rated, base_idxs, y_base, cache_idxs, y_cache)
                user_data_statistics = self.get_data_statistics_for_user(y_train_rated, base_n, cache_n)
                fold_results, fold_predictions, _ = self.train_model_for_user(X_train, y_train, X_train_rated, y_train_rated, X_val, y_val, user_data_statistics, user_info["voting_weight"])
                user_results_dict[fold_idx] = fold_results
                if self.config["save_users_predictions"]:
                    user_predictions_dict[fold_idx] = {**self.get_papers_ids(train_rated_idxs, y_train_rated, val_rated_idxs, y_val),
                                                       **{"train_predictions": fold_predictions["train_predictions"], "val_predictions": fold_predictions["val_predictions"], 
                                                          "tfidf_coefs": fold_predictions["tfidf_coefs"]}}
        self.save_user_info(user_id, user_info)
        self.save_user_results(user_id, user_results_dict)
        if self.config["save_users_predictions"]:
            self.save_users_predictions(user_id, user_predictions_dict)
        if "save_coefs" in self.config and self.config["save_coefs"]:
            self.save_user_coefs(user_id, user_coefs)
        print(f"User {user_id} done.")
    
    def set_cache_for_user(self, user_id : int, posrated_n : int, negrated_n : int, base_n) -> tuple:
        pos_n = posrated_n + base_n
        if self.include_global_cache:
            cache_idxs, cache_n, y_cache = self.global_cache_idxs, self.global_cache_n, self.y_global_cache
        else:
            if self.config["include_cache"]:
                target_ratio = self.config["target_ratio"] if "target_ratio" in self.config else None
                assert target_ratio is None or (target_ratio > 0 and target_ratio < 1)
                cache_idxs, cache_n, y_cache = load_filtered_cache_for_user(self.embedding, self.config["cache_type"], user_id, self.config["max_cache"], self.random_state, 
                                                                            pos_n, negrated_n, target_ratio)
            else:
                cache_idxs, cache_n, y_cache = [], 0, []
        return cache_idxs, cache_n, y_cache
    
    def store_user_info(self, user_id : int, posrated_n : int, negrated_n : int, base_n : int = 0, cache_n = 0) -> dict:
        return {"voting_weight" : self.users_voting_weights[user_id] if self.wh.need_voting_weight else None,
                "n_posrated" : posrated_n, "n_negrated" : negrated_n, "n_base" : base_n , "n_cache" : cache_n}
    
    def load_data_for_user(self, train_rated_idxs : np.ndarray, val_rated_idxs : np.ndarray, y_train_rated : np.ndarray, 
                           base_idxs : np.ndarray, y_base : np.ndarray, cache_idxs : np.ndarray, y_cache : np.ndarray) -> tuple:
        X_train_rated, X_val = self.embedding.matrix[train_rated_idxs], self.embedding.matrix[val_rated_idxs]
        X_train, y_train = load_training_data_for_user(self.embedding, self.config["include_base"], self.config["include_cache"], 
                                                        train_rated_idxs, y_train_rated, base_idxs, y_base, cache_idxs, y_cache)
        return X_train_rated, X_val, X_train, y_train
    
    def get_data_statistics_for_user(self, y_train_rated : np.ndarray, base_n : int = 0, cache_n : int = 0) -> dict:
        user_data_statistics = {"base_n" : base_n, "cache_n" : cache_n}
        user_data_statistics["train_rated_n"] = len(y_train_rated)
        user_data_statistics["train_posrated_n"] = np.sum(y_train_rated)
        user_data_statistics["train_negrated_n"] = user_data_statistics["train_rated_n"] - user_data_statistics["train_posrated_n"]
        user_data_statistics["train_rated_base_n"] = user_data_statistics["train_rated_n"] + base_n
        user_data_statistics["total_n"] = user_data_statistics["train_rated_n"] + base_n + cache_n
        return user_data_statistics
    
    def train_model_for_user(self, X_train : np.ndarray, y_train : np.ndarray, X_train_rated : np.ndarray, y_train_rated : np.ndarray, 
                             X_val : np.ndarray, y_val : np.ndarray, user_data_statistics : dict, voting_weight : float = None) -> tuple:
        user_results, user_predictions, user_coefs = {}, {"train_predictions": {}, "val_predictions": {}, "tfidf_coefs": {}}, None
        sample_weights = np.empty(user_data_statistics["total_n"], dtype = np.float64)
        for combination_idx, hyperparameters_combination in enumerate(self.hyperparameters_combinations):
            w_p, w_n, w_b, w_c = self.wh.load_weights_for_user(self.hyperparameters, hyperparameters_combination, voting_weight, user_data_statistics["train_posrated_n"], 
                                                               user_data_statistics["train_negrated_n"], user_data_statistics["base_n"], user_data_statistics["cache_n"])
            sample_weights[:user_data_statistics["train_posrated_n"]] = w_p
            sample_weights[user_data_statistics["train_posrated_n"]:user_data_statistics["train_rated_n"]] = w_n
            sample_weights[user_data_statistics["train_rated_n"]:user_data_statistics["train_rated_base_n"]] = w_b
            sample_weights[user_data_statistics["train_rated_base_n"]:user_data_statistics["total_n"]] = w_c
            model = self.get_model_for_user(hyperparameters_combination)
            model.fit(X_train, y_train, sample_weight = sample_weights)
            y_train_rated_pred, y_val_pred = model.predict(X_train_rated), model.predict(X_val)
            y_train_rated_proba = model.predict_proba(X_train_rated)[:, 1].tolist()
            y_val_proba = model.predict_proba(X_val)[:, 1].tolist()
            scores = self.get_scores_for_user(y_train_rated, y_train_rated_pred, y_train_rated_proba, y_val, y_val_pred, y_val_proba)
            user_results[combination_idx] = scores
            if self.config["save_users_predictions"]:
                user_predictions["train_predictions"][combination_idx] = y_train_rated_proba
                user_predictions["val_predictions"][combination_idx] = y_val_proba
            if self.config["save_tfidf_coefs"]:
                user_predictions["tfidf_coefs"][combination_idx] = model.coef_[0].tolist()
            if "save_coefs" in self.config and self.config["save_coefs"]:
                user_coefs = np.hstack([model.coef_[0], model.intercept_[0]]) 
        return user_results, user_predictions, user_coefs

    def get_model_for_user(self, hyperparameters_combination : tuple) -> object:
        clf_C = hyperparameters_combination[self.hyperparameters["clf_C"]]
        return get_model(self.config["algorithm"], self.config["max_iter"], clf_C, self.random_state, 
                         logreg_solver = self.config["logreg_solver"] if self.config["algorithm"] == Algorithm.LOGREG else None,
                         svm_kernel = self.config["svm_kernel"] if self.config["algorithm"] == Algorithm.SVM else None)
    
    def get_scores_for_user(self, y_train_rated : np.ndarray, y_train_rated_pred : np.ndarray, y_train_rated_proba : np.ndarray,
                                  y_val : np.ndarray, y_val_pred : np.ndarray, y_val_proba : np.ndarray) -> tuple:
        user_scores = [None, ] * self.scores_n
        for score in Score:
            user_scores[self.scores[f"train_{score.name.lower()}"]] = get_score(score, y_train_rated, y_train_rated_pred, y_train_rated_proba)
            user_scores[self.scores[f"val_{score.name.lower()}"]] = get_score(score, y_val, y_val_pred, y_val_proba)
        return tuple(user_scores)
    
    def get_papers_ids(self, train_rated_idxs : np.ndarray, y_train_rated : np.ndarray, val_rated_idxs : np.ndarray, y_val : np.ndarray) -> dict:
        return {"train_ids" : self.embedding.get_papers_ids(train_rated_idxs), "train_labels" : y_train_rated.tolist(), 
                "val_ids" : self.embedding.get_papers_ids(val_rated_idxs), "val_labels" : y_val.tolist()}
    
    def save_user_info(self, user_id : int, user_info : dict) -> None:
        folder = f"{self.config['outputs_dir']}/tmp/user_{user_id}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        json.dump(user_info, open(folder + "/user_info.json", 'w'), indent = 1)

    def save_user_results(self, user_id : int, user_results_dict : dict) -> None:
        folder = f"{self.config['outputs_dir']}/tmp/user_{user_id}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        json.dump(user_results_dict, open(folder + "/user_results.json", 'w'), indent = 1)

    def save_users_predictions(self, user_id : int, user_predictions_dict : dict) -> None:
        folder = f"{self.config['outputs_dir']}/users_predictions/user_{user_id}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        json.dump(user_predictions_dict, open(folder + "/user_predictions.json", 'w'), indent = 1)

    def save_user_coefs(self, user_id : int, user_coefs : np.ndarray) -> None:
        folder = f"{self.config['outputs_dir']}/tmp/user_{user_id}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(folder + "/user_coefs.npy", user_coefs)