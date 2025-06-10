import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[3]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_src_paths_to_sys()

import json, os, pickle
import numpy as np
from joblib import Parallel, delayed

from algorithm import Algorithm, Evaluation, Score, get_cross_val, get_model, get_score, get_ranking_scores, derive_score, SCORES_DICT
from compute_tfidf import load_vectorizer
from embedding import Embedding
from load_files import load_papers
from training_data import *
from weights_handler import Weights_Handler

class Evaluator:
    def __init__(self, config: dict, hyperparameters_combinations: list, wh: Weights_Handler) -> None:
        self.config = config
        self.hyperparameters, self.hyperparameters_combinations = config["hyperparameters"], hyperparameters_combinations
        self.wh = wh
        self.scores, self.scores_n = config["scores"], len(config["scores"])
        self.non_derivable_scores, self.derivable_scores = [], []
        for score in Score:
            self.derivable_scores.append(score) if SCORES_DICT[score]["derivable"] else self.non_derivable_scores.append(score)
        self.include_global_cache = self.config["include_cache"] and self.config["cache_type"] == "global"
        if self.config["evaluation"] == Evaluation.CROSS_VALIDATION:
            self.cross_val = get_cross_val(config["stratified"], config["k_folds"], self.config["model_random_state"])
        
    def evaluate_embedding(self, embedding : Embedding, users_ratings: pd.DataFrame) -> None:
        self.embedding = embedding
        if self.config["save_tfidf_coefs"]:
            vectorizer = load_vectorizer(self.config["embedding_folder"])
            feature_names = vectorizer.get_feature_names_out()
            with open(f"{self.config['outputs_dir']}" / "feature_names.pkl", "wb") as f:
                pickle.dump(feature_names, f)

        if "save_users_coefs" in self.config and self.config["save_users_coefs"]:
            if self.config["evaluation"] not in [Evaluation.TRAIN_TEST_SPLIT, Evaluation.SESSION_BASED]:
                raise ValueError("Users coefficient saving is only supported with train-test split evaluation.")
        self.load_users_coefs = ("load_users_coefs" in self.config and self.config["load_users_coefs"])
        if self.load_users_coefs:
            if self.config["evaluation"] not in [Evaluation.TRAIN_TEST_SPLIT, Evaluation.SESSION_BASED]:
                raise ValueError("Users coefficient loading is only supported with train-test split evaluation.")
            if len(self.hyperparameters_combinations) > 1:
                raise ValueError("Users coefficient loading is not supported with multiple hyperparameter combinations.")
            users_coefs_path = Path(self.config["users_coefs_path"]).resolve()
            self.users_coefs = np.load(users_coefs_path / "users_coefs.npy")
            with open(users_coefs_path / "users_coefs_ids_to_idxs.pkl", "rb") as f:
                self.users_coefs_ids_to_idxs = pickle.load(f)

        papers = load_papers(ProjectPaths.data_db_backup_date_path() / "papers.parquet", relevant_columns = ["paper_id", "in_ratings", "in_cache", "l1", "l2"])
        if self.include_global_cache:
            self.global_cache_ids, self.global_cache_idxs, self.global_cache_n, self.y_global_cache = load_global_cache(
                        self.embedding, papers, self.config["max_cache"], self.config["cache_random_state"])
        else:
            if self.config["include_cache"]:
                self.cache_papers_ids = papers[papers["in_cache"]]["paper_id"].tolist()
                assert self.cache_papers_ids == sorted(self.cache_papers_ids)
        self.negative_samples_ids, self.negative_samples_embeddings = load_negative_samples_embeddings(
            self.embedding, papers, self.config["n_negative_samples"], self.config["ranking_random_state"], exclude_in_ratings = True, exclude_in_cache = True)
        self.cache_attached_ids = load_negative_samples_embeddings(
            self.embedding, papers, self.config["n_cache_attached"], self.config["cache_random_state"], papers_to_exclude = self.negative_samples_ids)[0]

        self.cache_attached_idxs = self.embedding.get_idxs(self.cache_attached_ids)
        assert len(self.cache_attached_ids) == len(self.cache_attached_idxs) == self.config["n_cache_attached"]
        users_ids = users_ratings["user_id"].unique().tolist()
        users_ratings = users_ratings.merge(papers[["paper_id", "l1", "l2"]], on = "paper_id", how = "left")
        if self.config["n_jobs"] == 1:
            for user_id in users_ids:
                user_ratings = users_ratings[users_ratings["user_id"] == user_id].reset_index(drop = True)
                self.evaluate_user(user_id, user_ratings)
        else:
            users_ratings_list = [(user_id, users_ratings[users_ratings["user_id"] == user_id].reset_index(drop = True)) for user_id in users_ids]
            Parallel(n_jobs = self.config["n_jobs"])(delayed(self.evaluate_user)(user_id, user_ratings) for user_id, user_ratings in users_ratings_list)

    def evaluate_user(self, user_id: int, user_ratings: pd.DataFrame) -> None:
        user_results_dict, user_predictions_dict = {}, {}
        assert user_ratings["time"].is_monotonic_increasing
        posrated_ids = user_ratings[user_ratings["rating"] == 1]["paper_id"].tolist()
        negrated_ids = user_ratings[user_ratings["rating"] == 0]["paper_id"].tolist()
        rated_ids = posrated_ids + negrated_ids
        posrated_n, negrated_n = len(posrated_ids), len(negrated_ids)
        rated_labels = np.concatenate((np.ones(posrated_n, dtype = LABEL_DTYPE), np.zeros(negrated_n, dtype = LABEL_DTYPE)))
        base_ids, base_idxs, base_n, y_base, user_predictions_dict["base_ids"] = [], [], 0, [], []
        zerorated_ids, zerorated_idxs, zerorated_n, y_zerorated, user_predictions_dict["zerorated_ids"] = [], [], 0, [], []
        cache_ids, cache_idxs, cache_n, y_cache = self.set_cache_for_user(rated_ids)
        user_info = self.store_user_info(user_id, posrated_n, negrated_n, base_n, zerorated_n, cache_n)
        user_predictions_dict["negative_samples_ids"] = self.negative_samples_ids
        if self.config["evaluation"] in [Evaluation.TRAIN_TEST_SPLIT, Evaluation.SESSION_BASED]:
            train_mask, val_mask = user_ratings["split"] == "train", user_ratings["split"] == "val"
            train_rated_ids, val_rated_ids = user_ratings.loc[train_mask, "paper_id"].values.tolist(), user_ratings.loc[val_mask, "paper_id"].values.tolist()
            y_train_rated, y_val = user_ratings.loc[train_mask, "rating"].values, user_ratings.loc[val_mask, "rating"].values
            val_rated_negative_ids = [id for id in val_rated_ids if id in negrated_ids]
            negrated_ranking_ids = load_negrated_ranking_ids_for_user(val_rated_negative_ids, self.config["ranking_random_state"])
            train_rated_idxs, val_rated_idxs = self.embedding.get_idxs(train_rated_ids), self.embedding.get_idxs(val_rated_ids)
            X_train_rated, X_val, X_train, y_train = self.load_data_for_user(train_rated_idxs, val_rated_idxs, y_train_rated, base_idxs, y_base, zerorated_idxs, y_zerorated, cache_idxs, y_cache)
            user_data_statistics = self.get_data_statistics_for_user(y_train_rated, base_n, zerorated_n, cache_n)
            user_predictions_dict[0] = self.get_papers_ids(train_rated_idxs, y_train_rated, val_rated_idxs, y_val, negrated_ranking_ids)
            negrated_ranking_val_idxs = np.array([user_predictions_dict[0]["val_ids"].index(id) for id in negrated_ranking_ids])
            user_results, user_predictions, user_coefs = self.train_model_for_user(user_id, X_train, y_train, X_train_rated, y_train_rated, X_val, y_val, user_data_statistics, negrated_ranking_val_idxs)
            user_results_dict[0] = user_results
            if self.config["save_users_predictions"]:
                 user_predictions_dict[0].update({"train_predictions": user_predictions["train_predictions"], "val_predictions": user_predictions["val_predictions"],
                                                  "negrated_ranking_predictions": user_predictions["negrated_ranking_predictions"], 
                                                  "negative_samples_predictions": user_predictions["negative_samples_predictions"], "tfidf_coefs": user_predictions["tfidf_coefs"]})

        elif self.config["evaluation"] == Evaluation.CROSS_VALIDATION:
            rated_idxs = self.embedding.get_idxs(rated_ids)
            split = self.cross_val.split(X = rated_ids, y = rated_labels)
            for fold_idx, (fold_train_idxs, fold_val_idxs) in enumerate(split):
                train_rated_idxs, val_rated_idxs = rated_idxs[fold_train_idxs], rated_idxs[fold_val_idxs]
                val_rated_negative_ids = [id for id in self.embedding.get_papers_ids(val_rated_idxs) if id in negrated_ids]
                negrated_ranking_ids = load_negrated_ranking_ids_for_user(val_rated_negative_ids, self.config["ranking_random_state"])
                y_train_rated, y_val = rated_labels[fold_train_idxs], rated_labels[fold_val_idxs]
                X_train_rated, X_val, X_train, y_train = self.load_data_for_user(train_rated_idxs, val_rated_idxs, y_train_rated, base_idxs, y_base, zerorated_idxs, y_zerorated, cache_idxs, y_cache)
                user_data_statistics = self.get_data_statistics_for_user(y_train_rated, base_n, zerorated_n, cache_n)
                user_predictions_dict[fold_idx] = self.get_papers_ids(train_rated_idxs, y_train_rated, val_rated_idxs, y_val, negrated_ranking_ids)
                negrated_ranking_val_idxs = np.array([user_predictions_dict[fold_idx]["val_ids"].index(id) for id in negrated_ranking_ids])
                fold_results, fold_predictions, _ = self.train_model_for_user(user_id, X_train, y_train, X_train_rated, y_train_rated, X_val, y_val, user_data_statistics, negrated_ranking_val_idxs)
                user_results_dict[fold_idx] = fold_results
                if self.config["save_users_predictions"]:
                    user_predictions_dict[fold_idx].update({"train_predictions": fold_predictions["train_predictions"], "val_predictions": fold_predictions["val_predictions"],
                                                            "negrated_ranking_predictions": fold_predictions["negrated_ranking_predictions"],
                                                            "negative_samples_predictions": fold_predictions["negative_samples_predictions"], "tfidf_coefs": fold_predictions["tfidf_coefs"]})
        self.save_user_info(user_id, user_info)
        self.save_user_results(user_id, user_results_dict)
        self.save_users_predictions(user_id, user_predictions_dict)
        if "save_users_coefs" in self.config and self.config["save_users_coefs"]:
            self.save_users_coefs(user_id, user_coefs)
        print(f"User {user_id} done.")
    
    def set_cache_for_user(self, rated_ids: list) -> tuple:
        if self.include_global_cache:
            cache_ids, cache_idxs, cache_n, y_cache = self.global_cache_ids, self.global_cache_idxs, self.global_cache_n, self.y_global_cache
        else:
            if self.config["include_cache"]:
                cache_ids, cache_idxs, cache_n, y_cache = load_filtered_cache_for_user(self.embedding, self.cache_papers_ids, rated_ids, self.config["max_cache"], 
                                                                                       self.config["cache_random_state"])
            else:
                cache_ids, cache_idxs, cache_n, y_cache = [], [], 0, []
        if self.include_global_cache or self.config["include_cache"]:
            cache_ids += self.cache_attached_ids
            cache_idxs = np.concatenate((cache_idxs, self.cache_attached_idxs))
            cache_n += len(self.cache_attached_idxs)
            y_cache = np.concatenate((y_cache, np.zeros(len(self.cache_attached_idxs), dtype = LABEL_DTYPE)))
        return cache_ids, cache_idxs, cache_n, y_cache
        
    def store_user_info(self, user_id: int, posrated_n: int, negrated_n: int, base_n: int = 0, zerorated_n: int = 0, cache_n: int = 0) -> dict:
        return {"n_posrated" : posrated_n, "n_negrated" : negrated_n, "n_base" : base_n , "n_zerorated": zerorated_n, "n_cache" : cache_n}
    
    def load_data_for_user(self, train_rated_idxs: np.ndarray, val_rated_idxs: np.ndarray, y_train_rated: np.ndarray, base_idxs: np.ndarray, y_base: np.ndarray, 
                           zerorated_idxs: np.ndarray, y_zerorated: np.ndarray, cache_idxs: np.ndarray, y_cache: np.ndarray) -> tuple:
        X_train_rated, X_val = self.embedding.matrix[train_rated_idxs], self.embedding.matrix[val_rated_idxs]
        X_train, y_train = load_training_data_for_user(self.embedding, self.config["include_base"], self.config["include_zerorated"], self.config["include_cache"],
                                                       train_rated_idxs, y_train_rated, base_idxs, y_base, zerorated_idxs, y_zerorated, cache_idxs, y_cache)
        return X_train_rated, X_val, X_train, y_train
    
    def get_data_statistics_for_user(self, y_train_rated: np.ndarray, base_n: int = 0, zerorated_n: int = 0, cache_n: int = 0) -> dict:
        user_data_statistics = {"base_n" : base_n, "zerorated_n": zerorated_n, "cache_n" : cache_n}
        user_data_statistics["train_rated_n"] = len(y_train_rated)
        user_data_statistics["train_posrated_n"] = np.sum(y_train_rated)
        user_data_statistics["train_negrated_n"] = user_data_statistics["train_rated_n"] - user_data_statistics["train_posrated_n"]
        user_data_statistics["train_rated_base_n"] = user_data_statistics["train_rated_n"] + base_n
        user_data_statistics["train_rated_base_zerorated_n"] = user_data_statistics["train_rated_base_n"] + zerorated_n
        user_data_statistics["total_n"] = user_data_statistics["train_rated_base_zerorated_n"] + cache_n
        return user_data_statistics
    
    def train_model_for_user(self, user_id: int, X_train: np.ndarray, y_train: np.ndarray, X_train_rated: np.ndarray, y_train_rated: np.ndarray, 
                             X_val: np.ndarray, y_val: np.ndarray, user_data_statistics: dict, negrated_ranking_val_idxs: np.ndarray, voting_weight: float = None) -> tuple:
        user_results, user_predictions, user_coefs = {}, {"train_predictions": {}, "val_predictions": {}, "negrated_ranking_predictions": {}, "negative_samples_predictions": {}, 
                                                          "tfidf_coefs": {}}, None
        sample_weights = np.empty(user_data_statistics["total_n"], dtype = np.float64)
        for combination_idx, hyperparameters_combination in enumerate(self.hyperparameters_combinations):
            model = self.get_model_for_user(hyperparameters_combination)
            if self.load_users_coefs:
                user_coefs = self.users_coefs[self.users_coefs_ids_to_idxs[user_id], :]
                model.n_features_in = user_coefs.shape[0]
                model.coef_ = user_coefs[:-1].reshape(1, -1)
                model.intercept_ = user_coefs[-1].reshape(1, -1)
                model.classes_ = np.array([0, 1])
            else:
                w_p, w_n, w_b, w_z, w_c = self.wh.load_weights_for_user(self.hyperparameters, hyperparameters_combination, voting_weight, user_data_statistics["train_posrated_n"], 
                                                                user_data_statistics["train_negrated_n"], user_data_statistics["base_n"], user_data_statistics["zerorated_n"],
                                                                user_data_statistics["cache_n"])
                sample_weights[:user_data_statistics["train_rated_n"]][y_train_rated == 1] = w_p
                sample_weights[:user_data_statistics["train_rated_n"]][y_train_rated == 0] = w_n
                sample_weights[user_data_statistics["train_rated_n"]:user_data_statistics["train_rated_base_n"]] = w_b
                sample_weights[user_data_statistics["train_rated_base_n"]:user_data_statistics["train_rated_base_zerorated_n"]] = w_z
                sample_weights[user_data_statistics["train_rated_base_zerorated_n"]:user_data_statistics["total_n"]] = w_c
                model.fit(X_train, y_train, sample_weight = sample_weights)
            y_train_rated_pred, y_val_pred, y_negative_samples_pred = model.predict(X_train_rated), model.predict(X_val), model.predict(self.negative_samples_embeddings)
            y_train_rated_proba, y_train_rated_logits = model.predict_proba(X_train_rated)[:, 1], model.decision_function(X_train_rated)
            y_val_proba, y_val_logits = model.predict_proba(X_val)[:, 1], model.decision_function(X_val)
            y_negrated_ranking_proba, y_negrated_ranking_logits = y_val_proba[negrated_ranking_val_idxs], y_val_logits[negrated_ranking_val_idxs]
            y_negative_samples_proba, y_negative_samples_logits = model.predict_proba(self.negative_samples_embeddings)[:, 1], model.decision_function(self.negative_samples_embeddings)
            scores = self.get_scores_for_user(y_train_rated, y_train_rated_pred, y_train_rated_proba, y_train_rated_logits, y_val, y_val_pred, y_val_proba, y_val_logits,
                                              y_negrated_ranking_proba, y_negrated_ranking_logits, y_negative_samples_pred, y_negative_samples_proba, y_negative_samples_logits)
            user_results[combination_idx] = scores
            if self.config["save_users_predictions"]:
                user_predictions["train_predictions"][combination_idx] = y_train_rated_proba.tolist()
                user_predictions["val_predictions"][combination_idx] = y_val_proba.tolist()
                user_predictions["negrated_ranking_predictions"][combination_idx] = y_negrated_ranking_proba.tolist()
                user_predictions["negative_samples_predictions"][combination_idx] = y_negative_samples_proba.tolist()
            if self.config["save_tfidf_coefs"]:
                user_predictions["tfidf_coefs"][combination_idx] = model.coef_[0].tolist()
            if "save_users_coefs" in self.config and self.config["save_users_coefs"]:
                user_coefs = np.hstack([model.coef_[0], model.intercept_[0]]) 
        return user_results, user_predictions, user_coefs

    def get_model_for_user(self, hyperparameters_combination: tuple) -> object:
        clf_C = hyperparameters_combination[self.hyperparameters["clf_C"]]
        return get_model(self.config["algorithm"], self.config["max_iter"], clf_C, self.config["model_random_state"], 
                         logreg_solver = self.config["logreg_solver"] if self.config["algorithm"] == Algorithm.LOGREG else None,
                         svm_kernel = self.config["svm_kernel"] if self.config["algorithm"] == Algorithm.SVM else None)
    
    def get_scores_for_user(self, y_train_rated: np.ndarray, y_train_rated_pred: np.ndarray, y_train_rated_proba: np.ndarray, y_train_rated_logits: np.ndarray,
                            y_val: np.ndarray, y_val_pred: np.ndarray, y_val_proba: np.ndarray, y_val_logits: np.ndarray, y_negrated_ranking_proba: np.ndarray, 
                            y_negrated_ranking_logits: np.ndarray, y_negative_samples_pred: np.ndarray, y_negative_samples_proba: np.ndarray, y_negative_samples_logits: np.ndarray) -> tuple:
        user_scores = [0] * self.scores_n
        for score in self.non_derivable_scores:
            if not SCORES_DICT[score]["ranking"]:
                user_scores[self.scores[f"train_{score.name.lower()}"]] = get_score(score, y_train_rated, y_train_rated_pred, y_train_rated_proba, 
                                                                                    y_negative_samples_pred, y_negative_samples_proba)
                user_scores[self.scores[f"val_{score.name.lower()}"]] = get_score(score, y_val, y_val_pred, y_val_proba, 
                                                                                  y_negative_samples_pred, y_negative_samples_proba)
        ranking_scores = get_ranking_scores(y_train_rated, y_train_rated_logits, y_val, y_val_logits, y_negrated_ranking_logits, y_negative_samples_logits, self.config["info_nce_temperature"])
        for ranking_score in ranking_scores:
            user_scores[self.scores[ranking_score]] = ranking_scores[ranking_score]
        user_scores_copy = user_scores.copy()
        for score in self.derivable_scores:
            user_scores[self.scores[f"train_{score.name.lower()}"]] = derive_score(score, user_scores_copy, self.scores, validation = False)
            user_scores[self.scores[f"val_{score.name.lower()}"]] = derive_score(score, user_scores_copy, self.scores, validation = True)
        return tuple(user_scores)
    
    def get_papers_ids(self, train_rated_idxs: np.ndarray, y_train_rated: np.ndarray, val_rated_idxs: np.ndarray, y_val: np.ndarray, negrated_ranking_ids: list) -> dict:
        return {"train_ids": self.embedding.get_papers_ids(train_rated_idxs), "train_labels" : y_train_rated.tolist(), 
                "val_ids": self.embedding.get_papers_ids(val_rated_idxs), "val_labels": y_val.tolist(), "negrated_ranking_ids": negrated_ranking_ids}
    
    def save_user_info(self, user_id : int, user_info : dict) -> None:
        folder = self.config["outputs_dir"] / "tmp" / f"user_{user_id}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        json.dump(user_info, open(folder / "user_info.json", 'w'), indent = 1)

    def save_user_results(self, user_id: int, user_results_dict: dict) -> None:
        folder = self.config["outputs_dir"] / "tmp" / f"user_{user_id}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        try:
            json.dump(user_results_dict, open(folder / "user_results.json", 'w'), indent = 1)
        except:
            print(f"Error saving user {user_id} results.")
            raise

    def save_users_predictions(self, user_id: int, user_predictions_dict: dict) -> None:
        folder = self.config["outputs_dir"] / "users_predictions" / f"user_{user_id}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        json.dump(user_predictions_dict, open(folder / "user_predictions.json", 'w'), indent = 1)

    def save_users_coefs(self, user_id: int, user_coefs: np.ndarray) -> None:
        folder = self.config["outputs_dir"] / "tmp" / f"user_{user_id}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(folder / "user_coefs.npy", user_coefs)
