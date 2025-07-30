import json
import os
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ....src.load_files import load_papers, load_users_significant_categories
from ..embeddings.compute_tfidf import load_vectorizer
from ..embeddings.embedding import Embedding
from .algorithm import (
    SCORES_DICT,
    Algorithm,
    Evaluation,
    Score,
    derive_score,
    get_category_scores,
    get_cross_val,
    get_model,
    get_ranking_scores,
    get_score,
)
from .training_data import (
    LABEL_DTYPE,
    get_val_cache_attached_negative_samples_ids,
    load_filtered_cache_for_user,
    load_global_cache,
    load_negrated_ranking_idxs_for_user,
)
from .weights_handler import Weights_Handler


class Evaluator:
    def __init__(
        self, config: dict, hyperparameters_combinations: list, wh: Weights_Handler
    ) -> None:
        self.config = config
        self.hyperparameters, self.hyperparameters_combinations = (
            config["hyperparameters"],
            hyperparameters_combinations,
        )
        self.wh = wh
        self.scores, self.scores_n = config["scores"], len(config["scores"])
        self.non_derivable_scores, self.derivable_scores = [], []
        for score in Score:
            (
                self.derivable_scores.append(score)
                if SCORES_DICT[score]["derivable"]
                else self.non_derivable_scores.append(score)
            )
        self.include_global_cache = (
            self.config["include_cache"] and self.config["cache_type"] == "global"
        )
        if self.config["evaluation"] == Evaluation.CROSS_VALIDATION:
            self.cross_val = get_cross_val(
                config["stratified"], config["k_folds"], self.config["model_random_state"]
            )

    def evaluate_embedding(self, embedding: Embedding, users_ratings: pd.DataFrame) -> None:
        self.embedding = embedding
        if self.config["save_tfidf_coefs"]:
            vectorizer = load_vectorizer(self.config["embedding_folder"])
            feature_names = vectorizer.get_feature_names_out()
            with open(f"{self.config['outputs_dir']}" / "feature_names.pkl", "wb") as f:
                pickle.dump(feature_names, f)

        if "save_users_coefs" in self.config and self.config["save_users_coefs"]:
            if self.config["evaluation"] not in [
                Evaluation.TRAIN_TEST_SPLIT,
                Evaluation.SESSION_BASED,
            ]:
                raise ValueError(
                    "Users coefficient saving is only supported with train-test split evaluation."
                )
        self.load_users_coefs = (
            "load_users_coefs" in self.config and self.config["load_users_coefs"]
        )
        if self.load_users_coefs:
            if self.config["evaluation"] not in [
                Evaluation.TRAIN_TEST_SPLIT,
                Evaluation.SESSION_BASED,
            ]:
                raise ValueError(
                    "Users coefficient loading is only supported with train-test split evaluation."
                )
            if len(self.hyperparameters_combinations) > 1:
                raise ValueError(
                    "Users coefficient loading is not supported with multiple hyperparameter combinations."
                )
            users_coefs_path = Path(self.config["users_coefs_path"]).resolve()
            self.users_coefs = np.load(users_coefs_path / "users_coefs.npy")
            with open(users_coefs_path / "users_coefs_ids_to_idxs.pkl", "rb") as f:
                self.users_coefs_ids_to_idxs = pickle.load(f)

        papers = load_papers(relevant_columns=["paper_id", "in_ratings", "in_cache", "l1", "l2"])
        if self.include_global_cache:
            (
                self.global_cache_ids,
                self.global_cache_idxs,
                self.global_cache_n,
                self.y_global_cache,
            ) = load_global_cache(
                self.embedding, papers, self.config["max_cache"], self.config["cache_random_state"]
            )
        else:
            if self.config["include_cache"]:
                self.cache_papers_ids = papers[papers["in_cache"]]["paper_id"].tolist()
                assert self.cache_papers_ids == sorted(self.cache_papers_ids)

        users_ids = users_ratings["user_id"].unique().tolist()
        users_ratings = users_ratings.merge(
            papers[["paper_id", "l1", "l2"]], on="paper_id", how="left"
        )
        users_significant_categories = load_users_significant_categories(
            relevant_users_ids=users_ids,
        )
        
        val_negative_samples_ids, cache_attached_papers_ids, _ = (
            get_val_cache_attached_negative_samples_ids(
                users_ids=users_ids,
                papers=papers,
                n_val_negative_samples=self.config["n_negative_samples"],
                ranking_random_state=self.config["ranking_random_state"],
                n_cache_attached=self.config["n_cache_attached"],
                cache_random_state=self.config["cache_random_state"],
                cache_attached_user_specific=True,
                return_all_papers_ids=False,
            )
        )

        if self.config["n_jobs"] == 1:
            for i, user_id in enumerate(users_ids):
                user_ratings = users_ratings[users_ratings["user_id"] == user_id].reset_index(
                    drop=True
                )
                user_significant_categories = users_significant_categories[
                    users_significant_categories["user_id"] == user_id
                ]["category"].tolist()
                user_val_negative_samples_ids = val_negative_samples_ids[i]
                user_cache_attached_papers_ids = None
                if cache_attached_papers_ids is not None:
                    user_cache_attached_papers_ids = cache_attached_papers_ids[i]
                self.evaluate_user(
                    user_id,
                    user_ratings,
                    user_significant_categories,
                    user_val_negative_samples_ids,
                    user_cache_attached_papers_ids,
                )
        else:
            users_list = []
            for i, user_id in enumerate(users_ids):
                users_list.append(
                    (
                        user_id,
                        users_ratings[users_ratings["user_id"] == user_id].reset_index(drop=True),
                        users_significant_categories[users_significant_categories["user_id"] == user_id][
                            "category"
                        ].tolist(),
                        val_negative_samples_ids[i],
                        (
                            cache_attached_papers_ids[i]
                            if cache_attached_papers_ids is not None
                            else None
                        ),
                    )
                )
            Parallel(n_jobs=self.config["n_jobs"])(
                delayed(self.evaluate_user)(
                    user_id,
                    user_ratings,
                    user_significant_categories,
                    user_val_negative_samples_ids,
                    user_cache_attached_papers_ids,
                )
                for user_id, user_ratings, user_significant_categories, user_val_negative_samples_ids, user_cache_attached_papers_ids in users_list
            )

    def evaluate_user(
        self,
        user_id: int,
        user_ratings: pd.DataFrame,
        user_significant_categories: list,
        user_val_negative_samples_ids: np.ndarray = None,
        user_cache_attached_papers_ids: np.ndarray = None,
    ) -> None:
        user_val_negative_samples_ids = user_val_negative_samples_ids.tolist()
        negative_samples_embeddings = self.embedding.matrix[
            self.embedding.get_idxs(user_val_negative_samples_ids)
        ]
        user_results_dict, user_predictions_dict = {}, {}
        posrated_n, negrated_n = len(user_ratings[user_ratings["rating"] == 1]), len(
            user_ratings[user_ratings["rating"] == 0]
        )
        assert posrated_n + negrated_n == len(user_ratings)
        _, cache_embedding_idxs, cache_n, y_cache = self.set_cache_for_user(
            user_ratings["paper_id"].tolist(), user_cache_attached_papers_ids
        )
        user_info = self.store_user_info(posrated_n, negrated_n, cache_n)
        user_predictions_dict["negative_samples_ids"] = user_val_negative_samples_ids
        try:
            if self.config["evaluation"] in [
                Evaluation.TRAIN_TEST_SPLIT,
                Evaluation.SESSION_BASED,
            ]:
                train_ratings, val_ratings = split_ratings(user_ratings)
                user_info["train_rated_ratio"] = len(train_ratings) / len(user_ratings)
                X_train, y_train, pos_idxs, neg_idxs, cache_idxs = self.load_train_data_for_user(
                    train_ratings, cache_embedding_idxs, y_cache, self.config["model_random_state"]
                )
                (
                    X_train_rated,
                    y_train_rated,
                    X_train_rated_papers_ids,
                    X_val,
                    y_val,
                    X_val_papers_ids,
                    categories_dict,
                ) = self.load_val_data_for_user(train_ratings, val_ratings)
                user_predictions_dict[0] = {
                    "train_ids": X_train_rated_papers_ids,
                    "train_labels": y_train_rated.tolist(),
                    "val_ids": X_val_papers_ids,
                    "val_labels": y_val.tolist(),
                }
                train_negrated_ranking_idxs, val_negrated_ranking_idxs = (
                    load_negrated_ranking_idxs_for_user(
                        train_ratings=train_ratings,
                        val_ratings=val_ratings,
                        random_neg=(self.config["evaluation"] != Evaluation.SESSION_BASED),
                        random_state=self.config["ranking_random_state"],
                        same_negrated_for_all_pos=self.config["same_negrated_for_all_pos"],
                    )
                )
                user_results, user_predictions, user_coefs = self.train_model_for_user(
                    user_id,
                    X_train,
                    y_train,
                    X_train_rated,
                    y_train_rated,
                    X_val,
                    y_val,
                    negative_samples_embeddings,
                    categories_dict,
                    train_negrated_ranking_idxs,
                    val_negrated_ranking_idxs,
                    pos_idxs,
                    neg_idxs,
                    cache_idxs,
                )
                user_results_dict[0] = user_results
                if self.config["save_users_predictions"]:
                    user_predictions_dict[0].update(
                        {
                            "train_predictions": user_predictions["train_predictions"],
                            "val_predictions": user_predictions["val_predictions"],
                            "negative_samples_predictions": user_predictions[
                                "negative_samples_predictions"
                            ],
                            "tfidf_coefs": user_predictions["tfidf_coefs"],
                        }
                    )

            elif self.config["evaluation"] == Evaluation.CROSS_VALIDATION:
                user_ratings["split"] = None
                split = self.cross_val.split(X=range(len(user_ratings)), y=user_ratings["rating"])
                train_rated_ratios = []
                for fold_idx, (fold_train_idxs, fold_val_idxs) in enumerate(split):
                    (
                        user_ratings.loc[fold_train_idxs, "split"],
                        user_ratings.loc[fold_val_idxs, "split"],
                    ) = ("train", "val")
                    train_ratings, val_ratings = split_ratings(user_ratings)
                    train_rated_ratios.append(len(train_ratings) / len(user_ratings))
                    X_train, y_train, pos_idxs, neg_idxs, cache_idxs = (
                        self.load_train_data_for_user(
                            train_ratings,
                            cache_embedding_idxs,
                            y_cache,
                            self.config["model_random_state"],
                        )
                    )
                    (
                        X_train_rated,
                        y_train_rated,
                        X_train_rated_papers_ids,
                        X_val,
                        y_val,
                        X_val_papers_ids,
                        categories_dict,
                    ) = self.load_val_data_for_user(train_ratings, val_ratings)
                    user_predictions_dict[fold_idx] = {
                        "train_ids": X_train_rated_papers_ids,
                        "train_labels": y_train_rated.tolist(),
                        "val_ids": X_val_papers_ids,
                        "val_labels": y_val.tolist(),
                    }
                    train_negrated_ranking_idxs, val_negrated_ranking_idxs = (
                        load_negrated_ranking_idxs_for_user(
                            train_ratings=train_ratings,
                            val_ratings=val_ratings,
                            random_neg=True,
                            random_state=self.config["ranking_random_state"],
                            same_negrated_for_all_pos=self.config["same_negrated_for_all_pos"],
                        )
                    )
                    fold_results, fold_predictions, _ = self.train_model_for_user(
                        user_id,
                        X_train,
                        y_train,
                        X_train_rated,
                        y_train_rated,
                        X_val,
                        y_val,
                        negative_samples_embeddings,
                        categories_dict,
                        train_negrated_ranking_idxs,
                        val_negrated_ranking_idxs,
                        pos_idxs,
                        neg_idxs,
                        cache_idxs,
                    )
                    user_results_dict[fold_idx] = fold_results
                    if self.config["save_users_predictions"]:
                        user_predictions_dict[fold_idx].update(
                            {
                                "train_predictions": fold_predictions["train_predictions"],
                                "val_predictions": fold_predictions["val_predictions"],
                                "negative_samples_predictions": fold_predictions[
                                    "negative_samples_predictions"
                                ],
                                "tfidf_coefs": fold_predictions["tfidf_coefs"],
                            }
                        )
                user_info["train_rated_ratio"] = np.mean(train_rated_ratios)
            self.save_user_info(user_id, user_info)
            self.save_user_results(user_id, user_results_dict)
            self.save_user_predictions(user_id, user_predictions_dict)
            if "save_users_coefs" in self.config and self.config["save_users_coefs"]:
                self.save_user_coefs(user_id, user_coefs)
            print(f"User {user_id} done.")
        except Exception as e:
            print(f"Error evaluating user {user_id}: {e}")
            traceback.print_exc()

    def set_cache_for_user(self, rated_ids: list, cache_attached_papers_ids: np.ndarray) -> tuple:
        cache_attached_papers_ids = (
            [] if cache_attached_papers_ids is None else cache_attached_papers_ids.tolist()
        )
        cache_attached_papers_idxs = self.embedding.get_idxs(cache_attached_papers_ids)
        if self.include_global_cache:
            cache_ids, cache_idxs, cache_n, y_cache = (
                self.global_cache_ids,
                self.global_cache_idxs,
                self.global_cache_n,
                self.y_global_cache,
            )
        else:
            if self.config["include_cache"]:
                cache_ids, cache_idxs, cache_n, y_cache = load_filtered_cache_for_user(
                    self.embedding,
                    self.cache_papers_ids,
                    rated_ids,
                    self.config["max_cache"],
                    self.config["cache_random_state"],
                )
            else:
                cache_ids, cache_idxs, cache_n, y_cache = [], [], 0, []
        if self.include_global_cache or self.config["include_cache"]:
            cache_ids += cache_attached_papers_ids
            cache_idxs = np.concatenate((cache_idxs, cache_attached_papers_idxs))
            cache_n += len(cache_attached_papers_ids)
            y_cache = np.concatenate(
                (y_cache, np.zeros(len(cache_attached_papers_ids), dtype=LABEL_DTYPE))
            )
        return cache_ids, cache_idxs, cache_n, y_cache

    def store_user_info(
        self,
        posrated_n: int,
        negrated_n: int,
        cache_n: int = 0,
        base_n: int = 0,
        zerorated_n: int = 0,
    ) -> dict:
        return {
            "n_posrated": posrated_n,
            "n_negrated": negrated_n,
            "n_cache": cache_n,
            "n_base": base_n,
            "n_zerorated": zerorated_n,
        }

    def train_model_for_user(
        self,
        user_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_train_rated: np.ndarray,
        y_train_rated: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        negative_samples_embeddings: np.ndarray,
        categories_dict: dict,
        train_negrated_ranking_idxs: np.ndarray,
        val_negrated_ranking_idxs: np.ndarray,
        pos_idxs: np.ndarray,
        neg_idxs: np.ndarray,
        cache_idxs: np.ndarray,
        voting_weight: float = None,
    ) -> tuple:
        user_results, user_predictions, user_coefs = (
            {},
            {
                "train_predictions": {},
                "val_predictions": {},
                "negrated_ranking_predictions": {},
                "negative_samples_predictions": {},
                "tfidf_coefs": {},
            },
            None,
        )
        sample_weights = np.empty(X_train.shape[0], dtype=np.float64)
        train_posrated_n, train_negrated_n, cache_n = len(pos_idxs), len(neg_idxs), len(cache_idxs)
        for combination_idx, hyperparameters_combination in enumerate(
            self.hyperparameters_combinations
        ):
            model = self.get_model_for_user(hyperparameters_combination)
            if self.load_users_coefs:
                user_coefs = self.users_coefs[self.users_coefs_ids_to_idxs[user_id], :]
                model.n_features_in = user_coefs.shape[0]
                model.coef_ = user_coefs[:-1].reshape(1, -1)
                model.intercept_ = user_coefs[-1].reshape(1, -1)
                model.classes_ = np.array([0, 1])
            else:
                w_p, w_n, _, _, w_c = self.wh.load_weights_for_user(
                    self.hyperparameters,
                    hyperparameters_combination,
                    voting_weight,
                    train_posrated_n,
                    train_negrated_n,
                    0,
                    0,
                    cache_n,
                )
                sample_weights[pos_idxs], sample_weights[neg_idxs], sample_weights[cache_idxs] = (
                    w_p,
                    w_n,
                    w_c,
                )
                model.fit(X_train, y_train, sample_weight=sample_weights)
            user_coefs = np.hstack([model.coef_[0], model.intercept_[0]])
            if len(y_val) > 0:
                y_train_rated_pred, y_val_pred, y_negative_samples_pred = (
                    model.predict(X_train_rated),
                    model.predict(X_val),
                    model.predict(negative_samples_embeddings),
                )
                y_train_rated_proba, y_train_rated_logits = model.predict_proba(X_train_rated)[
                    :, 1
                ], model.decision_function(X_train_rated)
                y_val_proba, y_val_logits = model.predict_proba(X_val)[
                    :, 1
                ], model.decision_function(X_val)
                y_train_negrated_ranking_logits = y_train_rated_logits[train_negrated_ranking_idxs]
                y_val_negrated_ranking_logits = y_val_logits[val_negrated_ranking_idxs]
                y_negative_samples_proba, y_negative_samples_logits = model.predict_proba(
                    negative_samples_embeddings
                )[:, 1], model.decision_function(negative_samples_embeddings)
                scores = self.get_scores_for_user(
                    y_train_rated,
                    y_train_rated_pred,
                    y_train_rated_proba,
                    y_train_rated_logits,
                    y_val,
                    y_val_pred,
                    y_val_proba,
                    y_val_logits,
                    y_train_negrated_ranking_logits,
                    y_val_negrated_ranking_logits,
                    y_negative_samples_pred,
                    y_negative_samples_proba,
                    y_negative_samples_logits,
                    categories_dict,
                )
                user_results[combination_idx] = scores
                if self.config["save_users_predictions"]:
                    user_predictions["train_predictions"][
                        combination_idx
                    ] = y_train_rated_proba.tolist()
                    user_predictions["val_predictions"][combination_idx] = y_val_proba.tolist()
                    user_predictions["negative_samples_predictions"][
                        combination_idx
                    ] = y_negative_samples_proba.tolist()
                if self.config["save_tfidf_coefs"]:
                    user_predictions["tfidf_coefs"][combination_idx] = model.coef_[0].tolist()
        return user_results, user_predictions, user_coefs

    def get_model_for_user(self, hyperparameters_combination: tuple) -> object:
        clf_C = hyperparameters_combination[self.hyperparameters["clf_C"]]
        return get_model(
            self.config["algorithm"],
            self.config["max_iter"],
            clf_C,
            self.config["model_random_state"],
            logreg_solver=(
                self.config["logreg_solver"]
                if self.config["algorithm"] == Algorithm.LOGREG
                else None
            ),
            svm_kernel=(
                self.config["svm_kernel"] if self.config["algorithm"] == Algorithm.SVM else None
            ),
        )

    def get_scores_for_user(
        self,
        y_train_rated: np.ndarray,
        y_train_rated_pred: np.ndarray,
        y_train_rated_proba: np.ndarray,
        y_train_rated_logits: np.ndarray,
        y_val: np.ndarray,
        y_val_pred: np.ndarray,
        y_val_proba: np.ndarray,
        y_val_logits: np.ndarray,
        y_train_negrated_ranking_logits: np.ndarray,
        y_val_negrated_ranking_logits: np.ndarray,
        y_negative_samples_pred: np.ndarray,
        y_negative_samples_proba: np.ndarray,
        y_negative_samples_logits: np.ndarray,
        categories_dict: dict,
    ) -> tuple:
        user_scores = [0] * self.scores_n
        for score in self.non_derivable_scores:
            if not SCORES_DICT[score]["ranking"]:
                user_scores[self.scores[f"train_{score.name.lower()}"]] = get_score(
                    score,
                    y_train_rated,
                    y_train_rated_pred,
                    y_train_rated_proba,
                    y_negative_samples_pred,
                    y_negative_samples_proba,
                )
                user_scores[self.scores[f"val_{score.name.lower()}"]] = get_score(
                    score,
                    y_val,
                    y_val_pred,
                    y_val_proba,
                    y_negative_samples_pred,
                    y_negative_samples_proba,
                )
        ranking_scores = get_ranking_scores(
            y_train_rated,
            y_train_rated_logits,
            y_val,
            y_val_logits,
            y_train_negrated_ranking_logits,
            y_val_negrated_ranking_logits,
            y_negative_samples_logits,
        )
        for ranking_score in ranking_scores:
            user_scores[self.scores[ranking_score]] = ranking_scores[ranking_score]
        category_scores = get_category_scores(y_train_rated, y_val, categories_dict)
        for category_score in category_scores:
            user_scores[self.scores[category_score]] = category_scores[category_score]
        user_scores_copy = user_scores.copy()
        for score in self.derivable_scores:
            user_scores[self.scores[f"train_{score.name.lower()}"]] = derive_score(
                score, user_scores_copy, self.scores, validation=False
            )
            user_scores[self.scores[f"val_{score.name.lower()}"]] = derive_score(
                score, user_scores_copy, self.scores, validation=True
            )
        return tuple(user_scores)

    def get_papers_ids(
        self,
        train_rated_idxs: np.ndarray,
        y_train_rated: np.ndarray,
        val_rated_idxs: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        return {
            "train_ids": self.embedding.get_papers_ids(train_rated_idxs),
            "train_labels": y_train_rated.tolist(),
            "val_ids": self.embedding.get_papers_ids(val_rated_idxs),
            "val_labels": y_val.tolist(),
        }

    def load_train_data_for_user(
        self,
        train_ratings: pd.DataFrame,
        cache_embedding_idxs: np.ndarray,
        y_cache: np.ndarray,
        random_state: int,
    ) -> tuple:
        LABEL_NEGRATED, LABEL_POSRATED, LABEL_CACHE = 0, 1, 2
        rng = np.random.default_rng(random_state)
        train_ratings_ids, y_train_ratings = (
            train_ratings["paper_id"].tolist(),
            train_ratings["rating"].values,
        )
        train_rated_embedding_idxs = self.embedding.get_idxs(train_ratings_ids)
        train_source_labels = np.where(y_train_ratings > 0, LABEL_POSRATED, LABEL_NEGRATED)
        cache_source_labels = np.full(len(cache_embedding_idxs), LABEL_CACHE)

        embedding_idxs_full = np.concatenate((train_rated_embedding_idxs, cache_embedding_idxs))
        y_full = np.concatenate((y_train_ratings, y_cache))
        source_full = np.concatenate((train_source_labels, cache_source_labels))
        permuted_idxs = rng.permutation(len(embedding_idxs_full))
        embedding_idxs_full, y_full, source_full = (
            embedding_idxs_full[permuted_idxs],
            y_full[permuted_idxs],
            source_full[permuted_idxs],
        )
        pos_idxs = np.where(source_full == LABEL_POSRATED)[0]
        neg_idxs = np.where(source_full == LABEL_NEGRATED)[0]
        cache_idxs = np.where(source_full == LABEL_CACHE)[0]
        X_train = self.embedding.matrix[embedding_idxs_full]
        return X_train, y_full, pos_idxs, neg_idxs, cache_idxs

    def load_val_data_for_user(
        self, train_ratings: pd.DataFrame, val_ratings: pd.DataFrame
    ) -> tuple:
        X_train_rated_papers_ids, X_val_papers_ids = (
            train_ratings["paper_id"].tolist(),
            val_ratings["paper_id"].tolist(),
        )
        X_train_rated = self.embedding.matrix[self.embedding.get_idxs(X_train_rated_papers_ids)]
        y_train_rated = train_ratings["rating"].values
        X_val = self.embedding.matrix[self.embedding.get_idxs(X_val_papers_ids)]
        y_val = val_ratings["rating"].values
        categories_dict = {
            "l1_train_rated": train_ratings["l1"].values,
            "l1_val": val_ratings["l1"].values,
            "l2_train_rated": train_ratings["l2"].values,
            "l2_val": val_ratings["l2"].values,
        }
        return (
            X_train_rated,
            y_train_rated,
            X_train_rated_papers_ids,
            X_val,
            y_val,
            X_val_papers_ids,
            categories_dict,
        )

    def save_user_info(self, user_id: int, user_info: dict) -> None:
        folder = self.config["outputs_dir"] / "tmp" / f"user_{user_id}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        json.dump(user_info, open(folder / "user_info.json", "w"), indent=1)

    def save_user_results(self, user_id: int, user_results_dict: dict) -> None:
        folder = self.config["outputs_dir"] / "tmp" / f"user_{user_id}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        try:
            json.dump(user_results_dict, open(folder / "user_results.json", "w"), indent=1)
        except:
            print(f"Error saving user {user_id} results.")
            raise

    def save_user_predictions(self, user_id: int, user_predictions_dict: dict) -> None:
        folder = self.config["outputs_dir"] / "users_predictions" / f"user_{user_id}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        json.dump(user_predictions_dict, open(folder / "user_predictions.json", "w"), indent=1)

    def save_user_coefs(self, user_id: int, user_coefs: np.ndarray) -> None:
        folder = self.config["outputs_dir"] / "tmp" / f"user_{user_id}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(folder / "user_coefs.npy", user_coefs)


def split_ratings(user_ratings: pd.DataFrame) -> tuple:
    if "split" not in user_ratings.columns:
        raise ValueError("User ratings DataFrame must contain 'split' column.")
    train_mask = user_ratings["split"] == "train"
    val_mask = user_ratings["split"] == "val"
    train_ratings = user_ratings.loc[train_mask].reset_index(drop=True)
    val_ratings = user_ratings.loc[val_mask].reset_index(drop=True)
    assert len(train_ratings) + len(val_ratings) == len(user_ratings)
    assert (
        train_ratings["time"].is_monotonic_increasing
        and val_ratings["time"].is_monotonic_increasing
    )
    return train_ratings, val_ratings


def index_negative_samples_ids(
    negative_samples_ids: np.ndarray, embedding: Embedding
) -> np.ndarray:
    negative_samples_ids_shape = negative_samples_ids.shape
    negative_samples_idxs = embedding.get_idxs(negative_samples_ids.flatten())
    negative_samples_idxs = negative_samples_idxs.reshape(negative_samples_ids_shape)
    if negative_samples_idxs.shape != negative_samples_ids_shape:
        raise ValueError(
            f"Negative samples IDs shape {negative_samples_ids_shape} does not match indexed shape {negative_samples_idxs.shape}."
        )
    return negative_samples_idxs
