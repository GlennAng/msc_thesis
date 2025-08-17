import json
import os
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ....src.load_files import load_papers, load_users_significant_categories
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
from .get_users_ratings import N_NEGRATED_RANKING
from .training_data import (
    LABEL_DTYPE,
    get_cache_papers_ids,
    get_categories_samples_ids,
    get_user_cache_papers_ids,
    get_user_categories_ratios,
    get_user_categories_samples_ids,
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
        if self.config["evaluation"] == Evaluation.CROSS_VALIDATION:
            self.cross_val = get_cross_val(
                config["stratified"], config["k_folds"], self.config["model_random_state"]
            )

    def evaluate_embedding(
        self,
        embedding: Embedding,
        users_ratings: pd.DataFrame,
        users_embeddings: dict = None,
    ) -> None:
        self.embedding = embedding
        users_ids = users_ratings["user_id"].unique().tolist()
        users_significant_categories = load_users_significant_categories(
            relevant_users_ids=users_ids,
        )
        papers = load_papers(relevant_columns=["paper_id", "in_cache", "in_ratings", "l1", "l2"])
        users_ratings = users_ratings.merge(
            papers[["paper_id", "l1", "l2"]], on="paper_id", how="left"
        )

        self.manage_users_coefs()
        self.val_negative_samples_ids = get_val_negative_samples_ids(
            papers=papers,
            n_categories_samples=self.config["n_negative_samples"],
            random_state=self.config["ranking_random_state"],
            papers_ids_to_exclude=papers[papers["in_ratings"]]["paper_id"].tolist(),
        )
        self.cache_papers_categories_ids, self.cache_papers_ids = get_cache_papers_ids_full(
            papers=papers,
            cache_type=self.config["cache_type"],
            n_cache=self.config["n_cache"],
            random_state=self.config["cache_random_state"],
            n_categories_cache=self.config["n_categories_cache"],
        )

        if self.config["n_jobs"] == 1:
            self.evaluate_users_in_sequence(
                users_ids=users_ids,
                users_significant_categories=users_significant_categories,
                users_ratings=users_ratings,
                users_embeddings=users_embeddings,
            )
        else:
            self.evaluate_users_in_parallel(
                users_ids=users_ids,
                users_significant_categories=users_significant_categories,
                users_ratings=users_ratings,
                users_embeddings=users_embeddings,
            )

    def manage_users_coefs(self) -> None:
        if self.config.get("save_users_coefs", False):
            if self.config["evaluation"] not in [
                Evaluation.TRAIN_TEST_SPLIT,
                Evaluation.SESSION_BASED,
            ]:
                raise ValueError(
                    "Users coefficient saving is only supported with train-test split or session-based evaluation."
                )
        self.load_users_coefs = self.config.get("load_users_coefs", False)
        if self.load_users_coefs:
            if self.config["evaluation"] == Evaluation.SLIDING_WINDOW:
                return
            if self.config["evaluation"] not in [
                Evaluation.TRAIN_TEST_SPLIT,
                Evaluation.SESSION_BASED,
            ]:
                raise ValueError(
                    "Users coefficient loading is only supported with train-test split or session-based evaluation."
                )
            if len(self.hyperparameters_combinations) > 1:
                raise ValueError(
                    "Users coefficient loading is not supported with multiple hyperparameter combinations."
                )
            users_coefs_path = Path(self.config["users_coefs_path"]).resolve()
            self.users_coefs = np.load(users_coefs_path / "users_coefs.npy")
            with open(users_coefs_path / "users_coefs_ids_to_idxs.pkl", "rb") as f:
                self.users_coefs_ids_to_idxs = pickle.load(f)

    def evaluate_users_in_sequence(
        self,
        users_ids: list,
        users_significant_categories: pd.DataFrame,
        users_ratings: pd.DataFrame,
        users_embeddings: dict = None,
    ) -> None:
        for user_id in users_ids:
            user_significant_categories = users_significant_categories[
                users_significant_categories["user_id"] == user_id
            ]["category"].tolist()
            user_ratings = users_ratings[users_ratings["user_id"] == user_id].reset_index(drop=True)
            user_embeddings = None
            if users_embeddings is not None:
                user_embeddings = users_embeddings[user_id]
            self.evaluate_user(
                user_id=user_id,
                user_significant_categories=user_significant_categories,
                user_ratings=user_ratings,
                user_embeddings=user_embeddings,
            )

    def evaluate_users_in_parallel(
        self,
        users_ids: list,
        users_significant_categories: pd.DataFrame,
        users_ratings: pd.DataFrame,
        users_embeddings: dict = None,
    ) -> None:
        users_list = []
        for user_id in users_ids:
            users_list.append(
                (
                    user_id,
                    users_significant_categories[
                        users_significant_categories["user_id"] == user_id
                    ]["category"].tolist(),
                    users_ratings[users_ratings["user_id"] == user_id].reset_index(drop=True),
                    users_embeddings[user_id] if users_embeddings else None,
                )
            )
        Parallel(n_jobs=self.config["n_jobs"])(
            delayed(self.evaluate_user)(
                user_id=user_id,
                user_significant_categories=user_significant_categories,
                user_ratings=user_ratings,
                user_embeddings=user_embeddings,
            )
            for user_id, user_significant_categories, user_ratings, user_embeddings in users_list
        )

    def evaluate_user(
        self,
        user_id: int,
        user_significant_categories: list,
        user_ratings: pd.DataFrame,
        user_embeddings: dict = None,
    ) -> None:
        user_results_dict, user_predictions_dict = {}, {}

        user_categories_ratios = get_user_categories_ratios(
            categories_to_exclude=user_significant_categories
        )
        user_val_negative_samples = get_user_val_negative_samples(
            val_negative_samples_ids=self.val_negative_samples_ids,
            n_negative_samples=self.config["n_negative_samples"],
            random_state=self.config["ranking_random_state"],
            user_categories_ratios=user_categories_ratios,
            embedding=self.embedding,
        )
        user_predictions_dict["negative_samples_ids"] = user_val_negative_samples[
            "val_negative_samples_ids"
        ]
        papers_ids_to_exclude_from_cache = (
            user_ratings["paper_id"].tolist()
            + user_val_negative_samples["val_negative_samples_ids"]
        )
        user_cache_papers = get_user_cache_papers(
            cache_type=self.config["cache_type"],
            cache_papers_ids=self.cache_papers_ids,
            cache_papers_categories_ids=self.cache_papers_categories_ids,
            n_categories_cache=self.config["n_categories_cache"],
            random_state=self.config["cache_random_state"],
            papers_ids_to_exclude_from_cache=papers_ids_to_exclude_from_cache,
            user_categories_ratios=user_categories_ratios,
            embedding=self.embedding,
        )
        user_info = self.store_user_info(
            user_ratings=user_ratings,
            cache_n=user_cache_papers["cache_n"],
        )

        try:
            if self.config["evaluation"] in [
                Evaluation.TRAIN_TEST_SPLIT,
                Evaluation.SESSION_BASED,
            ]:
                self.evaluate_user_split(
                    user_id=user_id,
                    user_ratings=user_ratings,
                    user_val_negative_samples=user_val_negative_samples,
                    user_cache_papers=user_cache_papers,
                    user_info=user_info,
                    user_results_dict=user_results_dict,
                    user_predictions_dict=user_predictions_dict,
                )

            elif self.config["evaluation"] == Evaluation.CROSS_VALIDATION:
                self.evaluate_user_cross_validation(
                    user_id=user_id,
                    user_ratings=user_ratings,
                    user_val_negative_samples=user_val_negative_samples,
                    user_cache_papers=user_cache_papers,
                    user_info=user_info,
                    user_results_dict=user_results_dict,
                    user_predictions_dict=user_predictions_dict,
                )

            elif self.config["evaluation"] == Evaluation.SLIDING_WINDOW:
                self.evaluate_user_sliding_window(
                    user_id=user_id,
                    user_ratings=user_ratings,
                    user_embeddings=user_embeddings,
                    user_val_negative_samples=user_val_negative_samples,
                    user_info=user_info,
                    user_results_dict=user_results_dict,
                    user_predictions_dict=user_predictions_dict,
                )

            self.save_user_info(user_id, user_info)
            self.save_user_results(user_id, user_results_dict)
            self.save_user_predictions(user_id, user_predictions_dict)
            print(f"User {user_id} done.")
        except Exception as e:
            print(f"Error evaluating user {user_id}: {e}")
            traceback.print_exc()

    def store_user_info(self, user_ratings: pd.DataFrame, cache_n: int) -> dict:
        user_info = {"n_cache": cache_n, "n_base": 0, "n_zerorated": 0}
        posrated_n = len(user_ratings[user_ratings["rating"] == 1])
        negrated_n = len(user_ratings[user_ratings["rating"] == 0])
        assert posrated_n + negrated_n == len(user_ratings)
        user_info.update({"n_posrated": posrated_n, "n_negrated": negrated_n})
        return user_info

    def evaluate_user_split(
        self,
        user_id: int,
        user_ratings: pd.DataFrame,
        user_val_negative_samples: dict,
        user_cache_papers: dict,
        user_info: dict,
        user_results_dict: dict,
        user_predictions_dict: dict,
    ) -> None:
        train_ratings, val_ratings, removed_ratings = split_ratings(user_ratings)
        n_train, n_val = len(train_ratings), len(val_ratings)
        user_info["train_rated_ratio"] = n_train / (n_train + n_val)
        train_negrated_ranking, val_negrated_ranking = split_negrated_ranking(
            train_ratings, val_ratings, removed_ratings
        )

        user_data_dict = self.load_user_data_dict(
            train_ratings=train_ratings,
            val_ratings=val_ratings,
            train_negrated_ranking=train_negrated_ranking,
            val_negrated_ranking=val_negrated_ranking,
            cache_embedding_idxs=user_cache_papers["cache_embedding_idxs"],
            y_cache=user_cache_papers["y_cache"],
            random_state=self.config["model_random_state"],
        )
        user_predictions_dict[0] = fill_user_predictions_dict(user_data_dict)
        timesort = self.config["evaluation"] == Evaluation.SESSION_BASED
        val_causal_mask = self.config["filter_for_negrated_ranking"]
        train_negrated_ranking_idxs = load_negrated_ranking_idxs_for_user(
            ratings=train_ratings,
            negrated_ranking=train_negrated_ranking,
            timesort=timesort,
            causal_mask=False,
            random_state=self.config["ranking_random_state"],
            same_negrated_for_all_pos=self.config["same_negrated_for_all_pos"],
        )
        val_negrated_ranking_idxs = load_negrated_ranking_idxs_for_user(
            ratings=val_ratings,
            negrated_ranking=val_negrated_ranking,
            timesort=timesort,
            causal_mask=val_causal_mask,
            random_state=self.config["ranking_random_state"],
            same_negrated_for_all_pos=self.config["same_negrated_for_all_pos"],
        )
        user_models = self.train_user_models(user_data_dict=user_data_dict, user_id=user_id)
        user_results, user_predictions = self.score_user_models(
            user_data_dict=user_data_dict,
            user_models=user_models,
            negative_samples_embeddings=user_val_negative_samples[
                "val_negative_samples_embeddings"
            ],
            train_negrated_ranking_idxs=train_negrated_ranking_idxs,
            val_negrated_ranking_idxs=val_negrated_ranking_idxs,
        )
        user_results_dict[0] = user_results
        user_predictions_dict[0] = user_predictions
        if self.config["save_users_coefs"]:
            assert len(user_models) == 1
            model = user_models[0]
            user_coefs = np.hstack([model.coef_[0], model.intercept_[0]])
            self.save_user_coefs(user_id, user_coefs=user_coefs)

    def evaluate_user_cross_validation(
        self,
        user_id: int,
        user_ratings: pd.DataFrame,
        user_val_negative_samples: dict,
        user_cache_papers: dict,
        user_info: dict,
        user_results_dict: dict,
        user_predictions_dict: dict,
    ) -> None:
        assert user_ratings["split"].isnull().all()
        split = self.cross_val.split(X=range(len(user_ratings)), y=user_ratings["rating"])
        train_rated_ratios = []
        for fold_idx, (fold_train_idxs, fold_val_idxs) in enumerate(split):
            user_ratings.loc[fold_train_idxs, "split"] = "train"
            user_ratings.loc[fold_val_idxs, "split"] = "val"
            train_ratings, val_ratings, _ = split_ratings(user_ratings)
            train_rated_ratios.append(len(train_ratings) / len(user_ratings))
            train_negrated_ranking = train_ratings[train_ratings["rating"] == 0].reset_index(
                drop=True
            )
            val_negrated_ranking = val_ratings[val_ratings["rating"] == 0].reset_index(drop=True)

            user_data_dict = self.load_user_data_dict(
                train_ratings=train_ratings,
                val_ratings=val_ratings,
                train_negrated_ranking=train_negrated_ranking,
                val_negrated_ranking=val_negrated_ranking,
                cache_embedding_idxs=user_cache_papers["cache_embedding_idxs"],
                y_cache=user_cache_papers["y_cache"],
                random_state=self.config["model_random_state"],
            )
            user_predictions_dict[fold_idx] = fill_user_predictions_dict(user_data_dict)
            train_negrated_ranking_idxs = load_negrated_ranking_idxs_for_user(
                ratings=train_ratings,
                negrated_ranking=train_negrated_ranking,
                timesort=False,
                causal_mask=False,
                random_state=self.config["ranking_random_state"],
                same_negrated_for_all_pos=self.config["same_negrated_for_all_pos"],
            )
            val_negrated_ranking_idxs = load_negrated_ranking_idxs_for_user(
                ratings=val_ratings,
                negrated_ranking=val_negrated_ranking,
                timesort=False,
                causal_mask=False,
                random_state=self.config["ranking_random_state"],
                same_negrated_for_all_pos=self.config["same_negrated_for_all_pos"],
            )
            user_models = self.train_user_models(user_data_dict=user_data_dict, user_id=user_id)
            fold_results, fold_predictions = self.score_user_models(
                user_data_dict=user_data_dict,
                user_models=user_models,
                negative_samples_embeddings=user_val_negative_samples[
                    "val_negative_samples_embeddings"
                ],
                train_negrated_ranking_idxs=train_negrated_ranking_idxs,
                val_negrated_ranking_idxs=val_negrated_ranking_idxs,
            )
            user_results_dict[fold_idx] = fold_results
            if self.config["save_users_predictions"]:
                user_predictions_dict[fold_idx].update(
                    update_user_predictions_dict(fold_predictions)
                )
        user_info["train_rated_ratio"] = np.mean(train_rated_ratios)

    def evaluate_user_sliding_window(
        self,
        user_id: int,
        user_ratings: pd.DataFrame,
        user_embeddings: dict,
        user_val_negative_samples: dict,
        user_info: dict,
        user_results_dict: dict,
        user_predictions_dict: dict,
    ) -> None:
        train_ratings, val_ratings, removed_ratings = split_ratings(user_ratings)
        n_train, n_val = len(train_ratings), len(val_ratings)
        user_info["train_rated_ratio"] = n_train / (n_train + n_val)
        train_negrated_ranking, val_negrated_ranking = split_negrated_ranking(
            train_ratings, val_ratings, removed_ratings
        )

        user_data_dict = self.load_user_data_dict(
            train_ratings=train_ratings,
            val_ratings=val_ratings,
            train_negrated_ranking=train_negrated_ranking,
            val_negrated_ranking=val_negrated_ranking,
            cache_embedding_idxs=np.array([], dtype=self.embedding.papers_idxs_dtype),
            y_cache=np.array([], dtype=LABEL_DTYPE),
            random_state=self.config["model_random_state"],
        )
        user_predictions_dict[0] = fill_user_predictions_dict(user_data_dict)
        train_negrated_ranking_idxs = load_negrated_ranking_idxs_for_user(
            ratings=train_ratings,
            negrated_ranking=train_negrated_ranking,
            timesort=True,
            causal_mask=False,
            random_state=self.config["ranking_random_state"],
            same_negrated_for_all_pos=self.config["same_negrated_for_all_pos"],
        )
        val_negrated_ranking_idxs = load_negrated_ranking_idxs_for_user(
            ratings=val_ratings,
            negrated_ranking=val_negrated_ranking,
            timesort=True,
            causal_mask=True,
            random_state=self.config["ranking_random_state"],
            same_negrated_for_all_pos=self.config["same_negrated_for_all_pos"],
        )

        for i, session_id in enumerate(user_embeddings["sessions_ids"]):
            session_embedding = user_embeddings["sessions_embeddings"][i]
            model = self.load_user_model(session_embedding)

    def train_user_models(self, user_data_dict: dict, user_id: int = None) -> list:
        if self.load_users_coefs:
            return [
                self.load_user_model(self.users_coefs[self.users_coefs_ids_to_idxs[user_id], :])
            ]
        user_models = []
        for hyperparameters_combination in self.hyperparameters_combinations:
            model = self.get_model_for_user(hyperparameters_combination)
            sample_weights = np.empty(user_data_dict["X_train"].shape[0], dtype=np.float64)
            train_posrated_n = user_data_dict["pos_idxs"].shape[0]
            train_negrated_n = user_data_dict["neg_idxs"].shape[0]
            cache_n = user_data_dict["cache_idxs"].shape[0]
            w_p, w_n, _, _, w_c = self.wh.load_weights_for_user(
                hyperparameters=self.hyperparameters,
                hyperparameters_combination=hyperparameters_combination,
                train_posrated_n=train_posrated_n,
                train_negrated_n=train_negrated_n,
                cache_n=cache_n,
            )
            sample_weights[user_data_dict["pos_idxs"]] = w_p
            sample_weights[user_data_dict["neg_idxs"]] = w_n
            sample_weights[user_data_dict["cache_idxs"]] = w_c
            model.fit(
                user_data_dict["X_train"],
                user_data_dict["y_train"],
                sample_weight=sample_weights,
            )
            user_models.append(model)
        return user_models

    def load_user_model(self, user_coefs: np.ndarray) -> object:
        assert len(self.hyperparameters_combinations) == 1
        model = self.get_model_for_user(self.hyperparameters_combinations[0])
        model.n_features_in = user_coefs.shape[0]
        model.coef_ = user_coefs[:-1].reshape(1, -1)
        model.intercept_ = user_coefs[-1].reshape(1, -1)
        model.classes_ = np.array([0, 1])
        return model

    def score_user_models(
        self,
        user_data_dict: dict,
        user_models: list,
        negative_samples_embeddings: np.ndarray,
        train_negrated_ranking_idxs: np.ndarray,
        val_negrated_ranking_idxs: np.ndarray,
    ) -> tuple:
        user_results = {}
        user_predictions = {
            "train_predictions": {},
            "val_predictions": {},
            "negative_samples_predictions": {},
        }
        for i, model in enumerate(user_models):
            if len(user_data_dict["y_val"]) > 0:
                user_outputs_dict = get_user_outputs_dict(
                    model=model,
                    user_data_dict=user_data_dict,
                    negative_samples_embeddings=negative_samples_embeddings,
                    train_negrated_ranking_idxs=train_negrated_ranking_idxs,
                    val_negrated_ranking_idxs=val_negrated_ranking_idxs,
                )
                scores = self.get_user_scores(
                    user_data_dict=user_data_dict, user_outputs_dict=user_outputs_dict
                )
                user_results[i] = scores
                if self.config["save_users_predictions"]:
                    user_predictions["train_predictions"][i] = user_outputs_dict[
                        "y_train_rated_proba"
                    ].tolist()
                    user_predictions["val_predictions"][i] = user_outputs_dict[
                        "y_val_proba"
                    ].tolist()
                    user_predictions["negative_samples_predictions"][i] = user_outputs_dict[
                        "y_negative_samples_proba"
                    ].tolist()
                if self.config["save_tfidf_coefs"]:
                    user_predictions["tfidf_coefs"][i] = model.coef_[0].tolist()
        return user_results, user_predictions

    def get_model_for_user(self, hyperparameters_combination: tuple) -> object:
        clf_C = hyperparameters_combination[self.hyperparameters["clf_C"]]
        logreg_solver = None
        if self.config["algorithm"] == Algorithm.LOGREG:
            logreg_solver = self.config["logreg_solver"]
        svm_kernel = None
        if self.config["algorithm"] == Algorithm.SVM:
            svm_kernel = self.config["svm_kernel"]
        return get_model(
            algorithm=self.config["algorithm"],
            max_iter=self.config["max_iter"],
            clf_C=clf_C,
            random_state=self.config["model_random_state"],
            logreg_solver=logreg_solver,
            svm_kernel=svm_kernel,
        )

    def get_user_scores(
        self,
        user_data_dict: dict,
        user_outputs_dict: dict,
    ) -> tuple:
        user_scores = [0] * self.scores_n
        for score in self.non_derivable_scores:
            if not SCORES_DICT[score]["ranking"]:
                user_scores[self.scores[f"train_{score.name.lower()}"]] = get_score(
                    score=score,
                    y_true=user_data_dict["y_train_rated"],
                    y_pred=user_outputs_dict["y_train_rated_pred"],
                    y_proba=user_outputs_dict["y_train_rated_proba"],
                    y_negative_samples_pred=user_outputs_dict["y_negative_samples_pred"],
                    y_negative_samples_proba=user_outputs_dict["y_negative_samples_proba"],
                )
                user_scores[self.scores[f"val_{score.name.lower()}"]] = get_score(
                    score=score,
                    y_true=user_data_dict["y_val"],
                    y_pred=user_outputs_dict["y_val_pred"],
                    y_proba=user_outputs_dict["y_val_proba"],
                    y_negative_samples_pred=user_outputs_dict["y_negative_samples_pred"],
                    y_negative_samples_proba=user_outputs_dict["y_negative_samples_proba"],
                )
        ranking_scores = get_ranking_scores(
            y_train_rated=user_data_dict["y_train_rated"],
            y_train_rated_logits=user_outputs_dict["y_train_rated_logits"],
            y_val=user_data_dict["y_val"],
            y_val_logits=user_outputs_dict["y_val_logits"],
            y_train_negrated_ranking_logits=user_outputs_dict["y_train_negrated_ranking_logits"],
            y_val_negrated_ranking_logits=user_outputs_dict["y_val_negrated_ranking_logits"],
            y_negative_samples_logits=user_outputs_dict["y_negative_samples_logits"],
        )
        for ranking_score in ranking_scores:
            user_scores[self.scores[ranking_score]] = ranking_scores[ranking_score]
        category_scores = get_category_scores(
            y_train_rated=user_data_dict["y_train_rated"],
            y_val=user_data_dict["y_val"],
            categories_dict=user_data_dict["categories_dict"],
        )
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

    def load_user_data_dict(
        self,
        train_ratings: pd.DataFrame,
        val_ratings: pd.DataFrame,
        train_negrated_ranking: pd.DataFrame,
        val_negrated_ranking: pd.DataFrame,
        cache_embedding_idxs: np.ndarray,
        y_cache: np.ndarray,
        random_state: int,
    ) -> dict:
        train_data_dict = self.load_user_train_data_dict(
            train_ratings, cache_embedding_idxs, y_cache, random_state, train_negrated_ranking
        )
        val_data_dict = self.load_user_val_data_dict(
            train_ratings, val_ratings, val_negrated_ranking
        )
        return {**train_data_dict, **val_data_dict}

    def load_user_train_data_dict(
        self,
        train_ratings: pd.DataFrame,
        cache_embedding_idxs: np.ndarray,
        y_cache: np.ndarray,
        random_state: int,
        train_negrated_ranking: pd.DataFrame = None,
    ) -> dict:
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
        train_data_dict = {
            "X_train": X_train,
            "y_train": y_full,
            "pos_idxs": pos_idxs,
            "neg_idxs": neg_idxs,
            "cache_idxs": cache_idxs,
        }
        if train_negrated_ranking is not None:
            X_train_negrated_ranking = self.embedding.matrix[
                self.embedding.get_idxs(train_negrated_ranking["paper_id"].tolist())
            ]
            train_data_dict["X_train_negrated_ranking"] = X_train_negrated_ranking
        return train_data_dict

    def load_user_val_data_dict(
        self,
        train_ratings: pd.DataFrame,
        val_ratings: pd.DataFrame,
        val_negrated_ranking: pd.DataFrame = None,
    ) -> dict:
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
        val_data_dict = {
            "X_train_rated": X_train_rated,
            "y_train_rated": y_train_rated,
            "X_train_rated_papers_ids": X_train_rated_papers_ids,
            "X_val": X_val,
            "y_val": y_val,
            "X_val_papers_ids": X_val_papers_ids,
            "categories_dict": categories_dict,
        }
        if val_negrated_ranking is not None:
            X_val_negrated_ranking = self.embedding.matrix[
                self.embedding.get_idxs(val_negrated_ranking["paper_id"].tolist())
            ]
            val_data_dict["X_val_negrated_ranking"] = X_val_negrated_ranking
        return val_data_dict

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


def get_val_negative_samples_ids(
    papers: pd.DataFrame,
    n_categories_samples: int,
    random_state: int,
    papers_ids_to_exclude: list,
) -> list:
    return get_categories_samples_ids(
        papers=papers,
        n_categories_samples=n_categories_samples,
        random_state=random_state,
        papers_ids_to_exclude=papers_ids_to_exclude,
    )[0]


def get_cache_papers_ids_full(
    papers: pd.DataFrame,
    cache_type: str,
    n_cache: int,
    random_state: int,
    n_categories_cache: int,
) -> tuple:
    cache_papers_ids = get_cache_papers_ids(
        cache_type=cache_type,
        papers=papers,
        n_cache=n_cache,
        random_state=random_state,
    )
    cache_papers_categories_ids = get_categories_samples_ids(
        papers=papers,
        n_categories_samples=n_categories_cache,
        random_state=random_state,
        papers_ids_to_exclude=cache_papers_ids,
    )[0]
    return cache_papers_categories_ids, cache_papers_ids


def get_user_val_negative_samples(
    val_negative_samples_ids: list,
    n_negative_samples: int,
    random_state: int,
    user_categories_ratios: dict,
    embedding: Embedding = None,
) -> dict:
    val_negative_samples_ids = get_user_categories_samples_ids(
        categories_samples_ids=val_negative_samples_ids,
        n_categories_samples=n_negative_samples,
        random_state=random_state,
        sort_samples=True,
        user_categories_ratios=user_categories_ratios,
    )
    negative_samples_embeddings = None
    if embedding is not None:
        negative_samples_embeddings = embedding.matrix[embedding.get_idxs(val_negative_samples_ids)]
    return {
        "val_negative_samples_ids": val_negative_samples_ids,
        "val_negative_samples_embeddings": negative_samples_embeddings,
    }


def get_user_cache_papers(
    cache_type: str,
    cache_papers_ids: list,
    cache_papers_categories_ids: list,
    n_categories_cache: int,
    random_state: int,
    papers_ids_to_exclude_from_cache: list,
    user_categories_ratios: dict,
    embedding: Embedding = None,
) -> dict:
    user_cache_papers_ids = get_user_cache_papers_ids(
        cache_type=cache_type,
        cache_papers_ids=cache_papers_ids,
    )
    user_cache_papers_categories_ids = get_user_categories_samples_ids(
        categories_samples_ids=cache_papers_categories_ids,
        n_categories_samples=n_categories_cache,
        random_state=random_state,
        sort_samples=False,
        user_categories_ratios=user_categories_ratios,
    )
    cache_papers_ids = sorted(
        list(set(user_cache_papers_categories_ids) | set(user_cache_papers_ids))
    )
    cache_papers_ids = [
        paper_id
        for paper_id in cache_papers_ids
        if paper_id not in papers_ids_to_exclude_from_cache
    ]
    assert user_cache_papers_ids == sorted(user_cache_papers_ids)
    assert len(user_cache_papers_ids) == len(set(user_cache_papers_ids))
    cache_n = len(cache_papers_ids)
    y_cache = np.zeros(cache_n, dtype=LABEL_DTYPE)
    cache_embedding_idxs = None
    if embedding is not None:
        cache_embedding_idxs = embedding.get_idxs(cache_papers_ids)
    return {
        "user_cache_papers_categories_ids": user_cache_papers_categories_ids,
        "cache_embedding_idxs": cache_embedding_idxs,
        "cache_n": cache_n,
        "y_cache": y_cache,
    }


def split_ratings(user_ratings: pd.DataFrame) -> tuple:
    if "split" not in user_ratings.columns:
        raise ValueError("User ratings DataFrame must contain 'split' column.")
    train_mask = user_ratings["split"] == "train"
    val_mask = user_ratings["split"] == "val"
    removed_mask = user_ratings["split"] == "removed"
    train_ratings = user_ratings.loc[train_mask]
    val_ratings = user_ratings.loc[val_mask]
    removed_ratings = user_ratings.loc[removed_mask]
    assert len(train_ratings) + len(val_ratings) + len(removed_ratings) == len(user_ratings)
    removed_ratings = removed_ratings[removed_ratings["rating"] == 0]
    removed_ratings = removed_ratings.iloc[:N_NEGRATED_RANKING]
    assert train_ratings["time"].is_monotonic_increasing
    assert val_ratings["time"].is_monotonic_increasing
    assert removed_ratings["time"].is_monotonic_increasing
    return train_ratings, val_ratings, removed_ratings


def split_negrated_ranking(
    train_ratings: pd.DataFrame, val_ratings: pd.DataFrame, removed_ratings: pd.DataFrame
) -> tuple:
    train_negrated_ranking = train_ratings[train_ratings["rating"] == 0]
    val_negrated_ranking = pd.concat([val_ratings[val_ratings["rating"] == 0], removed_ratings])
    return (
        train_negrated_ranking.reset_index(drop=True),
        val_negrated_ranking.reset_index(drop=True),
    )


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


def get_user_outputs_dict(
    model: object,
    user_data_dict: dict,
    negative_samples_embeddings: np.ndarray,
    train_negrated_ranking_idxs: np.ndarray,
    val_negrated_ranking_idxs: np.ndarray,
) -> dict:
    user_outputs_dict = {}
    user_outputs_dict["y_train_rated_pred"] = model.predict(user_data_dict["X_train_rated"])
    user_outputs_dict["y_train_rated_proba"] = model.predict_proba(user_data_dict["X_train_rated"])[
        :, 1
    ]
    user_outputs_dict["y_train_rated_logits"] = model.decision_function(
        user_data_dict["X_train_rated"]
    )

    user_outputs_dict["y_val_pred"] = model.predict(user_data_dict["X_val"])
    user_outputs_dict["y_val_proba"] = model.predict_proba(user_data_dict["X_val"])[:, 1]
    user_outputs_dict["y_val_logits"] = model.decision_function(user_data_dict["X_val"])

    y_train_negrated_ranking_logits = model.decision_function(
        user_data_dict["X_train_negrated_ranking"]
    )
    y_val_negrated_ranking_logits = model.decision_function(
        user_data_dict["X_val_negrated_ranking"]
    )
    user_outputs_dict["y_train_negrated_ranking_logits"] = y_train_negrated_ranking_logits[
        train_negrated_ranking_idxs
    ]
    user_outputs_dict["y_val_negrated_ranking_logits"] = y_val_negrated_ranking_logits[
        val_negrated_ranking_idxs
    ]

    user_outputs_dict["y_negative_samples_pred"] = model.predict(negative_samples_embeddings)
    user_outputs_dict["y_negative_samples_proba"] = model.predict_proba(
        negative_samples_embeddings
    )[:, 1]
    user_outputs_dict["y_negative_samples_logits"] = model.decision_function(
        negative_samples_embeddings
    )
    return user_outputs_dict


def fill_user_predictions_dict(user_data_dict: dict) -> dict:
    return {
        "train_ids": user_data_dict["X_train_rated_papers_ids"],
        "train_labels": user_data_dict["y_train_rated"].tolist(),
        "val_ids": user_data_dict["X_val_papers_ids"],
        "val_labels": user_data_dict["y_val"].tolist(),
    }


def update_user_predictions_dict(user_predictions: dict) -> dict:
    return {
        "train_predictions": user_predictions["train_predictions"],
        "val_predictions": user_predictions["val_predictions"],
        "negative_samples_predictions": user_predictions["negative_samples_predictions"],
        "tfidf_coefs": user_predictions["tfidf_coefs"],
    }
