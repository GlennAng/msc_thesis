import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ....src.load_files import load_papers, load_users_significant_categories
from ..embeddings.embedding import Embedding
from .algorithm import (
    Algorithm,
    Evaluation,
    get_cross_val,
    get_model,
)
from .scores import (
    fill_user_predictions_dict,
    load_user_data_dicts,
    score_user_models,
    score_user_models_sliding_window,
    update_user_predictions_dict,
)
from .training_data import (
    get_cache_papers_ids_full,
    get_user_cache_papers,
    get_user_categories_ratios,
    get_user_val_negative_samples,
    get_val_negative_samples_ids,
    load_negrated_ranking_idxs_for_user,
    save_user_coefs,
    save_user_info,
    save_user_predictions,
    save_user_results,
    split_negrated_ranking,
    split_ratings,
    store_user_info_initial,
    update_user_info_split,
)
from .users_ratings import UsersRatingsSelection
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
        val_negative_samples_embeddings = user_val_negative_samples[
            "val_negative_samples_embeddings"
        ]
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
        user_info = store_user_info_initial(user_ratings, user_cache_papers["cache_n"])

        try:
            if self.config["evaluation"] in [
                Evaluation.TRAIN_TEST_SPLIT,
                Evaluation.SESSION_BASED,
            ]:
                self.evaluate_user_split(
                    user_id=user_id,
                    user_ratings=user_ratings,
                    val_negative_samples_embeddings=val_negative_samples_embeddings,
                    user_cache_papers=user_cache_papers,
                    user_info=user_info,
                    user_results_dict=user_results_dict,
                    user_predictions_dict=user_predictions_dict,
                )

            elif self.config["evaluation"] == Evaluation.CROSS_VALIDATION:
                self.evaluate_user_cross_validation(
                    user_id=user_id,
                    user_ratings=user_ratings,
                    val_negative_samples_embeddings=val_negative_samples_embeddings,
                    user_cache_papers=user_cache_papers,
                    user_info=user_info,
                    user_results_dict=user_results_dict,
                    user_predictions_dict=user_predictions_dict,
                )

            elif self.config["evaluation"] == Evaluation.SLIDING_WINDOW:
                self.evaluate_user_sliding_window(
                    user_ratings=user_ratings,
                    user_embeddings=user_embeddings,
                    val_negative_samples_embeddings=val_negative_samples_embeddings,
                    user_info=user_info,
                    user_results_dict=user_results_dict,
                    user_predictions_dict=user_predictions_dict,
                )

            save_user_info(self.config["outputs_dir"], user_id, user_info)
            save_user_results(self.config["outputs_dir"], user_id, user_results_dict)
            save_user_predictions(self.config["outputs_dir"], user_id, user_predictions_dict)
            print(f"User {user_id} done.")
        except Exception as e:
            print(f"Error evaluating user {user_id}: {e}")
            traceback.print_exc()

    def evaluate_user_split(
        self,
        user_id: int,
        user_ratings: pd.DataFrame,
        val_negative_samples_embeddings: np.ndarray,
        user_cache_papers: dict,
        user_info: dict,
        user_results_dict: dict,
        user_predictions_dict: dict,
    ) -> None:
        train_ratings, val_ratings, removed_ratings = split_ratings(user_ratings)
        update_user_info_split(user_info, train_ratings, val_ratings)
        train_negrated_ranking, val_negrated_ranking = split_negrated_ranking(
            train_ratings, val_ratings, removed_ratings
        )

        train_data_dict, val_data_dict = load_user_data_dicts(
            train_ratings=train_ratings,
            val_ratings=val_ratings,
            train_negrated_ranking=train_negrated_ranking,
            val_negrated_ranking=val_negrated_ranking,
            embedding=self.embedding,
            load_user_train_data_dict_bool=True,
            cache_embedding_idxs=user_cache_papers["cache_embedding_idxs"],
            y_cache=user_cache_papers["y_cache"],
            random_state=self.config["model_random_state"],
        )
        user_predictions_dict[0] = fill_user_predictions_dict(val_data_dict)
        timesort = self.config["evaluation"] == Evaluation.SESSION_BASED
        val_causal_mask = self.config["users_ratings_selection"] in [
            UsersRatingsSelection.SESSION_BASED_FILTERING,
            UsersRatingsSelection.SESSION_BASED_FILTERING_OLD,
        ]
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
        user_models = self.train_user_models(train_data_dict=train_data_dict, user_id=user_id)
        user_results, user_predictions = score_user_models(
            scores_to_indices_dict=self.config["scores"],
            val_data_dict=val_data_dict,
            user_models=user_models,
            negative_samples_embeddings=val_negative_samples_embeddings,
            train_negrated_ranking_idxs=train_negrated_ranking_idxs,
            val_negrated_ranking_idxs=val_negrated_ranking_idxs,
            user_info=user_info,
            save_users_predictions_bool=self.config["save_users_predictions"],
        )
        user_results_dict[0] = user_results
        user_predictions_dict[0].update(user_predictions)
        if self.config["save_users_coefs"]:
            assert len(user_models) == 1
            model = user_models[0]
            user_coefs = np.hstack([model.coef_[0], model.intercept_[0]])
            save_user_coefs(self.config["outputs_dir"], user_id, user_coefs=user_coefs)

    def evaluate_user_cross_validation(
        self,
        user_id: int,
        user_ratings: pd.DataFrame,
        val_negative_samples_embeddings: np.ndarray,
        user_cache_papers: dict,
        user_info: dict,
        user_results_dict: dict,
        user_predictions_dict: dict,
    ) -> None:        
        split = self.cross_val.split(X=range(len(user_ratings)), y=user_ratings["rating"])
        train_rated_ratios = []
        for fold_idx, (fold_train_idxs, fold_val_idxs) in enumerate(split):
            user_ratings.loc[fold_train_idxs, "split"] = "train"
            user_ratings.loc[fold_val_idxs, "split"] = "val"
            train_ratings, val_ratings, _ = split_ratings(user_ratings)
            update_user_info_split(user_info, train_ratings, val_ratings)
            train_rated_ratios.append(len(train_ratings) / len(user_ratings))
            train_negrated_ranking = train_ratings[train_ratings["rating"] == 0].reset_index(
                drop=True
            )
            val_negrated_ranking = val_ratings[val_ratings["rating"] == 0].reset_index(drop=True)

            train_data_dict, val_data_dict = load_user_data_dicts(
                train_ratings=train_ratings,
                val_ratings=val_ratings,
                train_negrated_ranking=train_negrated_ranking,
                val_negrated_ranking=val_negrated_ranking,
                embedding=self.embedding,
                load_user_train_data_dict_bool=True,
                cache_embedding_idxs=user_cache_papers["cache_embedding_idxs"],
                y_cache=user_cache_papers["y_cache"],
                random_state=self.config["model_random_state"],
            )
            user_predictions_dict[fold_idx] = fill_user_predictions_dict(val_data_dict)
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
            user_models = self.train_user_models(train_data_dict=train_data_dict, user_id=user_id)
            fold_results, fold_predictions = score_user_models(
                scores_to_indices_dict=self.config["scores"],
                val_data_dict=val_data_dict,
                user_models=user_models,
                negative_samples_embeddings=val_negative_samples_embeddings,
                train_negrated_ranking_idxs=train_negrated_ranking_idxs,
                val_negrated_ranking_idxs=val_negrated_ranking_idxs,
                user_info=user_info,
                save_users_predictions_bool=self.config["save_users_predictions"],
            )
            user_results_dict[fold_idx] = fold_results
            if self.config["save_users_predictions"]:
                user_predictions_dict[fold_idx].update(
                    update_user_predictions_dict(fold_predictions)
                )
        user_info["train_rated_ratio"] = np.mean(train_rated_ratios)

    def evaluate_user_sliding_window(
        self,
        user_ratings: pd.DataFrame,
        user_embeddings: dict,
        val_negative_samples_embeddings: np.ndarray,
        user_info: dict,
        user_results_dict: dict,
        user_predictions_dict: dict,
    ) -> None:
        train_ratings, val_ratings, removed_ratings = split_ratings(user_ratings)
        update_user_info_split(user_info, train_ratings, val_ratings)
        train_negrated_ranking, val_negrated_ranking = split_negrated_ranking(
            train_ratings, val_ratings, removed_ratings
        )

        val_sessions_idxs = val_ratings["session_id"] - val_ratings["session_id"].min()
        val_idxs_to_val_sessions_idxs = val_sessions_idxs.tolist()
        val_pos_idxs_to_val_sessions_idxs = val_sessions_idxs[val_ratings["rating"] > 0].tolist()

        _, val_data_dict = load_user_data_dicts(
            train_ratings=train_ratings,
            val_ratings=val_ratings,
            train_negrated_ranking=train_negrated_ranking,
            val_negrated_ranking=val_negrated_ranking,
            embedding=self.embedding,
            load_user_train_data_dict_bool=False,
        )
        user_predictions_dict[0] = fill_user_predictions_dict(val_data_dict)
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
        models = [
            self.load_user_model(user_embeddings["sessions_embeddings"][i])
            for i in range(len(user_embeddings["sessions_ids"]))
        ]
        user_results, user_predictions = score_user_models_sliding_window(
            val_idxs_to_val_sessions_idxs=val_idxs_to_val_sessions_idxs,
            val_pos_idxs_to_val_sessions_idxs=val_pos_idxs_to_val_sessions_idxs,
            scores_to_indices_dict=self.config["scores"],
            val_data_dict=val_data_dict,
            user_models=models,
            negative_samples_embeddings=val_negative_samples_embeddings,
            train_negrated_ranking_idxs=train_negrated_ranking_idxs,
            val_negrated_ranking_idxs=val_negrated_ranking_idxs,
            user_info=user_info,
            save_users_predictions_bool=self.config["save_users_predictions"],
        )
        user_results_dict[0] = user_results
        user_predictions_dict[0].update(user_predictions)

    def load_user_model(self, user_coefs: np.ndarray) -> object:
        assert len(self.hyperparameters_combinations) == 1
        model = self.get_model_for_user(self.hyperparameters_combinations[0])
        model.n_features_in = user_coefs.shape[0]
        model.coef_ = user_coefs[:-1].reshape(1, -1)
        model.intercept_ = user_coefs[-1].reshape(1, -1)
        model.classes_ = np.array([0, 1])
        return model

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

    def train_user_models(self, train_data_dict: dict, user_id: int = None) -> list:
        if self.load_users_coefs:
            return [
                self.load_user_model(self.users_coefs[self.users_coefs_ids_to_idxs[user_id], :])
            ]
        user_models = []
        for hyperparameters_combination in self.hyperparameters_combinations:
            model = self.get_model_for_user(hyperparameters_combination)
            sample_weights = np.empty(train_data_dict["X_train"].shape[0], dtype=np.float64)
            train_posrated_n = train_data_dict["pos_idxs"].shape[0]
            train_negrated_n = train_data_dict["neg_idxs"].shape[0]
            cache_n = train_data_dict["cache_idxs"].shape[0]
            w_p, w_n, _, _, w_c = self.wh.load_weights_for_user(
                hyperparameters=self.hyperparameters,
                hyperparameters_combination=hyperparameters_combination,
                train_posrated_n=train_posrated_n,
                train_negrated_n=train_negrated_n,
                cache_n=cache_n,
            )
            sample_weights[train_data_dict["pos_idxs"]] = w_p
            sample_weights[train_data_dict["neg_idxs"]] = w_n
            sample_weights[train_data_dict["cache_idxs"]] = w_c
            model.fit(
                train_data_dict["X_train"],
                train_data_dict["y_train"],
                sample_weight=sample_weights,
            )
            user_models.append(model)
        return user_models
