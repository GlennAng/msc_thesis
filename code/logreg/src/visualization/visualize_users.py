import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from ....src.load_files import load_papers_texts
from ..embeddings.embedding import compute_cosine_similarities
from .results_handling import average_over_folds_with_std
from .visualization_tools import (
    HYPERPARAMETERS_ABBREVIATIONS,
    Classification_Outcome,
    generate_wordclouds,
    get_papers_table_data,
    get_user_folds_tables,
    get_user_info_table,
    load_outputs_files,
    merge_tfidf_scores,
    plot_false_validation_papers,
    plot_ranking_predictions,
    plot_training_papers,
    plot_true_validation_papers,
    print_first_page_for_user,
    randomly_select_training_papers_label,
    select_n_most_extreme_val_papers_outcome,
    select_n_most_extreme_val_pos_papers,
    train_wordclouds,
    turn_predictions_into_df,
)
from .visualize_globally import Global_Visualizer


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_folder", type=Path)
    parser.add_argument("--users", type=str, nargs="+", default="all")
    parser.add_argument("--hyperparameters_combination", type=int, default=-1)
    parser.add_argument("--folds", type=str, default="0")
    parser.add_argument("--visualize_papers", action="store_true", default=True)
    args = vars(parser.parse_args())
    if not isinstance(args["outputs_folder"], Path):
        args["outputs_folder"] = Path(args["outputs_folder"]).resolve()
    return args


def preprocess_args(args: dict, gv: Global_Visualizer) -> None:
    args["folds_idxs"] = (
        list(range(gv.n_folds)) if args["folds"].lower() == "all" else [int(args["folds"])]
    )
    args["users_ids"] = sorted(
        gv.users_ids if args["users"] == "all" else [int(user_id) for user_id in args["users"]]
    )
    if args["hyperparameters_combination"] == -1:
        args["hyperparameters_combination"] = gv.best_global_hyperparameters_combination_idx


class User_Visualizer:
    def __init__(self, user_id: int) -> None:
        self.user_id = user_id
        self.user_info = gv.users_info[gv.users_info["user_id"] == user_id]
        self.extract_results_for_user()

    def extract_results_for_user(self) -> None:
        hyperparameters_combination = args["hyperparameters_combination"]
        user_results_before_averaging_over_folds = results_before_averaging_over_folds[
            results_before_averaging_over_folds["user_id"] == self.user_id
        ]
        self.user_results_before_averaging_over_folds = (
            user_results_before_averaging_over_folds.loc[
                user_results_before_averaging_over_folds["combination_idx"]
                == hyperparameters_combination
            ]
        )
        self.user_results_after_averaging_over_folds = average_over_folds_with_std(
            self.user_results_before_averaging_over_folds
        )

    def visualize_user_info(self, pdf: PdfPages) -> None:
        user_info_table = get_user_info_table(self.user_info)
        columns_list, val_table_list, train_table_list = get_user_folds_tables(
            self.user_results_before_averaging_over_folds,
            self.user_results_after_averaging_over_folds,
        )
        hyperparameters_combination = gv.hyperparameters_combinations[
            gv.hyperparameters_combinations["combination_idx"]
            == args["hyperparameters_combination"]
        ]
        title = f"User <{self.user_id}> Info and Scores on Combi <{args['hyperparameters_combination']}>"
        for i, hyperparameter in enumerate(gv.hyperparameters):
            title += " (" if i == 0 else ", "
            title += (
                HYPERPARAMETERS_ABBREVIATIONS[hyperparameter]
                if hyperparameter in HYPERPARAMETERS_ABBREVIATIONS
                else hyperparameter
            )
            title += f" = {hyperparameters_combination[hyperparameter].values[0]}"
        title += "):"
        n_tables = len(columns_list)
        for i in range(n_tables):
            columns, val_table, train_table = (
                columns_list[i],
                val_table_list[i],
                train_table_list[i],
            )
            table_title = f"{i + 1}/{n_tables}   -   " + title
            print_first_page_for_user(
                pdf, table_title, user_info_table, columns, val_table, train_table
            )

    def visualize_wordclouds(self, pdf: PdfPages, fold_idx: int) -> dict:
        fold_predictions = self.user_predictions[str(fold_idx)]
        wc_pos_train_scores, wc_neg_train_scores = train_wordclouds(
            fold_predictions["train_ids"],
            fold_predictions["train_labels"],
            fold_predictions["val_ids"],
        )
        wc_words_scores = merge_tfidf_scores(wc_pos_train_scores, wc_neg_train_scores)
        n_pos_train_papers_full = sum(fold_predictions["train_labels"])
        n_neg_train_papers_full = len(fold_predictions["train_ids"]) - n_pos_train_papers_full
        generate_wordclouds(
            pdf=pdf,
            wc_pos_train_scores=wc_pos_train_scores,
            wc_neg_train_scores=wc_neg_train_scores,
            use_tfidf_coefs=False,
            n_pos_train_papers_full=n_pos_train_papers_full,
            n_neg_train_papers_full=n_neg_train_papers_full,
        )
        return wc_words_scores

    def load_cosine_similarities_for_user(self) -> None:
        embedding_folder = Path(gv.config["embedding_folder"]).resolve()
        cosine_folder = embedding_folder / "cosine_similarities" / f"user_{self.user_id}"
        self.cosine_similarities = {}
        self.cosine_similarities["rated_cosine_similarities"] = np.load(
            cosine_folder / "rated_cosine_similarities.npy"
        )
        self.cosine_similarities["negative_samples_cosine_similarities"] = np.load(
            cosine_folder / "negative_samples_cosine_similarities.npy"
        )
        if not gv.config["embedding_is_sparse"]:
            self.cosine_similarities["rated_cosine_similarities"] = (
                self.cosine_similarities["rated_cosine_similarities"] + 1
            ) / 2
            self.cosine_similarities["negative_samples_cosine_similarities"] = (
                self.cosine_similarities["negative_samples_cosine_similarities"] + 1
            ) / 2
        with open(cosine_folder / "posrated_ids.pkl", "rb") as file:
            self.cosine_similarities["posrated_ids"] = pickle.load(file)
        with open(cosine_folder / "negrated_ids.pkl", "rb") as file:
            self.cosine_similarities["negrated_ids"] = pickle.load(file)
        with open(cosine_folder / "negative_samples_ids.pkl", "rb") as file:
            self.cosine_similarities["negative_samples_ids"] = pickle.load(file)

    def visualize_papers(self, pdf: PdfPages, fold_idx: int, wc_words_scores: dict) -> None:
        papers_texts = load_papers_texts(relevant_columns=["paper_id", "title", "abstract"])
        fold_predictions = self.user_predictions[str(fold_idx)]
        hyperparameters_combination = str(args["hyperparameters_combination"])
        fold_train_predictions = fold_predictions["train_predictions"][hyperparameters_combination]
        fold_val_predictions = fold_predictions["val_predictions"][hyperparameters_combination]
        negative_samples_predictions = fold_predictions["negative_samples_predictions"][
            hyperparameters_combination
        ]
        fold_train_predictions_df = turn_predictions_into_df(
            fold_train_predictions, fold_predictions["train_ids"], fold_predictions["train_labels"]
        )
        fold_val_predictions_df = turn_predictions_into_df(
            fold_val_predictions, fold_predictions["val_ids"], fold_predictions["val_labels"]
        )
        negative_samples_predictions_df = turn_predictions_into_df(
            negative_samples_predictions,
            self.user_predictions["negative_samples_ids"],
            [0] * len(negative_samples_predictions),
        )
        n_negative_samples_full = len(negative_samples_predictions_df)
        negative_samples_predictions_df = negative_samples_predictions_df.nlargest(
            100, "class_1_proba"
        ).sort_values("class_1_proba", ascending=False)
        negative_samples_selection = get_papers_table_data(
            papers_texts, negative_samples_predictions_df
        )

        pos_train_papers_selection, n_pos_train_papers_full = randomly_select_training_papers_label(
            papers_texts, fold_train_predictions_df, True
        )
        neg_train_papers_selection, n_neg_train_papers_full = randomly_select_training_papers_label(
            papers_texts, fold_train_predictions_df, False
        )
        true_pos_val_papers_selection, n_true_pos_val_papers_full = (
            select_n_most_extreme_val_papers_outcome(
                papers_texts, fold_val_predictions_df, Classification_Outcome.TRUE_POSITIVE, True
            )
        )
        true_neg_val_papers_selection, n_true_neg_val_papers_full = (
            select_n_most_extreme_val_papers_outcome(
                papers_texts, fold_val_predictions_df, Classification_Outcome.TRUE_NEGATIVE, False
            )
        )
        false_pos_val_papers_selection, n_false_pos_val_papers_full = (
            select_n_most_extreme_val_papers_outcome(
                papers_texts, fold_val_predictions_df, Classification_Outcome.FALSE_POSITIVE, True
            )
        )
        false_neg_val_papers_selection, n_false_neg_val_papers_full = (
            select_n_most_extreme_val_papers_outcome(
                papers_texts, fold_val_predictions_df, Classification_Outcome.FALSE_NEGATIVE, False
            )
        )
        pos_val_papers_selection, n_pos_val_papers_full = select_n_most_extreme_val_pos_papers(
            papers_texts, fold_val_predictions_df
        )

        plot_training_papers(
            pdf,
            pos_train_papers_selection,
            n_pos_train_papers_full,
            neg_train_papers_selection,
            n_neg_train_papers_full,
            wc_words_scores,
        )
        plot_true_validation_papers(
            pdf,
            true_pos_val_papers_selection,
            n_true_pos_val_papers_full,
            true_neg_val_papers_selection,
            n_true_neg_val_papers_full,
            wc_words_scores,
        )
        plot_false_validation_papers(
            pdf,
            false_pos_val_papers_selection,
            n_false_pos_val_papers_full,
            false_neg_val_papers_selection,
            n_false_neg_val_papers_full,
            wc_words_scores,
            self.cosine_similarities,
            pos_train_papers_selection,
            neg_train_papers_selection,
        )
        plot_ranking_predictions(
            pdf,
            negative_samples_selection,
            n_negative_samples_full,
            pos_val_papers_selection,
            n_pos_val_papers_full,
            wc_words_scores,
            self.cosine_similarities,
            pos_train_papers_selection,
            neg_train_papers_selection,
            is_negrated_ranking=False,
        )


if __name__ == "__main__":
    args = parse_args()
    config, users_info, hyperparameters_combinations, results_before_averaging_over_folds = (
        load_outputs_files(args["outputs_folder"])
    )
    gv = Global_Visualizer(
        config,
        users_info,
        hyperparameters_combinations,
        results_before_averaging_over_folds,
        args["outputs_folder"],
    )
    preprocess_args(args, gv)
    if args["visualize_papers"]:
        compute_cosine_similarities(
            gv.config["embedding_folder"],
            args["users_ids"],
            args["outputs_folder"] / "users_predictions",
        )
    for user_id in args["users_ids"]:
        uv = User_Visualizer(user_id)
        user_predictions_path = (
            gv.folder / "users_predictions" / f"user_{user_id}" / "user_predictions.json"
        )
        with open(user_predictions_path, "r") as file:
            uv.user_predictions = json.load(file)
        user_folder = gv.folder / "users_visualizations" / f"user_{user_id}"
        if not os.path.exists(user_folder):
            os.makedirs(user_folder, exist_ok=True)
        for fold_idx in args["folds_idxs"]:
            file_name = user_folder / f"user_{user_id}_fold_{fold_idx}.pdf"
            with PdfPages(file_name) as pdf:
                uv.visualize_user_info(pdf)
                wc_words_scores = uv.visualize_wordclouds(pdf, fold_idx)
                if args["visualize_papers"]:
                    uv.load_cosine_similarities_for_user()
                    uv.visualize_papers(pdf, fold_idx, wc_words_scores)
