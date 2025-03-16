import argparse
import json
import numpy as np
import os
import pickle
from embedding import compute_cosine_similarities
from results_handling import average_over_folds_with_std
from visualize_globally import Global_Visualizer
from visualization_tools import *
from matplotlib.backends.backend_pdf import PdfPages

def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_folder', type = str)
    parser.add_argument('--users', type = str, nargs = '+', default = "all")
    parser.add_argument('--hyperparameters_combination', type = int, default = -1)
    parser.add_argument('--folds', type = str, default = '0')
    parser.add_argument('--visualize_papers', action = 'store_true', default = True)
    args = parser.parse_args()
    return vars(args)

def preprocess_args(args : dict, gv : Global_Visualizer) -> None:
    args["folds_idxs"] = list(range(gv.n_folds)) if args["folds"].lower() == "all" else [int(args["folds"])]
    args["users_ids"] = sorted(gv.users_ids if args["users"] == "all" else [int(user_id) for user_id in args["users"]])
    if args["hyperparameters_combination"] == -1:
        args["hyperparameters_combination"] = gv.best_global_hyperparameters_combination_idx

class User_Visualizer:
    def __init__(self, user_id : int) -> None:
        self.user_id = user_id
        self.user_info = gv.users_info[gv.users_info["user_id"] == user_id]
        self.extract_results_for_user()

    def extract_results_for_user(self) -> None:
        hyperparameters_combination = args["hyperparameters_combination"]
        user_results_before_averaging_over_folds = results_before_averaging_over_folds[results_before_averaging_over_folds["user_id"] == self.user_id]
        self.user_results_before_averaging_over_folds = user_results_before_averaging_over_folds.loc[user_results_before_averaging_over_folds["combination_idx"] == hyperparameters_combination]
        self.user_results_after_averaging_over_folds = average_over_folds_with_std(self.user_results_before_averaging_over_folds)

    def visualize_user_info(self, pdf : PdfPages) -> None:
        user_info_table = get_user_info_table(self.user_info)
        columns_1, val_table_1, train_table_1, columns_2, val_table_2, train_table_2 = get_user_folds_tables(self.user_results_before_averaging_over_folds, self.user_results_after_averaging_over_folds)
        hyperparameters_combination = gv.hyperparameters_combinations[gv.hyperparameters_combinations["combination_idx"] == args["hyperparameters_combination"]]
        title = f"User <{self.user_id}> Info and Scores on Combi <{args['hyperparameters_combination']}>"
        for i, hyperparameter in enumerate(gv.hyperparameters):
            title += " (" if i == 0 else ", "
            title += HYPERPARAMETERS_ABBREVIATIONS[hyperparameter] if hyperparameter in HYPERPARAMETERS_ABBREVIATIONS else hyperparameter
            title += f" = {hyperparameters_combination[hyperparameter].values[0]}"
        title += "):"
        print_first_page_for_user(pdf, "1/2   -   " + title, user_info_table, columns_1, val_table_1, train_table_1)
        print_first_page_for_user(pdf, "2/2   -   " + title, user_info_table, columns_2, val_table_2, train_table_2)

    def visualize_wordclouds(self, pdf : PdfPages, fold_idx : int) -> dict:
        fold_predictions = self.user_predictions[str(fold_idx)]
        wc_pos_train_scores, wc_neg_train_scores = train_wordclouds(fold_predictions["train_ids"], fold_predictions["train_labels"], fold_predictions["val_ids"])
        wc_words_scores = merge_tfidf_scores(wc_pos_train_scores, wc_neg_train_scores)
        n_pos_train_papers_full = sum(fold_predictions["train_labels"])
        n_neg_train_papers_full = len(fold_predictions["train_ids"]) - n_pos_train_papers_full
        generate_wordclouds(pdf = pdf, wc_pos_train_scores = wc_pos_train_scores, wc_neg_train_scores = wc_neg_train_scores, use_tfidf_coefs = False,
                            n_pos_train_papers_full = n_pos_train_papers_full, n_neg_train_papers_full = n_neg_train_papers_full)
        return wc_words_scores

    def load_cosine_similarities_for_user(self) -> None:
        cosine_folder = f"{gv.config['embedding_folder']}/cosine_similarities/user_{self.user_id}"
        self.cosine_similarities = {}
        self.cosine_similarities["rated_cosine_similarities"] = np.load(f"{cosine_folder}/rated_cosine_similarities.npy")
        self.cosine_similarities["negative_samples_cosine_similarities"] = np.load(f"{cosine_folder}/negative_samples_cosine_similarities.npy")
        if not gv.config['embedding_is_sparse']:
            self.cosine_similarities["rated_cosine_similarities"] = (self.cosine_similarities["rated_cosine_similarities"] + 1) / 2
            self.cosine_similarities["negative_samples_cosine_similarities"] = (self.cosine_similarities["negative_samples_cosine_similarities"] + 1) / 2
        with open(f"{cosine_folder}/posrated_ids.pkl", 'rb') as file:
            self.cosine_similarities["posrated_ids"] = pickle.load(file)
        with open(f"{cosine_folder}/negrated_ids.pkl", 'rb') as file:
            self.cosine_similarities["negrated_ids"] = pickle.load(file)
        with open(f"{cosine_folder}/negative_samples_ids.pkl", 'rb') as file:
            self.cosine_similarities["negative_samples_ids"] = pickle.load(file)

    def visualize_papers(self, pdf : PdfPages, fold_idx : int, wc_words_scores : dict) -> None:
        fold_predictions = self.user_predictions[str(fold_idx)]
        hyperparameters_combination = str(args["hyperparameters_combination"])
        fold_train_predictions = fold_predictions["train_predictions"][hyperparameters_combination]
        fold_val_predictions = fold_predictions["val_predictions"][hyperparameters_combination]
        negative_samples_predictions = fold_predictions["negative_samples_predictions"][hyperparameters_combination]
        fold_train_predictions_df = turn_predictions_into_df(fold_train_predictions, fold_predictions["train_ids"], fold_predictions["train_labels"])
        fold_val_predictions_df = turn_predictions_into_df(fold_val_predictions, fold_predictions["val_ids"], fold_predictions["val_labels"])
        negative_samples_predictions_df = turn_predictions_into_df(negative_samples_predictions, self.user_predictions["negative_samples_ids"], 
                                                                  [0] * len(negative_samples_predictions))
        n_negative_samples_full = len(negative_samples_predictions_df)
        negative_samples_predictions_df = negative_samples_predictions_df.nlargest(100, "class_1_proba").sort_values("class_1_proba", ascending = False)
        negative_samples_selection = get_papers_table_data(negative_samples_predictions_df)

        pos_train_papers_selection, n_pos_train_papers_full = randomly_select_training_papers_label(fold_train_predictions_df, True)
        neg_train_papers_selection, n_neg_train_papers_full = randomly_select_training_papers_label(fold_train_predictions_df, False)
        true_pos_val_papers_selection, n_true_pos_val_papers_full = select_n_most_extreme_val_papers_outcome(fold_val_predictions_df, Classification_Outcome.TRUE_POSITIVE, True)
        true_neg_val_papers_selection, n_true_neg_val_papers_full = select_n_most_extreme_val_papers_outcome(fold_val_predictions_df, Classification_Outcome.TRUE_NEGATIVE, False)
        false_pos_val_papers_selection, n_false_pos_val_papers_full = select_n_most_extreme_val_papers_outcome(fold_val_predictions_df, Classification_Outcome.FALSE_POSITIVE, True)
        false_neg_val_papers_selection, n_false_neg_val_papers_full = select_n_most_extreme_val_papers_outcome(fold_val_predictions_df, Classification_Outcome.FALSE_NEGATIVE, False)
        pos_val_papers_selection, n_pos_val_papers_full = select_n_most_extreme_val_pos_papers(fold_val_predictions_df)

        plot_training_papers(pdf, pos_train_papers_selection, n_pos_train_papers_full, neg_train_papers_selection, n_neg_train_papers_full, wc_words_scores)
        plot_true_validation_papers(pdf, true_pos_val_papers_selection, n_true_pos_val_papers_full, true_neg_val_papers_selection, n_true_neg_val_papers_full, wc_words_scores)
        plot_false_validation_papers(pdf, false_pos_val_papers_selection, n_false_pos_val_papers_full, false_neg_val_papers_selection, n_false_neg_val_papers_full, wc_words_scores,
                                     self.cosine_similarities, pos_train_papers_selection, neg_train_papers_selection)
        plot_negative_samples(pdf, negative_samples_selection, n_negative_samples_full, pos_val_papers_selection, n_pos_val_papers_full, wc_words_scores, 
                              self.cosine_similarities, pos_train_papers_selection, neg_train_papers_selection)

if __name__ == "__main__":
    args = parse_args()
    config, users_info, hyperparameters_combinations, results_before_averaging_over_folds = load_outputs_files(args["outputs_folder"])
    gv = Global_Visualizer(config, users_info, hyperparameters_combinations, results_before_averaging_over_folds, args["outputs_folder"])
    preprocess_args(args, gv)
    if args["visualize_papers"]:
        compute_cosine_similarities(gv.config["embedding_folder"], args["users_ids"], args["outputs_folder"] + "/users_predictions")
    for user_id in args["users_ids"]:
        uv = User_Visualizer(user_id)
        with open(f'{gv.folder}/users_predictions/user_{user_id}/user_predictions.json', 'r') as file:
            uv.user_predictions = json.load(file)
        user_folder = f"{gv.folder}/users_visualizations/user_{user_id}"
        if not os.path.exists(user_folder):
            os.makedirs(user_folder, exist_ok = True)
        for fold_idx in args["folds_idxs"]:
            file_name = f"{user_folder}/user_{user_id}_fold_{fold_idx}.pdf"
            with PdfPages(file_name) as pdf:
                uv.visualize_user_info(pdf)
                wc_words_scores = uv.visualize_wordclouds(pdf, fold_idx)
                if args["visualize_papers"]:
                    uv.load_cosine_similarities_for_user()
                    uv.visualize_papers(pdf, fold_idx, wc_words_scores)