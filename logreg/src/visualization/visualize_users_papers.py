import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[3]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_src_paths_to_sys()

import argparse, os, pickle
from enum import Enum
from matplotlib.backends.backend_pdf import PdfPages

from compute_cosine import compute_cosine_similarities
from results_handling import average_over_folds, average_over_folds_with_std
from visualize_globally import Global_Visualizer
from visualization_tools import *

class Users_Selection_Criterion(Enum):
    PERFORMANCE_ON_BEST_GLOBAL = 1
    PERFORMANCE_ON_BEST_INDIVIDUAL = 2
    PERFORMANCE_ON_BEST_EITHER = 3

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='User Visualization Parameters')
    parser.add_argument('--outputs_folder', type = Path)
    parser.add_argument('--n_print_interesting_users', type = str, default = '10')
    parser.add_argument('--visualize_worst_users', action = 'store_true', default = False)
    parser.add_argument('--visualize_best_users', action = 'store_true', default = False)
    parser.add_argument('--specific_users', type = int, nargs = '+', default = [])
    parser.add_argument('--use_tfidf_coefs', action = 'store_true', default = False)
    parser.add_argument('--visualize_best_global-hyperparameters_combination', action = 'store_true', default = True)
    parser.add_argument('--visualize_best_individual-hyperparameters_combination', action = 'store_true', default = False)
    parser.add_argument('--fold_idx', type = str, default = '0')
    parser.add_argument('--worst_users_selection_criterion', type = str, choices=[e.value for e in Users_Selection_Criterion], default = Users_Selection_Criterion.PERFORMANCE_ON_BEST_EITHER.value)
    parser.add_argument('--best_users_selection_criterion', type = str, choices=[e.value for e in Users_Selection_Criterion], default = Users_Selection_Criterion.PERFORMANCE_ON_BEST_EITHER.value)
    args = parser.parse_args()    
    args.outputs_folder = args.outputs_folder if args.outputs_folder[-1] != "/" else args.outputs_folder[:-1]
    args.n_print_interesting_users = int(args.n_print_interesting_users) if args.n_print_interesting_users != "all" else "all"
    args.worst_users_selection_criterion = Users_Selection_Criterion(args.worst_users_selection_criterion)
    args.best_users_selection_criterion = Users_Selection_Criterion(args.best_users_selection_criterion)
    return args

USERS_VISUALIZATION_CONSTANTS = {"N_TFIDF_FEATURES": 5000}

class User_Papers_Visualizer:
    def __init__(self, user_id : int) -> None:
        self.user_id = user_id
        self.user_info = gv.users_info[gv.users_info["user_id"] == user_id]
        self.use_tfidf_coefs = args.use_tfidf_coefs
        self.extract_results_for_user()

    def extract_results_for_user(self) -> None:
        self.best_individual_hyperparameters_combination_idx = (gv.best_individual_hyperparameters_combination_df.loc
                                                                [gv.best_individual_hyperparameters_combination_df["user_id"] == self.user_id]["combination_idx"].values[0])
        user_results_before_averaging_over_folds = results_before_averaging_over_folds[results_before_averaging_over_folds["user_id"] == self.user_id]
        self.user_results_before_averaging_over_folds_best_global_hyperparameters_combination = (user_results_before_averaging_over_folds.loc
                                                                                                [user_results_before_averaging_over_folds["combination_idx"] == gv.best_global_hyperparameters_combination_idx])
        self.user_results_before_averaging_over_folds_best_individual_hyperparameters_combination = (user_results_before_averaging_over_folds.loc
                                                                                                    [user_results_before_averaging_over_folds["combination_idx"] == self.best_individual_hyperparameters_combination_idx])
        self.user_results_after_averaging_over_folds_with_std = average_over_folds_with_std(user_results_before_averaging_over_folds)
        self.user_results_after_averaging_over_folds_best_global_hyperparameters_combination = average_over_folds(self.user_results_before_averaging_over_folds_best_global_hyperparameters_combination)
        self.user_results_after_averaging_over_folds_best_individual_hyperparameters_combination = average_over_folds(self.user_results_before_averaging_over_folds_best_individual_hyperparameters_combination) 

    def load_cosine_similarities_for_user(self) -> None:
        cosine_folder = f"{gv.config['embedding_folder']}/cosine_similarities/user_{self.user_id}"
        self.cosine_similarities = {"similarities": np.load(f"{cosine_folder}/cosine.npy")}
        if not gv.config['embedding_is_sparse']:
            self.cosine_similarities["similarities"] = (self.cosine_similarities["similarities"] + 1) / 2
        with open(f"{cosine_folder}/posrated_ids.pkl", 'rb') as file:
            self.cosine_similarities["posrated_ids"] = pickle.load(file)
        with open(f"{cosine_folder}/negrated_ids.pkl", 'rb') as file:
            self.cosine_similarities["negrated_ids"] = pickle.load(file)

    def visualize_user(self, folder : str) -> None:
        with open(f'{gv.folder}/users_predictions/user_{self.user_id}/user_predictions.json', 'r') as file:
            self.user_predictions = json.load(file)
        self.load_cosine_similarities_for_user()
        self.visualize_papers_for_user(folder)

    def visualize_papers_for_user(self, folder : str) -> None:
        folds_idxs = range(gv.n_folds) if args.fold_idx.lower() == "all" else [args.fold_idx]
        for fold_idx in folds_idxs:
            fold_predictions = self.user_predictions[str(fold_idx)]
            wc_pos_train_scores, wc_neg_train_scores, wc_words_scores = None, None, None
            if not self.use_tfidf_coefs:
                wc_pos_train_scores, wc_neg_train_scores = train_wordclouds(fold_predictions["train_ids"], fold_predictions["train_labels"], fold_predictions["val_ids"],
                                                                            self.user_predictions["base_ids"], USERS_VISUALIZATION_CONSTANTS["N_TFIDF_FEATURES"])
                wc_words_scores = merge_tfidf_scores(wc_pos_train_scores, wc_neg_train_scores)
            if args.visualize_best_global_hyperparameters_combination:
                self.visualize_papers_for_fold_category(folder, fold_idx, fold_predictions, True, wc_pos_train_scores, wc_neg_train_scores, wc_words_scores)
            if args.visualize_best_individual_hyperparameters_combination:
                self.visualize_papers_for_fold_category(folder, fold_idx, fold_predictions, False, wc_pos_train_scores, wc_neg_train_scores, wc_words_scores)
                
    def visualize_papers_for_fold_category(self, folder : str, fold_idx : int, fold_predictions : pd.DataFrame, best_global_hyperparameters_combination : bool,
                                           wc_pos_train_scores : dict, wc_neg_train_scores : dict, wc_words_scores : dict) -> None:
        if best_global_hyperparameters_combination:
            fold_user_file = f"{folder}/global_{fold_idx}{'_tfidf' if self.use_tfidf_coefs else ''}_{self.user_id}.pdf"
            fold_train_predictions = fold_predictions["train_predictions"][str(gv.best_global_hyperparameters_combination_idx)]
            fold_val_predictions = fold_predictions["val_predictions"][str(gv.best_global_hyperparameters_combination_idx)]
            if self.use_tfidf_coefs:
                fold_tfidf_coefs = fold_predictions["tfidf_coefs"][str(gv.best_global_hyperparameters_combination_idx)]
                tfidf_coefs_scores = get_tfidf_coefs_scores(gv.feature_names, fold_tfidf_coefs)
                wc_pos_train_scores_tfidf_params, wc_neg_train_scores_tfidf_params, wc_words_scores_tfidf_params, largest_pos_tfidf_coef, largest_neg_tfidf_coef = tfidf_coefs_scores        
        else:
            fold_user_file = f"{folder}/individual_{fold_idx}{'_tfidf' if self.use_tfidf_coefs else ''}_{self.user_id}.pdf"
            fold_train_predictions = fold_predictions["train_predictions"][str(self.best_individual_hyperparameters_combination_idx)]
            fold_val_predictions = fold_predictions["val_predictions"][str(self.best_individual_hyperparameters_combination_idx)]
            if self.use_tfidf_coefs:
                fold_tfidf_coefs = fold_predictions["tfidf_coefs"][str(self.best_individual_hyperparameters_combination_idx)]
                tfidf_coefs_scores = get_tfidf_coefs_scores(gv.feature_names, fold_tfidf_coefs)
                wc_pos_train_scores_tfidf_params, wc_neg_train_scores_tfidf_params, wc_words_scores_tfidf_params, largest_pos_tfidf_coef, largest_neg_tfidf_coef = tfidf_coefs_scores
        fold_train_predictions_df = turn_predictions_into_df(fold_train_predictions, fold_predictions["train_ids"], fold_predictions["train_labels"])
        fold_val_predictions_df = turn_predictions_into_df(fold_val_predictions, fold_predictions["val_ids"], fold_predictions["val_labels"])

        pos_train_papers_selection, n_pos_train_papers_full = randomly_select_training_papers_label(fold_train_predictions_df, True)
        neg_train_papers_selection, n_neg_train_papers_full = randomly_select_training_papers_label(fold_train_predictions_df, False)

        true_pos_val_papers_selection, n_true_pos_val_papers_full = select_n_most_extreme_val_papers_outcome(fold_val_predictions_df, Classification_Outcome.TRUE_POSITIVE, True)
        true_neg_val_papers_selection, n_true_neg_val_papers_full = select_n_most_extreme_val_papers_outcome(fold_val_predictions_df, Classification_Outcome.TRUE_NEGATIVE, False)
        false_pos_val_papers_selection, n_false_pos_val_papers_full = select_n_most_extreme_val_papers_outcome(fold_val_predictions_df, Classification_Outcome.FALSE_POSITIVE, True)
        false_neg_val_papers_selection, n_false_neg_val_papers_full = select_n_most_extreme_val_papers_outcome(fold_val_predictions_df, Classification_Outcome.FALSE_NEGATIVE, False)

        with PdfPages(fold_user_file) as pdf:
            if self.use_tfidf_coefs:
                generate_wordclouds(pdf = pdf, wc_pos_train_scores = wc_pos_train_scores_tfidf_params, wc_neg_train_scores = wc_neg_train_scores_tfidf_params, use_tfidf_coefs = True, 
                                    largest_pos_tfidf_coef = largest_pos_tfidf_coef, largest_neg_tfidf_coef = largest_neg_tfidf_coef)
                plot_training_papers(pdf, pos_train_papers_selection, n_pos_train_papers_full, neg_train_papers_selection, n_neg_train_papers_full, wc_words_scores_tfidf_params)
                plot_true_validation_papers(pdf, true_pos_val_papers_selection, n_true_pos_val_papers_full, true_neg_val_papers_selection, n_true_neg_val_papers_full, wc_words_scores_tfidf_params)
                plot_false_validation_papers(pdf, false_pos_val_papers_selection, n_false_pos_val_papers_full, false_neg_val_papers_selection, n_false_neg_val_papers_full, wc_words_scores_tfidf_params,
                                             self.cosine_similarities, pos_train_papers_selection, neg_train_papers_selection)
            else:
                generate_wordclouds(pdf = pdf, wc_pos_train_scores = wc_pos_train_scores, wc_neg_train_scores = wc_neg_train_scores, use_tfidf_coefs = False,
                                    include_base = gv.config["include_base"], n_pos_train_papers_full = n_pos_train_papers_full, n_neg_train_papers_full = n_neg_train_papers_full)
                plot_training_papers(pdf, pos_train_papers_selection, n_pos_train_papers_full, neg_train_papers_selection, n_neg_train_papers_full, wc_words_scores)
                plot_true_validation_papers(pdf, true_pos_val_papers_selection, n_true_pos_val_papers_full, true_neg_val_papers_selection, n_true_neg_val_papers_full, wc_words_scores)
                plot_false_validation_papers(pdf, false_pos_val_papers_selection, n_false_pos_val_papers_full, false_neg_val_papers_selection, n_false_neg_val_papers_full, wc_words_scores,
                                             self.cosine_similarities, pos_train_papers_selection, neg_train_papers_selection)
        
def select_interesting_users(worst_users : bool) -> list:
    selection_criterion = args.worst_users_selection_criterion if worst_users else args.best_users_selection_criterion
    max_n = args.n_print_interesting_users if args.n_print_interesting_users != "all" else len(gv.users_ids)
    if selection_criterion == Users_Selection_Criterion.PERFORMANCE_ON_BEST_GLOBAL:
        if worst_users:
            interesting_users = gv.best_global_hyperparameters_combination_df.nsmallest(max_n, f'val_{gv.score.name.lower()}')["user_id"].values
        else:
            interesting_users = gv.best_global_hyperparameters_combination_df.nlargest(max_n, f'val_{gv.score.name.lower()}')["user_id"].values
    elif selection_criterion == Users_Selection_Criterion.PERFORMANCE_ON_BEST_INDIVIDUAL:
        if worst_users:
            interesting_users = gv.best_individual_hyperparameters_combination_df.nsmallest(max_n, f'val_{gv.score.name.lower()}')["user_id"].values
        else:
            interesting_users = gv.best_individual_hyperparameters_combination_df.nlargest(max_n, f'val_{gv.score.name.lower()}')["user_id"].values
    elif selection_criterion == Users_Selection_Criterion.PERFORMANCE_ON_BEST_EITHER:
        if worst_users:
            interesting_users = np.concatenate((gv.best_global_hyperparameters_combination_df.nsmallest(max_n, f'val_{gv.score.name.lower()}')["user_id"].values,
                                                gv.best_individual_hyperparameters_combination_df.nsmallest(max_n, f'val_{gv.score.name.lower()}')["user_id"].values))
        else:
            interesting_users = np.concatenate((gv.best_global_hyperparameters_combination_df.nlargest(max_n, f'val_{gv.score.name.lower()}')["user_id"].values,
                                                gv.best_individual_hyperparameters_combination_df.nlargest(max_n, f'val_{gv.score.name.lower()}')["user_id"].values))
    return interesting_users.tolist()

def visualize_interesting_users() -> None:
    folder = f"{gv.folder}/users_visualizations"
    if not os.path.exists(folder):
        os.makedirs(folder)
    worst_users_ids = select_interesting_users(worst_users = True) if args.visualize_worst_users else []
    best_users_ids = select_interesting_users(worst_users = False) if args.visualize_best_users else []
    specific_users_ids = args.specific_users
    interesting_users_ids = sorted(list(set(worst_users_ids + best_users_ids + specific_users_ids)))
    compute_cosine_similarities(gv.config["embedding_folder"], interesting_users_ids)
    for user_id in interesting_users_ids:
        try:
            user_folder = f"{folder}/user_{user_id}"
            if not os.path.exists(user_folder):
                os.makedirs(user_folder)
            uv = User_Papers_Visualizer(user_id)
            uv.visualize_user(user_folder)
        except Exception as e:
            print(f"Error for visualizations of user {user_id}: {e}. Skipped.")
    
if __name__ == '__main__':
    args = parse_args()
    config, users_info, hyperparameters_combinations, results_before_averaging_over_folds = load_outputs_files(args.outputs_folder)
    gv = Global_Visualizer(config, users_info, hyperparameters_combinations, results_before_averaging_over_folds, args.outputs_folder)
    if args.use_tfidf_coefs:
        with open(f"{gv.folder}/feature_names.pkl", "rb") as f:
            gv.feature_names = pickle.load(f)
    visualize_interesting_users()