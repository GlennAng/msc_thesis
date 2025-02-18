from algorithm import Algorithm, Evaluation, Score, SCORES_ABBREVIATIONS_DICT, SCORES_INCREASE_BETTER_DICT, get_score_from_arg
from data_handling import Paper_Removal
from results_handling import *
from visualization_tools import *

from matplotlib.backends.backend_pdf import PdfPages
import argparse
import sys

OPTIMIZATION_CONSTANTS = {"N_TAIL_USERS": 15, "N_PRINT_BEST_HYPERPARAMETERS_COMBINATIONS": 25}

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description = "Global Visualization Parameters")
    parser.add_argument("--outputs_folder", type = str)
    parser.add_argument("--score", type = str, default = "balanced_accuracy")
    parser.add_argument("--optimize_tail", action = "store_true", default = False)
    parser.add_argument("--no-optimize_tail", action = "store_false", dest = "optimize_tail")
    args_dict = vars(parser.parse_args())
    args_dict["outputs_folder"] = args_dict["outputs_folder"].rstrip("/")
    args_dict["score"] = get_score_from_arg(args_dict["score"])
    return args_dict

class Global_Visualizer:
    def __init__(self, config : dict, users_info : pd.DataFrame, hyperparameters_combinations : pd.DataFrame, results_before_averaging_over_folds : pd.DataFrame, 
                 folder : str, score : Score = Score.BALANCED_ACCURACY, tail : bool = False) -> None:
        self.config = config
        self.users_info = clean_users_info(users_info, config["include_base"], config["include_cache"])
        self.hyperparameters_combinations = hyperparameters_combinations
        self.results_before_averaging_over_folds = results_before_averaging_over_folds
        self.folder = folder.rstrip("/")
        self.score, self.tail = score, tail
        self.extract_users_data()
        self.extract_optimization_data()
        self.extract_results_data()
        self.extract_best_global_hyperparameters_combination_data()
        self.extract_best_individual_hyperparameters_combination_data()
        self.extract_largest_performance_gain_data()

    def extract_users_data(self) -> None:
        self.n_folds = self.results_before_averaging_over_folds["fold_idx"].nunique()
        self.users_ids = sorted(list(self.users_info["user_id"].unique()))
        self.n_users = len(self.users_ids)
        self.n_tail_users = min(OPTIMIZATION_CONSTANTS["N_TAIL_USERS"], self.n_users)
        self.n_print_interesting_users = min(15, self.n_users)
        self.n_print_largest_performance_gain = min(32, self.n_users)

    def extract_optimization_data(self) -> None:
        self.hyperparameters = list(self.config["hyperparameters"].keys())
        self.hyperparameters_ranges = get_hyperparameters_ranges(self.hyperparameters_combinations)
        self.n_print_best_hyperparameters_combinations = min(OPTIMIZATION_CONSTANTS["N_PRINT_BEST_HYPERPARAMETERS_COMBINATIONS"], len(self.hyperparameters_combinations))

    def extract_results_data(self) -> None:
        self.results_after_averaging_over_folds = average_over_folds(self.results_before_averaging_over_folds)
        self.results_after_averaging_over_users = average_over_users(self.results_after_averaging_over_folds)
        self.results_after_averaging_over_tails = average_over_n_most_extreme_users_for_all_hyperparameters_combinations(self.results_after_averaging_over_folds, self.n_tail_users, True)
        self.val_upper_bounds = get_val_upper_bounds(self.results_after_averaging_over_folds, self.n_tail_users)

    def extract_best_global_hyperparameters_combination_data(self) -> None:
        self.best_global_hyperparameters_combinations_idxs = get_n_best_hyperparameters_combinations_score(self.results_after_averaging_over_tails if self.tail else self.results_after_averaging_over_users,
                                                                                                      self.score, self.n_print_best_hyperparameters_combinations)
        self.best_global_hyperparameters_combination_idx = self.best_global_hyperparameters_combinations_idxs[0]
        self.best_global_hyperparameters_combination_df = self.results_after_averaging_over_folds[self.results_after_averaging_over_folds["combination_idx"] == self.best_global_hyperparameters_combination_idx]

        if SCORES_INCREASE_BETTER_DICT[self.score]:
            self.worst_users_best_global_hyperparameters_combination_df = self.best_global_hyperparameters_combination_df.nsmallest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
            self.best_users_best_global_hyperparameters_combination_df = self.best_global_hyperparameters_combination_df.nlargest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
        else:
            self.worst_users_best_global_hyperparameters_combination_df = self.best_global_hyperparameters_combination_df.nlargest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
            self.best_users_best_global_hyperparameters_combination_df = self.best_global_hyperparameters_combination_df.nsmallest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
        
        """
        users = [3613, 6537, 17739, 18417]
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', None)        # Don't limit the display width
        pd.set_option('display.max_colwidth', None)
        filtered_df = self.best_global_hyperparameters_combination_df[
            self.best_global_hyperparameters_combination_df['user_id'].isin(users)]
        merged_df = filtered_df.merge(self.users_info, on='user_id', how='left')
        print(merged_df[["user_id", "combination_idx", "n_posrated", "n_negrated", "val_balanced_accuracy", "train_balanced_accuracy", "val_recall", "train_recall", 
                         "val_specificity", "train_specificity", "val_cel", "train_cel"]])
        """
        
        
        self.worst_users_best_global_hyperparameters_combination = self.worst_users_best_global_hyperparameters_combination_df["user_id"].values
        self.best_users_best_global_hyperparameters_combination = self.best_users_best_global_hyperparameters_combination_df["user_id"].values
        self.hyperparameters_combinations_with_explicit_X_hyperparameter = get_hyperparameters_combinations_with_explicit_X_hyperparameter(self.hyperparameters, self.hyperparameters_combinations)
        self.globally_optimal_X_hyperparameter = (self.hyperparameters_combinations[self.hyperparameters_combinations["combination_idx"] == 
                                                                                    self.best_global_hyperparameters_combination_idx][PLOT_CONSTANTS["X_HYPERPARAMETER"]].values[0])

    def extract_best_individual_hyperparameters_combination_data(self) -> None:
        self.best_individual_hyperparameters_combination_df = keep_only_n_most_extreme_hyperparameters_combinations_for_all_users_score(self.results_after_averaging_over_folds, self.score, 1, False)
        if SCORES_INCREASE_BETTER_DICT[self.score]:
            self.worst_users_best_individual_hyperparameters_combination_df = self.best_individual_hyperparameters_combination_df.nsmallest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
            self.best_users_best_individual_hyperparameters_combination_df = self.best_individual_hyperparameters_combination_df.nlargest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
        else:
            self.worst_users_best_individual_hyperparameters_combination_df = self.best_individual_hyperparameters_combination_df.nlargest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
            self.best_users_best_individual_hyperparameters_combination_df = self.best_individual_hyperparameters_combination_df.nsmallest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
        self.worst_users_best_individual_hyperparameters_combination = self.worst_users_best_individual_hyperparameters_combination_df["user_id"].values
        self.best_users_best_individual_hyperparameters_combination = self.best_users_best_individual_hyperparameters_combination_df["user_id"].values
        """
        users = [3613, 6537, 17739, 18417]
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', None)        # Don't limit the display width
        pd.set_option('display.max_colwidth', None)
        filtered_df = self.best_individual_hyperparameters_combination_df[
            self.best_individual_hyperparameters_combination_df['user_id'].isin(users)]
        merged_df = filtered_df.merge(self.users_info, on='user_id', how='left')
        print(merged_df[["user_id", "combination_idx", "n_posrated", "n_negrated", "val_balanced_accuracy", "train_balanced_accuracy", "val_recall", "train_recall", 
        "val_specificity", "train_specificity", "val_cel", "train_cel", "val_cel_pos", "train_cel_pos", "val_cel_neg", "train_cel_neg"]])
        """

    def extract_largest_performance_gain_data(self) -> None:
        column_renames_global = {f'val_{score.name.lower()}': f'val_{score.name.lower()}_global' for score in list(Score)}
        column_renames_individual = {f'val_{score.name.lower()}': f'val_{score.name.lower()}_individual' for score in list(Score)}
        best_global_hyperparameters_combination_df = self.best_global_hyperparameters_combination_df.rename(columns = column_renames_global)
        best_global_hyperparameters_combination_df = best_global_hyperparameters_combination_df.drop(columns = ["combination_idx"] + [f'train_{score.name.lower()}' for score in list(Score)])
        best_individual_hyperparameters_combination_df = self.best_individual_hyperparameters_combination_df.rename(columns = column_renames_individual)
        best_individual_hyperparameters_combination_df = best_individual_hyperparameters_combination_df.drop(columns = [f'train_{score.name.lower()}' for score in list(Score)])

        self.largest_performance_gain_df = best_global_hyperparameters_combination_df.merge(best_individual_hyperparameters_combination_df, on = "user_id")
        for score in list(Score):
            self.largest_performance_gain_df[f'{score.name.lower()}_gain'] = np.abs(
                self.largest_performance_gain_df[f'val_{score.name.lower()}_individual'] - self.largest_performance_gain_df[f'val_{score.name.lower()}_global'])
        self.largest_performance_gain_df = self.largest_performance_gain_df.drop(columns = [f'val_{score.name.lower()}_individual' for score in list(Score)])
        self.largest_performance_gain_df = self.largest_performance_gain_df.merge(self.users_info, on = "user_id")
        self.largest_performance_gain_df = self.largest_performance_gain_df.nlargest(self.n_print_largest_performance_gain, f'{self.score.name.lower()}_gain')

    def get_config_str(self) -> str:
        config_string = []
        config_string.append(f"Embedding Folder: '{self.config['embedding_folder'].split('/')[-1]}'.")
        config_string.append(f"Embedding Info: {'Sparse' if self.config['embedding_is_sparse'] else 'Dense'} with {self.config['embedding_n_dimensions']} Dimensions.")
        config_string.append(f"Database: '{self.config['db_name']}' (Backup Date: {self.config['db_backup_date']}).")
        if "time_elapsed" in self.config:
            config_string.append(f"Time Elapsed: {(self.config['time_elapsed'] / 60):.2f} Minutes.")

        if type(self.config["algorithm"]) == str:
            self.config["algorithm"] = Algorithm[self.config["algorithm"]]
        if self.config["algorithm"] == Algorithm.LOGREG:
            config_string.append(f"Alogrithm: Logistic Regression.")
            config_string.append(f"Logistic Regression Solver: {self.config['logreg_solver'].capitalize()}.")
        elif self.config["algorithm"] == Algorithm.SVM:
            config_string.append(f"Algorithm: Support Vector Machine.")
            config_string.append(f"SVM Kernel: {self.config['svm_kernel'].capitalize()}.")
        config_string.append(f"Maximum Number of Optimization iterations: {self.config['max_iter']}.")

        if type(self.config["evaluation"]) == str:
            self.config["evaluation"] = Evaluation[self.config["evaluation"]]
        if self.config["evaluation"] == Evaluation.CROSS_VALIDATION:
            config_string.append(f"Evaluation Method: Cross-Validation.")
            config_string.append(f"Number of Cross-Validation Folds: {self.config['k_folds']}.")
        elif self.config["evaluation"] == Evaluation.TRAIN_TEST_SPLIT:
            config_string.append(f"Evaluation Method: Train-Test Split.")
            config_string.append(f"Test Size: {self.config['test_size']}.")
        config_string.append(f"Were the Training Sets stratified? {'Yes' if self.config['stratified'] else 'No'}.")
        config_string.append(f"Random State: {self.config['random_state']}.")
        config_string.append("\n")

        config_string.append(f"Weights: {self.config['weights'].upper()}.")
        config_string.append(f"Training Data includes Base: {'Yes' if self.config['include_base'] else 'No'}.")
        include_cache = self.config["include_cache"]
        config_string.append(f"Training Data includes Cache: {'Yes' if include_cache else 'No'}.")
        if include_cache:
            config_string.append(get_cache_type_str(self.config["cache_type"], self.config["max_cache"]))
        config_string.append("\n")

        config_string.append(f"Minimum Number of required negative Votes per User: {self.config['min_n_negrated']}.")
        config_string.append(f"Minimum Number of required positive Votes per User: {self.config['min_n_posrated']}.")
        config_string.append(f"Number of selected Users: {self.n_users}.")
        config_string.append(get_users_selection_str(self.config["users_selection"], self.users_ids))

        if type(self.config["rated_paper_removal"]) == str:
            self.config["rated_paper_removal"] = Paper_Removal[self.config["rated_paper_removal"]]
        if type(self.config["base_paper_removal"]) == str:
            self.config["base_paper_removal"] = Paper_Removal[self.config["base_paper_removal"]]
        if self.config["rated_paper_removal"] != Paper_Removal.NONE or self.config["base_paper_removal"] != Paper_Removal.NONE:
            config_string.append("\n")
            config_string.append(f"Rated Papers Removal: {str(self.config['rated_paper_removal']).capitalize()} (remaining percentage {self.config['remaining_percentage']}).")
            config_string.append(f"Base Papers Removal: {str(self.config['base_paper_removal']).capitalize()} (remaining percentage {self.config['remaining_percentage']}).")
        return "\n\n".join(config_string)

    def generate_first_page(self, pdf : PdfPages) -> None:
        config_str = self.get_config_str()
        print_first_page(pdf, config_str)

    def generate_second_page(self, pdf : PdfPages) -> None:
        hyperparameters_ranges_str = get_hyperparameters_ranges_str(self.hyperparameters_ranges)
        print_second_page(pdf, hyperparameters_ranges_str)

    def generate_third_page(self, pdf : PdfPages) -> None:
        users_info_table = get_users_info_table(self.users_info)
        print_third_page(pdf, self.n_users, users_info_table, self.config["users_selection"], self.users_ids)

    def generate_fourth_page(self, pdf : PdfPages) -> None:
        hyperparameters_combinations_table = get_hyperparameters_combinations_table(self.val_upper_bounds, self.score, self.tail, self.best_global_hyperparameters_combinations_idxs, 
                                                                                    self.results_after_averaging_over_users, self.results_after_averaging_over_tails)
        print_fourth_page(pdf, hyperparameters_combinations_table, self.score, self.tail, self.n_users, self.n_tail_users)

    def generate_fifth_page(self, pdf : PdfPages) -> None:
        title = "Worst"
        interesting_users_best_global_hyperparameters_combination_df = self.worst_users_best_global_hyperparameters_combination_df
        best_global_merged_with_users_info = interesting_users_best_global_hyperparameters_combination_df.merge(self.users_info, on = "user_id")
        interesting_users_best_individual_hyperparameters_combination_df = self.worst_users_best_individual_hyperparameters_combination_df
        best_individual_merged_with_users_info = interesting_users_best_individual_hyperparameters_combination_df.merge(self.users_info, on = "user_id")
        print_interesting_users(pdf, self.score, title, best_global_merged_with_users_info, best_individual_merged_with_users_info)

    def generate_sixth_page(self, pdf : PdfPages) -> None:
        title = "Best"
        interesting_users_best_global_hyperparameters_combination_df = self.best_users_best_global_hyperparameters_combination_df
        best_global_merged_with_users_info = interesting_users_best_global_hyperparameters_combination_df.merge(self.users_info, on = "user_id")
        interesting_users_best_individual_hyperparameters_combination_df = self.best_users_best_individual_hyperparameters_combination_df
        best_individual_merged_with_users_info = interesting_users_best_individual_hyperparameters_combination_df.merge(self.users_info, on = "user_id")
        print_interesting_users(pdf, self.score, title, best_global_merged_with_users_info, best_individual_merged_with_users_info)

    def generate_seventh_page(self, pdf : PdfPages) -> None:
        print_largest_performance_gain(pdf, self.largest_performance_gain_df, self.score, self.best_global_hyperparameters_combination_idx)

    def generate_plots(self, pdf : PdfPages) -> None:
        plot_df = get_plot_df(self.results_after_averaging_over_users, self.hyperparameters_combinations_with_explicit_X_hyperparameter)
        plot_tail_df = get_plot_df(self.results_after_averaging_over_tails, self.hyperparameters_combinations_with_explicit_X_hyperparameter)
        plot_hyperparameter_for_all_combinations(pdf, self.hyperparameters, self.hyperparameters_combinations_with_explicit_X_hyperparameter, plot_df, plot_tail_df)

    def generate_pdf(self):
        file_name = f"{self.folder}/global_visu_{SCORES_ABBREVIATIONS_DICT[self.score]}.pdf"
        with PdfPages(file_name) as pdf:
            self.generate_first_page(pdf)
            self.generate_second_page(pdf)
            self.generate_third_page(pdf)
            self.generate_fourth_page(pdf)
            self.generate_fifth_page(pdf)
            self.generate_sixth_page(pdf)
            self.generate_seventh_page(pdf)
            self.generate_plots(pdf)

if __name__ == '__main__':
    args_dict = parse_args()
    config, users_info, hyperparameters_combinations, results_before_averaging_over_folds = load_outputs_files(args_dict["outputs_folder"])
    global_visualizer = Global_Visualizer(config, users_info, hyperparameters_combinations, results_before_averaging_over_folds, args_dict["outputs_folder"],
                                          args_dict["score"], args_dict["optimize_tail"])
    global_visualizer.generate_pdf()