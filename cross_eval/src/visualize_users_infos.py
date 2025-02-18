from results_handling import average_over_folds, average_over_folds_with_std
from visualize_globally import Global_Visualizer
from visualization_tools import *
from matplotlib.backends.backend_pdf import PdfPages
import os
import sys

class User_Info_Visualizer:
    def __init__(self, user_id : int) -> None:
        self.user_id = user_id
        self.user_info = gv.users_info[gv.users_info["user_id"] == user_id]
        self.extract_results_for_user()
        self.folder = f"{gv.folder}/users_visualizations/user_{self.user_id}"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder, exist_ok = True)

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

    def visualize_user_info(self) -> None:
        file_name = f'{self.folder}/info_{self.user_id}.pdf'
        with PdfPages(file_name) as pdf:
            self.generate_first_page_for_user(pdf)
            self.generate_plots_for_user(pdf)

    def generate_first_page_for_user(self, pdf : PdfPages) -> None:
        user_info_table = get_user_info_table(self.user_info)
        user_optimizer_column = get_optimizer_column(gv.score, gv.tail) + (1 if not gv.tail else 0)
        user_folds_table_best_global_hyperparameters_combination = get_user_folds_table(self.user_results_before_averaging_over_folds_best_global_hyperparameters_combination,
                                                                                        self.user_results_after_averaging_over_folds_best_global_hyperparameters_combination)
        user_folds_table_best_individual_hyperparameters_combination = get_user_folds_table(self.user_results_before_averaging_over_folds_best_individual_hyperparameters_combination,
                                                                                            self.user_results_after_averaging_over_folds_best_individual_hyperparameters_combination)
        print_first_page_for_user(pdf, self.user_id, user_info_table, user_optimizer_column,
                                  user_folds_table_best_global_hyperparameters_combination, user_folds_table_best_individual_hyperparameters_combination,
                                  gv.best_global_hyperparameters_combination_idx, self.best_individual_hyperparameters_combination_idx)
        

    def generate_plots_for_user(self, pdf : PdfPages) -> None:
        plot_df_user = get_plot_df(self.user_results_after_averaging_over_folds_with_std, gv.hyperparameters_combinations_with_explicit_X_hyperparameter)
        plot_hyperparameter_for_all_combinations_for_user(pdf, plot_df_user, gv.hyperparameters, gv.hyperparameters_combinations_with_explicit_X_hyperparameter,
                                                          gv.globally_optimal_X_hyperparameter, gv.best_global_hyperparameters_combination_idx) 

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Usage: python visualize_users_info.py <outputs_folder>")
    outputs_folder = sys.argv[1]
    config, users_info, hyperparameters_combinations, results_before_averaging_over_folds = load_outputs_files(outputs_folder)
    gv = Global_Visualizer(config, users_info, hyperparameters_combinations, results_before_averaging_over_folds, outputs_folder)
    for user_id in gv.users_ids:
        user_info_visualizer = User_Info_Visualizer(user_id)
        user_info_visualizer.visualize_user_info()