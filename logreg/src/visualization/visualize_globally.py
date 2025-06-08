import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[3]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_src_paths_to_sys()

import argparse, pickle, os, sys
from matplotlib.backends.backend_pdf import PdfPages
from algorithm import Algorithm, Evaluation, Score, get_score_from_arg, SCORES_DICT
from results_handling import *
from visualization_tools import *

OPTIMIZATION_CONSTANTS = {"N_TAIL_USERS": 15, "N_PRINT_BEST_HYPERPARAMETERS_COMBINATIONS": 25, "PERCENTAGE_HI/LO_VOTES": 0.1, 
                          "PERCENTAGE_HI/LO_RATIO": 0.1}

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description = "Global Visualization Parameters")
    parser.add_argument("--outputs_folder", type = Path)
    parser.add_argument("--score", type = str, default = "balanced_accuracy")
    parser.add_argument("--optimize_tail", action = "store_true", default = False)
    parser.add_argument("--no-optimize_tail", action = "store_false", dest = "optimize_tail")
    parser.add_argument("--save_scores_tables", action = "store_true", default = False)
    args_dict = vars(parser.parse_args())
    if not isinstance(args_dict["outputs_folder"], Path):
        args_dict["outputs_folder"] = Path(args_dict["outputs_folder"])
    args_dict["score"] = get_score_from_arg(args_dict["score"])
    return args_dict

class Global_Visualizer:
    def __init__(self, config : dict, users_info : pd.DataFrame, hyperparameters_combinations : pd.DataFrame, results_before_averaging_over_folds : pd.DataFrame, 
                 folder : Path, score : Score = Score.BALANCED_ACCURACY, tail : bool = False) -> None:
        self.config = config
        self.users_info = clean_users_info(users_info, config["include_base"], config["include_cache"])
        self.hyperparameters_combinations = hyperparameters_combinations
        self.results_before_averaging_over_folds = results_before_averaging_over_folds
        self.folder = folder
        self.score, self.tail = score, tail
        self.extract_users_data()
        self.extract_optimization_data()
        self.extract_results_data()
        self.extract_best_global_hyperparameters_combination_data()
        self.extract_high_low_users()
        self.extract_best_individual_hyperparameters_combination_data()
        self.extract_largest_performance_gain_data()

    def extract_users_data(self) -> None:
        self.n_folds = self.results_before_averaging_over_folds["fold_idx"].nunique()
        self.users_ids = sorted(list(self.users_info["user_id"].unique()))
        self.n_users = len(self.users_ids)
        self.n_tail_users = min(OPTIMIZATION_CONSTANTS["N_TAIL_USERS"], self.n_users)
        self.n_print_interesting_users = min(24, self.n_users)
        self.n_print_largest_performance_gain = min(25, self.n_users)

    def extract_optimization_data(self) -> None:
        self.hyperparameters = list(self.hyperparameters_combinations.columns)[1:]
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

        if SCORES_DICT[self.score]["increase_better"]:
            self.worst_users_best_global_hyperparameters_combination_df = self.best_global_hyperparameters_combination_df.nsmallest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
            self.best_users_best_global_hyperparameters_combination_df = self.best_global_hyperparameters_combination_df.nlargest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
        else:
            self.worst_users_best_global_hyperparameters_combination_df = self.best_global_hyperparameters_combination_df.nlargest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
            self.best_users_best_global_hyperparameters_combination_df = self.best_global_hyperparameters_combination_df.nsmallest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
        
        self.worst_users_best_global_hyperparameters_combination = self.worst_users_best_global_hyperparameters_combination_df["user_id"].values
        self.best_users_best_global_hyperparameters_combination = self.best_users_best_global_hyperparameters_combination_df["user_id"].values
        self.hyperparameters_combinations_with_explicit_X_hyperparameter = get_hyperparameters_combinations_with_explicit_X_hyperparameter(self.hyperparameters, self.hyperparameters_combinations)
        self.globally_optimal_X_hyperparameter = (self.hyperparameters_combinations[self.hyperparameters_combinations["combination_idx"] == 
                                                                                    self.best_global_hyperparameters_combination_idx][PLOT_CONSTANTS["X_HYPERPARAMETER"]].values[0])

    def extract_high_low_users(self) -> None:
        n_hi_lo_votes_users = max(1, min(int(self.n_users * OPTIMIZATION_CONSTANTS["PERCENTAGE_HI/LO_VOTES"]), self.n_users))
        n_hi_lo_ratio_users = max(1, min(int(self.n_users * OPTIMIZATION_CONSTANTS["PERCENTAGE_HI/LO_RATIO"]), self.n_users))
        self.users_info["n_rated"] = self.users_info["n_posrated"] + self.users_info["n_negrated"]
        self.users_info["posrated_ratio"] = self.users_info["n_posrated"] / self.users_info["n_rated"]
        self.high_votes_users = self.users_info.nlargest(n_hi_lo_votes_users, "n_rated")["user_id"].values
        self.low_votes_users = self.users_info.nsmallest(n_hi_lo_votes_users, "n_rated")["user_id"].values
        self.high_ratio_users = self.users_info.nlargest(n_hi_lo_ratio_users, "posrated_ratio")["user_id"].values
        self.low_ratio_users = self.users_info.nsmallest(n_hi_lo_ratio_users, "posrated_ratio")["user_id"].values
        if SCORES_DICT[self.score]["increase_better"]:
            tail_df = self.best_global_hyperparameters_combination_df.nsmallest(self.n_tail_users, f'val_{self.score.name.lower()}')
        else:
            tail_df = self.best_global_hyperparameters_combination_df.nlargest(self.n_tail_users, f'val_{self.score.name.lower()}')
        self.tail_users = tail_df["user_id"].values

    def extract_best_individual_hyperparameters_combination_data(self) -> None:
        self.best_individual_hyperparameters_combination_df = keep_only_n_most_extreme_hyperparameters_combinations_for_all_users_score(self.results_after_averaging_over_folds, self.score, 1, False)
        if SCORES_DICT[self.score]["increase_better"]:
            self.worst_users_best_individual_hyperparameters_combination_df = self.best_individual_hyperparameters_combination_df.nsmallest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
            self.best_users_best_individual_hyperparameters_combination_df = self.best_individual_hyperparameters_combination_df.nlargest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
        else:
            self.worst_users_best_individual_hyperparameters_combination_df = self.best_individual_hyperparameters_combination_df.nlargest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
            self.best_users_best_individual_hyperparameters_combination_df = self.best_individual_hyperparameters_combination_df.nsmallest(self.n_print_interesting_users, f'val_{self.score.name.lower()}')
        self.worst_users_best_individual_hyperparameters_combination = self.worst_users_best_individual_hyperparameters_combination_df["user_id"].values
        self.best_users_best_individual_hyperparameters_combination = self.best_users_best_individual_hyperparameters_combination_df["user_id"].values

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
        if "time_elapsed" in self.config:
            config_string.append(f"Time Elapsed: {(self.config['time_elapsed'] / 60):.2f} Minutes.")

        if type(self.config["algorithm"]) == str:
            self.config["algorithm"] = Algorithm[self.config["algorithm"]]
        if self.config["algorithm"] == Algorithm.LOGREG:
            config_string.append(f"Algorithm: Logistic Regression.")
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
        urs, mrs, crs, rrs = self.config["users_random_state"], self.config["model_random_state"], self.config["cache_random_state"], self.config["ranking_random_state"]
        config_string.append(f"Random States:  Users: {urs}  |  Model: {mrs}  |  Cache: {crs}  |  Ranking: {rrs}.")
        config_string.append("\n")

        config_string.append(f"Weights: {self.config['weights'].upper()}.")
        config_string.append(f"Training Data includes Base: {'Yes' if self.config['include_base'] else 'No'}.")
        include_cache = self.config["include_cache"]
        config_string.append(f"Training Data includes Cache: {'Yes' if include_cache else 'No'}.")
        if include_cache:
            config_string.append(get_cache_type_str(self.config["cache_type"], self.config["max_cache"], self.config["n_cache_attached"]))
        config_string.append(f"Number of Negative Samples per User: {self.config['n_negative_samples']}.")
        config_string.append("\n")

        config_string.append(f"Minimum Number of required negative Votes per User: {self.config['min_n_negrated']}.")
        config_string.append(f"Minimum Number of required positive Votes per User: {self.config['min_n_posrated']}.")
        config_string.append(f"Number of selected Users: {self.n_users}.")
        config_string.append(get_users_selection_str(self.config["users_selection"], self.users_ids))

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
        hyperparameters_combinations_table_val = get_hyperparameters_combinations_table(self.val_upper_bounds, self.score, self.best_global_hyperparameters_combinations_idxs, 
                                                                                        self.results_after_averaging_over_users, self.hyperparameters_combinations)
        print_fourth_page(pdf, hyperparameters_combinations_table_val, self.score, self.hyperparameters)

    def generate_fifth_page(self, pdf : PdfPages, scores_tables_save_path : str = None) -> None:
        title = f"Scores of the Best Global Combi {self.best_global_hyperparameters_combination_idx}"
        best_global_hyperparameters_combination = self.hyperparameters_combinations[self.hyperparameters_combinations["combination_idx"] == self.best_global_hyperparameters_combination_idx]
        for i, hyperparameter in enumerate(self.hyperparameters):
            title += " (" if i == 0 else ", "
            title += HYPERPARAMETERS_ABBREVIATIONS[hyperparameter] if hyperparameter in HYPERPARAMETERS_ABBREVIATIONS else hyperparameter
            title += f" = {best_global_hyperparameters_combination[hyperparameter].values[0]}"
        title += "):"
        legend_text = "Legend:   "
        legend_text += f"All: All {self.n_users} Users | HiVote/LoVote: The {len(self.high_votes_users)} Users with the highest/lowest number of positively + negatively rated Papers\n"
        legend_text += f"HiPosi/LoPosi: The {len(self.high_ratio_users)} Users with the highest/lowest ratio of positively to negatively rated Papers | "
        legend_text += f"Tail: The {self.n_tail_users} Users with the worst Validation Performance on the Score in the grey Row."
        scores_tables = get_best_global_hyperparameters_combination_table(self.best_global_hyperparameters_combination_df, self.tail_users,
                                                                          self.high_votes_users, self.low_votes_users, self.high_ratio_users, self.low_ratio_users)
        score_index = list(Score).index(self.score)
        optimizer_page, optimizer_row = score_index // (n_scores_halved), score_index % (n_scores_halved)
        for s in range(len(scores_tables)):
            s_title = f"{s + 1}/{len(scores_tables)}   -   {title}"
            grey_row = optimizer_row + 1 if s == optimizer_page else -1
            scores_table_save_path = f"{scores_tables_save_path}_{s + 1}.pkl" if scores_tables_save_path else None
            print_fifth_page(pdf, s_title, legend_text, scores_tables[s], grey_row, scores_table_save_path)

    def generate_sixth_page(self, pdf : PdfPages) -> None:
        title = "Worst"
        interesting_users_best_global_hyperparameters_combination_df = self.worst_users_best_global_hyperparameters_combination_df
        best_global_merged_with_users_info = interesting_users_best_global_hyperparameters_combination_df.merge(self.users_info, on = "user_id")
        interesting_users_best_individual_hyperparameters_combination_df = self.worst_users_best_individual_hyperparameters_combination_df
        best_individual_merged_with_users_info = interesting_users_best_individual_hyperparameters_combination_df.merge(self.users_info, on = "user_id")
        print_interesting_users(pdf, self.score, title, best_global_merged_with_users_info, best_individual_merged_with_users_info)

    def generate_seventh_page(self, pdf : PdfPages) -> None:
        title = "Best"
        interesting_users_best_global_hyperparameters_combination_df = self.best_users_best_global_hyperparameters_combination_df
        best_global_merged_with_users_info = interesting_users_best_global_hyperparameters_combination_df.merge(self.users_info, on = "user_id")
        interesting_users_best_individual_hyperparameters_combination_df = self.best_users_best_individual_hyperparameters_combination_df
        best_individual_merged_with_users_info = interesting_users_best_individual_hyperparameters_combination_df.merge(self.users_info, on = "user_id")
        print_interesting_users(pdf, self.score, title, best_global_merged_with_users_info, best_individual_merged_with_users_info)

    def generate_eighth_page(self, pdf : PdfPages) -> None:
        print_largest_performance_gain(pdf, self.largest_performance_gain_df, self.score, self.best_global_hyperparameters_combination_idx)

    def generate_plots(self, pdf : PdfPages) -> None:
        plot_df = get_plot_df(self.results_after_averaging_over_users, self.hyperparameters_combinations_with_explicit_X_hyperparameter)
        plot_tail_df = get_plot_df(self.results_after_averaging_over_tails, self.hyperparameters_combinations_with_explicit_X_hyperparameter)
        plot_hyperparameter_for_all_combinations(pdf, self.hyperparameters, self.hyperparameters_combinations_with_explicit_X_hyperparameter, plot_df, plot_tail_df)

    def generate_pdf(self, save_scores_tables : bool = False) -> None:
        file_name = self.folder / f"global_visu_{SCORES_DICT[self.score]['abbreviation'].lower()}.pdf"
        scores_tables_save_path = self.folder / "scores_table" if save_scores_tables else None
        with PdfPages(file_name) as pdf:
            self.generate_first_page(pdf)
            self.generate_second_page(pdf)
            self.generate_third_page(pdf)
            self.generate_fourth_page(pdf)
            self.generate_fifth_page(pdf, scores_tables_save_path)
            self.generate_sixth_page(pdf)
            self.generate_seventh_page(pdf)
            if len(self.hyperparameters_combinations) > 1:
                self.generate_eighth_page(pdf)
                self.generate_plots(pdf)

    def print_fold_stds(self):
        results_before_averaging_over_folds = self.results_before_averaging_over_folds[self.results_before_averaging_over_folds["combination_idx"] == self.best_global_hyperparameters_combination_idx]
        results_before_averaging_over_folds = results_before_averaging_over_folds.drop(columns = ["combination_idx"])
        group_columns = ["fold_idx"]
        results_before_averaging_over_folds = results_before_averaging_over_folds.groupby(group_columns).mean()
        print(results_before_averaging_over_folds["val_balanced_accuracy"].std())
        print(results_before_averaging_over_folds["val_cel"].std())

    def print_survey_correlations(self):
        from data_handling import get_users_survey_ratings
        survey_ratings = get_users_survey_ratings()
        survey_ratings = survey_ratings.drop_duplicates(subset=["user_id"], keep = "last")
        users_ids_with_sufficient_votes = self.best_global_hyperparameters_combination_df["user_id"].values
        survey_ratings = survey_ratings[survey_ratings["user_id"].isin(users_ids_with_sufficient_votes)]
        survey_ratings = survey_ratings.merge(self.best_global_hyperparameters_combination_df, on = "user_id")
        for score in list(Score):
            print(f"Correlation between {score.name} and Survey Ratings: {survey_ratings[f'val_{score.name.lower()}'].corr(survey_ratings['survey_rating'])}")

    def print_negative_samples(self):
        outputs_folder_2 = ProjectPaths.logreg_outputs_path() / "tfidf2"
        config2, users_info2, hyperparameters_combinations2, results_before_averaging_over_folds2 = load_outputs_files(outputs_folder_2)
        global_visualizer2 = Global_Visualizer(config2, users_info2, hyperparameters_combinations2, results_before_averaging_over_folds2, outputs_folder_2,
                                                self.score, self.tail)
        df_merged = self.best_global_hyperparameters_combination_df.merge(global_visualizer2.best_global_hyperparameters_combination_df, on = "user_id")
        metric = "val_ndcg_samples"
        df_merged[metric] = df_merged[f"{metric}_x"] - df_merged[f"{metric}_y"]
        with pd.option_context('display.max_rows', None):
            print(df_merged[["user_id", f"{metric}_x", f"{metric}_y", metric]].sort_values(metric))

if __name__ == '__main__':
    args_dict = parse_args()
    if args_dict["score"] not in PRINT_SCORES:
        raise ValueError(f"Score {args_dict['score']} not in {PRINT_SCORES}.")
    config, users_info, hyperparameters_combinations, results_before_averaging_over_folds = load_outputs_files(args_dict["outputs_folder"])
    global_visualizer = Global_Visualizer(config, users_info, hyperparameters_combinations, results_before_averaging_over_folds, args_dict["outputs_folder"],
                                          args_dict["score"], args_dict["optimize_tail"])
    global_visualizer.generate_pdf(args_dict["save_scores_tables"])