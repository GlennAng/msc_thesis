import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from ....src.load_files import load_users_significant_categories
from ..training.algorithm import Algorithm, Evaluation
from ..training.scores_definitions import SCORES_DICT, Score, get_score_from_arg
from .results_handling import (
    average_over_folds,
    average_over_n_most_extreme_users_for_all_hyperparameters_combinations,
    average_over_users,
    get_n_best_hyperparameters_combinations_score,
    get_val_upper_bounds,
    keep_only_n_most_extreme_hyperparameters_combinations_for_all_users_score,
)
from .visualization_tools import (
    HYPERPARAMETERS_ABBREVIATIONS,
    PLOT_CONSTANTS,
    PRINT_SCORES,
    clean_users_info,
    get_best_global_hyperparameters_combination_tables,
    get_cache_type_str,
    get_hyperparameters_combinations_table,
    get_hyperparameters_combinations_with_explicit_X_hyperparameter,
    get_hyperparameters_ranges,
    get_hyperparameters_ranges_str,
    get_plot_df,
    get_users_info_table,
    load_outputs_files,
    plot_hyperparameter_for_all_combinations,
    print_fifth_page,
    print_first_page,
    print_fourth_page,
    print_interesting_users,
    print_largest_performance_gain,
    print_second_page,
    print_third_page,
)

OPTIMIZATION_CONSTANTS = {
    "N_TAIL_USERS": 15,
    "N_PRINT_BEST_HYPERPARAMETERS_COMBINATIONS": 25,
    "PERCENTAGE_USERS_SPECIAL_GROUPS": 0.1,
}


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Global Visualization Parameters")
    parser.add_argument("--outputs_folder", type=Path)
    parser.add_argument("--score", type=str, default="ndcg_all")
    parser.add_argument("--optimize_tail", action="store_true", default=False)
    parser.add_argument("--no-optimize_tail", action="store_false", dest="optimize_tail")
    parser.add_argument("--save_scores_tables", action="store_true", default=False)
    args_dict = vars(parser.parse_args())
    if not isinstance(args_dict["outputs_folder"], Path):
        args_dict["outputs_folder"] = Path(args_dict["outputs_folder"])
    args_dict["score"] = get_score_from_arg(args_dict["score"])
    return args_dict


class Global_Visualizer:
    def __init__(
        self,
        config: dict,
        users_info: pd.DataFrame,
        hyperparameters_combinations: pd.DataFrame,
        results_before_averaging_over_folds: pd.DataFrame,
        folder: Path,
        score: Score = Score.NDCG_ALL,
        tail: bool = False,
    ) -> None:
        self.config = config
        self.users_info = clean_users_info(users_info, include_base=False, include_cache=True)
        self.hyperparameters_combinations = hyperparameters_combinations
        self.results_before_averaging_over_folds = results_before_averaging_over_folds
        self.folder = folder
        self.score, self.tail = score, tail
        self.users_significant_categories = load_users_significant_categories(
            relevant_users_ids=self.users_info["user_id"].unique().tolist(),
        )
        self.users_significant_categories = self.users_significant_categories[
            self.users_significant_categories["rank"] == 1
        ]
        self.extract_users_data()
        self.extract_optimization_data()
        self.extract_results_data()
        self.extract_best_global_hyperparameters_combination_data()
        self.extract_high_low_users()
        self.extract_best_individual_hyperparameters_combination_data()
        self.extract_largest_performance_gain_data()

    def extract_users_data(self) -> None:
        self.n_folds = self.results_before_averaging_over_folds["fold_idx"].nunique()
        self.users_ids = sorted(self.users_info["user_id"].unique().tolist())
        self.n_users = len(self.users_ids)
        self.n_tail_users = min(OPTIMIZATION_CONSTANTS["N_TAIL_USERS"], self.n_users)
        self.n_print_interesting_users = min(24, self.n_users)
        self.n_print_largest_performance_gain = min(25, self.n_users)
        if self.n_users >= 500:
            self.n_users_special_groups = 75
        else:
            self.n_users_special_groups = int(
                self.n_users * OPTIMIZATION_CONSTANTS["PERCENTAGE_USERS_SPECIAL_GROUPS"]
            )
        self.n_users_special_groups = max(1, min(self.n_users, self.n_users_special_groups))

    def extract_optimization_data(self) -> None:
        self.hyperparameters = list(self.hyperparameters_combinations.columns)[1:]
        self.hyperparameters_ranges = get_hyperparameters_ranges(self.hyperparameters_combinations)
        self.n_print_best_hyperparameters_combinations = min(
            OPTIMIZATION_CONSTANTS["N_PRINT_BEST_HYPERPARAMETERS_COMBINATIONS"],
            len(self.hyperparameters_combinations),
        )

    def extract_results_data(self) -> None:
        self.results_after_averaging_over_folds = average_over_folds(
            self.results_before_averaging_over_folds
        )
        self.results_after_averaging_over_users = average_over_users(
            self.results_after_averaging_over_folds
        )
        self.results_after_averaging_over_tails = (
            average_over_n_most_extreme_users_for_all_hyperparameters_combinations(
                self.results_after_averaging_over_folds, self.n_tail_users, True
            )
        )
        self.val_upper_bounds = get_val_upper_bounds(
            self.results_after_averaging_over_folds, self.n_tail_users
        )

    def extract_best_global_hyperparameters_combination_data(self) -> None:
        self.best_global_hyperparameters_combinations_idxs = (
            get_n_best_hyperparameters_combinations_score(
                (
                    self.results_after_averaging_over_tails
                    if self.tail
                    else self.results_after_averaging_over_users
                ),
                self.score,
                self.n_print_best_hyperparameters_combinations,
            )
        )
        self.best_global_hyperparameters_combination_idx = (
            self.best_global_hyperparameters_combinations_idxs[0]
        )
        self.best_global_hyperparameters_combination_df = self.results_after_averaging_over_folds[
            self.results_after_averaging_over_folds["combination_idx"]
            == self.best_global_hyperparameters_combination_idx
        ]

        if SCORES_DICT[self.score]["increase_better"]:
            self.worst_users_best_global_hyperparameters_combination_df = (
                self.best_global_hyperparameters_combination_df.nsmallest(
                    self.n_print_interesting_users, f"val_{self.score.name.lower()}"
                )
            )
            self.best_users_best_global_hyperparameters_combination_df = (
                self.best_global_hyperparameters_combination_df.nlargest(
                    self.n_print_interesting_users, f"val_{self.score.name.lower()}"
                )
            )
        else:
            self.worst_users_best_global_hyperparameters_combination_df = (
                self.best_global_hyperparameters_combination_df.nlargest(
                    self.n_print_interesting_users, f"val_{self.score.name.lower()}"
                )
            )
            self.best_users_best_global_hyperparameters_combination_df = (
                self.best_global_hyperparameters_combination_df.nsmallest(
                    self.n_print_interesting_users, f"val_{self.score.name.lower()}"
                )
            )

        self.worst_users_best_global_hyperparameters_combination = (
            self.worst_users_best_global_hyperparameters_combination_df["user_id"].values
        )
        self.best_users_best_global_hyperparameters_combination = (
            self.best_users_best_global_hyperparameters_combination_df["user_id"].values
        )
        self.hyperparameters_combinations_with_explicit_X_hyperparameter = (
            get_hyperparameters_combinations_with_explicit_X_hyperparameter(
                self.hyperparameters, self.hyperparameters_combinations
            )
        )
        self.globally_optimal_X_hyperparameter = self.hyperparameters_combinations[
            self.hyperparameters_combinations["combination_idx"]
            == self.best_global_hyperparameters_combination_idx
        ][PLOT_CONSTANTS["X_HYPERPARAMETER"]].values[0]

    def extract_head_middle_tail_users(self) -> None:
        score_abb = SCORES_DICT[self.score]["abbreviation_for_visu_file"]
        sorted_df = self.best_global_hyperparameters_combination_df.sort_values(
            f"val_{self.score.name.lower()}",
            ascending=not SCORES_DICT[self.score]["increase_better"],
        )
        assert len(sorted_df) == self.n_users
        n = self.n_users_special_groups
        head_df, tail_df = (sorted_df.head(n), sorted_df.tail(n))
        start_idx = (self.n_users - n) // 2
        middle_df = sorted_df.iloc[start_idx : start_idx + n]
        n_head, n_tail, n_middle = len(head_df), len(tail_df), len(middle_df)
        assert n == n_head == n_tail == n_middle
        legend = f"Head/Middle/Tail: The {n_head}/{n_middle}/{n_tail} Users with the "
        legend += f"best/medium/worst {score_abb}."
        self.users_groups_dict["Head"] = {"users_ids": head_df["user_id"].values, "legend": legend}
        self.users_groups_dict["Middle"] = {"users_ids": middle_df["user_id"].values}
        self.users_groups_dict["Tail"] = {"users_ids": tail_df["user_id"].values}

    def extract_high_low_negrated_train_users(self) -> None:
        n = self.n_users_special_groups
        hi_negrated_train_users = self.users_info.nlargest(n, "n_negrated_train")["user_id"].values
        lo_negrated_train_users = self.users_info.nsmallest(n, "n_negrated_train")["user_id"].values
        legend = f"HiNegTr/LoNegTr: The {n} Users with the highest/lowest Negative Training Votes."
        self.users_groups_dict["HiNegTr"] = {"users_ids": hi_negrated_train_users, "legend": legend}
        self.users_groups_dict["LoNegTr"] = {"users_ids": lo_negrated_train_users}

    def extract_high_pos_val_ratings_sessions_time_users(self) -> None:
        n = self.n_users_special_groups
        hi_pos_val_ratings_users = self.users_info.nlargest(n, "n_posrated_val")["user_id"].values
        hi_pos_val_ratings_legend = (
            f"VotePV: The {n} Users with the most Positive Validation Votes."
        )
        hi_pos_val_sessions_users = self.users_info.nlargest(n, "n_sessions_pos_val")[
            "user_id"
        ].values
        hi_pos_val_sessions_legend = (
            f"SessPV: The {n} Users with the most Positive Validation Sessions."
        )
        hi_pos_val_time_users = self.users_info.nlargest(n, "time_range_days_pos_val")[
            "user_id"
        ].values
        hi_pos_val_time_legend = (
            f"TimePV: The {n} Users with the largest Time Range for Positive Validation."
        )
        self.users_groups_dict["VotePV"] = {
            "users_ids": hi_pos_val_ratings_users,
            "legend": hi_pos_val_ratings_legend,
        }
        self.users_groups_dict["SessPV"] = {
            "users_ids": hi_pos_val_sessions_users,
            "legend": hi_pos_val_sessions_legend,
        }
        self.users_groups_dict["TimePV"] = {
            "users_ids": hi_pos_val_time_users,
            "legend": hi_pos_val_time_legend,
        }

    def extract_cs_non_cs_users(self) -> None:
        cs_users = self.users_significant_categories[
            self.users_significant_categories["category"] == "Computer Science"
        ]["user_id"].values
        non_cs_users = [user_id for user_id in self.users_ids if user_id not in cs_users]
        n_cs, n_non_cs = len(cs_users), len(non_cs_users)
        cs_legend = (
            f"CS/NonCS: The {n_cs}/{n_non_cs} Users whose main Category is/isn't Computer Science."
        )
        self.users_groups_dict["CS"] = {"users_ids": cs_users, "legend": cs_legend}
        self.users_groups_dict["NonCS"] = {"users_ids": non_cs_users}

    def extract_high_low_users(self) -> None:
        self.users_groups_dict = {}
        self.extract_head_middle_tail_users()
        self.extract_high_low_negrated_train_users()
        self.extract_high_pos_val_ratings_sessions_time_users()
        self.extract_cs_non_cs_users()

    def extract_best_individual_hyperparameters_combination_data(self) -> None:
        self.best_individual_hyperparameters_combination_df = (
            keep_only_n_most_extreme_hyperparameters_combinations_for_all_users_score(
                self.results_after_averaging_over_folds, self.score, 1, False
            )
        )
        if SCORES_DICT[self.score]["increase_better"]:
            self.worst_users_best_individual_hyperparameters_combination_df = (
                self.best_individual_hyperparameters_combination_df.nsmallest(
                    self.n_print_interesting_users, f"val_{self.score.name.lower()}"
                )
            )
            self.best_users_best_individual_hyperparameters_combination_df = (
                self.best_individual_hyperparameters_combination_df.nlargest(
                    self.n_print_interesting_users, f"val_{self.score.name.lower()}"
                )
            )
        else:
            self.worst_users_best_individual_hyperparameters_combination_df = (
                self.best_individual_hyperparameters_combination_df.nlargest(
                    self.n_print_interesting_users, f"val_{self.score.name.lower()}"
                )
            )
            self.best_users_best_individual_hyperparameters_combination_df = (
                self.best_individual_hyperparameters_combination_df.nsmallest(
                    self.n_print_interesting_users, f"val_{self.score.name.lower()}"
                )
            )
        self.worst_users_best_individual_hyperparameters_combination = (
            self.worst_users_best_individual_hyperparameters_combination_df["user_id"].values
        )
        self.best_users_best_individual_hyperparameters_combination = (
            self.best_users_best_individual_hyperparameters_combination_df["user_id"].values
        )

    def extract_largest_performance_gain_data(self) -> None:
        column_renames_global = {
            f"val_{score.name.lower()}": f"val_{score.name.lower()}_global" for score in list(Score)
        }
        column_renames_individual = {
            f"val_{score.name.lower()}": f"val_{score.name.lower()}_individual"
            for score in list(Score)
        }
        best_global_hyperparameters_combination_df = (
            self.best_global_hyperparameters_combination_df.rename(columns=column_renames_global)
        )
        best_global_hyperparameters_combination_df = (
            best_global_hyperparameters_combination_df.drop(
                columns=["combination_idx"]
                + [f"train_{score.name.lower()}" for score in list(Score)]
            )
        )
        best_individual_hyperparameters_combination_df = (
            self.best_individual_hyperparameters_combination_df.rename(
                columns=column_renames_individual
            )
        )
        best_individual_hyperparameters_combination_df = (
            best_individual_hyperparameters_combination_df.drop(
                columns=[f"train_{score.name.lower()}" for score in list(Score)]
            )
        )

        self.largest_performance_gain_df = best_global_hyperparameters_combination_df.merge(
            best_individual_hyperparameters_combination_df, on="user_id"
        )
        for score in list(Score):
            self.largest_performance_gain_df[f"{score.name.lower()}_gain"] = np.abs(
                self.largest_performance_gain_df[f"val_{score.name.lower()}_individual"]
                - self.largest_performance_gain_df[f"val_{score.name.lower()}_global"]
            )
        self.largest_performance_gain_df = self.largest_performance_gain_df.drop(
            columns=[f"val_{score.name.lower()}_individual" for score in list(Score)]
        )
        self.largest_performance_gain_df = self.largest_performance_gain_df.merge(
            self.users_info, on="user_id"
        )
        self.largest_performance_gain_df = self.largest_performance_gain_df.nlargest(
            self.n_print_largest_performance_gain, f"{self.score.name.lower()}_gain"
        )

    def get_config_str(self) -> str:
        config_string = []
        config_string.append(
            f"Embedding Folder: '{self.config['embedding_folder'].split('/')[-1]}'."
        )
        config_string.append(
            f"Embedding Info: {'Sparse' if self.config['embedding_is_sparse'] else 'Dense'} "
            f"with {self.config['embedding_n_dimensions']} Dimensions."
        )
        if "time_elapsed" in self.config:
            config_string.append(f"Time Elapsed: {(self.config['time_elapsed'] / 60):.2f} Minutes.")

        if isinstance(self.config["algorithm"], str):
            self.config["algorithm"] = Algorithm[self.config["algorithm"]]
        if self.config["algorithm"] == Algorithm.LOGREG:
            config_string.append("Algorithm: Logistic Regression.")
            config_string.append(
                f"Logistic Regression Solver: {self.config['logreg_solver'].capitalize()}."
            )
        elif self.config["algorithm"] == Algorithm.SVM:
            config_string.append("Algorithm: Support Vector Machine.")
            config_string.append(f"SVM Kernel: {self.config['svm_kernel'].capitalize()}.")
        config_string.append(
            f"Maximum Number of Optimization iterations: {self.config['max_iter']}."
        )

        if isinstance(self.config["evaluation"], str):
            self.config["evaluation"] = Evaluation[self.config["evaluation"]]
        if self.config["evaluation"] == Evaluation.CROSS_VALIDATION:
            config_string.append("Evaluation Method: Cross-Validation.")
            config_string.append(f"Number of Cross-Validation Folds: {self.config['k_folds']}.")
        elif self.config["evaluation"] == Evaluation.TRAIN_TEST_SPLIT:
            config_string.append("Evaluation Method: Train-Test Split.")
        elif self.config["evaluation"] == Evaluation.SESSION_BASED:
            config_string.append(
                "Evaluation Method: Session-Based (with Time Sorting for Ranking Negatives)."
            )
        config_string.append(f"Users Ratings Selection: {self.config['users_ratings_selection']}.")
        if self.config["relevant_users_ids"] is not None:
            urs_appendix = " (specifically chosen)."
        else:
            urs_appendix = " (all chosen)."
        config_string.append(f"Number of selected Users: {self.n_users}{urs_appendix}")
        if self.config["evaluation"] != Evaluation.SESSION_BASED:
            config_string.append(
                "Same Ranking Negatives for all Positives? "
                f"{'Yes' if self.config['same_negrated_for_all_pos'] else 'No'}."
            )
        config_string.append(
            f"Were the Training Sets stratified? {'Yes' if self.config['stratified'] else 'No'}."
        )
        mrs, crs, rrs = (
            self.config["model_random_state"],
            self.config["cache_random_state"],
            self.config["ranking_random_state"],
        )
        config_string.append(f"Random States:  Model: {mrs}  |  Cache: {crs}  |  Ranking: {rrs}.")
        config_string.append("\n")
        config_string.append(
            get_cache_type_str(
                self.config["cache_type"], self.config["n_cache"], self.config["n_categories_cache"]
            )
        )
        if self.config["categories_dim"] is not None:
            config_string.append(f"Categories Scale: {self.config['categories_scale']}.")

        return "\n\n".join(config_string)

    def generate_first_page(self, pdf: PdfPages) -> None:
        config_str = self.get_config_str()
        print_first_page(pdf, config_str)

    def generate_second_page(self, pdf: PdfPages) -> None:
        hyperparameters_ranges_str = get_hyperparameters_ranges_str(self.hyperparameters_ranges)
        print_second_page(pdf, hyperparameters_ranges_str)

    def generate_third_page(self, pdf: PdfPages) -> None:
        users_info_table = get_users_info_table(self.users_info)
        print_third_page(pdf, self.n_users, users_info_table, self.users_ids)

    def generate_fourth_page(self, pdf: PdfPages) -> None:
        hyperparameters_combinations_table_val = get_hyperparameters_combinations_table(
            self.val_upper_bounds,
            self.score,
            self.best_global_hyperparameters_combinations_idxs,
            self.results_after_averaging_over_users,
            self.hyperparameters_combinations,
        )
        print_fourth_page(
            pdf, hyperparameters_combinations_table_val, self.score, self.hyperparameters
        )

    def generate_fifth_page(
        self, pdf: PdfPages, scores_tables_save_path: Path = None, is_validation: bool = True
    ) -> None:
        title = "Validation: " if is_validation else "Training: "
        title += (
            f"Scores of the Best Global Combi {self.best_global_hyperparameters_combination_idx}"
        )
        best_global_hyperparameters_combination = self.hyperparameters_combinations[
            self.hyperparameters_combinations["combination_idx"]
            == self.best_global_hyperparameters_combination_idx
        ]
        for i, hyperparameter in enumerate(self.hyperparameters):
            title += " (" if i == 0 else ", "
            title += (
                HYPERPARAMETERS_ABBREVIATIONS[hyperparameter]
                if hyperparameter in HYPERPARAMETERS_ABBREVIATIONS
                else hyperparameter
            )
            title += f" = {best_global_hyperparameters_combination[hyperparameter].values[0]}"
        title += "):"
        legend_text = "Legend:   "
        legend_text += f"All: All {self.n_users} Users"
        legend_counter = 1
        for i, users_group in enumerate(self.users_groups_dict.values()):
            if "legend" in users_group:
                if legend_counter % 2 == 0:
                    legend_text += "\n"
                else:
                    legend_text += "| "
                legend_counter += 1
                legend_text += f"{users_group['legend']}"
        scores_tables = get_best_global_hyperparameters_combination_tables(
            self.best_global_hyperparameters_combination_df,
            self.users_groups_dict,
            is_val=is_validation,
        )
        optimizer_page = SCORES_DICT[self.score]["page"]
        scores_same_page = [
            score for score in SCORES_DICT if SCORES_DICT[score]["page"] == optimizer_page
        ]
        optimizer_idx = scores_same_page.index(self.score) + 1
        for s in range(len(scores_tables)):
            grey_row = optimizer_idx if s == optimizer_page else -1
            s_title = f"{s + 1}/{len(scores_tables)}   -   {title}"
            scores_table_save_path = (
                f"{scores_tables_save_path}_{s + 1}.pkl" if scores_tables_save_path else None
            )
            print_fifth_page(
                pdf=pdf,
                users_groups_dict=self.users_groups_dict,
                title=s_title,
                legend_text=legend_text,
                best_global_hyperparameters_combination_table=scores_tables[s],
                optimizer_row=grey_row,
                save_path=scores_table_save_path,
            )

    def generate_sixth_page(self, pdf: PdfPages) -> None:
        title = "Worst"
        interesting_users_best_global_hyperparameters_combination_df = (
            self.worst_users_best_global_hyperparameters_combination_df
        )
        best_global_merged_with_users_info = (
            interesting_users_best_global_hyperparameters_combination_df.merge(
                self.users_info, on="user_id"
            )
        )
        best_global_merged_with_users_info = best_global_merged_with_users_info.merge(
            self.users_significant_categories, on="user_id", how="left"
        )
        print_interesting_users(
            pdf,
            self.score,
            title,
            best_global_merged_with_users_info,
        )

    def generate_seventh_page(self, pdf: PdfPages) -> None:
        title = "Best"
        interesting_users_best_global_hyperparameters_combination_df = (
            self.best_users_best_global_hyperparameters_combination_df
        )
        best_global_merged_with_users_info = (
            interesting_users_best_global_hyperparameters_combination_df.merge(
                self.users_info, on="user_id"
            )
        )
        best_global_merged_with_users_info = best_global_merged_with_users_info.merge(
            self.users_significant_categories, on="user_id", how="left"
        )
        print_interesting_users(
            pdf,
            self.score,
            title,
            best_global_merged_with_users_info,
        )

    def generate_eighth_page(self, pdf: PdfPages) -> None:
        self.largest_performance_gain_df = self.largest_performance_gain_df.merge(
            self.users_significant_categories, on="user_id"
        )
        print_largest_performance_gain(
            pdf,
            self.largest_performance_gain_df,
            self.score,
            self.best_global_hyperparameters_combination_idx,
        )

    def generate_plots(self, pdf: PdfPages) -> None:
        plot_df = get_plot_df(
            self.results_after_averaging_over_users,
            self.hyperparameters_combinations_with_explicit_X_hyperparameter,
        )
        plot_tail_df = get_plot_df(
            self.results_after_averaging_over_tails,
            self.hyperparameters_combinations_with_explicit_X_hyperparameter,
        )
        plot_hyperparameter_for_all_combinations(
            pdf,
            self.hyperparameters,
            self.hyperparameters_combinations_with_explicit_X_hyperparameter,
            plot_df,
            plot_tail_df,
        )

    def generate_pdf(self, save_scores_tables: bool = False) -> None:
        file_name = (
            self.folder
            / f"global_visu_{SCORES_DICT[self.score]['abbreviation_for_visu_file'].lower()}.pdf"
        )
        scores_tables_save_path = self.folder / "scores_table" if save_scores_tables else None
        with PdfPages(file_name) as pdf:
            self.generate_first_page(pdf)
            self.generate_second_page(pdf)
            self.generate_third_page(pdf)
            self.generate_fourth_page(pdf)
            self.generate_fifth_page(pdf, scores_tables_save_path, is_validation=True)
            self.generate_sixth_page(pdf)
            self.generate_seventh_page(pdf)
            self.generate_fifth_page(pdf, scores_tables_save_path, is_validation=False)
            if len(self.hyperparameters_combinations) > 1:
                self.generate_eighth_page(pdf)
                self.generate_plots(pdf)

    def print_fold_stds(self):
        results_before_averaging_over_folds = self.results_before_averaging_over_folds[
            self.results_before_averaging_over_folds["combination_idx"]
            == self.best_global_hyperparameters_combination_idx
        ]
        results_before_averaging_over_folds = results_before_averaging_over_folds.drop(
            columns=["combination_idx"]
        )
        group_columns = ["fold_idx"]
        results_before_averaging_over_folds = results_before_averaging_over_folds.groupby(
            group_columns
        ).mean()
        print(results_before_averaging_over_folds["val_balanced_accuracy"].std())
        print(results_before_averaging_over_folds["val_cel"].std())


if __name__ == "__main__":
    try:
        args_dict = parse_args()
        if args_dict["score"] not in PRINT_SCORES:
            raise ValueError(f"Score {args_dict['score']} not in {PRINT_SCORES}.")
        config, users_info, hyperparameters_combinations, results_before_averaging_over_folds = (
            load_outputs_files(args_dict["outputs_folder"])
        )
        global_visualizer = Global_Visualizer(
            config,
            users_info,
            hyperparameters_combinations,
            results_before_averaging_over_folds,
            args_dict["outputs_folder"],
            args_dict["score"],
            args_dict["optimize_tail"],
        )
        global_visualizer.generate_pdf(args_dict["save_scores_tables"])
    except Exception as e:
        print(f"Remark: Were not able to visualize the results globally. Error: {e}")
        traceback.print_exc()
