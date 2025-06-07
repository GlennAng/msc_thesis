from algorithm import Score, SCORES_DICT
from compute_tfidf import train_vectorizer_for_user, get_mean_embedding
from data_handling import sql_execute
from results_handling import average_over_users

from enum import Enum
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
from wordcloud import WordCloud
import json
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re

HYPERPARAMETERS_ABBREVIATIONS = {"clf_C": "C", "weights_cache_v": "v", "weights_negrated_importance": "v", "weights_cache_v1_v2": "v1_v2", "weights_neg_scale": "S", "cache_scale": "CS"}

PLOT_CONSTANTS = {"FIG_SIZE": (11, 8.5), "ALPHA_PLOT": 0.5, "ALPHA_FILL": 0.2, "LINE_WIDTH": 2.5, "X_HYPERPARAMETER": "clf_C",
                  "N_PAPERS_PER_PAGE": 7, "N_PAPERS_IN_TOTAL" : 70, "MAX_LINES": 5, "LINE_HEIGHT": 0.025, "WORD_SPACING": 0.0075, "X_LOCATION": -0.125, 
                  "PLOT_SCORES" : [Score.BALANCED_ACCURACY, Score.RECALL, Score.PRECISION, Score.SPECIFICITY]}
PRINT_SCORES = [Score.RECALL, Score.SPECIFICITY, Score.BALANCED_ACCURACY, Score.CEL, Score.NDCG, Score.MRR,
                Score.NDCG_SAMPLES, Score.MRR_SAMPLES, Score.NDCG_ALL, Score.MRR_ALL, Score.INFO_NCE_ALL]
n_scores_halved = len(Score) // 2

def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def format_number(value : float, max_decimals : int = 4) -> str:
    if not is_number(value):
        return value
    if value == np.float64('nan'):
        return value
    try:
        n_digits_before_decimal = len(str(int(value)))
    except ValueError:
        n_digits_before_decimal = max_decimals
    if n_digits_before_decimal == 3:
        max_decimals = 3
    elif n_digits_before_decimal == 4:
        max_decimals = 2
    elif n_digits_before_decimal == 5:
        max_decimals = 1
    elif n_digits_before_decimal >= 6:
        max_decimals = 0
    formatted = f"{value:.{max_decimals}f}"
    stripped = formatted.rstrip('0')
    if stripped.endswith('.'):
        return stripped + '0'
    return stripped

def load_outputs_files(folder : str) -> tuple:
    folder = folder[-1] if folder[-1] == "/" else folder
    config = json.load(open(folder + "/config.json", 'r'))
    users_info = pd.read_csv(folder + "/users_info.csv")
    hyperparameters_combinations = pd.read_csv(folder + "/hyperparameters_combinations.csv")
    results_before_averaging_over_folds = pd.read_csv(folder + "/users_results.csv")
    return config, users_info, hyperparameters_combinations, results_before_averaging_over_folds

def get_hyperparameters_ranges(hyperparameters_combinations : pd.DataFrame) -> dict:
    return {col: hyperparameters_combinations[col].unique().tolist() for col in hyperparameters_combinations.drop(columns = ["combination_idx"]).columns}

def print_table(data : list, bbox : list, col_labels : list, col_widths : list, bold_row : int = -1, bold_column : int = -1, grey_row : int = -1, grey_column : int = -1) -> None:
    bold_row = [bold_row] if type(bold_row) == int else bold_row
    bold_column = [bold_column] if type(bold_column) == int else bold_column
    grey_row = [grey_row] if type(grey_row) == int else grey_row
    grey_column = [grey_column] if type(grey_column) == int else grey_column
    table = plt.table(cellText = data, loc = 'center', cellLoc = 'center', bbox = bbox, colWidths = col_widths, colLabels = col_labels)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 0.5)
    for key, cell in table._cells.items():
        if key[0] in bold_row or key[1] in bold_column:
            cell.set_text_props(fontproperties = FontProperties(weight = 'bold'))
        if key[0] in grey_row or key[1] in grey_column:
            cell.set_facecolor('#d3d3d3')

def clean_users_info(user_info: pd.DataFrame, include_base : bool, include_cache : bool) -> pd.DataFrame:
    if not include_base:
        user_info["n_base"] = np.float64('nan')
    if not include_cache:
        user_info["n_cache"] = np.float64('nan')
    return user_info

def get_cache_type_str(cache_type : str, max_cache : int, n_cache_attached : int) -> str:
    s = "Cache Type: "
    s_cache = str(max_cache // 1000) + 'K' if max_cache % 1000 == 0 else str(max_cache)
    s_cache_attached = str(n_cache_attached // 1000) + 'K' if n_cache_attached % 1000 == 0 else str(n_cache_attached)
    if cache_type == "global":
        s += "Global"
    elif cache_type == "user_filtered":
        s += "User-filtered"
    s += f" ({s_cache} Papers, {s_cache_attached} Attached). "
    return s

def get_users_selection_str(users_selection : str, users_ids : list) -> str:
    s = "User Selection Criterion among those who qualified: "
    if users_selection == "random":
        s += "Random."
    elif users_selection == "largest_n":
        s += "Largest Number of rated Papers."
    elif users_selection == "smallest_n":
        s += "Smallest Number of rated Papers."
    else:
        s += f"Specifically chosen."
        if len(users_ids) <= 100:
            s += f"\n{users_ids}."
    return s

def print_first_page(pdf : PdfPages, config_str : str) -> None:
    fig, ax = plt.subplots(figsize = PLOT_CONSTANTS["FIG_SIZE"])
    ax.axis("off")
    ax.text(0.5, 1.08, "Evaluation Configuration:", fontsize = 18, ha = 'center', va = 'center', fontweight = 'bold')
    ax.text(0.5, 1.0, config_str, fontsize = 9, ha = 'center', va = 'top', wrap = True)
    pdf.savefig(fig)
    plt.close(fig)

def get_hyperparameters_ranges_str(hyperparameters_ranges : dict) -> str:
    hyperparameters_ranges_text = []
    for hyperparameter, values in hyperparameters_ranges.items():
        formatted_values = ", ".join(format_number(value) for value in values)
        hyperparameters_ranges_text.append(f"{hyperparameter}: \n[{formatted_values}].\n")
    return "\n".join(hyperparameters_ranges_text)

def get_users_info_table(users_info : pd.DataFrame) -> list:
    data = []
    rows = {"Voting Weight": users_info["voting_weight"], "Number of positively rated Papers": users_info["n_posrated"],
            "Number of negatively rated Papers": users_info["n_negrated"], "Number of base Papers": users_info["n_base"],
            "Percentage of positively rated among all rated": users_info["n_posrated"] / (users_info["n_posrated"] + users_info["n_negrated"])}
    n_base = users_info["n_base"].fillna(0)
    n_zerorated = users_info["n_zerorated"].fillna(0)
    rows["Percentage of positively rated + base \n among all rated + base + zerorated"] = (
        (users_info["n_posrated"] + n_base) / (users_info["n_posrated"] + users_info["n_negrated"] + n_base + n_zerorated))
    for row_name, row in rows.items():
        if row.isnull().all():
            data.append([row_name] + [np.float64('nan')] * 5)
        else:
            row_min = str(row.min()) if row.dtype == 'int64' else format_number(row.min())
            row_max = str(row.max()) if row.dtype == 'int64' else format_number(row.max())
            data.append([row_name, row_min, row_max, format_number(row.median()), format_number(row.mean()), format_number(row.std())])
    return data

def print_second_page(pdf : PdfPages, hyperparameters_ranges_str : str) -> None:
    fig, ax = plt.subplots(figsize = PLOT_CONSTANTS["FIG_SIZE"])
    ax.axis("off")
    ax.text(0.5, 1.075, "Hyperparameters Ranges:\n", fontsize = 17, ha = 'center', va = 'center', fontweight = 'bold')
    ax.text(0.5, 1.05, hyperparameters_ranges_str, fontsize = 10, ha = 'center', va = 'top', wrap = True)
    pdf.savefig(fig)
    plt.close(fig)

def get_hyperparameters_combinations_table(val_upper_bounds : pd.DataFrame, optimizer_score : Score, best_global_hyperparameters_combinations_idxs : list, 
                                           results_after_averaging_over_users : pd.DataFrame, hyperparameters_combinations : pd.DataFrame) -> list:
    data = []
    n_hyperparameters = len(hyperparameters_combinations.columns) - 1
    first_row = ["N/A"] * n_hyperparameters
    for score in list(PRINT_SCORES):
        first_row.append(format_number(val_upper_bounds[f"val_{score.name.lower()}"]))
    data.append(first_row)
    for combination_idx in best_global_hyperparameters_combinations_idxs:
        row = hyperparameters_combinations.loc[hyperparameters_combinations["combination_idx"] == combination_idx].values[0][1:].tolist()
        row = [format_number(value) for value in row]
        for score in list(PRINT_SCORES):
            row.append(format_number(results_after_averaging_over_users.loc[results_after_averaging_over_users["combination_idx"] == combination_idx, f"val_{score.name.lower()}_mean"].values[0]))
        data.append(row)
    return data

def print_third_page(pdf : PdfPages, n_users : int, users_info_table : list, users_selection: str, users_ids : list) -> None:
    fig, ax = plt.subplots(figsize = PLOT_CONSTANTS["FIG_SIZE"])
    ax.axis("off")
    ax.text(0.5, 1.1285, f"Users Info (N = {n_users}):\n", fontsize = 17, ha = 'center', va = 'top', fontweight = 'bold')
    print_table(users_info_table, [-0.125, 0.765, 1.2, 0.325], ["", "Minimum", "Maximum", "Median", "Mean", "Standard Dev."], [0.5] + (5 * [0.175]))
    ax.text(0.5, 0.745, f"{users_ids}", fontsize = 9, ha = 'center', va = 'top', wrap = True)
    pdf.savefig(fig)
    plt.close(fig)

def print_fourth_page(pdf : PdfPages, hyperparameters_combinations_table : list, optimizer_score : Score, hyperparameters : list) -> None:
    fig, ax = plt.subplots(figsize = PLOT_CONSTANTS["FIG_SIZE"])
    ax.axis("off")
    ax.text(0.5, 1.1, f"Validation Scores for Hyperparameters Combinations:\n", fontsize = 16, ha = 'center', va = 'center', fontweight = 'bold')
    columns = [HYPERPARAMETERS_ABBREVIATIONS[hyperparameter] for hyperparameter in hyperparameters]
    for i, score in enumerate(PRINT_SCORES):
        columns.append(SCORES_DICT[score]['abbreviation'])
    optimizer_column = columns.index(SCORES_DICT[optimizer_score]['abbreviation'])
    print_table(hyperparameters_combinations_table, [-0.14, -0.1, 1.25, 1.18], columns, len(hyperparameters) * [0.125] + len(PRINT_SCORES) * [0.15], bold_row = 1, 
                grey_column = optimizer_column)
    pdf.savefig(fig)
    plt.close(fig)

def get_best_global_hyperparameters_combination_table(best_global_hyperparameters_combination_df : pd.DataFrame, tail_users : np.ndarray, high_votes_users : np.ndarray, 
                                                      low_votes_users : np.ndarray, high_ratio_users : np.ndarray, low_ratio_users : np.ndarray) -> tuple:
    averaged_over_all_users = average_over_users(best_global_hyperparameters_combination_df)
    averaged_over_high_votes_users = average_over_users(best_global_hyperparameters_combination_df.loc[best_global_hyperparameters_combination_df["user_id"].isin(high_votes_users)])
    averaged_over_low_votes_users = average_over_users(best_global_hyperparameters_combination_df.loc[best_global_hyperparameters_combination_df["user_id"].isin(low_votes_users)])
    averaged_over_high_ratio_users = average_over_users(best_global_hyperparameters_combination_df.loc[best_global_hyperparameters_combination_df["user_id"].isin(high_ratio_users)])
    averaged_over_low_ratio_users = average_over_users(best_global_hyperparameters_combination_df.loc[best_global_hyperparameters_combination_df["user_id"].isin(low_ratio_users)])
    averaged_over_tail_users = average_over_users(best_global_hyperparameters_combination_df.loc[best_global_hyperparameters_combination_df["user_id"].isin(tail_users)])
    dfs = [averaged_over_all_users, averaged_over_high_votes_users, averaged_over_low_votes_users, averaged_over_high_ratio_users, averaged_over_low_ratio_users, averaged_over_tail_users]
    tables = ([], [])
    for s, score in enumerate(list(Score)):
        score_name = score.name.lower()
        row = [SCORES_DICT[score]["abbreviation"]]
        for i, df in enumerate(dfs):
            row.append(format_number(df[f"val_{score_name}_mean"].values[0]))
            row.append(format_number(df[f"train_{score_name}_mean"].values[0]))
            if i == 0:
                row.append(format_number(df[f"val_{score_name}_std"].values[0]))
        if s < n_scores_halved:
            tables[0].append(row)
        else:
            tables[1].append(row)
    return tables

def print_fifth_page(pdf : PdfPages, title : str, legend_text : str, best_global_hyperparameters_combination_table : list, optimizer_row : int, save_path : str = None) -> None:
    fig, ax = plt.subplots(figsize = PLOT_CONSTANTS["FIG_SIZE"])
    ax.axis("off")
    ax.text(0.5, 1.12, title, fontsize = 15, ha = 'center', va = 'center', fontweight = 'bold')
    groups = ["All", "HiVote", "LoVote", "HiPosi", "LoPosi", "Tail"]
    columns = ["Score"]
    for group in groups:
        columns.extend([group, f"{group}_T"])
        if group == "All":
            columns.append("All_σ")
    if save_path is not None:
        full_table = [columns] + best_global_hyperparameters_combination_table
        with open(save_path, "wb") as f:
            pickle.dump(full_table, f)
    print_table(best_global_hyperparameters_combination_table, [-0.14, -0.025, 1.25, 1.11], columns, [0.1] + (len(groups) * 2 + 1) * [0.15], bold_row = 0, grey_row = optimizer_row)
    ax.text(0.5, -0.08, legend_text, fontsize = 8, ha = 'center', va = 'center')
    pdf.savefig(fig)
    plt.close(fig)

def get_interesting_users_table(interesting_users_df : pd.DataFrame) -> list:
    data = []
    for _, row in interesting_users_df.iterrows():
        data_row = [int(row["user_id"]), int(row["combination_idx"]), int(row["n_posrated"]), int(row["n_negrated"]), row["n_base"] if np.isnan(row["n_base"]) else int(row["n_base"])]
        for score in PRINT_SCORES:
            data_row.append(format_number(row[f"val_{score.name.lower()}"]))
        data.append(data_row)
    return data

def print_interesting_users(pdf : PdfPages, gv_score : Score, title : str, interesting_users_best_global_hyperparameters_combination_df : pd.DataFrame, 
                                                                        interesting_users_best_individual_hyperparameters_combination_df : pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize = PLOT_CONSTANTS["FIG_SIZE"])
    ax.axis('off')
    ax.text(0.5, 1.1, f"Users with {title} Performance for Best Global Hyperparameters Combination:\n", fontsize = 12, ha = 'center', va = 'center', fontweight = 'bold')
    columns = ["User ID", "Combi", "N_POS", "N_NEG", "N_BASE"] + [SCORES_DICT[score]["abbreviation"] for score in PRINT_SCORES]
    print_table(get_interesting_users_table(interesting_users_best_global_hyperparameters_combination_df), [-0.14, -0.12, 1.25, 1.19], columns, 2 * [0.11] + 3 * [0.1] + len(Score) * [0.125], 
                grey_column = columns.index(SCORES_DICT[gv_score]["abbreviation"]))
    pdf.savefig(fig)
    plt.close(fig)

def get_largest_performance_gain_table(largest_performance_gain_df : pd.DataFrame, gv_score : Score) -> list:
    data = []
    for _, row in largest_performance_gain_df.iterrows():
        data_row = [int(row["user_id"]), int(row["combination_idx"]), int(row["n_posrated"]), int(row["n_negrated"])]
        for score in PRINT_SCORES:
            gain = row[f"{score.name.lower()}_gain"]
            if gain >= 0:
                data_row.append(("+" if SCORES_DICT[score]["increase_better"] else "-") + format_number(gain))
            else:
                data_row.append(("-" if SCORES_DICT[score]["increase_better"] else "+") + format_number(-gain))
        data_row.append(format_number(row[f"val_{gv_score.name.lower()}_global"]))
        data.append(data_row)
    return data

def print_largest_performance_gain(pdf : PdfPages, largest_performance_gain_df : pd.DataFrame, gv_score : Score, best_global_hyperparameters_combination_idx : int) -> None:
    fig, ax = plt.subplots(figsize = PLOT_CONSTANTS["FIG_SIZE"])
    ax.axis('off')
    ax.text(0.49, 1.1, f"Users with Largest Performance Gain between Best Global Hyperparameters Combination {best_global_hyperparameters_combination_idx} and Best Individual:\n", 
            fontsize = 12, ha = 'center', va = 'center', fontweight = 'bold')
    columns = (["User ID", "Combi", "N_POS", "N_NEG"] + [SCORES_DICT[score]["abbreviation"] for score in PRINT_SCORES] + 
               [SCORES_DICT[gv_score]["abbreviation"] + "_Val"])
    print_table(get_largest_performance_gain_table(largest_performance_gain_df, gv_score), [-0.14, -0.1, 1.25, 1.19], columns, [0.11] + [0.085] + 2 * [0.09] + (2 * len(PRINT_SCORES) + 1) * [0.125],
                grey_column = columns.index(SCORES_DICT[gv_score]["abbreviation"]))
    pdf.savefig(fig)
    plt.close(fig)

def get_hyperparameters_combinations_with_explicit_X_hyperparameter(hyperparameters : list, hyperparameters_combinations : pd.DataFrame) -> pd.DataFrame:
    hyperparameters_without_X_hyperparameter = [hyperparameter for hyperparameter in hyperparameters if hyperparameter not in ["combination_idx", PLOT_CONSTANTS["X_HYPERPARAMETER"]]]
    hyperparameters_combinations_with_explicit_X_hyperparameter = hyperparameters_combinations.copy()
    if len(hyperparameters_without_X_hyperparameter) > 0:
        hyperparameters_combinations_with_explicit_X_hyperparameter["non_X_hyperparameter_combination_idx"] = (
                    hyperparameters_combinations_with_explicit_X_hyperparameter.groupby(hyperparameters_without_X_hyperparameter).ngroup())
    else:
        hyperparameters_combinations_with_explicit_X_hyperparameter["non_X_hyperparameter_combination_idx"] = -1
    return hyperparameters_combinations_with_explicit_X_hyperparameter

def get_plot_df(results_df : pd.DataFrame, hyperparameters_combinations_with_explicit_X_hyperparameter : pd.DataFrame) -> pd.DataFrame:
    relevant_score_columns = [f"{score.name.lower()}" for score in PLOT_CONSTANTS["PLOT_SCORES"]]
    relevant_train_columns = [f"train_{score}_mean" for score in relevant_score_columns] + [f"train_{score}_std" for score in relevant_score_columns]
    relevant_val_columns = [f"val_{score}_mean" for score in relevant_score_columns] + [f"val_{score}_std" for score in relevant_score_columns]
    relevant_columns = relevant_train_columns + relevant_val_columns
    plot_df = pd.merge(hyperparameters_combinations_with_explicit_X_hyperparameter[["combination_idx", PLOT_CONSTANTS["X_HYPERPARAMETER"], "non_X_hyperparameter_combination_idx"]], 
                       results_df[["combination_idx"] + relevant_columns], on = "combination_idx")
    plot_df = plot_df[["non_X_hyperparameter_combination_idx", PLOT_CONSTANTS["X_HYPERPARAMETER"]] + relevant_columns]
    return plot_df

def plot_score(axs, results_df : pd.DataFrame, score : Score, vertical_line : float = None, results_tail_df : pd.DataFrame = None) -> tuple:
    score_name, subtitle = score.name.lower(), SCORES_DICT[score]["name"]
    alpha_plot, alpha_fill, line_width, X_hyperparameter = PLOT_CONSTANTS["ALPHA_PLOT"], PLOT_CONSTANTS["ALPHA_FILL"], PLOT_CONSTANTS["LINE_WIDTH"], PLOT_CONSTANTS["X_HYPERPARAMETER"]
    axs.set_title(f"{subtitle}.", fontsize = 14)

    if SCORES_DICT[score]["increase_better"]:
        best_X_hyperparameter_value = results_df.loc[results_df[f"val_{score_name}_mean"].idxmax()][X_hyperparameter]
        best_X_hyperparameter_val_score = results_df.loc[results_df[f"val_{score_name}_mean"].idxmax()][f"val_{score_name}_mean"]
        axs.set_ylim(-0.01, 1.01)
    else:
        best_X_hyperparameter_value = results_df.loc[results_df[f"val_{score_name}_mean"].idxmin()][X_hyperparameter]
        best_X_hyperparameter_val_score = results_df.loc[results_df[f"val_{score_name}_mean"].idxmin()][f"val_{score_name}_mean"]
        axs.set_ylim(-0.01, 2.01)
    train_score, val_score = results_df[f'train_{score_name}_mean'], results_df[f'val_{score_name}_mean']
    train_line, = axs.plot(results_df[X_hyperparameter], train_score, label = f'Performance on Training Set', alpha = alpha_plot, linewidth = line_width)
    val_line, = axs.plot(results_df[X_hyperparameter], val_score, label = f'Performance on Validation Set', alpha = alpha_plot, linewidth = line_width)
    axs.scatter(best_X_hyperparameter_value, best_X_hyperparameter_val_score, s = 50, zorder = 5, facecolor = val_line.get_color(), edgecolor = 'black', label = 'Best Hyperparameter')

    train_std, val_std = results_df[f'train_{score_name}_std'], results_df[f'val_{score_name}_std']
    axs.fill_between(results_df[X_hyperparameter], train_score - train_std, train_score + train_std, alpha = alpha_fill)
    axs.fill_between(results_df[X_hyperparameter], val_score - val_std, val_score + val_std, color = "#FFD700", alpha = alpha_fill)

    if results_tail_df is not None:
        if SCORES_DICT[score]["increase_better"]:
            best_X_hyperparameter_tail = results_tail_df.loc[results_tail_df[f"val_{score_name}_mean"].idxmax()][X_hyperparameter]
            best_X_hyperparameter_tail_val_score = results_tail_df.loc[results_tail_df[f"val_{score_name}_mean"].idxmax()][f"val_{score_name}_mean"]
        else:
            best_X_hyperparameter_tail = results_tail_df.loc[results_tail_df[f"val_{score_name}_mean"].idxmin()][X_hyperparameter]
            best_X_hyperparameter_tail_val_score = results_tail_df.loc[results_tail_df[f"val_{score_name}_mean"].idxmin()][f"val_{score_name}_mean"]
        tail_score = results_tail_df[f"val_{score_name}_mean"]
        tail_line, = axs.plot(results_tail_df[X_hyperparameter], tail_score, label = f'Tail Performance on Validation Set', alpha = alpha_plot, 
                              linewidth = line_width, color = "red")
        axs.scatter(best_X_hyperparameter_tail, best_X_hyperparameter_tail_val_score, s = 50, zorder = 5, facecolor = "red", edgecolor = 'black', label = 'Best Hyperparameter (Tail)')

    secax = axs.secondary_yaxis('right')
    secax.set_yticks([best_X_hyperparameter_val_score, best_X_hyperparameter_tail_val_score] if results_tail_df is not None else [best_X_hyperparameter_val_score])
    axs.set_xscale("log")
    if vertical_line is not None:
        axs.axvline(x = vertical_line, color = 'grey', linestyle = '--')
    return train_line, val_line, tail_line if results_tail_df is not None else None

def plot_hyperparameter(pdf : PdfPages, results_df : pd.DataFrame, title : str, multiple_users : bool, vertical_line : float = None, results_tail_df : pd.DataFrame = None) -> None:
    plot_scores, X_hyperparameter = PLOT_CONSTANTS["PLOT_SCORES"], PLOT_CONSTANTS["X_HYPERPARAMETER"]
    fig, ax = plt.subplots(2, 2, figsize = (18, 11))
    fig.suptitle(title, fontsize = 14)
    plot_tail = results_tail_df is not None

    for i, axs in enumerate(ax.flat):
        score = plot_scores[i]
        train_line, val_line, tail_line = plot_score(axs, results_df, score, vertical_line, results_tail_df)
        if i > 1:
            axs.set_xlabel(f"'{X_hyperparameter}'", fontsize = 12)

    legend_lines = [train_line, val_line, tail_line] if plot_tail else [train_line, val_line]
    legend_titles = (["Performance on Training Set", "Performance on Validation Set", "Tail Performance on Validation Set"] if results_tail_df is not None
                      else ["Performance on Training Set", "Performance on Validation Set"])
    fig.legend(legend_lines, legend_titles, loc = "lower right", ncol = 1, fontsize = 12)
    fig_text = f"Remark: The filled area represents +- 1 standard deviation from the mean over all {'Users' if multiple_users else 'Folds'}." 
    fig.text(0.6, 0.02, fig_text, wrap = False, horizontalalignment = 'right', fontsize = 10)
    pdf.savefig(fig)
    plt.close(fig)

def plot_hyperparameter_for_all_combinations(pdf : PdfPages, hyperparameters : list, hyperparameters_combinations_with_explicit_X_hyperparameter : pd.DataFrame, plot_df : pd.DataFrame, plot_tail_df : pd.DataFrame) -> None:
    X_hyperparameter = PLOT_CONSTANTS["X_HYPERPARAMETER"]
    hyperparameters_without_X_hyperparameter = [hyperparameter for hyperparameter in hyperparameters if hyperparameter != X_hyperparameter]
    if len(hyperparameters_without_X_hyperparameter) == 0:
        included_combinations = (sorted(hyperparameters_combinations_with_explicit_X_hyperparameter.loc[hyperparameters_combinations_with_explicit_X_hyperparameter
                                ["non_X_hyperparameter_combination_idx"] == -1]["combination_idx"]))
        title = f"Combinations: {str(included_combinations)}.\n"
        plot_hyperparameter(pdf, plot_df, title, True, None, plot_tail_df)
    else:
        non_X_hyperparameter_combinations = len(hyperparameters_combinations_with_explicit_X_hyperparameter["non_X_hyperparameter_combination_idx"].unique())
        for non_X_hyperparameter_combination_idx in range(non_X_hyperparameter_combinations):
            included_combinations = (sorted(hyperparameters_combinations_with_explicit_X_hyperparameter.loc[hyperparameters_combinations_with_explicit_X_hyperparameter
                                    ["non_X_hyperparameter_combination_idx"] == non_X_hyperparameter_combination_idx]["combination_idx"]))
            plot_subset_title = f"Combinations: {str(included_combinations)}.\n"
            non_X_hyparameter_combinations_first_row_with_idx = (hyperparameters_combinations_with_explicit_X_hyperparameter.loc
                    [hyperparameters_combinations_with_explicit_X_hyperparameter["non_X_hyperparameter_combination_idx"] == non_X_hyperparameter_combination_idx].iloc[0])
            plot_df_subset = plot_df[plot_df["non_X_hyperparameter_combination_idx"] == non_X_hyperparameter_combination_idx]
            plot_tail_df_subset = plot_tail_df[plot_tail_df["non_X_hyperparameter_combination_idx"] == non_X_hyperparameter_combination_idx]
            for i, hyperparameter in enumerate(hyperparameters_without_X_hyperparameter):
                plot_subset_title += f"{hyperparameter}: {format_number(non_X_hyparameter_combinations_first_row_with_idx[hyperparameter])}"
                if i < len(hyperparameters_without_X_hyperparameter) - 1:
                    plot_subset_title += ","
                    if i % 4 == 3:
                        plot_subset_title += "\n"
                    else:
                        plot_subset_title += "  "
            plot_hyperparameter(pdf, plot_df_subset, plot_subset_title + ".", True, None, plot_tail_df_subset)
    
def get_user_info_table(user_info : pd.DataFrame) -> tuple:
    columns = ["Voting Weight", "Number of Positively Rated", "Number of Negatively Rated", "Number of Base", "Number of Cache"]
    data = [[format_number(user_info["voting_weight"].iloc[0]), user_info["n_posrated"].iloc[0], user_info["n_negrated"].iloc[0], user_info["n_base"].iloc[0], user_info["n_cache"].iloc[0]]]
    return columns, data

def get_user_folds_tables(user_results_before_averaging_over_folds : pd.DataFrame, user_results_after_averaging_over_folds : pd.DataFrame) -> tuple:
    columns_1, columns_2 = ["Fold"], ["Fold"]
    for i, score in enumerate(list(Score)):
        score_abbreviation = SCORES_DICT[score]["abbreviation"]
        if i < n_scores_halved:
            columns_1.append(score_abbreviation)
        else:
            columns_2.append(score_abbreviation)
    val_first_row_1, val_first_row_2, train_first_row_1, train_first_row_2 = ["μ"], ["μ"], ["μ"], ["μ"]
    for i, score in enumerate(list(Score)):
        if i < n_scores_halved:
            val_first_row_1.append(format_number(user_results_after_averaging_over_folds[f"val_{score.name.lower()}_mean"].values[0]))
            train_first_row_1.append(format_number(user_results_after_averaging_over_folds[f"train_{score.name.lower()}_mean"].values[0]))
        else:
            val_first_row_2.append(format_number(user_results_after_averaging_over_folds[f"val_{score.name.lower()}_mean"].values[0]))
            train_first_row_2.append(format_number(user_results_after_averaging_over_folds[f"train_{score.name.lower()}_mean"].values[0]))
    val_table_1, val_table_2, train_table_1, train_table_2 = [val_first_row_1], [val_first_row_2], [train_first_row_1], [train_first_row_2]
    n_folds = len(user_results_before_averaging_over_folds)
    for fold_idx in range(n_folds):
        val_row_1, val_row_2, train_row_1, train_row_2 = [fold_idx], [fold_idx], [fold_idx], [fold_idx]
        for i, score in enumerate(list(Score)):
            if i < n_scores_halved:
                val_row_1.append(format_number(user_results_before_averaging_over_folds.loc[user_results_before_averaging_over_folds["fold_idx"] == fold_idx, f"val_{score.name.lower()}"].values[0]))
                train_row_1.append(format_number(user_results_before_averaging_over_folds.loc[user_results_before_averaging_over_folds["fold_idx"] == fold_idx, f"train_{score.name.lower()}"].values[0]))
            else:
                val_row_2.append(format_number(user_results_before_averaging_over_folds.loc[user_results_before_averaging_over_folds["fold_idx"] == fold_idx, f"val_{score.name.lower()}"].values[0]))
                train_row_2.append(format_number(user_results_before_averaging_over_folds.loc[user_results_before_averaging_over_folds["fold_idx"] == fold_idx, f"train_{score.name.lower()}"].values[0]))
        val_table_1.append(val_row_1)
        val_table_2.append(val_row_2)
        train_table_1.append(train_row_1)
        train_table_2.append(train_row_2)
    val_last_row_1, val_last_row_2, train_last_row_1, train_last_row_2 = ["σ"], ["σ"], ["σ"], ["σ"]
    for i, score in enumerate(list(Score)):
        if i < n_scores_halved:
            val_last_row_1.append(format_number(user_results_after_averaging_over_folds[f"val_{score.name.lower()}_std"].values[0]))
            train_last_row_1.append(format_number(user_results_after_averaging_over_folds[f"train_{score.name.lower()}_std"].values[0]))
        else:
            val_last_row_2.append(format_number(user_results_after_averaging_over_folds[f"val_{score.name.lower()}_std"].values[0]))
            train_last_row_2.append(format_number(user_results_after_averaging_over_folds[f"train_{score.name.lower()}_std"].values[0]))
    val_table_1.append(val_last_row_1)
    val_table_2.append(val_last_row_2)
    train_table_1.append(train_last_row_1)
    train_table_2.append(train_last_row_2)
    return columns_1, val_table_1, train_table_1, columns_2, val_table_2, train_table_2

    
def print_first_page_for_user(pdf : PdfPages, title : str, user_info_table : tuple, columns : list, val_table : list, train_table : list) -> None:
    fig, ax = plt.subplots(figsize = (13, 8.5))
    ax.axis("off")
    ax.text(0.5, 1.11, title, fontsize = 16, ha = 'center', va = 'center', fontweight = 'bold')
    user_info_columns, data = user_info_table
    print_table(data, [-0.125, 0.98, 1.2, 0.075], user_info_columns, 5 * [0.2])
    ax.text(0.5, 0.945, f"Validation Scores:", fontsize = 13, ha = 'center', va = 'top', fontweight = 'bold')
    print_table(val_table, [-0.155, 0.45, 1.275, 0.46], columns, [0.1] + (len(columns) - 1) * [0.15], bold_row = 1)
    ax.text(0.5, 0.42, f"Training Scores:", fontsize = 13, ha = 'center', va = 'top', fontweight = 'bold')
    print_table(train_table, [-0.155, -0.075, 1.275, 0.46], columns, [0.1] + (len(columns) - 1) * [0.15], bold_row = 1)
    pdf.savefig(fig)
    plt.close(fig)

def plot_hyperparameter_for_all_combinations_for_user(pdf : PdfPages, plot_df_user : pd.DataFrame, hyperparameters : list, hyperparameters_combinations_with_explicit_X_hyperparameter : pd.DataFrame, 
                                                      globally_optimal_X_hyperparameter : float, best_global_hyperparameters_combination_idx : int) -> None:
    X_hyperparameter = PLOT_CONSTANTS["X_HYPERPARAMETER"]
    title = f"X_Hyperparameter: {X_hyperparameter}.\n"
    hyperparameters_without_X_hyperparameter = [hyperparameter for hyperparameter in hyperparameters if hyperparameter != X_hyperparameter]
    if len(hyperparameters_without_X_hyperparameter) == 0:
        included_combinations = (sorted(hyperparameters_combinations_with_explicit_X_hyperparameter.loc[hyperparameters_combinations_with_explicit_X_hyperparameter
                                ["non_X_hyperparameter_combination_idx"] == -1]["combination_idx"]))
        title += f"Combinations: {str(included_combinations)}.\n"
        plot_hyperparameter(pdf, plot_df_user, title, False, globally_optimal_X_hyperparameter, None)
    else:
        non_X_hyperparameter_combinations = len(hyperparameters_combinations_with_explicit_X_hyperparameter["non_X_hyperparameter_combination_idx"].unique())
        for non_X_hyperparameter_combination_idx in range(non_X_hyperparameter_combinations):
            included_combinations = (sorted(hyperparameters_combinations_with_explicit_X_hyperparameter.loc[hyperparameters_combinations_with_explicit_X_hyperparameter
                                    ["non_X_hyperparameter_combination_idx"] == non_X_hyperparameter_combination_idx]["combination_idx"]))
            plot_subset_title = title + f"Combinations: {str(included_combinations)}.\n"
            non_X_hyparameter_combinations_first_row_with_idx = (hyperparameters_combinations_with_explicit_X_hyperparameter.loc
                        [hyperparameters_combinations_with_explicit_X_hyperparameter["non_X_hyperparameter_combination_idx"] == non_X_hyperparameter_combination_idx].iloc[0])
            plot_df_subset = plot_df_user[plot_df_user["non_X_hyperparameter_combination_idx"] == non_X_hyperparameter_combination_idx]
            for i, hyperparameter in enumerate(hyperparameters_without_X_hyperparameter):
                plot_subset_title += f"{',  ' if i > 0 else ''}{hyperparameter}: {format_number(non_X_hyparameter_combinations_first_row_with_idx[hyperparameter])}"
            vertical_line = globally_optimal_X_hyperparameter if best_global_hyperparameters_combination_idx in included_combinations else None
            plot_hyperparameter(pdf, plot_df_subset, plot_subset_title + ".", False, vertical_line, None)

class Classification_Outcome(Enum):
    TRUE_POSITIVE = 0
    FALSE_POSITIVE = 1
    TRUE_NEGATIVE = 2
    FALSE_NEGATIVE = 3

def get_classification_outcome(gt_label : int, prediction : int) -> Classification_Outcome:
    if gt_label == 1:
        return Classification_Outcome.TRUE_POSITIVE if prediction == 1 else Classification_Outcome.FALSE_NEGATIVE
    elif gt_label == 0:
        return Classification_Outcome.TRUE_NEGATIVE if prediction == 0 else Classification_Outcome.FALSE_POSITIVE

def merge_tfidf_scores(pos_train_scores : dict, neg_train_scores : dict) -> dict:
    pos_train_values_array, neg_train_values_array = np.array(list(pos_train_scores.values())), np.array(list(neg_train_scores.values()))
    max_word_score = max(np.max(pos_train_values_array), np.max(neg_train_values_array))
    pos_train_values_array, neg_train_values_array = pos_train_values_array / max_word_score, neg_train_values_array / max_word_score
    merged_word_scores = pos_train_values_array - neg_train_values_array
    return {word: score for word, score in zip(pos_train_scores.keys(), merged_word_scores)}

def turn_predictions_into_df(predictions : list, ids : list, labels : list) -> pd.DataFrame:
    data = []
    for paper_idx, val_prediction in enumerate(predictions):
        data.append({"paper_id": ids[paper_idx], "gt_label": labels[paper_idx], "class_1_proba": val_prediction})
    predictions_df = pd.DataFrame(data)
    predictions_df["prediction"] = predictions_df["class_1_proba"].apply(lambda x: 1 if x >= 0.5 else 0)
    predictions_df["classification_outcome"] = predictions_df.apply(lambda row: get_classification_outcome(row["gt_label"], row["prediction"]), axis = 1)
    return predictions_df

def sanitize_latex(text):
    return re.sub(r'\$.*?\$', "'LATEX_CODE'", text)

def load_title_abstract(paper_id: int) -> tuple:
    title = sanitize_latex(sql_execute("select title from papers where paper_id = :id", id = paper_id)[0][0])
    abstract = sanitize_latex(sql_execute("select abstract from papers where paper_id = :id", id = paper_id)[0][0])
    return title.rstrip(), abstract.lstrip()

def get_papers_table_data(outcome_df : pd.DataFrame) -> list:
    table_data = []
    i = 0
    for _, row in outcome_df.iterrows():
        i += 1
        paper_id = int(row["paper_id"])
        class_1_proba = format_number(row["class_1_proba"])
        title, abstract = load_title_abstract(paper_id)
        table_data.append([i, paper_id, class_1_proba, title, abstract])
    return table_data

def randomly_select_training_papers_label(fold_predictions_df : pd.DataFrame, is_positive : bool, random_seed : int = 42) -> tuple:
    fold_predictions_subset_df = fold_predictions_df[fold_predictions_df["gt_label"] == int(is_positive)]
    n_train_papers_full = len(fold_predictions_subset_df)
    n_train_papers_selection = min(PLOT_CONSTANTS["N_PAPERS_IN_TOTAL"], n_train_papers_full)
    if n_train_papers_selection < n_train_papers_full:
        fold_predictions_subset_df = fold_predictions_subset_df.sample(n = n_train_papers_selection, random_state = random_seed)
    fold_predictions_subset_df = fold_predictions_subset_df.sort_values(by = "class_1_proba", ascending = not is_positive)
    data = get_papers_table_data(fold_predictions_subset_df)
    return data, n_train_papers_full

def select_n_most_extreme_val_papers_outcome(fold_predictions_df : pd.DataFrame, classification_outcome : Classification_Outcome, descending : bool) -> tuple:
    fold_predictions_df_classification_outcome = fold_predictions_df[fold_predictions_df["classification_outcome"] == classification_outcome]
    n_val_papers_outcome_full = len(fold_predictions_df_classification_outcome)
    n_val_papers_outcome_selection = min(PLOT_CONSTANTS["N_PAPERS_IN_TOTAL"], n_val_papers_outcome_full)
    fold_predictions_df_classification_outcome = (fold_predictions_df_classification_outcome.nlargest(n_val_papers_outcome_selection, "class_1_proba") if descending 
                                                 else fold_predictions_df_classification_outcome.nsmallest(n_val_papers_outcome_selection, "class_1_proba"))
    data = get_papers_table_data(fold_predictions_df_classification_outcome)
    return data, n_val_papers_outcome_full

def select_n_most_extreme_val_pos_papers(fold_predictions_df : pd.DataFrame) -> tuple:
    fold_predictions_df_pos = fold_predictions_df[fold_predictions_df["gt_label"] == 1]
    n_val_pos_papers_full = len(fold_predictions_df_pos)
    n_val_pos_papers_selection = min(100, n_val_pos_papers_full)
    fold_predictions_df_pos = fold_predictions_df_pos.nsmallest(n_val_pos_papers_selection, "class_1_proba").sort_values(by = "class_1_proba", ascending = False)
    data = get_papers_table_data(fold_predictions_df_pos)
    return data, n_val_pos_papers_full

def train_wordclouds(train_ids : list, train_labels : list, val_ids : list,  n_tfidf_features : int = 5000) -> tuple:
    paper_ids = train_ids + val_ids
    v = train_vectorizer_for_user(paper_ids, n_tfidf_features)
    features = v.get_feature_names_out()
    pos_train_ids = [paper_id for paper_id, label in zip(train_ids, train_labels) if label == 1]
    neg_train_ids = [paper_id for paper_id, label in zip(train_ids, train_labels) if label == 0]
    pos_train_mean_embedding, neg_train_mean_embedding = get_mean_embedding(v, pos_train_ids), get_mean_embedding(v, neg_train_ids)
    pos_train_scores = {word: score for word, score in zip(features, pos_train_mean_embedding.A1)}
    neg_train_scores = {word: score for word, score in zip(features, neg_train_mean_embedding.A1)}
    return pos_train_scores, neg_train_scores

def generate_wordclouds(pdf : PdfPages, wc_pos_train_scores : dict, wc_neg_train_scores : dict, use_tfidf_coefs : bool,
                        n_pos_train_papers_full : int = None, n_neg_train_papers_full : int = None,
                        largest_pos_tfidf_coef : float = None, largest_neg_tfidf_coef : float = None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = PLOT_CONSTANTS["FIG_SIZE"])
    suptitle = "Word Clouds based on Coefficients of TF-IDF Classifer." if use_tfidf_coefs else "Word Clouds based on Mean Embeddings of Training Set."
    fig.suptitle(suptitle, fontsize = 16, fontweight = 'bold')
    color_palette = ["#453430", "#7D3E3E", "#9E7F61", "#362B2B", "#564747", "#827676", "#9B8D8C", "#B7A5A0", "#D1BDB5"]

    def generate_wordcloud(word_scores, ax, title):
        if word_scores is None:
            ax.text(0.5, 0.5, "No data available.", ha='center', va='center', fontsize = 18)
            ax.axis('off')
            ax.set_title(title, fontweight = 'bold', pad = 10)
            return
        wordcloud = WordCloud(width = 400, height = 550, background_color = '#F8F8F8', max_words = 50, random_state = 42, 
                              colormap = colors.ListedColormap(color_palette)).generate_from_frequencies(word_scores)
        ax.imshow(wordcloud, interpolation = 'bilinear')
        ax.axis('off')
        ax.set_title(title, fontweight = 'bold', pad = 15)
        
    if use_tfidf_coefs:
        pos_title = f"Positive TF-IDF Classifier Coefficients\n (Total N = {len(wc_pos_train_scores)}, Largest Coef = {format_number(largest_pos_tfidf_coef, 3)})."
        neg_title = f"Negative TF-IDF Classifier Coefficients\n (Total N = {len(wc_neg_train_scores)}, Largest Coef = {format_number(largest_neg_tfidf_coef, 3)})."
    else:
        pos_title = f"Positively Rated Training Papers\n (Total N = {n_pos_train_papers_full})."
        neg_title = f"Negatively Rated Training Papers\n (Total N = {n_neg_train_papers_full})."
    generate_wordcloud(wc_pos_train_scores, ax1, pos_title)
    generate_wordcloud(wc_neg_train_scores, ax2, neg_title)
    plt.tight_layout(pad=2.5)
    pdf.savefig(fig)
    plt.close(fig)

def score_to_color(score):
    if score > 0:
        return '#0000FF'
    elif score < 0:
        return '#FF0000'
    else:
        return '#000000'

def score_to_weight(score):
    if abs(score) <= 0.05:
        return 475
    return int(450 + min(abs(score), 1) * 550)

def score_text_with_tfidf(text: str, tfidf_scores: dict) -> list:
    max_ngram = max(len(key.split()) for key in tfidf_scores.keys())
    words, scored_words = text.split(), []
    i = 0
    while i < len(words):
        best_match = ""
        best_score = 0.0
        for n in range(min(max_ngram, len(words) - i), 0, -1):
            ngram = " ".join(words[i:i+n])
            ngram_lower = ngram.lower()
            if ngram_lower in tfidf_scores:
                best_match = ngram
                best_score = tfidf_scores[ngram_lower]
                break
        if best_match:
            best_match_joined = best_match.split()
            scored_words.extend([(best_match_joined[i], best_score) for i in range(len(best_match_joined))])
            i += len(best_match_joined)
        else:
            scored_words.append((words[i], 0.0))
            i += 1
    return scored_words

def plot_papers(pdf : PdfPages, table_data : list, table_name : str, word_scores : dict) -> None:
    if len(table_data) == 0:
        return
    plt.rcParams["text.usetex"] = False
    fig, ax = plt.subplots(figsize = PLOT_CONSTANTS["FIG_SIZE"])
    ax.axis('off')
    ax.text(0.5, 1.12, table_name, fontsize = 12, ha = 'center', va = 'center', weight = 'bold')
    for i in range(len(table_data)):
        s = f"#{table_data[i][0]}:   ID: {table_data[i][1]},   Class 1 Proba: {table_data[i][2]}." 
        ax.text(0.5, 1.0675 - i * 0.175, s, fontsize = 10.5, ha = 'center', va = 'center', wrap = True, weight = 'bold')
        title, abstract = table_data[i][-2], table_data[i][-1]
        title_scores, abstract_scores = score_text_with_tfidf(title, word_scores), score_text_with_tfidf(abstract, word_scores)
        tfidf_scores = title_scores + abstract_scores
        last_title_index = len(title_scores) - 1
        x, y = PLOT_CONSTANTS["X_LOCATION"], 1.07 - i * 0.175 - 0.0175
        word_counter, line_counter = 0, 0
        for word, score in tfidf_scores:
            if word_counter == last_title_index:
                text_obj = ax.text(x, y, f"{word}.  ", fontsize = 10, ha = 'left', va = 'top', color = score_to_color(score), weight = score_to_weight(score))
            else:
                text_obj = ax.text(x, y, word, fontsize = 10, ha = 'left', va = 'top', color = score_to_color(score), weight = score_to_weight(score))
            bb = text_obj.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
            bb_data = bb.transformed(ax.transData.inverted())
            x = bb_data.x1 + PLOT_CONSTANTS["WORD_SPACING"]
            if x > 0.97:
                line_counter += 1
                if line_counter == PLOT_CONSTANTS["MAX_LINES"]:
                    text_obj.set_text("...")
                    break
                x = PLOT_CONSTANTS["X_LOCATION"]
                y -= PLOT_CONSTANTS["LINE_HEIGHT"]
            word_counter += 1
    pdf.savefig(fig)
    plt.close()

def plot_training_papers_class(pdf : PdfPages, train_papers_selection : list, n_train_papers_full : int, word_scores : dict, is_positive : bool) -> None:
    n_train_papers_selection = len(train_papers_selection)
    train_papers_selection_n_pages = ((n_train_papers_selection - 1) // PLOT_CONSTANTS["N_PAPERS_PER_PAGE"]) + 1
    for i in range(train_papers_selection_n_pages):
        title = (f"Random Selection of {n_train_papers_selection} {'Positively' if is_positive else 'Negatively'} " +
                f"Rated Training Papers (Total N = {n_train_papers_full}), Page {i+1} / {train_papers_selection_n_pages}.") 
        plot_papers(pdf, train_papers_selection[i * PLOT_CONSTANTS["N_PAPERS_PER_PAGE"] : (i + 1) * PLOT_CONSTANTS["N_PAPERS_PER_PAGE"]], title, word_scores)

def plot_training_papers(pdf : PdfPages, pos_train_papers_selection : list, n_pos_train_papers_full : int, neg_train_papers_selection : list, n_neg_train_papers_full : int, 
                         word_scores : dict) -> None:
    plot_training_papers_class(pdf, pos_train_papers_selection, n_pos_train_papers_full, word_scores, True)
    plot_training_papers_class(pdf, neg_train_papers_selection, n_neg_train_papers_full, word_scores, False)

def plot_true_validation_papers_classification_outcome(pdf : PdfPages, true_val_papers_selection : list, n_true_val_papers_full : int, word_scores : dict, 
                                                       classification_outcome : Classification_Outcome) -> None:
    n_true_val_papers_selection = len(true_val_papers_selection)
    true_val_papers_selection_n_pages = ((n_true_val_papers_selection - 1) // PLOT_CONSTANTS["N_PAPERS_PER_PAGE"]) + 1
    title = (f"Collection of the {n_true_val_papers_selection} Most Extremely Rated Validation " + 
            f"{'True Positives' if classification_outcome == Classification_Outcome.TRUE_POSITIVE else 'True Negatives'} (Total N = {n_true_val_papers_full}),")
    for i in range(true_val_papers_selection_n_pages):
        i_title = title + f" Page {i+1} / {true_val_papers_selection_n_pages}."
        plot_papers(pdf, true_val_papers_selection[i * PLOT_CONSTANTS["N_PAPERS_PER_PAGE"] : (i + 1) * PLOT_CONSTANTS["N_PAPERS_PER_PAGE"]], i_title, word_scores)

def plot_true_validation_papers(pdf : PdfPages, true_pos_val_papers_selection : list, n_true_pos_val_papers_full : int, true_neg_val_papers_selection : list, n_true_neg_val_papers_full : int, 
                                word_scores : dict) -> None:
        plot_true_validation_papers_classification_outcome(pdf, true_pos_val_papers_selection, n_true_pos_val_papers_full, word_scores, Classification_Outcome.TRUE_POSITIVE)
        plot_true_validation_papers_classification_outcome(pdf, true_neg_val_papers_selection, n_true_neg_val_papers_full, word_scores, Classification_Outcome.TRUE_NEGATIVE)

def plot_false_validation_papers_classification_outcome(pdf : PdfPages, false_val_papers_selection : list, n_false_val_papers_full : int, word_scores : dict, cosine_similarities : dict, 
                                                        pos_train_papers_selection : list, neg_train_papers_selection : list, classification_outcome : Classification_Outcome) -> None:
    n_false_val_papers_selection = len(false_val_papers_selection)
    title = (f"Collection of the {n_false_val_papers_selection} Most Extremely Rated Validation " + 
             f"{'False Positives' if classification_outcome == Classification_Outcome.FALSE_POSITIVE else 'False Negatives'} (Total N = {n_false_val_papers_full})")
    for i in range(n_false_val_papers_selection):
        plot_single_paper(pdf, false_val_papers_selection[i], title, word_scores, cosine_similarities, pos_train_papers_selection, neg_train_papers_selection)

def plot_false_validation_papers(pdf : PdfPages, false_pos_val_papers_selection : list, n_false_pos_val_papers_full : int, false_neg_val_papers_selection : list, n_false_neg_val_papers_full : int, 
                                 word_scores : dict, cosine_similarities : dict, pos_train_papers_selection : list, neg_train_papers_selection : list) -> None:
    plot_false_validation_papers_classification_outcome(pdf, false_pos_val_papers_selection, n_false_pos_val_papers_full, word_scores, cosine_similarities, 
                                                        pos_train_papers_selection, neg_train_papers_selection, Classification_Outcome.FALSE_POSITIVE)
    plot_false_validation_papers_classification_outcome(pdf, false_neg_val_papers_selection, n_false_neg_val_papers_full, word_scores, cosine_similarities, 
                                                        pos_train_papers_selection, neg_train_papers_selection, Classification_Outcome.FALSE_NEGATIVE)

def plot_ranking_predictions(pdf : PdfPages, papers_selection : list, n_papers_full : int, pos_val_papers_selection : list, n_pos_val_papers_full : int,
                             words_scores : dict, cosine_similarities : dict, pos_train_papers_selection : list, neg_train_papers_selection : list, is_negrated_ranking : bool) -> None:
    plot_ranking_overview(pdf, papers_selection, n_papers_full, pos_val_papers_selection, n_pos_val_papers_full)
    n_papers_selection = min(25, len(papers_selection))
    if is_negrated_ranking:
        title = f"Collection of all randomly selected Explicit Negative Papers (Total N = {n_papers_full})"
    else:
        title = f"Collection of the {n_papers_selection} Most Highly Rated Random Negative Samples (Total N = {n_papers_full})"
    for i in range(n_papers_selection):
        plot_single_paper(pdf, papers_selection[i], title, words_scores, cosine_similarities, pos_train_papers_selection, neg_train_papers_selection, samples = not is_negrated_ranking)

def plot_ranking_overview(pdf : PdfPages, papers_selection : list, n_papers_full : int, pos_val_papers_selection : list, n_pos_val_papers_full : int) -> None:
    fig, ax = plt.subplots(figsize = PLOT_CONSTANTS["FIG_SIZE"])
    ax.axis('off')

    colors = ['#ffffff', '#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000']
    n_bins = len(colors)
    custom_cmap = LinearSegmentedColormap.from_list('custom_YlOrRd', colors, N = n_bins)

    papers_ax = fig.add_axes([0.01, 0.52, 1.1, 0.45])
    papers_data = reshape_to_n_rows_10_cols(np.array([paper[2] for paper in papers_selection]), 10)
    im_papers = papers_ax.imshow(papers_data, cmap = custom_cmap, vmin=0, vmax=1, aspect='auto')
    add_grid_lines(papers_ax, papers_data)
    counter = 1
    for j in range(papers_data.shape[1]):
        for i in range(papers_data.shape[0]):
            formatted_number = format_number(papers_data[i, j], max_decimals = 3)
            if formatted_number != "nan":
                papers_ax.text(j, i, f"#{counter}:  {formatted_number}\n ID: {papers_selection[counter-1][1]}", ha='center', va='center', color = 'black', fontsize=8)
            counter += 1     
    papers_ax.set_title(f'Model Probabilities for the {len(papers_selection)} Most Highly Rated Papers (Total N = {n_papers_full})', pad = 3, fontsize = 9, fontweight = 'bold')
    papers_ax.set_xticks([])
    papers_ax.set_yticks([])
    cbar_papers = plt.colorbar(im_papers, ax=papers_ax, orientation='vertical', pad = 0.01)
    cbar_papers.ax.tick_params(labelsize=9)

    pos_val_ax = fig.add_axes([0.01, 0.03, 1.1, 0.45])
    pos_val_data = reshape_to_n_rows_10_cols(np.array([pos_val_paper[2] for pos_val_paper in pos_val_papers_selection]), 10)
    im_pos_val = pos_val_ax.imshow(pos_val_data,
                                   cmap = custom_cmap, vmin=0, vmax=1, aspect='auto')
    add_grid_lines(pos_val_ax, pos_val_data)
    counter = 1
    for j in range(pos_val_data.shape[1]):
        for i in range(pos_val_data.shape[0]):
            formatted_number = format_number(pos_val_data[i, j], max_decimals = 3)
            if formatted_number != "nan":
                pos_val_ax.text(j, i, f"#{counter}:  {formatted_number}\n ID: {pos_val_papers_selection[counter-1][1]}", ha='center', va='center', color = 'black', fontsize=8)
            counter += 1
    pos_val_ax.set_title(f'Model Probabilities for the {len(pos_val_papers_selection)} Least Highly Rated Validation GT-Positives (Total N = {n_pos_val_papers_full})', pad = 3, fontsize = 9, fontweight = 'bold')
    pos_val_ax.set_xticks([])
    pos_val_ax.set_yticks([])
    cbar_pos_val = plt.colorbar(im_pos_val, ax=pos_val_ax, orientation='vertical', pad = 0.01)
    cbar_pos_val.ax.tick_params(labelsize=9)
    pdf.savefig(fig)
    plt.close()

def plot_ranking_overview(pdf : PdfPages, negative_samples_selection : list, n_negative_samples_full : int, pos_val_papers_selection : list, n_pos_val_papers_full : int) -> None:
    fig, ax = plt.subplots(figsize = PLOT_CONSTANTS["FIG_SIZE"])
    ax.axis('off')

    colors = ['#ffffff', '#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000']
    n_bins = len(colors)
    custom_cmap = LinearSegmentedColormap.from_list('custom_YlOrRd', colors, N = n_bins)

    neg_samples_ax = fig.add_axes([0.01, 0.52, 1.1, 0.45])
    neg_samples_data = reshape_to_n_rows_10_cols(np.array([neg_sample[2] for neg_sample in negative_samples_selection]), 10)
    im_neg_samples = neg_samples_ax.imshow(neg_samples_data, cmap = custom_cmap, vmin=0, vmax=1, aspect='auto')
    add_grid_lines(neg_samples_ax, neg_samples_data)
    counter = 1
    for j in range(neg_samples_data.shape[1]):
        for i in range(neg_samples_data.shape[0]):
            formatted_number = format_number(neg_samples_data[i, j], max_decimals = 3)
            if formatted_number != "nan":
                neg_samples_ax.text(j, i, f"#{counter}:  {formatted_number}\n ID: {negative_samples_selection[counter-1][1]}", ha='center', va='center', color = 'black', fontsize=8)
            counter += 1     
    neg_samples_ax.set_title(f'Model Probabilities for the {len(negative_samples_selection)} Most Highly Rated Negative Samples (Total N = {n_negative_samples_full})', pad = 3, fontsize = 9, fontweight = 'bold')
    neg_samples_ax.set_xticks([])
    neg_samples_ax.set_yticks([])
    cbar_neg_samples = plt.colorbar(im_neg_samples, ax=neg_samples_ax, orientation='vertical', pad = 0.01)
    cbar_neg_samples.ax.tick_params(labelsize=9)

    pos_val_ax = fig.add_axes([0.01, 0.03, 1.1, 0.45])
    pos_val_data = reshape_to_n_rows_10_cols(np.array([pos_val_paper[2] for pos_val_paper in pos_val_papers_selection]), 10)
    im_pos_val = pos_val_ax.imshow(pos_val_data, cmap = custom_cmap, vmin=0, vmax=1, aspect='auto')
    add_grid_lines(pos_val_ax, pos_val_data)
    counter = 1
    for j in range(pos_val_data.shape[1]):
        for i in range(pos_val_data.shape[0]):
            formatted_number = format_number(pos_val_data[i, j], max_decimals = 3)
            if formatted_number != "nan":
                pos_val_ax.text(j, i, f"#{counter}:  {formatted_number}\n ID: {pos_val_papers_selection[counter-1][1]}", ha='center', va='center', color = 'black', fontsize=8)
            counter += 1
    pos_val_ax.set_title(f'Model Probabilities for the {len(pos_val_papers_selection)} Least Highly Rated Validation GT-Positives (Total N = {n_pos_val_papers_full})', pad = 3, fontsize = 9, fontweight = 'bold')
    pos_val_ax.set_xticks([])
    pos_val_ax.set_yticks([])
    cbar_pos_val = plt.colorbar(im_pos_val, ax=pos_val_ax, orientation='vertical', pad = 0.01)
    cbar_pos_val.ax.tick_params(labelsize=9)

    pdf.savefig(fig)
    plt.close()

def preprocess_cosine_similarities(paper_id : int, cosine_similarities : dict, pos_train_papers_selection : list, neg_train_papers_selection : list, samples : bool = False) -> tuple:
    ids_to_idxs = {paper_id: idx for idx, paper_id in enumerate(cosine_similarities["posrated_ids"] + cosine_similarities["negrated_ids"])}
    if samples:
        ids_to_idxs_samples = {paper_id: idx for idx, paper_id in enumerate(cosine_similarities["negative_samples_ids"])}
        relevant_row = cosine_similarities["negative_samples_cosine_similarities"][ids_to_idxs_samples[paper_id]]
    else:
        relevant_row = cosine_similarities["rated_cosine_similarities"][ids_to_idxs[paper_id]]
    pos_relevant_row, neg_relevant_row = np.zeros(len(pos_train_papers_selection)), np.zeros(len(neg_train_papers_selection))
    pos_relevant_ids_row, neg_relevant_ids_row = np.zeros(len(pos_train_papers_selection), dtype = np.int32), np.zeros(len(neg_train_papers_selection), dtype = np.int32)
    for i in range(len(pos_relevant_row)):
        pos_relevant_row[i] = relevant_row[ids_to_idxs[pos_train_papers_selection[i][1]]]
        pos_relevant_ids_row[i] = pos_train_papers_selection[i][1]
    for i in range(len(neg_relevant_row)):
        neg_relevant_row[i] = relevant_row[ids_to_idxs[neg_train_papers_selection[i][1]]]
        neg_relevant_ids_row[i] = neg_train_papers_selection[i][1]
    return pos_relevant_row, pos_relevant_ids_row, neg_relevant_row, neg_relevant_ids_row

def reshape_to_n_rows_10_cols(arr, n):
    result = np.full((n, 10), np.nan)
    n_entries = len(arr)
    n_full_cols = min(n_entries // n, 10)
    for col in range(n_full_cols):
        result[:, col] = arr[col*n:(col+1)*n]    
    remaining_entries = n_entries - (n_full_cols * n)
    if remaining_entries > 0 and n_full_cols < 10:
        result[:remaining_entries, n_full_cols] = arr[n_full_cols*n:n_entries]
    return result

def add_grid_lines(ax, data):
        for x in range(data.shape[1] + 1):
            ax.axvline(x - 0.5, color='black', linewidth=0.5, alpha=0.5)
        for y in range(data.shape[0] + 1):
            ax.axhline(y - 0.5, color='black', linewidth=0.5, alpha=0.5)

def plot_single_paper(pdf: PdfPages, table_data: list, page_title: str, word_scores: dict, cosine_similarities: dict, pos_train_papers_selection: list, neg_train_papers_selection: list,
                      samples : bool = False) -> None:
    preprocess_cosine = preprocess_cosine_similarities(table_data[1], cosine_similarities, pos_train_papers_selection, neg_train_papers_selection, samples)
    pos_relevant_row, pos_relevant_ids_row, neg_relevant_row, neg_relevant_ids_row = preprocess_cosine
    plt.rcParams["text.usetex"] = False
    fig, ax = plt.subplots(figsize = PLOT_CONSTANTS["FIG_SIZE"])
    ax.axis('off')
    s = f"#{table_data[0]}:   ID: {table_data[1]},   Class 1 Proba: {table_data[2]}." 
    ax.text(0.5, 1.12, page_title + f"\n{s}", fontsize = 11, ha = 'center', va = 'center', weight = 'bold')

    title, abstract = table_data[-2], table_data[-1]
    title_scores, abstract_scores = score_text_with_tfidf(title, word_scores), score_text_with_tfidf(abstract, word_scores)
    tfidf_scores = title_scores + abstract_scores
    last_title_index = len(title_scores) - 1
    x, y = PLOT_CONSTANTS["X_LOCATION"], 1.08
    word_counter, line_counter = 0, 0
    for word, score in tfidf_scores:
        if word_counter == last_title_index:
            text_obj = ax.text(x, y, f"{word}.  ", fontsize = 10, ha = 'left', va = 'top', color = score_to_color(score), weight = score_to_weight(score))
        else:
            text_obj = ax.text(x, y, word, fontsize = 10, ha = 'left', va = 'top', color = score_to_color(score), weight = score_to_weight(score))
        bb = text_obj.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        bb_data = bb.transformed(ax.transData.inverted())
        x = bb_data.x1 + PLOT_CONSTANTS["WORD_SPACING"]
        if x > 0.97:
            line_counter += 1
            if line_counter == 15:
                text_obj.set_text("...")
                break
            x = PLOT_CONSTANTS["X_LOCATION"]
            y -= PLOT_CONSTANTS["LINE_HEIGHT"]
        word_counter += 1

    colors = ['#ffffff', '#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000']
    n_bins = len(colors)
    custom_cmap = LinearSegmentedColormap.from_list('custom_YlOrRd', colors, N=n_bins)

    pos_ax = fig.add_axes([0.01, 0.33, 1.1, 0.29])
    pos_data = reshape_to_n_rows_10_cols(pos_relevant_row, 7)
    pos_data_ids = reshape_to_n_rows_10_cols(pos_relevant_ids_row, 7)
    im_pos = pos_ax.imshow(pos_data, cmap = custom_cmap, vmin=0, vmax=1, aspect='auto')
    add_grid_lines(pos_ax, pos_data)

    counter = 1
    for j in range(pos_data.shape[1]):
        for i in range(pos_data.shape[0]):
            formatted_number = format_number(pos_data[i, j], max_decimals = 3)
            if formatted_number != "nan":
                pos_ax.text(j, i, f"#{counter}:  {formatted_number}\nID: {int(pos_data_ids[i, j])}", ha='center', va='center', color = 'black', fontsize=8)
            counter += 1
            
    pos_ax.set_title(f'Cosine Similarities to the Selection of {len(pos_relevant_row)} positively rated Papers:', pad = 3, fontsize = 9, fontweight = 'bold')
    pos_ax.set_xticks([])
    pos_ax.set_yticks([])
    cbar_pos = plt.colorbar(im_pos, ax=pos_ax, orientation='vertical', pad = 0.01)
    cbar_pos.ax.tick_params(labelsize=9)

    neg_ax = fig.add_axes([0.01, 0.01, 1.1, 0.29])
    neg_data = reshape_to_n_rows_10_cols(neg_relevant_row, 7)
    neg_data_ids = reshape_to_n_rows_10_cols(neg_relevant_ids_row, 7)
    im_neg = neg_ax.imshow(neg_data, cmap = custom_cmap, vmin=0, vmax=1, aspect='auto')
    add_grid_lines(neg_ax, neg_data)

    counter = 1
    for j in range(neg_data.shape[1]):
        for i in range(neg_data.shape[0]):
            formatted_number = format_number(neg_data[i, j], max_decimals = 3)
            if formatted_number != "nan":
                neg_ax.text(j, i, f"#{counter}:  {formatted_number}\nID: {int(neg_data_ids[i, j])}", ha='center', va='center', color = 'black', fontsize=8)
            counter += 1
    neg_ax.set_title(f'Cosine Similarities to the Selection of {len(neg_relevant_row)} negatively rated Papers:', pad = 3, fontsize = 9, fontweight = 'bold')
    neg_ax.set_xticks([])
    neg_ax.set_yticks([])
    cbar_neg = plt.colorbar(im_neg, ax=neg_ax, orientation='vertical', pad = 0.01)
    cbar_neg.ax.tick_params(labelsize=9)
    
    pdf.savefig(fig)
    plt.close()

def get_tfidf_coefs_scores(feature_names : list, tfidf_coefs : list) -> tuple:
    tfidf_coefs = np.array(tfidf_coefs)
    largest_pos_tfidf_coef, largest_neg_tfidf_coef = np.max(tfidf_coefs[tfidf_coefs >= 0]), np.min(tfidf_coefs[tfidf_coefs < 0])
    largest_abs_coef = max(largest_pos_tfidf_coef, -largest_neg_tfidf_coef)
    tfidf_coefs /= largest_abs_coef
    wc_words_scores = {feature_names[i]: tfidf_coefs[i] for i in range(len(feature_names))}
    wc_pos_train_scores = {feature_names[i]: tfidf_coefs[i] for i in range(len(feature_names)) if tfidf_coefs[i] >= 0}
    wc_neg_train_scores = {feature_names[i]: tfidf_coefs[i] for i in range(len(feature_names)) if tfidf_coefs[i] < 0}
    return wc_pos_train_scores, wc_neg_train_scores, wc_words_scores, largest_pos_tfidf_coef, largest_neg_tfidf_coef