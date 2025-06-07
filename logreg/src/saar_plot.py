from visualize_globally import Global_Visualizer
from visualization_tools import *
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
import os
scores = [Score.RECALL, Score.SPECIFICITY, Score.BALANCED_ACCURACY]
scores_abbrevations = {Score.RECALL: "Recall", Score.SPECIFICITY: "Specificity", Score.BALANCED_ACCURACY: "Balanced Accuracy"}
scores_mean_columns = [f"train_{score.name.lower()}_mean" for score in scores] + [f"val_{score.name.lower()}_mean" for score in scores]
scores_std_columns = [f"train_{score.name.lower()}_std" for score in scores] + [f"val_{score.name.lower()}_std" for score in scores]
scores_columns = ["combination_idx"] + scores_mean_columns + scores_std_columns

hyperparameters_abbreviations = {"clf_C": "C"}
hyperparameters = list(hyperparameters_abbreviations.keys())
hyperparameters_gvs = {}

outputs_folder = "outputs/c"
config, users_info, hyperparameters_combinations, results_before_averaging_over_folds = load_outputs_files(outputs_folder)
gv = Global_Visualizer(config, users_info, hyperparameters_combinations, results_before_averaging_over_folds, outputs_folder, Score.BALANCED_ACCURACY, False)
hyperparameter_results_df = gv.results_after_averaging_over_users[scores_columns]
hyperparameter_results_df = hyperparameter_results_df.merge(gv.hyperparameters_combinations[["combination_idx", "clf_C"]], on = "combination_idx")


def plot_score(axs, results_df : pd.DataFrame, score : Score, hyperparameter : str) -> tuple:
    score_name, subtitle = score.name.lower(), SCORES_DICT[score]["name"]
    alpha_plot, alpha_fill, line_width, X_hyperparameter = PLOT_CONSTANTS["ALPHA_PLOT"], PLOT_CONSTANTS["ALPHA_FILL"], PLOT_CONSTANTS["LINE_WIDTH"], hyperparameter

    best_X_hyperparameter_idx = results_df[f"val_{score_name}_mean"].idxmax()
    best_X_hyperparameter_value = results_df.loc[best_X_hyperparameter_idx][X_hyperparameter]
    best_X_hyperparameter_val_score = results_df.loc[best_X_hyperparameter_idx][f"val_{score_name}_mean"]
    amount = 0
    if best_X_hyperparameter_idx == results_df.index.min():
        best_X_hyperparameter_value += amount
    elif best_X_hyperparameter_idx == results_df.index.max():
        best_X_hyperparameter_value -= amount
    best_X_train_hyperparameter_idx = results_df[f"train_{score_name}_mean"].idxmax()
    best_X_train_hyperparameter_value = results_df.loc[best_X_train_hyperparameter_idx][X_hyperparameter]
    best_X_train_hyperparameter_val_score = results_df.loc[best_X_train_hyperparameter_idx][f"train_{score_name}_mean"]
    if best_X_train_hyperparameter_idx == results_df.index.min():
        best_X_train_hyperparameter_value += amount
    elif best_X_train_hyperparameter_idx == results_df.index.max():
        best_X_train_hyperparameter_value -= amount

    train_score, val_score = results_df[f'train_{score_name}_mean'], results_df[f'val_{score_name}_mean']
    train_line, = axs.plot(results_df[X_hyperparameter], train_score, label = f'Performance on Training Set', alpha = alpha_plot, linewidth = line_width)
    val_line, = axs.plot(results_df[X_hyperparameter], val_score, label = f'Performance on Validation Set', alpha = alpha_plot, linewidth = line_width)
    axs.scatter(best_X_hyperparameter_value, best_X_hyperparameter_val_score, s = 50, zorder = 5, facecolor = val_line.get_color(), edgecolor = 'black', label = 'Best Hyperparameter')
    axs.scatter(best_X_train_hyperparameter_value, best_X_train_hyperparameter_val_score, s = 50, zorder = 5, facecolor = train_line.get_color(), edgecolor = 'black', label = 'Best Hyperparameter')
    train_std_users, val_std_users = results_df[f'train_{score_name}_std'], results_df[f'val_{score_name}_std']
    axs.fill_between(results_df[X_hyperparameter], train_score - train_std_users, train_score + train_std_users, color = train_line.get_color(), alpha = alpha_fill)
    axs.fill_between(results_df[X_hyperparameter], val_score - val_std_users, val_score + val_std_users, color = "#FFD700", alpha = alpha_fill)
    return train_line, val_line



fig, ax = plt.subplots(1, 3, fig_size = PLOT_CONSTANTS["FIG_SIZE"], sharey=True)
fig.subplots_adjust(left=0.75) 
fig.tight_layout(pad = 0.5, w_pad = 3.5, h_pad = 0.05)
train_line, val_line = None, None
for i, axs in enumerate(ax.flat):
    score, hyperparameter = scores[i], "clf_C"
    results_df = hyperparameter_results_df
    axs.grid(True, linestyle='-', linewidth=1, alpha=0.2)
    axs.set_facecolor('#f8f8f8')
    if hyperparameter == "clf_C":
        log_min, log_max = np.log10(0.001), np.log10(100)
        padding_ratio = 0.02
        log_range = log_max - log_min
        log_padding = log_range * padding_ratio
        axs.set_xlim(10**(log_min - log_padding), 10**(log_max + log_padding))
        x_ticks = [0.001, 0.01, 0.1, 1, 10, 100]
        axs.set_xticks(x_ticks)
        xlabels = ["10⁻³", "10⁻²", "10⁻¹", "10⁰", "10¹", "10²"]
        axs.set_xticklabels(xlabels, fontsize=17)

        
    axs.tick_params(axis='x', pad=10) 
    axs.set_ylim(0.5, 1.01)
    yticks = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    axs.set_yticks(yticks) 
    ylabels = [f"{y:.1f}" if y in [0.5, 0.6, 0.7, 0.9, 1.0] else "" for y in yticks]
    axs.set_yticklabels(ylabels, fontsize=17)
    
    if hyperparameter == "clf_C":
        axs.set_xscale("log")
    if i % 3 != 0:
        axs.set_yticklabels([])
    if i == 0:
        axs.set_ylabel(scores_abbrevations[scores[0]], fontsize = 18, fontfamily='sans-serif', fontweight = "bold", rotation=0, color='#333333', va='center')
    if i == 3:
        axs.set_ylabel(scores_abbrevations[scores[1]], fontsize = 18, fontfamily='sans-serif', fontweight = "bold", rotation=0, color='#333333', va='center')
    if i == 6:
        axs.set_ylabel(scores_abbrevations[scores[2]], fontsize = 18, fontfamily='sans-serif', fontweight = "bold", rotation=0, color='#333333', va='center')
    if i < 6:
        axs.set_xticklabels([])
    if i > 5:
        axs.set_xlabel(hyperparameters_abbreviations[hyperparameter], fontsize=23, fontweight='bold', color='#333333')
    train_line_this, val_line_this = plot_score(axs, results_df, score, hyperparameter)
    if i == 8:
        train_line, val_line = train_line_this, val_line_this


handles = [train_line, val_line]
labels = ['Training Score', 'Validation Score']
"""
legend = ax[2].legend(handles=handles, labels=labels, loc='lower right', fontsize=17.5, framealpha=0.7)
for line in legend.get_lines():
    line.set_linewidth(3.25)
"""
plt.show()
fig.savefig('/home/scholar/glenn_rp/msc_thesis/cross_eval/ablation_study.pdf', dpi = 300, bbox_inches = 'tight')
plt.close(fig)

