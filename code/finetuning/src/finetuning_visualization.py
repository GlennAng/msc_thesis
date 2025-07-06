import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from ...logreg.src.training.algorithm import SCORES_DICT
from ...logreg.src.visualization.visualization_tools import (
    PLOT_CONSTANTS,
    PRINT_SCORES,
    format_number,
    print_table,
)


def load_visualization_files(finetuning_model_path: Path) -> tuple:
    with open(finetuning_model_path / "config.json", "r") as f:
        config = json.load(f)
    with open(finetuning_model_path / "train_losses.pkl", "rb") as f:
        train_losses = pickle.load(f)
    with open(finetuning_model_path / "val_scores.pkl", "rb") as f:
        val_scores = pickle.load(f)
    return config, train_losses, val_scores


def scores_table_to_df(scores_table: list) -> pd.DataFrame:
    scores_names, val_scores, train_scores = [], [], []
    scores_table_first_row = scores_table.pop(0)
    index_score, index_val, index_train = (
        scores_table_first_row.index("Score"),
        scores_table_first_row.index("All"),
        scores_table_first_row.index("All_T"),
    )
    for i, row in enumerate(scores_table):
        scores_names.append(row[index_score])
        val_scores.append(row[index_val])
        train_scores.append(row[index_train])
    scores_df = pd.DataFrame(
        {"Score": scores_names, "Validation": val_scores, "Train": train_scores}
    )
    scores_df["Validation"] = pd.to_numeric(scores_df["Validation"], errors="coerce")
    scores_df["Train"] = pd.to_numeric(scores_df["Train"], errors="coerce")
    return scores_df


def get_scores_dict(outputs_folder: Path) -> dict:
    if not isinstance(outputs_folder, Path):
        outputs_folder = Path(outputs_folder).resolve()
    scores_dict = {}
    for folder in os.listdir(outputs_folder):
        try:
            folder_path = outputs_folder / folder
            folder_name = " ".join([w.capitalize() for w in folder.split("_")])
            scores_dfs, random_states = [], []
            for random_state in os.listdir(folder_path):
                random_states.append(random_state.split("_")[-1][1:])
                scores_tables = []
                table_index = 1
                while True:
                    table_path = folder_path / random_state / f"scores_table_{table_index}.pkl"
                    if table_path.exists():
                        with open(table_path, "rb") as f:
                            scores_table = scores_table_to_df(pickle.load(f))
                        scores_tables.append(scores_table)
                        table_index += 1
                    else:
                        break
                if scores_tables:
                    if len(scores_tables) == 1:
                        scores_df = scores_tables[0]
                    else:
                        tables_to_concat = [scores_tables[0]] + [
                            table.iloc[1:] for table in scores_tables[1:]
                        ]
                        scores_df = pd.concat(tables_to_concat, axis=0, ignore_index=True)
                    scores_df.set_index("Score", inplace=True)
                    scores_dfs.append(scores_df)
            mean_scores_df = pd.concat(scores_dfs).groupby(level=0).mean()
            std_scores_df = pd.concat(scores_dfs).groupby(level=0).std()
            std_scores_df.fillna(0, inplace=True)
            random_states = sorted(random_states, key=lambda x: int(x))
            scores_dict[folder_name] = {
                "mean": mean_scores_df,
                "std": std_scores_df,
                "random_states": random_states,
            }
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            continue
    return scores_dict


def scores_df_to_table(mean_df: pd.DataFrame, std_df: pd.DataFrame, random_states: list) -> list:
    mean_row_val, std_row_val, mean_row_train, std_row_train = (
        ["Val_μ"],
        ["Val_σ"],
        ["Train_μ"],
        ["Train_σ"],
    )
    for i in range(len(mean_df)):
        mean_row_val.append(format_number(mean_df.iloc[i, 0]))
        std_row_val.append(format_number(std_df.iloc[i, 0]))
        mean_row_train.append(format_number(mean_df.iloc[i, 1]))
        std_row_train.append(format_number(std_df.iloc[i, 1]))
    if len(random_states) > 1:
        return [mean_row_val, std_row_val, mean_row_train, std_row_train]
    else:
        return [mean_row_val, mean_row_train]


def visualize_testing(
    finetuning_model_path: Path,
    outputs_folder: Path,
    testing_metrics: list = PRINT_SCORES,
) -> None:
    if not isinstance(outputs_folder, Path):
        outputs_folder = Path(outputs_folder).resolve()
    scores_dict = get_scores_dict(outputs_folder)
    file_name = outputs_folder.parent / "visualization.pdf"
    with PdfPages(file_name) as pdf:
        try:
            if not isinstance(finetuning_model_path, Path):
                finetuning_model_path = Path(finetuning_model_path).resolve()
            config, train_losses, val_scores = load_visualization_files(finetuning_model_path)
        except Exception as _:
            config, train_losses, val_scores = None, None, None
        if config is not None:
            visualize_config(pdf, config)
        visualize_scores(pdf, scores_dict, testing_metrics)
        if config is not None:
            visualize_training_curve(
                pdf,
                train_losses,
                val_scores,
                config["loss_function"],
                config["n_batches_per_val"],
            )


def visualize_scores(pdf: PdfPages, scores_dict: dict, testing_metrics: list) -> None:
    fig, ax = plt.subplots(figsize=PLOT_CONSTANTS["FIG_SIZE"])
    ax.axis("off")
    ax.text(
        0.5,
        1.11,
        "Testing Scores:",
        fontsize=16,
        ha="center",
        va="center",
        fontweight="bold",
    )
    testing_metrics_abbreviations = [
        SCORES_DICT[metric]["abbreviation"] for metric in testing_metrics
    ]
    scores_dict_keys_sorted = sorted(scores_dict.keys())
    num_keys = len(scores_dict_keys_sorted)
    if num_keys > 0:
        text_positions = [1.07, 0.77, 0.47, 0.17][:num_keys]
        y_height = 0.255
        columns = [""]
        for i, key in enumerate(scores_dict_keys_sorted):
            mean_df, std_df = scores_dict[key]["mean"], scores_dict[key]["std"]
            mean_df, std_df = (
                mean_df[mean_df.index.isin(testing_metrics_abbreviations)],
                std_df[std_df.index.isin(testing_metrics_abbreviations)],
            )
            mean_df, std_df = mean_df.reindex(testing_metrics_abbreviations), std_df.reindex(
                testing_metrics_abbreviations
            )
            if columns == [""]:
                columns += mean_df.index.tolist()
            random_states = scores_dict[key]["random_states"]
            scores_table = scores_df_to_table(mean_df, std_df, random_states)
            if len(random_states) > 1:
                seeds_title = f" (Seeds {', '.join(random_states)}):"
            else:
                seeds_title = f" (Seed {random_states[0]}):"
            ax.text(
                0.5,
                text_positions[i],
                key + seeds_title,
                fontsize=11,
                ha="center",
                va="center",
                fontweight="bold",
            )
            y_pos = 0.7975 - i * (y_height + 0.045)
            print_table(
                scores_table,
                [-0.14, y_pos, 1.25, y_height],
                columns,
                [0.1] + len(testing_metrics) * [0.15],
                grey_row=[1, 3],
            )
    pdf.savefig(fig)
    plt.close(fig)


def visualize_config(pdf: PdfPages, config: dict) -> None:
    fig, ax = plt.subplots(figsize=PLOT_CONSTANTS["FIG_SIZE"])
    ax.axis("off")
    ax.text(
        0.5,
        1.1,
        "Finetuning Configuration:",
        fontsize=16,
        ha="center",
        va="center",
        fontweight="bold",
    )
    ax.text(
        0.5,
        1.0,
        get_config_string(config),
        fontsize=9,
        ha="center",
        va="top",
        wrap=True,
    )
    pdf.savefig(fig)
    plt.close(fig)


def get_config_string(config: dict) -> str:
    config_string = []
    hours, minutes = divmod(config["time_elapsed"], 3600)
    config_string.append(
        f"Folder Name: {config['outputs_folder'].split('/')[-1]}   "
        f"|   Random Seed: {config['seed']}"
    )
    frozen_string = (
        f"Number of unfrozen Transformer Layers: {config['n_unfreeze_layers']} "
        f"(out of {config['n_transformer_layers']})"
    )
    n_transformer_params, n_unfrozen_transformer_params = round(
        config["n_transformer_parameters"] / 1000000
    ), round(config["n_unfrozen_transformer_parameters"] / 1000000)
    frozen_string += (
        f"   |   Number of unfrozen Transformer Parameters: {n_unfrozen_transformer_params}M "
        f" (out of {n_transformer_params}M)"
    )
    config_string.append(frozen_string)
    pretrained_string = "Layers pretrained?   "
    pretrained_string += f"Projection: {'Yes' if config['pretrained_projection'] else 'No'}"
    pretrained_string += (
        "   |   Users Embeddings: " f"{'Yes' if config['pretrained_users_embeddings'] else 'No'}"
    )
    pretrained_string += (
        "   |   Categories Embeddings: "
        f"{'Yes' if config['pretrained_categories_embeddings'] else 'No'}"
    )
    config_string.append(pretrained_string)
    config_string.append("\n")

    config_string.append(
        f"Training Time: {int(hours)}h {int(minutes)%60}m   "
        f"|   Max Number of Batches: {config['n_batches_total']//1000}K"
    )
    val_string = f"Validation Metric: {config['val_metric']}"
    val_string += (
        f"   |   Number of Batches per Validation Check: {config['n_batches_per_val']}   "
        f"|   Early Stopping Patience: {config['early_stopping_patience']}"
    )
    config_string.append(val_string)
    loss_string = f"Train Loss Function: {config['loss_function']}"
    if config["loss_function"] == "info_nce":
        loss_string += (
            f" (Temperate: {config['info_nce_temperature']}   "
            f"|   Log-Q Correction? {config['info_nce_log_q_correction']})"
        )
    config_string.append(loss_string)
    config_string.append(
        f"Number of negative Samples during Training: {config['n_train_negative_samples']}"
    )
    config_string.append("\n")

    batch_string = (
        f"Batch Size: {config['batch_size']}   "
        f"|   Users Sampling Strategy: {config['users_sampling_strategy']}"
    )
    batch_string += (
        "   |   Number of Samples in Batch per selected User: " f"{config['n_samples_per_user']}"
    )
    config_string.append(batch_string)
    min_max_string = "Min / Max Number of Samples for each User?   "
    min_max_string += (
        f"Positive: {config['n_min_positive_samples_per_user']} "
        f"/ {config['n_max_positive_samples_per_user']}"
    )
    min_max_string += (
        f"   |   Negative: {config['n_min_negative_samples_per_user']} "
        f"/ {config['n_max_negative_samples_per_user']}"
    )
    config_string.append(min_max_string)
    config_string.append("\n")

    scheduler_string = f"Learning Rate Scheduler: {config['lr_scheduler']}"
    if config["lr_scheduler"] == "linear_decay":
        scheduler_string += (
            f"   |   Percentage of Warmup Steps: {config['percentage_warmup_steps']}"
        )
    config_string.append(scheduler_string)
    config_string.append(
        f"Learning Rates:   Transformer: {config['lr_transformer_model']}   "
        f"|   Other: {config['lr_other']}"
    )
    config_string.append(
        f"L2 Regularization:   Transformer: {config['l2_regularization_transformer_model']}   "
        "|   Other: {config['l2_regularization_other']}"
    )
    return "\n\n".join(config_string)


def visualize_training_curve(
    pdf: PdfPages,
    train_losses: list,
    val_scores: list,
    loss_function: str,
    n_batches_per_val: int,
) -> None:
    train_idxs, train_losses = zip(*train_losses)
    train_idxs, train_losses = np.array(train_idxs), np.array(train_losses)
    assert len(train_losses) % n_batches_per_val == 0
    val_idxs, val_losses = zip(*val_scores)
    val_losses = [val_dict[f"val_{loss_function}"] for val_dict in val_losses]
    val_idxs, val_losses = np.array(val_idxs), np.array(val_losses)

    num_chunks = len(train_losses) // n_batches_per_val
    chunks = [
        train_losses[i * n_batches_per_val : (i + 1) * n_batches_per_val] for i in range(num_chunks)
    ]
    chunks_idxs, chunks_means, chunks_stds = [], [], []
    for i in range(num_chunks):
        chunks_idxs.append(i * n_batches_per_val + n_batches_per_val // 2)
        chunks_means.append(np.mean(chunks[i]))
        chunks_stds.append(np.std(chunks[i]))
    chunks_idxs, chunks_means, chunks_stds = (
        np.array(chunks_idxs),
        np.array(chunks_means),
        np.array(chunks_stds),
    )
    min_train_loss_idx = np.argmin(chunks_means)
    min_val_loss_idx = np.argmin(val_losses)
    min_train_loss, min_val_loss = (
        chunks_means[min_train_loss_idx],
        val_losses[min_val_loss_idx],
    )
    x, y = PLOT_CONSTANTS["FIG_SIZE"]
    fig, ax = plt.subplots(figsize=(x * 1.25, y))
    very_light_grey = "#e6e6e6"
    ax.set_facecolor(very_light_grey)
    formatted_labels = []
    for x in val_idxs:
        if x % 1000 == 0:
            formatted_labels.append(f"{int(x / 1000)}")
        else:
            formatted_labels.append("")
    ax.set_xticks(val_idxs)
    ax.set_xticklabels(formatted_labels, fontsize=10)
    all_yticks = np.arange(0, 0.65, 0.05)
    if all_yticks[-1] != 0.6:
        all_yticks = np.append(all_yticks, 0.6)
    y_tick_labels = [f"{y:.1f}" if y * 10 % 1 == 0 else "" for y in all_yticks]
    ax.set_yticks(all_yticks)
    ax.set_yticklabels(y_tick_labels, fontsize=10)
    ax.set_xlim(-50, len(train_losses) + 50)
    ax.set_ylim(0, 0.6)
    ax2 = ax.twinx()
    if loss_function == "bcel":
        ax2.set_ylim(0, 0.6)
        ax.set_ylabel("BCEL", fontsize=13)
    ax2.set_yticks([min_train_loss, min_val_loss])
    ax2.set_yticklabels([f"{min_train_loss:.3f}", f"{min_val_loss:.3f}"], fontsize=10)
    ax.plot(
        train_idxs,
        train_losses,
        label="Train Loss (per Batch)",
        linewidth=2,
        color="blue",
        alpha=0.25,
    )
    (train_line,) = ax.plot(
        chunks_idxs,
        chunks_means,
        label="Train Loss (Mean of each Chunk)",
        linewidth=2,
        color="royalblue",
        marker="o",
        markersize=3,
    )
    (val_line,) = ax.plot(
        val_idxs,
        val_losses,
        label="Validation Loss",
        linewidth=2,
        color="darkorange",
        marker="o",
        markersize=3,
    )
    ax.scatter(
        chunks_idxs[min_train_loss_idx],
        chunks_means[min_train_loss_idx],
        s=50,
        zorder=5,
        facecolor=train_line.get_color(),
        edgecolor="black",
    )
    ax.scatter(
        val_idxs[min_val_loss_idx],
        val_losses[min_val_loss_idx],
        s=50,
        zorder=5,
        facecolor=val_line.get_color(),
        edgecolor="black",
    )
    ax.set_xlabel("Batch Number (in K)", fontsize=13)

    ax.legend(fontsize=12)
    ax.grid(True, linestyle="-", alpha=0.4)
    pdf.savefig(fig)
    plt.close(fig)
