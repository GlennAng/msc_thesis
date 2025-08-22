from typing import Callable

import pandas as pd

from ..training.scores_definitions import SCORES_DICT, Score


def throw_if_missing(column_names: list, method_name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(results: pd.DataFrame, *args, **kwargs):
            for column_name in column_names:
                if column_name not in results.columns:
                    raise ValueError(
                        f"Column '{column_name}' not found in the DataFrame for '{method_name}'."
                    )
                return func(results, *args, **kwargs)

        return wrapper

    return decorator


def throw_if_present(column_names: list, method_name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(results: pd.DataFrame, *args, **kwargs):
            for column_name in column_names:
                if column_name in results.columns:
                    raise ValueError(
                        f"Column '{column_name}' should not be present in the DataFrame "
                        f"for '{method_name}'."
                    )
                return func(results, *args, **kwargs)

        return wrapper

    return decorator


def throw_if_fold_idx_missing(method_name):
    return throw_if_missing(["fold_idx"], method_name)


def throw_if_fold_idx_present(method_name):
    return throw_if_present(["fold_idx"], method_name)


def throw_if_user_id_missing(method_name):
    return throw_if_missing(["user_id"], method_name)


def throw_if_user_id_present(method_name):
    return throw_if_present(["user_id"], method_name)


def throw_if_any_score_missing(method_name):
    return throw_if_missing(list(Score), method_name)


def average_over_folds(results_before_avging_over_folds: pd.DataFrame) -> pd.DataFrame:
    """
    output columns: ['user_id', 'combination_idx'] + ['train_{score}', 'val_{score}']
    for each score in the DF
    """
    group_columns = ["user_id", "combination_idx"]
    results_after_avging_over_folds = (
        results_before_avging_over_folds.groupby(group_columns).mean().reset_index()
    )
    results_after_avging_over_folds = results_after_avging_over_folds.drop(columns=["fold_idx"])
    return results_after_avging_over_folds


def average_over_folds_with_std(results_before_avging_over_folds: pd.DataFrame) -> pd.DataFrame:
    """
    output columns: ['user_id', 'combination_idx'] + ['train_{score}_mean', 'val_{score}_mean',
    'train_{score}_std', 'val_{score}_std'] for each score in the DF
    """
    group_columns = ["user_id", "combination_idx"]
    results_means = results_before_avging_over_folds.groupby(group_columns).mean().reset_index()
    results_means = results_means.drop(columns=["fold_idx"])
    results_means = results_means.rename(
        columns={
            f"{score}": f"{score}_mean"
            for score in results_means.columns
            if score not in group_columns
        }
    )
    results_stds = results_before_avging_over_folds.groupby(group_columns).std(ddof=0).reset_index()
    results_stds = results_stds.drop(columns=["fold_idx"])
    results_stds = results_stds.rename(
        columns={
            f"{score}": f"{score}_std"
            for score in results_stds.columns
            if score not in group_columns
        }
    )
    results_after_avging_over_folds = pd.merge(
        results_means, results_stds, on=group_columns, how="inner"
    )
    return results_after_avging_over_folds


@throw_if_fold_idx_present("average_over_users")
def average_over_users(
    results_after_avging_over_folds: pd.DataFrame, medi: bool = False
) -> pd.DataFrame:
    """
    output columns: ['combination_idx] + ['train_{score}_mean', 'val_{score}_mean',
    'train_{score}_std', 'val_{score}_std'] for each score in the DF
    """
    group_columns = ["combination_idx"]
    non_group_columns = [
        column for column in results_after_avging_over_folds.columns if column not in group_columns
    ]
    results_stds = results_after_avging_over_folds.groupby(group_columns).std(ddof=1).reset_index()
    results_stds = results_stds.drop(columns=["user_id"])
    results_stds = results_stds.rename(
        columns={f"{score}": f"{score}_std" for score in non_group_columns}
    )
    if medi:
        results_avgs = results_after_avging_over_folds.groupby(group_columns).median().reset_index()
        results_avgs = results_avgs.drop(columns=["user_id"])
        results_avgs = results_avgs.rename(
            columns={f"{score}": f"{score}_medi" for score in non_group_columns}
        )
    else:
        results_avgs = results_after_avging_over_folds.groupby(group_columns).mean().reset_index()
        results_avgs = results_avgs.drop(columns=["user_id"])
        results_avgs = results_avgs.rename(
            columns={f"{score}": f"{score}_mean" for score in non_group_columns}
        )
    results_after_avging_over_users = pd.merge(
        results_avgs, results_stds, on=group_columns, how="inner"
    )
    return results_after_avging_over_users


@throw_if_fold_idx_present("get_val_upper_bounds")
def get_val_upper_bounds(results_after_avging_over_folds: pd.DataFrame, n_tail_users: int) -> dict:
    """
    output dict mapping to single floats with keys: 'val_{score}', 'val_{score}_tail'
    for each score
    """
    val_upper_bounds = {}
    for score in Score:
        val_score_name = f"val_{score.name.lower()}"
        train_score_name = f"train_{score.name.lower()}"
        if SCORES_DICT[score]["increase_better"]:
            best_combinations = (
                results_after_avging_over_folds.groupby("user_id")[
                    ["user_id", "combination_idx", val_score_name, train_score_name]
                ]
                .apply(lambda x: x.nlargest(1, val_score_name))
                .reset_index(drop=True)
            )
            best_combinations_tail = best_combinations.nsmallest(
                n_tail_users, val_score_name
            ).reset_index(drop=True)
        else:
            best_combinations = (
                results_after_avging_over_folds.groupby("user_id")[
                    ["user_id", "combination_idx", val_score_name, train_score_name]
                ]
                .apply(lambda x: x.nsmallest(1, val_score_name))
                .reset_index(drop=True)
            )
            best_combinations_tail = best_combinations.nlargest(
                n_tail_users, val_score_name
            ).reset_index(drop=True)
        val_upper_bounds[val_score_name] = best_combinations[val_score_name].mean()
        val_upper_bounds[train_score_name] = best_combinations[train_score_name].mean()
        val_upper_bounds[f"{val_score_name}_tail"] = best_combinations_tail[val_score_name].mean()
        val_upper_bounds[f"{train_score_name}_tail"] = best_combinations_tail[
            train_score_name
        ].mean()
    return val_upper_bounds


@throw_if_fold_idx_present("keep_only_n_most_extreme_hyperparameters_combinations_score")
def keep_only_n_most_extreme_hyperparameters_combinations_for_all_users_score(
    results_after_avging_over_folds: pd.DataFrame,
    score: Score,
    n_extreme_combinations: int,
    use_smallest: bool,
) -> pd.DataFrame:
    """
    output columns: just like the input DF
    """
    use_smallest_score = use_smallest if SCORES_DICT[score]["increase_better"] else not use_smallest
    extreme_func = pd.DataFrame.nsmallest if use_smallest_score else pd.DataFrame.nlargest
    # Get all columns that we need to preserve
    columns_to_preserve = list(results_after_avging_over_folds.columns)

    return (
        results_after_avging_over_folds.groupby(["user_id"])[columns_to_preserve]
        .apply(lambda x: extreme_func(x, n_extreme_combinations, f"val_{score.name.lower()}"))
        .reset_index(drop=True)
    )


@throw_if_fold_idx_present(
    "keep_only_n_most_extreme_users_for_all_hyperparameters_combinations_score"
)
def keep_only_n_most_extreme_users_for_all_hyperparameters_combinations_score(
    results_after_avging_over_folds: pd.DataFrame,
    score: Score,
    n_extreme_users: int,
    use_smallest: bool,
) -> tuple:
    """
    output columns 'n_most_extreme_users_train': ['user_id', 'combination_idx'] + ['train_{score}']
    for the sole argument score
    output columns: 'n_most_extreme_users_val': ['user_id', 'combination_idx] + ['val_{score}']
    for the sole argument score
    """
    score_name = score.name.lower()
    n_most_extreme_users_train = results_after_avging_over_folds[
        ["user_id", "combination_idx"] + [f"train_{score_name}"]
    ]
    n_most_extreme_users_val = results_after_avging_over_folds[
        ["user_id", "combination_idx"] + [f"val_{score_name}"]
    ]
    extreme_func = pd.DataFrame.nsmallest if use_smallest else pd.DataFrame.nlargest
    # Get the columns we need for each dataframe
    train_columns = ["user_id", "combination_idx", f"train_{score_name}"]
    val_columns = ["user_id", "combination_idx", f"val_{score_name}"]

    n_most_extreme_users_train = (
        n_most_extreme_users_train.groupby(["combination_idx"])[train_columns]
        .apply(lambda x: extreme_func(x, n_extreme_users, f"train_{score_name}"))
        .reset_index(drop=True)
    )
    n_most_extreme_users_val = (
        n_most_extreme_users_val.groupby(["combination_idx"])[val_columns]
        .apply(lambda x: extreme_func(x, n_extreme_users, f"val_{score_name}"))
        .reset_index(drop=True)
    )
    return n_most_extreme_users_train, n_most_extreme_users_val


@throw_if_fold_idx_present(
    "average_over_n_most_extreme_users_for_all_hyperparameters_combinations_score"
)
def average_over_n_most_extreme_users_for_all_hyperparameters_combinations_score(
    results_after_avging_over_folds: pd.DataFrame,
    score: Score,
    n_extreme_users: int,
    use_smallest: bool,
) -> pd.DataFrame:
    """
    output columns: ['combination_idx'] + ['train_{score}_mean', 'train_{score}_std',
    'val_{score}_mean', 'val_{score}_std'] for the sole argument score
    """
    n_most_extreme_users_train, n_most_extreme_users_val = (
        keep_only_n_most_extreme_users_for_all_hyperparameters_combinations_score(
            results_after_avging_over_folds, score, n_extreme_users, use_smallest
        )
    )
    results_after_avging_over_n_most_extreme_users_train = average_over_users(
        n_most_extreme_users_train
    )
    results_after_avging_over_n_most_extreme_users_val = average_over_users(
        n_most_extreme_users_val
    )
    return pd.merge(
        results_after_avging_over_n_most_extreme_users_train,
        results_after_avging_over_n_most_extreme_users_val,
        on=["combination_idx"],
        how="inner",
    )


@throw_if_fold_idx_present("average_over_n_most_extreme_users_for_all_hyperparameters_combinations")
def average_over_n_most_extreme_users_for_all_hyperparameters_combinations(
    results_after_avging_over_folds: pd.DataFrame, n_extreme_users: int, use_smallest: bool
) -> pd.DataFrame:
    """
    output columns: ['combination_ix] + ['train_{score}_mean', 'train_{score}_std',
    'val_{score}_mean', 'val_{scores}_std'] for each score
    """
    results_after_avging_over_n_most_extreme_users = results_after_avging_over_folds[
        ["combination_idx"]
    ].drop_duplicates()
    for score in Score:
        use_smallest_score = (
            use_smallest if SCORES_DICT[score]["increase_better"] else not use_smallest
        )
        results_after_avging_over_n_most_extreme_users_score = (
            average_over_n_most_extreme_users_for_all_hyperparameters_combinations_score(
                results_after_avging_over_folds, score, n_extreme_users, use_smallest_score
            )
        )
        results_after_avging_over_n_most_extreme_users = pd.merge(
            results_after_avging_over_n_most_extreme_users,
            results_after_avging_over_n_most_extreme_users_score,
            on=["combination_idx"],
            how="inner",
        )
    return results_after_avging_over_n_most_extreme_users


@throw_if_fold_idx_present("get_n_best_hyperparameters_combinations_score")
@throw_if_user_id_present("get_n_best_hyperparameters_combinations_score")
def get_n_best_hyperparameters_combinations_score(
    results_after_avging_over_users: pd.DataFrame, score: Score, n_best_combinations: int
) -> list:
    score_name = score.name.lower()
    column = f"val_{score_name}_mean"
    if SCORES_DICT[score]["increase_better"]:
        best_n_combinations = results_after_avging_over_users.nlargest(n_best_combinations, column)[
            ["combination_idx"]
        ].values.tolist()
    else:
        best_n_combinations = results_after_avging_over_users.nsmallest(
            n_best_combinations, column
        )[["combination_idx"]].values.tolist()
    return [combination_idx[0] for combination_idx in best_n_combinations]
