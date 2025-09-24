import time
from enum import Enum, auto
from pathlib import Path

from ....src.project_paths import ProjectPaths


class EmbedFunction(Enum):
    MEAN_POS_POOLING = auto()
    LOGISTIC_REGRESSION = auto()
    NEURAL_PRECOMPUTED = auto()


def get_embed_function_from_arg(embed_function_arg: str) -> EmbedFunction:
    valid_embed_function_args = [ef.name.lower() for ef in EmbedFunction]
    if embed_function_arg.lower() not in valid_embed_function_args:
        raise ValueError(
            f"Invalid argument {embed_function_arg} 'embed_function'. Possible values: {valid_embed_function_args}."
        )
    return EmbedFunction[embed_function_arg.upper()]


def get_users_selections_choices() -> list:
    return [None, "sequence_val", "sequence_test", "sequence_high_sessions"]


def get_eval_data_folder(
    embed_function: EmbedFunction, users_selection: str, single_val_session: bool
) -> Path:
    s = embed_function.name.lower()
    if users_selection is not None:
        s += f"_{users_selection}"
    else:
        s += "_all"
    if single_val_session:
        s += "_single_session"
    else:
        s += "_multi_session"
    s += time.strftime("_%Y_%m_%d_%H_%M_%S")
    return ProjectPaths.sequence_data_sliding_window_eval_path() / s
