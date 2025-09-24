from enum import Enum, auto

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC


class Algorithm(Enum):
    LOGREG = auto()
    SVM = auto()


def get_algorithm_from_arg(algorithm_arg: str) -> Algorithm:
    valid_algorithm_args = [algorithm.name.lower() for algorithm in Algorithm]
    if algorithm_arg.lower() not in valid_algorithm_args:
        raise ValueError(
            f"Invalid argument {algorithm_arg} 'algorithm'. Possible values: {valid_algorithm_args}."
        )
    return Algorithm[algorithm_arg.upper()]


def get_model(
    algorithm: Algorithm,
    max_iter: int,
    clf_C: float,
    random_state: int,
    logreg_solver: str = None,
    svm_kernel: str = None,
) -> object:
    if algorithm == Algorithm.LOGREG:
        return LogisticRegression(
            max_iter=max_iter, C=clf_C, random_state=random_state, solver=logreg_solver
        )
    elif algorithm == Algorithm.SVM:
        return SVC(
            max_iter=max_iter,
            C=clf_C,
            random_state=random_state,
            kernel=svm_kernel,
            probability=True,
        )


class Evaluation(Enum):
    CROSS_VALIDATION = auto()
    TRAIN_TEST_SPLIT = auto()
    SESSION_BASED = auto()
    SLIDING_WINDOW = auto()


def get_evaluation_from_arg(evaluation_arg: str) -> Evaluation:
    valid_evaluation_args = [evaluation.name.lower() for evaluation in Evaluation]
    if evaluation_arg.lower() not in valid_evaluation_args:
        raise ValueError(
            f"Invalid argument {evaluation_arg} 'evaluation'. Possible values: {valid_evaluation_args}."
        )
    return Evaluation[evaluation_arg.upper()]


def get_cross_val(stratified: bool, k_folds: int, random_state: int) -> object:
    if stratified:
        return StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    else:
        return KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
