import inspect
from enum import Enum, auto

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Score_Type(Enum):
    DEFAULT = auto()
    DEFAULT_DERIVABLE = auto()
    RANKING = auto()
    RANKING_TEMPORAL = auto()
    CATEGORY = auto()


CONFIDENCE_THRESHOLD = 0.5


def cel_score(y_proba: np.ndarray, positive: bool) -> float:
    eps = np.finfo(y_proba.dtype).eps
    y_proba = np.clip(y_proba, eps, 1 - eps)
    return -np.mean(np.log(y_proba)) if positive else -np.mean(np.log(1 - y_proba))


def calculate_positive_gt_ratio(y_true: np.ndarray) -> float:
    return np.mean(y_true)


def calculate_positive_pred_ratio(y_pred: np.ndarray) -> float:
    return np.mean(y_pred)


def calculate_true_positive_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true == 1) & (y_pred == 1))


def calculate_false_positive_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true == 0) & (y_pred == 1))


def calculate_true_negative_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true == 0) & (y_pred == 0))


def calculate_false_negative_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true == 1) & (y_pred == 0))


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return recall_score(y_true, y_pred, zero_division=0)


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if tn + fp != 0 else 0


def calculate_balanced_accuracy(
    user_scores: list, scores_to_indices_dict: dict, val: bool
) -> float:
    s = "val" if val else "train"
    recall = user_scores[scores_to_indices_dict[f"{s}_{Score.RECALL.name.lower()}"]]
    specificity = user_scores[scores_to_indices_dict[f"{s}_{Score.SPECIFICITY.name.lower()}"]]
    return (recall + specificity) / 2


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return precision_score(y_true, y_pred, zero_division=0)


def calculate_cel(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return log_loss(y_true, y_proba)


def calculate_cel_pos(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return cel_score(y_proba[y_true == 1], positive=True)


def calculate_cel_neg(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return cel_score(y_proba[y_true == 0], positive=False)


def calculate_auroc_classification(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return roc_auc_score(y_true, y_proba)


def calculate_confidence_all(y_proba: np.ndarray) -> float:
    return np.mean(y_proba)


def calculate_confidence_tp(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
    tp = (y_true == 1) & (y_pred == 1)
    return 0.5 if np.sum(tp) == 0 else np.mean(y_proba[tp])


def calculate_confidence_fp(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
    fp = (y_true == 0) & (y_pred == 1)
    return 0.5 if np.sum(fp) == 0 else np.mean(y_proba[fp])


def calculate_confidence_tn(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
    tn = (y_true == 0) & (y_pred == 0)
    return 0.5 if np.sum(tn) == 0 else np.mean(y_proba[tn])


def calculate_confidence_fn(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
    fn = (y_true == 1) & (y_pred == 0)
    return 0.5 if np.sum(fn) == 0 else np.mean(y_proba[fn])


def calculate_confidence_top_1_samples(y_negative_samples_proba: np.ndarray) -> float:
    return np.max(y_negative_samples_proba) if len(y_negative_samples_proba) > 0 else 0


def calculate_positive_gt_above_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    positive_gt = y_true == 1
    if np.sum(positive_gt) == 0:
        return 0
    return np.mean(y_proba[positive_gt] > CONFIDENCE_THRESHOLD)


def calculate_samples_above_threshold(y_negative_samples_proba: np.ndarray) -> float:
    return np.mean(y_negative_samples_proba > CONFIDENCE_THRESHOLD)


def calculate_specificity_samples(y_negative_samples_pred: np.ndarray) -> float:
    return np.mean(y_negative_samples_pred == 0)


def calculate_precision_samples(
    y_true: np.ndarray, y_pred: np.ndarray, y_negative_samples_pred: np.ndarray
) -> float:
    y = np.concatenate(
        (y_true[y_true == 1], np.zeros(len(y_negative_samples_pred), dtype=y_true.dtype))
    )
    y_predi = np.concatenate((y_pred[y_true == 1], y_negative_samples_pred))
    return precision_score(y, y_predi, zero_division=0)


def calculate_category_l1_most_frequent_identical(l1_pos: pd.Series, l1_neg: pd.Series) -> float:
    mode_pos, mode_neg = l1_pos.mode(), l1_neg.mode()
    if len(mode_pos) == 0 or len(mode_neg) == 0:
        return 0.0
    return float(mode_pos.iloc[0] == mode_neg.iloc[0])


def calculate_category_l1l2_most_frequent_identical(
    l1l2_pos: pd.Series, l1l2_neg: pd.Series
) -> float:
    mode_pos, mode_neg = l1l2_pos.mode(), l1l2_neg.mode()
    if len(mode_pos) == 0 or len(mode_neg) == 0:
        return 0.0
    return float(mode_pos.iloc[0] == mode_neg.iloc[0])


def calculate_confidence_pos(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return np.mean(y_proba[y_true == 1]) if np.sum(y_true == 1) > 0 else 0


def calculate_confidence_bottom_1_pos(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return np.min(y_proba[y_true == 1]) if np.sum(y_true == 1) > 0 else 0


SCORES_DICT = {
    "POSITIVE_GT_RATIO": {
        "abbreviation": "PGTRo",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 0,
        "calculator": calculate_positive_gt_ratio,
    },
    "POSITIVE_PRED_RATIO": {
        "abbreviation": "PPRRo",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 0,
        "calculator": calculate_positive_pred_ratio,
    },
    "TRUE_POSITIVE_RATIO": {
        "abbreviation": "TPRo",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 0,
        "calculator": calculate_true_positive_ratio,
    },
    "FALSE_POSITIVE_RATIO": {
        "abbreviation": "FPRo",
        "type": Score_Type.DEFAULT,
        "increase_better": False,
        "page": 0,
        "calculator": calculate_false_positive_ratio,
    },
    "TRUE_NEGATIVE_RATIO": {
        "abbreviation": "TNRo",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 0,
        "calculator": calculate_true_negative_ratio,
    },
    "FALSE_NEGATIVE_RATIO": {
        "abbreviation": "FNRo",
        "type": Score_Type.DEFAULT,
        "increase_better": False,
        "page": 0,
        "calculator": calculate_false_negative_ratio,
    },
    "ACCURACY": {
        "abbreviation": "ACC",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 0,
        "calculator": calculate_accuracy,
    },
    "RECALL": {
        "abbreviation": "REC",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 0,
        "calculator": calculate_recall,
    },
    "SPECIFICITY": {
        "abbreviation": "SPE",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 0,
        "calculator": calculate_specificity,
    },
    "BALANCED_ACCURACY": {
        "abbreviation": "BAL",
        "abbreviation_for_visu_file": "BAL_All",
        "type": Score_Type.DEFAULT_DERIVABLE,
        "increase_better": True,
        "page": 0,
        "calculator": calculate_balanced_accuracy,
    },
    "PRECISION": {
        "abbreviation": "PRE",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 0,
        "calculator": calculate_precision,
    },
    "CEL": {
        "abbreviation": "CEL",
        "type": Score_Type.DEFAULT,
        "increase_better": False,
        "page": 0,
        "calculator": calculate_cel,
    },
    "CEL_POS": {
        "abbreviation": "CELP",
        "type": Score_Type.DEFAULT,
        "increase_better": False,
        "page": 0,
        "calculator": calculate_cel_pos,
    },
    "CEL_NEG": {
        "abbreviation": "CELN",
        "type": Score_Type.DEFAULT,
        "increase_better": False,
        "page": 0,
        "calculator": calculate_cel_neg,
    },
    "AUROC_CLASSIFICATION": {
        "abbreviation": "AUC_C",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 0,
        "calculator": calculate_auroc_classification,
    },
    "NDCG": {
        "abbreviation": "NDCG",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 0,
    },
    "MRR": {
        "abbreviation": "MRR",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 0,
    },
    "HIT_RATE_AT_1": {
        "abbreviation": "HR@1",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 0,
    },
    "CONFIDENCE_ALL": {
        "abbreviation": "C_ALL",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 1,
        "calculator": calculate_confidence_all,
    },
    "CONFIDENCE_TP": {
        "abbreviation": "C_TP",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 1,
        "calculator": calculate_confidence_tp,
    },
    "CONFIDENCE_FP": {
        "abbreviation": "C_FP",
        "type": Score_Type.DEFAULT,
        "increase_better": False,
        "page": 1,
        "calculator": calculate_confidence_fp,
    },
    "CONFIDENCE_TN": {
        "abbreviation": "C_TN",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 1,
        "calculator": calculate_confidence_tn,
    },
    "CONFIDENCE_FN": {
        "abbreviation": "C_FN",
        "type": Score_Type.DEFAULT,
        "increase_better": False,
        "page": 1,
        "calculator": calculate_confidence_fn,
    },
    "CONFIDENCE_TOP_1_SAMPLES": {
        "abbreviation": "C_T1\nSmpl",
        "type": Score_Type.DEFAULT,
        "increase_better": False,
        "page": 1,
        "calculator": calculate_confidence_top_1_samples,
    },
    "POSITIVE_GT_ABOVE_THRESHOLD": {
        "abbreviation": f"PGTAT\n{CONFIDENCE_THRESHOLD}",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 1,
        "calculator": calculate_positive_gt_above_threshold,
    },
    "SAMPLES_ABOVE_THRESHOLD": {
        "abbreviation": f"SAT\n{CONFIDENCE_THRESHOLD}",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 1,
        "calculator": calculate_samples_above_threshold,
    },
    "SPECIFICITY_SAMPLES": {
        "abbreviation": "SPE\nSmpl",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 1,
        "calculator": calculate_specificity_samples,
    },
    "PRECISION_SAMPLES": {
        "abbreviation": "PRE\nSmpl",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 1,
        "calculator": calculate_precision_samples,
    },
    "NDCG_SAMPLES": {
        "abbreviation": "NDCG\nSmpl",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 1,
    },
    "NDCG_ALL": {
        "abbreviation": "NDCG\nAll",
        "abbreviation_for_visu_file": "NDCG_All",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 1,
    },
    "MRR_SAMPLES": {
        "abbreviation": "MRR\nSmpl",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 1,
    },
    "MRR_ALL": {
        "abbreviation": "MRR\nAll",
        "abbreviation_for_visu_file": "MRR_All",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 1,
    },
    "HIT_RATE_AT_1_SAMPLES": {
        "abbreviation": "HR@1\nSmpl",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 1,
    },
    "HIT_RATE_AT_1_ALL": {
        "abbreviation": "HR@1\nAll",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 1,
    },
    "CATEGORY_L1_MOST_FREQUENT_IDENTICAL": {
        "abbreviation": "L1\nMFI",
        "type": Score_Type.CATEGORY,
        "increase_better": True,
        "page": 1,
        "calculator": calculate_category_l1_most_frequent_identical,
    },
    "CATEGORY_L1L2_MOST_FREQUENT_IDENTICAL": {
        "abbreviation": "L1L2\nMFI",
        "type": Score_Type.CATEGORY,
        "increase_better": True,
        "page": 1,
        "calculator": calculate_category_l1l2_most_frequent_identical,
    },
    "CONFIDENCE_POS": {
        "abbreviation": "C_P",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 2,
        "calculator": calculate_confidence_pos,
    },
    "CONFIDENCE_BOTTOM_1_POS": {
        "abbreviation": "C_P↓1",
        "type": Score_Type.DEFAULT,
        "increase_better": True,
        "page": 2,
        "calculator": calculate_confidence_bottom_1_pos,
    },
    "SOFTMAX_POS_05": {
        "abbreviation": "SmP\n0.5",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 2,
    },
    "SOFTMAX_BOTTOM_1_POS_05": {
        "abbreviation": "SmP↓1\n0.5",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 2,
    },
    "SOFTMAX_RANKING_NEG_05": {
        "abbreviation": "SmRN\n0.5",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
    },
    "SOFTMAX_TOP_1_SAMPLES_05": {
        "abbreviation": "SmS↑1\n0.5",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
    },
    "INFO_NCE_05": {
        "abbreviation": "INCE\n0.5",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
    },
    "SOFTMAX_POS_1": {
        "abbreviation": "SmP\n1.0",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 2,
    },
    "SOFTMAX_BOTTOM_1_POS_1": {
        "abbreviation": "SmP↓1\n1.0",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 2,
    },
    "SOFTMAX_RANKING_NEG_1": {
        "abbreviation": "SmRN\n1.0",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
    },
    "SOFTMAX_TOP_1_SAMPLES_1": {
        "abbreviation": "SmS↑1\n1.0",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
    },
    "INFO_NCE_1": {
        "abbreviation": "INCE\n1.0",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
    },
    "SOFTMAX_POS_2": {
        "abbreviation": "SmP\n2.0",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 2,
    },
    "SOFTMAX_BOTTOM_1_POS_2": {
        "abbreviation": "SmP↓1\n2.0",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 2,
    },
    "SOFTMAX_RANKING_NEG_2": {
        "abbreviation": "SmRN\n2.0",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
    },
    "SOFTMAX_TOP_1_SAMPLES_2": {
        "abbreviation": "SmS↑1\n2.0",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
    },
    "INFO_NCE_2": {
        "abbreviation": "INCE\n2.0",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
    },
    "NDCG_ALL_TEMPORAL": {
        "abbreviation": "NDCG\nTemp",
        "type": Score_Type.RANKING_TEMPORAL,
        "increase_better": True,
        "page": 3,
        "lookup": "NDCG_ALL",
    },
    "MRR_ALL_TEMPORAL": {
        "abbreviation": "MRR\nTemp",
        "type": Score_Type.RANKING_TEMPORAL,
        "increase_better": True,
        "page": 3,
        "lookup": "MRR_ALL",
    },
    "HIT_RATE_AT_1_TEMPORAL": {
        "abbreviation": "HR@1\nTemp",
        "type": Score_Type.RANKING_TEMPORAL,
        "increase_better": True,
        "page": 3,
        "lookup": "HIT_RATE_AT_1_ALL",
    },
    "INFO_NCE_TEMP": {
        "abbreviation": "INCE\nTemp",
        "type": Score_Type.RANKING_TEMPORAL,
        "increase_better": False,
        "page": 3,
        "lookup": "INFO_NCE_1",
    },
}
Score = Enum("Score", list(SCORES_DICT.keys()))
SCORES_DICT = {Score[key]: value for key, value in SCORES_DICT.items()}
for score in SCORES_DICT:
    if "lookup" in SCORES_DICT[score]:
        SCORES_DICT[score]["lookup"] = Score[SCORES_DICT[score]["lookup"]]
SCORES_BY_TYPE = {}
for score in Score:
    score_type = SCORES_DICT[score]["type"]
    if score_type not in SCORES_BY_TYPE:
        SCORES_BY_TYPE[score_type] = []
    SCORES_BY_TYPE[score_type].append(score)


def get_score_from_arg(score_arg: str) -> Score:
    valid_score_args = [score.name.lower() for score in Score]
    if score_arg.lower() not in valid_score_args:
        raise ValueError(
            f"Invalid argument {score_arg} 'algorithm'. Possible values: {valid_score_args}."
        )
    return Score[score_arg.upper()]


def get_score_default(
    score: Score,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    y_negative_samples_pred: np.ndarray,
    y_negative_samples_proba: np.ndarray,
) -> float:
    func = SCORES_DICT[score]["calculator"]
    sig = inspect.signature(func)
    kwargs = {}
    if "y_true" in sig.parameters:
        kwargs["y_true"] = y_true
    if "y_pred" in sig.parameters:
        kwargs["y_pred"] = y_pred
    if "y_proba" in sig.parameters:
        kwargs["y_proba"] = y_proba
    if "y_negative_samples_pred" in sig.parameters:
        kwargs["y_negative_samples_pred"] = y_negative_samples_pred
    if "y_negative_samples_proba" in sig.parameters:
        kwargs["y_negative_samples_proba"] = y_negative_samples_proba
    return func(**kwargs)


def get_score_default_derivable(
    score: Score, user_scores: list, scores_to_indices_dict: dict, val: bool
) -> float:
    func = SCORES_DICT[score]["calculator"]
    return func(user_scores=user_scores, scores_to_indices_dict=scores_to_indices_dict, val=val)


def get_score_category(
    score: Score,
    l1_pos: pd.Series,
    l1_neg: pd.Series,
    l1l2_pos: pd.Series,
    l1l2_neg: pd.Series,
) -> float:
    func = SCORES_DICT[score]["calculator"]
    sig = inspect.signature(func)
    kwargs = {}
    if "l1_pos" in sig.parameters:
        kwargs["l1_pos"] = l1_pos
    if "l1_neg" in sig.parameters:
        kwargs["l1_neg"] = l1_neg
    if "l1l2_pos" in sig.parameters:
        kwargs["l1l2_pos"] = l1l2_pos
    if "l1l2_neg" in sig.parameters:
        kwargs["l1l2_neg"] = l1l2_neg
    return func(**kwargs)
