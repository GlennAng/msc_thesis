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

from ....finetuning.src.finetuning_compare_embeddings import (
    compute_sims,
    compute_sims_same_set,
)
from ..training.users_ratings import N_NEGRATED_RANKING

FIRST_LAST_PERCENT = 25


class Score_Type(Enum):
    DEFAULT = auto()
    DEFAULT_DERIVABLE = auto()
    RANKING = auto()
    RANKING_BOTTOM_1_POS = auto()
    RANKING_SESSION = auto()
    INFO = auto()
    CATEGORY = auto()


class Neg_Type(Enum):
    NEGRATED = auto()
    SAMPLES = auto()
    ALL = auto()


CONFIDENCE_THRESHOLD = 0.05


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


def calculate_softmax_pos(softmax_array: np.ndarray) -> float:
    return softmax_array[0] if len(softmax_array) > 0 else 0


def calculate_softmax_ranking_neg(softmax_array: np.ndarray) -> float:
    if len(softmax_array) <= 1:
        return 0
    return np.mean(softmax_array[1 : (N_NEGRATED_RANKING + 1)])


def calculate_softmax_top_1_samples(softmax_array: np.ndarray) -> float:
    start_idx = 1 + N_NEGRATED_RANKING
    return np.max(softmax_array[start_idx:]) if len(softmax_array) > start_idx else 0


def calculate_info_nce(softmax_array: np.ndarray) -> float:
    return -np.log(softmax_array[0] + 1e-10) if len(softmax_array) > 0 else 0


def calculate_ndcg(pos_rank: int) -> float:
    return 1.0 / np.log2(pos_rank + 1) if pos_rank > 0 else 0.0


def calculate_mrr(pos_rank: int) -> float:
    return 1.0 / pos_rank if pos_rank > 0 else 0.0


def calculate_hit_rate_at_1(pos_rank: int) -> float:
    return float(pos_rank == 1)


def calculate_train_single_session(user_info: dict) -> float:
    n_sessions_train = user_info["n_sessions_train"]
    if isinstance(n_sessions_train, list):
        n_sessions_train = n_sessions_train[-1]
    return float(n_sessions_train == 1)


def calculate_train_set_cosine_similarity(train_pos_embeddings_all_sessions: np.ndarray) -> float:
    return compute_sims_same_set(train_pos_embeddings_all_sessions)


def calculate_train_set_val_session_cosine_similarity(
    train_pos_embeddings_all_sessions: np.ndarray,
    val_pos_embeddings_per_session: dict,
    selection: str,
) -> float:
    sessions_ids = list(val_pos_embeddings_per_session)
    if selection == "first":
        session_id = sessions_ids[0]
    elif selection == "middle":
        session_id = sessions_ids[len(sessions_ids) // 2]
    elif selection == "last":
        session_id = sessions_ids[-1]
    else:
        raise ValueError(f"Unknown selection: {selection}")
    val_pos_embeddings = val_pos_embeddings_per_session[session_id]
    return compute_sims(train_pos_embeddings_all_sessions, val_pos_embeddings)


def calculate_val_set_cosine_similarity_sliding(val_pos_embeddings_per_session: dict) -> float:
    cosines = []
    sessions_ids = list(val_pos_embeddings_per_session)
    if len(sessions_ids) < 2:
        return None
    for i in range(len(sessions_ids) - 1):
        emb_before = val_pos_embeddings_per_session[sessions_ids[i]]
        emb_after = val_pos_embeddings_per_session[sessions_ids[i + 1]]
        cosines.append(compute_sims(emb_before, emb_after))
    return np.mean(cosines)


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
        "abbreviation_for_visu_file": "REC",
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
        "calculator": calculate_ndcg,
        "neg_type": Neg_Type.NEGRATED,
    },
    "MRR": {
        "abbreviation": "MRR",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 0,
        "calculator": calculate_mrr,
        "neg_type": Neg_Type.NEGRATED,
    },
    "HIT_RATE_AT_1": {
        "abbreviation": "HR@1",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 0,
        "calculator": calculate_hit_rate_at_1,
        "neg_type": Neg_Type.NEGRATED,
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
        "calculator": calculate_ndcg,
        "neg_type": Neg_Type.SAMPLES,
    },
    "NDCG_ALL": {
        "abbreviation": "NDCG\nAll",
        "abbreviation_for_visu_file": "NDCG_All",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 1,
        "calculator": calculate_ndcg,
        "neg_type": Neg_Type.ALL,
    },
    "NDCG_MEAN_PER_SESSION": {
        "abbreviation": "NDCG\nMean S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": True,
        "page": 1,
        "lookup": "NDCG_ALL",
        "selection": "mean",
    },
    "NDCG_MEDIAN_PER_SESSION": {
        "abbreviation": "NDCG\nMed S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": True,
        "page": 1,
        "lookup": "NDCG_ALL",
        "selection": "median",
    },
    "MRR_SAMPLES": {
        "abbreviation": "MRR\nSmpl",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 1,
        "calculator": calculate_mrr,
        "neg_type": Neg_Type.SAMPLES,
    },
    "MRR_ALL": {
        "abbreviation": "MRR\nAll",
        "abbreviation_for_visu_file": "MRR_All",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 1,
        "calculator": calculate_mrr,
        "neg_type": Neg_Type.ALL,
    },
    "HIT_RATE_AT_1_SAMPLES": {
        "abbreviation": "HR@1\nSmpl",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 1,
        "calculator": calculate_hit_rate_at_1,
        "neg_type": Neg_Type.SAMPLES,
    },
    "HIT_RATE_AT_1_ALL": {
        "abbreviation": "HR@1\nAll",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 1,
        "calculator": calculate_hit_rate_at_1,
        "neg_type": Neg_Type.ALL,
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
    "CONFIDENCE_TOP_1_SAMPLES": {
        "abbreviation": "C_S↑1",
        "type": Score_Type.DEFAULT,
        "increase_better": False,
        "page": 2,
        "calculator": calculate_confidence_top_1_samples,
    },
    "SOFTMAX_POS_05": {
        "abbreviation": "SmP\n0.5",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 2,
        "calculator": calculate_softmax_pos,
        "temperature": "05",
    },
    "SOFTMAX_BOTTOM_1_POS_05": {
        "abbreviation": "SmP↓1\n0.5",
        "type": Score_Type.RANKING_BOTTOM_1_POS,
        "increase_better": True,
        "page": 2,
        "lookup": "SOFTMAX_POS_05",
    },
    "SOFTMAX_RANKING_NEG_05": {
        "abbreviation": "SmRN\n0.5",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
        "calculator": calculate_softmax_ranking_neg,
        "temperature": "05",
    },
    "SOFTMAX_TOP_1_SAMPLES_05": {
        "abbreviation": "SmS↑1\n0.5",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
        "calculator": calculate_softmax_top_1_samples,
        "temperature": "05",
    },
    "INFO_NCE_01": {
        "abbreviation": "INCE\n0.1",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
        "calculator": calculate_info_nce,
        "temperature": "01",
    },
    "SOFTMAX_POS_1": {
        "abbreviation": "SmP\n1.0",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 2,
        "calculator": calculate_softmax_pos,
        "temperature": "1",
    },
    "SOFTMAX_BOTTOM_1_POS_1": {
        "abbreviation": "SmP↓1\n1.0",
        "type": Score_Type.RANKING_BOTTOM_1_POS,
        "increase_better": True,
        "page": 2,
        "lookup": "SOFTMAX_POS_1",
    },
    "SOFTMAX_RANKING_NEG_1": {
        "abbreviation": "SmRN\n1.0",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
        "calculator": calculate_softmax_ranking_neg,
        "temperature": "1",
    },
    "SOFTMAX_TOP_1_SAMPLES_1": {
        "abbreviation": "SmS↑1\n1.0",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
        "calculator": calculate_softmax_top_1_samples,
        "temperature": "1",
    },
    "INFO_NCE_1": {
        "abbreviation": "INCE\n1.0",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
        "calculator": calculate_info_nce,
        "temperature": "1",
    },
    "SOFTMAX_POS_2": {
        "abbreviation": "SmP\n2.0",
        "type": Score_Type.RANKING,
        "increase_better": True,
        "page": 2,
        "calculator": calculate_softmax_pos,
        "temperature": "2",
    },
    "SOFTMAX_BOTTOM_1_POS_2": {
        "abbreviation": "SmP↓1\n2.0",
        "type": Score_Type.RANKING_BOTTOM_1_POS,
        "increase_better": True,
        "page": 2,
        "lookup": "SOFTMAX_POS_2",
    },
    "SOFTMAX_RANKING_NEG_2": {
        "abbreviation": "SmRN\n2.0",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
        "calculator": calculate_softmax_ranking_neg,
        "temperature": "2",
    },
    "SOFTMAX_TOP_1_SAMPLES_2": {
        "abbreviation": "SmS↑1\n2.0",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
        "calculator": calculate_softmax_top_1_samples,
        "temperature": "2",
    },
    "INFO_NCE_2": {
        "abbreviation": "INCE\n2.0",
        "type": Score_Type.RANKING,
        "increase_better": False,
        "page": 2,
        "calculator": calculate_info_nce,
        "temperature": "2",
    },
    f"NDCG_FIRST_{FIRST_LAST_PERCENT}_PERCENT_SESSIONS": {
        "abbreviation": f"NDCG\nF{FIRST_LAST_PERCENT}% S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": True,
        "page": 3,
        "lookup": "NDCG_ALL",
        "selection": "first_sessions",
    },
    f"NDCG_LAST_{FIRST_LAST_PERCENT}_PERCENT_SESSIONS": {
        "abbreviation": f"NDCG\nL{FIRST_LAST_PERCENT}% S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": True,
        "page": 3,
        "lookup": "NDCG_ALL",
        "selection": "last_sessions",
    },
    f"NDCG_FIRST_{FIRST_LAST_PERCENT}_PERCENT_TIMES": {
        "abbreviation": f"NDCG\nF{FIRST_LAST_PERCENT}% T",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": True,
        "page": 3,
        "lookup": "NDCG_ALL",
        "selection": "first_times",
    },
    f"NDCG_LAST_{FIRST_LAST_PERCENT}_PERCENT_TIMES": {
        "abbreviation": f"NDCG\nL{FIRST_LAST_PERCENT}% T",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": True,
        "page": 3,
        "lookup": "NDCG_ALL",
        "selection": "last_times",
    },
    "NDCG_STD_PER_SESSION": {
        "abbreviation": "NDCG\nStd S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": True,
        "page": 3,
        "lookup": "NDCG_ALL",
        "selection": "std",
    },
    "NDCG_ALL_FIRST_SESSION": {
        "abbreviation": "NDCG\nFirst S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": True,
        "page": 3,
        "lookup": "NDCG_ALL",
        "selection": "first",
    },
    "NDCG_ALL_LAST_SESSION": {
        "abbreviation": "NDCG\nLast S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": True,
        "page": 3,
        "lookup": "NDCG_ALL",
        "selection": "last",
    },
    "MRR_ALL_FIRST_SESSION": {
        "abbreviation": "MRR\nFirst S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": True,
        "page": 3,
        "lookup": "MRR_ALL",
        "selection": "first",
    },
    "MRR_ALL_MIDDLE_SESSION": {
        "abbreviation": "MRR\nMid S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": True,
        "page": 3,
        "lookup": "MRR_ALL",
        "selection": "middle",
    },
    "MRR_ALL_LAST_SESSION": {
        "abbreviation": "MRR\nLast S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": True,
        "page": 3,
        "lookup": "MRR_ALL",
        "selection": "last",
    },
    "HIT_RATE_AT_1_ALL_FIRST_SESSION": {
        "abbreviation": "HR@1\nFirst S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": True,
        "page": 3,
        "lookup": "HIT_RATE_AT_1_ALL",
        "selection": "first",
    },
    "HIT_RATE_AT_1_ALL_MIDDLE_SESSION": {
        "abbreviation": "HR@1\nMid S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": True,
        "page": 3,
        "lookup": "HIT_RATE_AT_1_ALL",
        "selection": "middle",
    },
    "HIT_RATE_AT_1_ALL_LAST_SESSION": {
        "abbreviation": "HR@1\nLast S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": True,
        "page": 3,
        "lookup": "HIT_RATE_AT_1_ALL",
        "selection": "last",
    },
    "INFO_NCE_1_FIRST_SESSION": {
        "abbreviation": "INCE 1\nFirst S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": False,
        "page": 3,
        "lookup": "INFO_NCE_1",
        "selection": "first",
    },
    "INFO_NCE_1_MIDDLE_SESSION": {
        "abbreviation": "INCE 1\nMid S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": False,
        "page": 3,
        "lookup": "INFO_NCE_1",
        "selection": "middle",
    },
    "INFO_NCE_1_LAST_SESSION": {
        "abbreviation": "INCE 1\nLast S",
        "type": Score_Type.RANKING_SESSION,
        "increase_better": False,
        "page": 3,
        "lookup": "INFO_NCE_1",
        "selection": "last",
    },
    "N_TRAIN_POSRATED": {
        "abbreviation": "N Train\nPos R",
        "type": Score_Type.INFO,
        "increase_better": True,
        "page": 4,
        "lookup_key": "n_posrated_train",
    },
    "N_TRAIN_NEGRATED": {
        "abbreviation": "N Train\nNeg R",
        "type": Score_Type.INFO,
        "increase_better": True,
        "page": 4,
        "lookup_key": "n_negrated_train",
    },
    "N_VAL_POSRATED": {
        "abbreviation": "N Val\nPos R",
        "type": Score_Type.INFO,
        "increase_better": True,
        "page": 4,
        "lookup_key": "n_posrated_val",
    },
    "N_VAL_NEGRATED": {
        "abbreviation": "N Val\nNeg R",
        "type": Score_Type.INFO,
        "increase_better": True,
        "page": 4,
        "lookup_key": "n_negrated_val",
    },
    "TRAIN_SINGLE_SESSION": {
        "abbreviation": "Train\nSingle S",
        "type": Score_Type.INFO,
        "increase_better": True,
        "page": 4,
        "calculator": calculate_train_single_session,
    },
    "N_VAL_POS_SESSIONS": {
        "abbreviation": "N Val\nPos Ss",
        "type": Score_Type.INFO,
        "increase_better": True,
        "page": 4,
        "lookup_key": "n_sessions_pos_val",
    },
    "N_VAL_POS_DAYS": {
        "abbreviation": "N Val\nPos Day",
        "type": Score_Type.INFO,
        "increase_better": True,
        "page": 4,
        "lookup_key": "time_range_days_pos_val",
    },
    "TRAIN_SET_COSINE_SIMILARITY": {
        "abbreviation": "Cosine\nTrain",
        "type": Score_Type.INFO,
        "increase_better": True,
        "page": 4,
        "calculator": calculate_train_set_cosine_similarity,
    },
    "TRAIN_SET_FIRST_VAL_SESSION_COSINE_SIMILARITY": {
        "abbreviation": "Cos TrV\nFirst S",
        "type": Score_Type.INFO,
        "increase_better": True,
        "page": 4,
        "selection": "first",
        "calculator": calculate_train_set_val_session_cosine_similarity,
    },
    "TRAIN_SET_MIDDLE_VAL_SESSION_COSINE_SIMILARITY": {
        "abbreviation": "Cos TrV\nMid S",
        "type": Score_Type.INFO,
        "increase_better": True,
        "page": 4,
        "selection": "middle",
        "calculator": calculate_train_set_val_session_cosine_similarity,
    },
    "TRAIN_SET_LAST_VAL_SESSION_COSINE_SIMILARITY": {
        "abbreviation": "Cos TrV\nLast S",
        "type": Score_Type.INFO,
        "increase_better": True,
        "page": 4,
        "selection": "last",
        "calculator": calculate_train_set_val_session_cosine_similarity,
    },
    "VAL_SET_COSINE_SIMILARITY_SLIDING": {
        "abbreviation": "Cos Val\nSlide",
        "type": Score_Type.INFO,
        "increase_better": True,
        "page": 4,
        "calculator": calculate_val_set_cosine_similarity_sliding,
    },
    "CATEGORY_L1_MOST_FREQUENT_IDENTICAL": {
        "abbreviation": "L1\nMFI",
        "type": Score_Type.CATEGORY,
        "increase_better": True,
        "page": 4,
        "calculator": calculate_category_l1_most_frequent_identical,
    },
    "CATEGORY_L1L2_MOST_FREQUENT_IDENTICAL": {
        "abbreviation": "L1L2\nMFI",
        "type": Score_Type.CATEGORY,
        "increase_better": True,
        "page": 4,
        "calculator": calculate_category_l1l2_most_frequent_identical,
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


def get_score_ranking(
    score: Score,
    rank_pos: int,
    rank_pos_samples: int,
    rank_pos_all: int,
    softmax_dict: dict,
) -> float:
    func = SCORES_DICT[score]["calculator"]
    sig = inspect.signature(func)
    kwargs = {}
    if "softmax_array" in sig.parameters:
        temperature = SCORES_DICT[score]["temperature"]
        kwargs["softmax_array"] = softmax_dict[temperature]
    if "pos_rank" in sig.parameters:
        neg_type = SCORES_DICT[score]["neg_type"]
        if neg_type == Neg_Type.NEGRATED:
            kwargs["pos_rank"] = rank_pos
        elif neg_type == Neg_Type.SAMPLES:
            kwargs["pos_rank"] = rank_pos_samples
        elif neg_type == Neg_Type.ALL:
            kwargs["pos_rank"] = rank_pos_all
        else:
            raise ValueError(f"Invalid neg_type {neg_type} for score {score.name}.")
    return func(**kwargs)


def get_score_ranking_session(
    scores_per_session: dict,
    avgs_per_session: dict,
    selection: str,
    largest_session_idx_to_include_for_first_sessions: int = None,
    smallest_session_idx_to_include_for_last_sessions: int = None,
    largest_session_idx_to_include_for_first_times: int = None,
    smallest_session_idx_to_include_for_last_times: int = None,
) -> float:
    distinct_sessions_ids = list(avgs_per_session.keys())
    if selection in ["first", "middle", "last"]:
        if selection == "first":
            session_id = distinct_sessions_ids[0]
        elif selection == "middle":
            session_id = distinct_sessions_ids[len(distinct_sessions_ids) // 2]
        elif selection == "last":
            session_id = distinct_sessions_ids[-1]
        return avgs_per_session[session_id]
    elif selection in ["mean", "median", "std"]:
        values = np.array(list(avgs_per_session.values()))
        if selection == "mean":
            return float(np.mean(values))
        elif selection == "median":
            return float(np.median(values))
        elif selection == "std":
            return float(np.std(values))
    elif selection in ["first_sessions", "last_sessions", "first_times", "last_times"]:
        if selection == "first_sessions":
            sessions_ids = distinct_sessions_ids[
                : largest_session_idx_to_include_for_first_sessions + 1
            ]
        elif selection == "last_sessions":
            sessions_ids = distinct_sessions_ids[smallest_session_idx_to_include_for_last_sessions:]
        elif selection == "first_times":
            sessions_ids = distinct_sessions_ids[
                : largest_session_idx_to_include_for_first_times + 1
            ]
        elif selection == "last_times":
            sessions_ids = distinct_sessions_ids[smallest_session_idx_to_include_for_last_times:]
        if len(sessions_ids) == 0:
            sessions_ids = distinct_sessions_ids
        values = []
        for session_id in sessions_ids:
            values.extend(scores_per_session[session_id])
        return float(np.mean(values))


def get_score_info(
    score: Score,
    val_data_dict: dict,
    user_info: dict,
    train_pos_embeddings_all_sessions: np.ndarray,
    val_pos_embeddings_per_session: dict,
) -> float:
    func = SCORES_DICT[score]["calculator"]
    selection = SCORES_DICT[score].get("selection", None)
    sig = inspect.signature(func)
    kwargs = {}
    if "user_info" in sig.parameters:
        kwargs["user_info"] = user_info
    if "val_data_dict" in sig.parameters:
        kwargs["val_data_dict"] = val_data_dict
    if "selection" in sig.parameters:
        kwargs["selection"] = selection
    if "train_pos_embeddings_all_sessions" in sig.parameters:
        kwargs["train_pos_embeddings_all_sessions"] = train_pos_embeddings_all_sessions
    if "val_pos_embeddings_per_session" in sig.parameters:
        kwargs["val_pos_embeddings_per_session"] = val_pos_embeddings_per_session
    return_val = func(**kwargs)
    if return_val is not None:
        return float(return_val)
    return return_val
