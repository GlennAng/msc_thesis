from enum import Enum, auto
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, log_loss, roc_auc_score, ndcg_score
from sklearn.svm import SVC
import numpy as np
import random

class Score(Enum):
    POSITIVE_GT_RATIO = auto()
    POSITIVE_PRED_RATIO = auto()
    TRUE_POSITIVE_RATIO = auto()
    FALSE_POSITIVE_RATIO = auto()
    TRUE_NEGATIVE_RATIO = auto()
    FALSE_NEGATIVE_RATIO = auto()
    ACCURACY = auto()
    RECALL = auto()
    SPECIFICITY = auto()
    BALANCED_ACCURACY = auto()
    PRECISION = auto()
    F1_SCORE = auto()
    CEL = auto()
    CEL_POS = auto()
    CEL_NEG = auto()
    AUROC = auto()
    NDCG = auto()
    HIT_RATE_AT_1 = auto()
    HIT_RATE_AT_3 = auto()
    MRR = auto()
    CONFIDENCE_ALL = auto()
    CONFIDENCE_POS_GT = auto()
    CONFIDENCE_NEG_GT = auto()
    CONFIDENCE_BOTTOM_25_PERCENT_POS_GT = auto()
    CONFIDENCE_TOP_25_PERCENT_NEG_GT = auto()
    CONFIDENCE_TP = auto()
    CONFIDENCE_FP = auto()
    CONFIDENCE_TN = auto()
    CONFIDENCE_FN = auto()
    SPECIFICITY_SAMPLES = auto()
    PRECISION_SAMPLES = auto()
    F1_SCORE_SAMPLES = auto()
    CONFIDENCE_ALL_SAMPLES = auto()
    CONFIDENCE_TOP_3_SAMPLES = auto()
    CEL_SAMPLES = auto()
    AUROC_SAMPLES = auto()
    NDCG_SAMPLES = auto()
    HIT_RATE_AT_1_SAMPLES = auto()
    HIT_RATE_AT_3_SAMPLES = auto()
    MRR_SAMPLES = auto()
SCORES_DICT = { Score.POSITIVE_GT_RATIO : {"name": "Positive Ground Truth Ratio", "abbreviation": "PGTRo", "increase_better": True, "derivable": False, "ranking": False},
                Score.POSITIVE_PRED_RATIO : {"name": "Positive Prediction Ratio", "abbreviation": "PPRRo", "increase_better": True, "derivable": False, "ranking": False},
                Score.TRUE_POSITIVE_RATIO : {"name": "True Positive Ratio", "abbreviation": "TPRo", "increase_better": True, "derivable": False, "ranking": False},
                Score.FALSE_POSITIVE_RATIO : {"name": "False Positive Ratio", "abbreviation": "FPRo", "increase_better": False, "derivable": False, "ranking": False},
                Score.TRUE_NEGATIVE_RATIO : {"name": "True Negative Ratio", "abbreviation": "TNRo", "increase_better": True, "derivable": False, "ranking": False},
                Score.FALSE_NEGATIVE_RATIO : {"name": "False Negative Ratio", "abbreviation": "FNRo", "increase_better": False, "derivable": False, "ranking": False},
                Score.ACCURACY : {"name": "Accuracy", "abbreviation": "ACC", "increase_better": True, "derivable": False, "ranking": False},
                Score.RECALL : {"name": "Recall", "abbreviation": "REC", "increase_better": True, "derivable": False, "ranking": False},
                Score.SPECIFICITY : {"name": "Specificity", "abbreviation": "SPE", "increase_better": True, "derivable": False, "ranking": False},
                Score.BALANCED_ACCURACY : {"name": "Balanced Accuracy", "abbreviation": "BAL", "increase_better": True, "derivable": True, "ranking": False},
                Score.PRECISION : {"name": "Precision", "abbreviation": "PRE", "increase_better": True, "derivable": False, "ranking": False},
                Score.F1_SCORE : {"name": "F1 Score", "abbreviation": "F1", "increase_better": True, "derivable": True, "ranking": False},
                Score.CEL : {"name": "Cross-Entropy Loss", "abbreviation": "CEL", "increase_better": False, "derivable": False, "ranking": False},
                Score.CEL_POS : {"name": "CEL among positive GT", "abbreviation": "CELP", "increase_better": False, "derivable": False, "ranking": False},
                Score.CEL_NEG : {"name": "CEL among negative GT", "abbreviation": "CELN", "increase_better": False, "derivable": False, "ranking": False},
                Score.AUROC : {"name": "Area under Roc Curve", "abbreviation": "AUC", "increase_better": True, "derivable": False, "ranking": False},
                Score.NDCG : {"name": "Normalized Discounted Cumulative Gain", "abbreviation": "NDCG", "increase_better": True, "derivable": False, "ranking": True},
                Score.HIT_RATE_AT_1 : {"name": "Hit Rate at 1", "abbreviation": "HR@1", "increase_better": True, "derivable": False, "ranking": True},
                Score.HIT_RATE_AT_3 : {"name": "Hit Rate at 3", "abbreviation": "HR@3", "increase_better": True, "derivable": False, "ranking": True},
                Score.MRR : {"name": "Mean Reciprocal Rank", "abbreviation": "MRR", "increase_better": True, "derivable": False, "ranking": True},
                Score.CONFIDENCE_ALL : {"name": "Confidence All", "abbreviation": "CALL", "increase_better": True, "derivable": False, "ranking": False},
                Score.CONFIDENCE_POS_GT : {"name": "Confidence among positive GT", "abbreviation": "CPGT", "increase_better": True, "derivable": False, "ranking": False},
                Score.CONFIDENCE_NEG_GT : {"name": "Confidence among negative GT", "abbreviation": "CNGT", "increase_better": False, "derivable": False, "ranking": False},
                Score.CONFIDENCE_BOTTOM_25_PERCENT_POS_GT : {"name": "Confidence among 25 Percent least confident positive GT", "abbreviation": "CPGT25", "increase_better": True, 
                                                             "derivable": False, "ranking": False},
                Score.CONFIDENCE_TOP_25_PERCENT_NEG_GT : {"name": "Confidence among 25 Percent most confident negative GT", "abbreviation": "CNGT25", "increase_better": False, 
                                                          "derivable": False, "ranking": False},
                Score.CONFIDENCE_TP : {"name": "Confidence among True Positives", "abbreviation": "CTP", "increase_better": True, "derivable": False, "ranking": False},
                Score.CONFIDENCE_FP : {"name": "Confidence among False Positives", "abbreviation": "CFP", "increase_better": False, "derivable": False, "ranking": False},
                Score.CONFIDENCE_TN : {"name": "Confidence among True Negatives", "abbreviation": "CTN", "increase_better": True, "derivable": False, "ranking": False},
                Score.CONFIDENCE_FN : {"name": "Confidence among False Negatives", "abbreviation": "CFN", "increase_better": False, "derivable": False, "ranking": False},
                Score.SPECIFICITY_SAMPLES : {"name": "Specificity among Samples", "abbreviation": "SPE_S", "increase_better": True, "derivable": False, "ranking": False},
                Score.PRECISION_SAMPLES : {"name": "Precision among Samples", "abbreviation": "PRE_S", "increase_better": True, "derivable": False, "ranking": False},
                Score.F1_SCORE_SAMPLES : {"name": "F1 Score among Samples", "abbreviation": "F1_S", "increase_better": True, "derivable": True, "ranking": False},
                Score.CONFIDENCE_ALL_SAMPLES : {"name": "Confidence All among Samples", "abbreviation": "CALL_S", "increase_better": True, "derivable": False, "ranking": False},
                Score.CONFIDENCE_TOP_3_SAMPLES : {"name": "Confidence among Top 3 Samples", "abbreviation": "CT3_S", "increase_better": True, "derivable": False, "ranking": False},
                Score.CEL_SAMPLES : {"name": "Cross-Entropy Loss among Samples", "abbreviation": "CEL_S", "increase_better": False, "derivable": False, "ranking": False},
                Score.AUROC_SAMPLES : {"name": "Area under Roc Curve among Samples", "abbreviation": "AUC_S", "increase_better": True, "derivable": False, "ranking": False},
                Score.NDCG_SAMPLES : {"name": "Normalized Discounted Cumulative Gain among Samples", "abbreviation": "NDCG_S", "increase_better": True, "derivable": False, "ranking": True},
                Score.HIT_RATE_AT_1_SAMPLES : {"name": "Hit Rate at 1 among Samples", "abbreviation": "HR@1_S", "increase_better": True, "derivable": False, "ranking": True},
                Score.HIT_RATE_AT_3_SAMPLES : {"name": "Hit Rate at 3 among Samples", "abbreviation": "HR@3_S", "increase_better": True, "derivable": False, "ranking": True},
                Score.MRR_SAMPLES : {"name": "Mean Reciprocal Rank among Samples", "abbreviation": "MRR_S", "increase_better": True, "derivable": False, "ranking": True}}
RANKING_SCORES = [score for score in Score if SCORES_DICT[score]["ranking"]]

def get_score_from_arg(score_arg : str) -> Score:
    valid_score_args = [score.name.lower() for score in Score]
    if score_arg.lower() not in valid_score_args:
        raise ValueError(f"Invalid argument {score_arg} 'algorithm'. Possible values: {valid_score_args}.")
    return Score[score_arg.upper()]

def specificity_score(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if tn + fp != 0 else 0

def cel_score(y_proba : np.ndarray, positive : bool) -> float:
    eps = np.finfo(y_proba.dtype).eps
    y_proba = np.clip(y_proba, eps, 1 - eps)
    return -np.mean(np.log(y_proba)) if positive else -np.mean(np.log(1 - y_proba))

def get_score(score : Score, y_true : np.ndarray, y_pred : np.ndarray, y_proba : np.ndarray, 
              y_negative_samples_pred : np.ndarray, y_negative_samples_proba : np.ndarray) -> float:
    if score == Score.POSITIVE_GT_RATIO:
        return np.mean(y_true)
    elif score == Score.POSITIVE_PRED_RATIO:
        return np.mean(y_pred)
    elif score == Score.TRUE_POSITIVE_RATIO:
        return np.mean((y_true == 1) & (y_pred == 1))
    elif score == Score.FALSE_POSITIVE_RATIO:
        return np.mean((y_true == 0) & (y_pred == 1))
    elif score == Score.TRUE_NEGATIVE_RATIO:
        return np.mean((y_true == 0) & (y_pred == 0))
    elif score == Score.FALSE_NEGATIVE_RATIO:
        return np.mean((y_true == 1) & (y_pred == 0))
    elif score == Score.ACCURACY:
        return accuracy_score(y_true, y_pred)
    elif score == Score.RECALL:
        return recall_score(y_true, y_pred, zero_division = 0)
    elif score == Score.SPECIFICITY:
        return specificity_score(y_true, y_pred)
    elif score == Score.PRECISION:
        return precision_score(y_true, y_pred, zero_division = 0)
    elif score == Score.CEL:
        return log_loss(y_true, y_proba)
    elif score == Score.CEL_POS:
        return cel_score(y_proba[y_true == 1], True)
    elif score == Score.CEL_NEG:
        return cel_score(y_proba[y_true == 0], False)
    elif score == Score.AUROC:
        return roc_auc_score(y_true, y_proba)
    elif score == Score.CONFIDENCE_ALL:
        return np.mean(y_proba)
    elif score == Score.CONFIDENCE_POS_GT:
        return np.mean(y_proba[y_true == 1])
    elif score == Score.CONFIDENCE_NEG_GT:
        return np.mean(y_proba[y_true == 0])
    elif score == Score.CONFIDENCE_BOTTOM_25_PERCENT_POS_GT:
        n_vals = len(y_proba[y_true == 1]) // 4
        pos_idxs = np.argsort(y_proba[y_true == 1])[:n_vals]
        return np.mean(y_proba[y_true == 1][pos_idxs])
    elif score == Score.CONFIDENCE_TOP_25_PERCENT_NEG_GT:
        n_vals = len(y_proba[y_true == 0]) // 4
        neg_idxs = np.argsort(y_proba[y_true == 0])[-n_vals:]
        return np.mean(y_proba[y_true == 0][neg_idxs])
    elif score == Score.CONFIDENCE_TP:
        tp = (y_true == 1) & (y_pred == 1)
        return 0.5 if np.sum(tp) == 0 else np.mean(y_proba[tp])
    elif score == Score.CONFIDENCE_FP:
        fp = (y_true == 0) & (y_pred == 1)
        return 0.5 if np.sum(fp) == 0 else np.mean(y_proba[fp])
    elif score == Score.CONFIDENCE_TN:
        tn = (y_true == 0) & (y_pred == 0)
        return 0.5 if np.sum(tn) == 0 else np.mean(y_proba[tn])
    elif score == Score.CONFIDENCE_FN:
        fn = (y_true == 1) & (y_pred == 0)
        return 0.5 if np.sum(fn) == 0 else np.mean(y_proba[fn])
    elif score == Score.SPECIFICITY_SAMPLES:
        return np.mean(y_negative_samples_pred == 0)
    elif score == Score.PRECISION_SAMPLES:
        y_negative_samples = np.zeros(len(y_negative_samples_pred), dtype = y_true.dtype)
        return precision_score(np.concatenate((y_true, y_negative_samples)), np.concatenate((y_pred, y_negative_samples_pred)), zero_division = 0)
    elif score == Score.CONFIDENCE_ALL_SAMPLES:
        return np.mean(y_negative_samples_proba)
    elif score == Score.CONFIDENCE_TOP_3_SAMPLES:
        return np.mean(np.sort(y_negative_samples_proba)[-3:])
    elif score == Score.CEL_SAMPLES:
        return cel_score(y_negative_samples_proba, False)
    elif score == Score.AUROC_SAMPLES:
        y = np.concatenate((y_true[y_true == 1], np.zeros(len(y_negative_samples_pred), dtype = y_true.dtype)))
        y_prob = np.concatenate((y_proba[y_true == 1], y_negative_samples_proba))
        return roc_auc_score(y, y_prob)

def derive_score(score : Score, user_scores : list, scores_indices_dict : dict, validation : bool) -> float:
    if score == Score.BALANCED_ACCURACY:
        recall = user_scores[scores_indices_dict[f"{'val' if validation else 'train'}_{Score.RECALL.name.lower()}"]]
        specificity = user_scores[scores_indices_dict[f"{'val' if validation else 'train'}_{Score.SPECIFICITY.name.lower()}"]]
        return (recall + specificity) / 2
    elif score == Score.F1_SCORE:
        precision = user_scores[scores_indices_dict[f"{'val' if validation else 'train'}_{Score.PRECISION.name.lower()}"]]
        recall = user_scores[scores_indices_dict[f"{'val' if validation else 'train'}_{Score.RECALL.name.lower()}"]]
        if precision == 0 and recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    elif score == Score.F1_SCORE_SAMPLES:
        precision_samples = user_scores[scores_indices_dict[f"{'val' if validation else 'train'}_{Score.PRECISION_SAMPLES.name.lower()}"]]
        recall_samples = user_scores[scores_indices_dict[f"{'val' if validation else 'train'}_{Score.RECALL.name.lower()}"]]
        if precision_samples == 0 and recall_samples == 0:
            return 0
        return 2 * (precision_samples * recall_samples) / (precision_samples + recall_samples)

def get_ranking_scores(y_train_rated: np.ndarray, y_train_rated_proba: np.ndarray, y_val: np.ndarray, y_val_proba: np.ndarray, 
                       y_negative_samples_proba: np.ndarray) -> dict:
    ranking_scores = {}
    y_train_pos_proba, y_val_pos_proba = y_train_rated_proba[y_train_rated == 1], y_val_proba[y_val == 1]
    y_train_pos_proba_n, y_val_pos_proba_n = len(y_train_pos_proba), len(y_val_pos_proba)
    y_pos_proba = np.concatenate((y_train_pos_proba, y_val_pos_proba))
    for ranking_score in RANKING_SCORES:
        ranking_scores[f"train_{ranking_score.name.lower()}"] = np.empty(y_train_pos_proba_n)
        ranking_scores[f"val_{ranking_score.name.lower()}"] = np.empty(y_val_pos_proba_n)
    y_val_neg_proba = y_val_proba[y_val == 0]
    y_val_neg_proba_n = len(y_val_neg_proba)
    y_ranking = np.concatenate(([1], np.zeros(y_val_neg_proba_n, dtype = y_train_rated.dtype)))
    y_ranking_samples = np.concatenate(([1], np.zeros(len(y_negative_samples_proba), dtype = y_train_rated.dtype)))
    for i, y_pos_prob in enumerate(y_pos_proba):
        y_proba = np.concatenate(([y_pos_prob], y_val_neg_proba))
        y_proba_samples = np.concatenate(([y_pos_prob], y_negative_samples_proba))
        prefix, j = ("train", i) if i < y_train_pos_proba_n else ("val", i - y_train_pos_proba_n)
        for ranking_score in RANKING_SCORES:
            ranking_scores[f"{prefix}_{ranking_score.name.lower()}"][j] = get_ranking_score(ranking_score, y_ranking, y_proba, y_ranking_samples, y_proba_samples)
    for ranking_score in RANKING_SCORES:
        ranking_scores[f"train_{ranking_score.name.lower()}"] = np.mean(ranking_scores[f"train_{ranking_score.name.lower()}"])
        ranking_scores[f"val_{ranking_score.name.lower()}"] = np.mean(ranking_scores[f"val_{ranking_score.name.lower()}"])
    return ranking_scores

def get_ranking_score(ranking_score : Score, y_ranking : np.ndarray, y_proba : np.ndarray, y_ranking_samples : np.ndarray, y_proba_samples : np.ndarray) -> float:
    if ranking_score == Score.NDCG:
        return ndcg_score(y_ranking.reshape(1, -1), y_proba.reshape(1, -1))
    elif ranking_score == Score.NDCG_SAMPLES:
        return ndcg_score(y_ranking_samples.reshape(1, -1), y_proba_samples.reshape(1, -1))
    elif ranking_score == Score.HIT_RATE_AT_1:
        return float(y_ranking[np.argmax(y_proba)] > 0)
    elif ranking_score == Score.HIT_RATE_AT_1_SAMPLES:
        return float(y_ranking_samples[np.argmax(y_proba_samples)] > 0)
    elif ranking_score == Score.HIT_RATE_AT_3:
        top_3_idxs = np.argsort(y_proba)[-3:]
        return float(np.any(y_ranking[top_3_idxs] > 0))
    elif ranking_score == Score.HIT_RATE_AT_3_SAMPLES:
        top_3_idxs = np.argsort(y_proba_samples)[-3:]
        return float(np.any(y_ranking_samples[top_3_idxs] > 0))
    elif ranking_score in [Score.MRR, Score.MRR_SAMPLES]:
        if ranking_score == Score.MRR:
            ranking, proba = y_ranking, y_proba
        else:
            ranking, proba = y_ranking_samples, y_proba_samples
        sorted_indices = np.argsort(proba)[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            if ranking[idx] > 0:
                return 1.0 / rank
        return 0.0
   
class Algorithm(Enum):
    LOGREG = auto()
    SVM = auto()

def get_algorithm_from_arg(algorithm_arg : str) -> Algorithm:
    valid_algorithm_args = [algorithm.name.lower() for algorithm in Algorithm]
    if algorithm_arg.lower() not in valid_algorithm_args:
        raise ValueError(f"Invalid argument {algorithm_arg} 'algorithm'. Possible values: {valid_algorithm_args}.")
    return Algorithm[algorithm_arg.upper()]

def get_model(algorithm : Algorithm, max_iter : int, clf_C: float, random_state : int, logreg_solver : str = None, svm_kernel : str = None) -> object:
    if algorithm == Algorithm.LOGREG:
        return LogisticRegression(max_iter = max_iter, C = clf_C, random_state = random_state, solver = logreg_solver)
    elif algorithm == Algorithm.SVM:
        return SVC(max_iter = max_iter, C = clf_C, random_state = random_state, kernel = svm_kernel, probability = True)

class Evaluation(Enum):
    CROSS_VALIDATION = auto()
    TRAIN_TEST_SPLIT = auto()

def get_evaluation_from_arg(evaluation_arg : str) -> Evaluation:
    valid_evaluation_args = [evaluation.name.lower() for evaluation in Evaluation]
    if evaluation_arg.lower() not in valid_evaluation_args:
        raise ValueError(f"Invalid argument {evaluation_arg} 'evaluation'. Possible values: {valid_evaluation_args}.")
    return Evaluation[evaluation_arg.upper()]

def get_cross_val(stratified : bool, k_folds : int, random_state : int) -> object:
    if stratified:
        return StratifiedKFold(n_splits = k_folds, shuffle = True, random_state = random_state)
    else:
        return KFold(n_splits = k_folds, shuffle = True, random_state = random_state)