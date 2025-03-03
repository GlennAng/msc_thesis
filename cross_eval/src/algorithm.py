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
    AUROC = auto()
    NDCG_NEG = auto()
    CEL = auto()
    CEL_POS = auto()
    CEL_NEG = auto()
    CONFIDENCE_ALL = auto()
    CONFIDENCE_POS_GT = auto()
    CONFIDENCE_NEG_GT = auto()
    CONFIDENCE_POS_GT_LOWEST_25_PERCENT = auto()
    CONFIDENCE_NEG_GT_HIGHEST_25_PERCENT = auto()
    CONFIDENCE_POS_PRED = auto()
    CONFIDENCE_NEG_PRED = auto()
    CONFIDENCE_TP = auto()
    CONFIDENCE_FP = auto()
    CONFIDENCE_TN = auto()
    CONFIDENCE_FN = auto()
    CONFIDENCE_RANKING = auto()
    CONFIDENCE_RANKING_MAX_3 = auto()
    ACCURACY_RANKING = auto()
    SAMPLING_PRECISION = auto()
    SAMPLING_F1_SCORE = auto()
    NDCG = auto()
    PRECISION_AT_1 = auto()
    HIT_RATE_AT_3 = auto()
    MRR = auto()
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
                Score.NEGATIVE_PRECISION : {"name": "Negative Precision", "abbreviation": "NPRE", "increase_better": True, "derivable": False, "ranking": False},
                Score.NEGATIVE_F1_SCORE : {"name": "Negative F1 Score", "abbreviation": "NF1", "increase_better": True, "derivable": True, "ranking": False},
                Score.AUROC : {"name": "Area under Roc Curve", "abbreviation": "AUC", "increase_better": True, "derivable": False, "ranking": False},
                Score.CEL : {"name": "Cross-Entropy Loss", "abbreviation": "CEL", "increase_better": False, "derivable": False, "ranking": False},
                Score.CEL_POS : {"name": "CEL among positive GT", "abbreviation": "CELP", "increase_better": False, "derivable": False, "ranking": False},
                Score.CEL_NEG : {"name": "CEL among negative GT", "abbreviation": "CELN", "increase_better": False, "derivable": False, "ranking": False},
                Score.CONFIDENCE_ALL : {"name": "Confidence All", "abbreviation": "CALL", "increase_better": True, "derivable": False, "ranking": False},
                Score.CONFIDENCE_POS_GT : {"name": "Confidence among positive GT", "abbreviation": "CPGT", "increase_better": True, "derivable": False, "ranking": False},
                Score.CONFIDENCE_NEG_GT : {"name": "Confidence among negative GT", "abbreviation": "CNGT", "increase_better": False, "derivable": False, "ranking": False},
                Score.CONFIDENCE_POS_GT_LOWEST_25_PERCENT : {"name": "Confidence among 25 Percent least confident positive GT", "abbreviation": "CPGT25", "increase_better": True, "derivable": False, "ranking": False},
                Score.CONFIDENCE_NEG_GT_HIGHEST_25_PERCENT : {"name": "Confidence among 25 Percent most confident negative GT", "abbreviation": "CNGT25", "increase_better": False, "derivable": False, "ranking": False},
                Score.CONFIDENCE_POS_PRED : {"name": "Confidence among positive Predictions", "abbreviation": "CPPR", "increase_better": True, "derivable": False, "ranking": False},
                Score.CONFIDENCE_NEG_PRED : {"name": "Confidence among negative Predictions", "abbreviation": "CNPR", "increase_better": False, "derivable": False, "ranking": False},
                Score.CONFIDENCE_TP : {"name": "Confidence among True Positives", "abbreviation": "CTP", "increase_better": True, "derivable": False, "ranking": False},
                Score.CONFIDENCE_FP : {"name": "Confidence among False Positives", "abbreviation": "CFP", "increase_better": False, "derivable": False, "ranking": False},
                Score.CONFIDENCE_TN : {"name": "Confidence among True Negatives", "abbreviation": "CTN", "increase_better": True, "derivable": False, "ranking": False},
                Score.CONFIDENCE_FN : {"name": "Confidence among False Negatives", "abbreviation": "CFN", "increase_better": False, "derivable": False, "ranking": False},
                Score.CONFIDENCE_RANKING : {"name": "Confidence among Ranking Papers", "abbreviation": "CRk", "increase_better": True, "derivable": False, "ranking": True},
                Score.CONFIDENCE_RANKING_MAX_3 : {"name": "Confidence among 3 most confident Ranking Papers", "abbreviation": "C-Rk3", "increase_better": True, "derivable": False, "ranking": True},
                Score.ACCURACY_RANKING : {"name": "Accuracy on Ranking", "abbreviation": "ACCRk", "increase_better": True, "derivable": False, "ranking": True},
                Score.NDCG : {"name": "Normalized Discounted Cumulative Gain", "abbreviation": "NDCG", "increase_better": True, "derivable": False, "ranking": True},
                Score.PRECISION_AT_1 : {"name": "Precision at 1", "abbreviation": "PR@1", "increase_better": True, "derivable": False, "ranking": True},
                Score.HIT_RATE_AT_3 : {"name": "Hit Rate at 3", "abbreviation": "HR@3", "increase_better": True, "derivable": False, "ranking": True},
                Score.MRR : {"name": "Mean Reciprocal Rank", "abbreviation": "MRR", "increase_better": True, "derivable": False, "ranking": True},
                Score.NDCG_NEG : {"name": "Normalized Discounted Cumulative Gain for Negative Votes", "abbreviation": "NDCG-N", "increase_better": True, "derivable": False, "ranking": True}
                }
CLASSIFICATION_SCORES = [score for score in Score if not SCORES_DICT[score]["ranking"]]
RANKING_SCORES = [score for score in Score if SCORES_DICT[score]["ranking"]]

def get_score_from_arg(score_arg : str) -> Score:
    valid_score_args = [score.name.lower() for score in Score]
    if score_arg.lower() not in valid_score_args:
        raise ValueError(f"Invalid argument {score_arg} 'algorithm'. Possible values: {valid_score_args}.")
    return Score[score_arg.upper()]

def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if tn + fp != 0 else 0

def get_score(score : Score, y_true : np.ndarray, y_pred : np.ndarray, y_proba : np.ndarray) -> float:
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
    elif score == Score.NEGATIVE_PRECISION:
        return precision_score(y_true, y_pred, pos_label = 0, zero_division = 0)
    elif score == Score.AUROC:
        return roc_auc_score(y_true, y_proba)
    elif score == Score.CEL:
        return log_loss(y_true, y_proba)
    elif score == Score.CEL_POS:
        eps = np.finfo(y_proba.dtype).eps
        y_proba = np.clip(y_proba, eps, 1 - eps)
        pos_mask = y_true >= 0.5
        return -np.mean(np.log(y_proba[pos_mask]))
    elif score == Score.CEL_NEG:
        eps = np.finfo(y_proba.dtype).eps
        y_proba = np.clip(y_proba, eps, 1 - eps)
        neg_mask = y_true < 0.5
        return -np.mean(np.log(1 - y_proba[neg_mask]))
    elif score == Score.CONFIDENCE_ALL:
        return np.mean(y_proba)
    elif score == Score.CONFIDENCE_POS_GT:
        return np.mean(y_proba[y_true == 1])
    elif score == Score.CONFIDENCE_NEG_GT:
        return np.mean(y_proba[y_true == 0])
    elif score == Score.CONFIDENCE_POS_GT_LOWEST_25_PERCENT:
        n_vals = len(y_proba[y_true == 1]) // 4
        pos_idxs = np.argsort(y_proba[y_true == 1])[:n_vals]
        return np.mean(y_proba[y_true == 1][pos_idxs])
    elif score == Score.CONFIDENCE_NEG_GT_HIGHEST_25_PERCENT:
        n_vals = len(y_proba[y_true == 0]) // 4
        neg_idxs = np.argsort(y_proba[y_true == 0])[-n_vals:]
        return np.mean(y_proba[y_true == 0][neg_idxs])
    elif score == Score.CONFIDENCE_POS_PRED:
        ppred = y_pred == 1
        return 0.5 if np.sum(ppred) == 0 else np.mean(y_proba[ppred])
    elif score == Score.CONFIDENCE_NEG_PRED:
        npred = y_pred == 0
        return 0.5 if np.sum(npred) == 0 else np.mean(y_proba[npred])
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
    elif score == Score.NEGATIVE_F1_SCORE:
        negative_precision = user_scores[scores_indices_dict[f"{'val' if validation else 'train'}_{Score.NEGATIVE_PRECISION.name.lower()}"]]
        specificity = user_scores[scores_indices_dict[f"{'val' if validation else 'train'}_{Score.SPECIFICITY.name.lower()}"]]
        if negative_precision == 0 and specificity == 0:
            return 0
        return 2 * (negative_precision * specificity) / (negative_precision + specificity)

def get_ranking_scores(y_train_rated: np.ndarray, y_train_rated_proba: np.ndarray, y_val_rated: np.ndarray, y_val_rated_proba: np.ndarray, 
                       y_ranking_proba: np.ndarray) -> dict:
    y_train_pos_proba, y_val_pos_proba = y_train_rated_proba[y_train_rated == 1], y_val_rated_proba[y_val_rated == 1]
    y_train_pos_proba_n, y_val_pos_proba_n = len(y_train_pos_proba), len(y_val_pos_proba)
    ranking_scores = {f"train_{score.name.lower()}": np.empty(y_train_pos_proba_n) for score in RANKING_SCORES}
    ranking_scores.update({f"val_{score.name.lower()}": np.empty(y_val_pos_proba_n) for score in RANKING_SCORES})
    y_pos_proba = np.concatenate((y_train_pos_proba, y_val_pos_proba))
    y_ranking = np.concatenate(([1], np.zeros(y_ranking_proba.shape[0])))

    y_val_neg_proba = y_val_rated_proba[y_val_rated == 0]
    y_val_neg_proba_n = len(y_val_neg_proba)
    y_ranking_neg = np.concatenate(([1], np.zeros(y_val_neg_proba_n)))

    for i, y_pos_prob in enumerate(y_pos_proba):
        y_proba = np.concatenate(([y_pos_prob], y_ranking_proba))
        y_proba_neg = np.concatenate(([y_pos_prob], y_val_neg_proba))
        prefix, j = ("train", i) if i < y_train_pos_proba_n else ("val", i - y_train_pos_proba_n)
        for score in RANKING_SCORES:
            ranking_scores[f"{prefix}_{score.name.lower()}"][j] = get_ranking_score(score, y_proba, y_ranking, y_proba_neg, y_ranking_neg)
    for score in RANKING_SCORES:
        ranking_scores[f"train_{score.name.lower()}"] = np.mean(ranking_scores[f"train_{score.name.lower()}"])
        ranking_scores[f"val_{score.name.lower()}"] = np.mean(ranking_scores[f"val_{score.name.lower()}"])
    return ranking_scores
            
def get_ranking_score(ranking_score : Score, y_proba : np.ndarray, y_ranking : np.ndarray, y_proba_neg : np.ndarray, y_ranking_neg : np.ndarray) -> float:
    if ranking_score == Score.CONFIDENCE_RANKING:
        return np.mean(y_proba[1:])
    elif ranking_score == Score.CONFIDENCE_RANKING_MAX_3:
        return np.mean(np.sort(y_proba[1:])[-3:])
    elif ranking_score == Score.ACCURACY_RANKING:
        return accuracy_score(y_ranking[1:], y_proba[1:] > 0.5)
    elif ranking_score == Score.NDCG:
        return ndcg_score(y_ranking.reshape(1, -1), y_proba.reshape(1, -1), )
    elif ranking_score == Score.PRECISION_AT_1:
        top_idx = np.argmax(y_proba)
        return float(y_ranking[top_idx] > 0)
    elif ranking_score == Score.HIT_RATE_AT_3:
        top_3_idxs = np.argsort(y_proba)[-3:]
        return float(np.any(y_ranking[top_3_idxs] > 0))
    elif ranking_score == Score.MRR:
        sorted_indices = np.argsort(y_proba)[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            if y_ranking[idx] > 0:
                return 1.0 / rank
        return 0.0
    elif ranking_score == Score.NDCG_NEG:
        return ndcg_score(y_ranking_neg.reshape(1, -1), y_proba_neg.reshape(1, -1))
   
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