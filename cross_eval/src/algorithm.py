from enum import Enum, auto
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, log_loss, roc_auc_score, ndcg_score
from sklearn.svm import SVC
import numpy as np

class Score(Enum):
    ACCURACY = auto()
    RECALL = auto()
    PRECISION = auto()
    SPECIFICITY = auto()
    BALANCED_ACCURACY = auto()
    F1_SCORE = auto()
    AUROC = auto()
    NDCG = auto()
    CEL = auto()
    CEL_POS = auto()
    CEL_NEG = auto()
SCORES_DICT = { Score.ACCURACY : {"name": "Accuracy", "abbreviation": "ACC", "increase_better": True, "derivable": False},
                Score.RECALL : {"name": "Recall", "abbreviation": "REC", "increase_better": True, "derivable": False},
                Score.PRECISION : {"name": "Precision", "abbreviation": "PRE", "increase_better": True, "derivable": False},
                Score.SPECIFICITY : {"name": "Specificity", "abbreviation": "SPE", "increase_better": True, "derivable": False},
                Score.BALANCED_ACCURACY : {"name": "Balanced Accuracy", "abbreviation": "BAL", "increase_better": True, "derivable": True},
                Score.F1_SCORE : {"name": "F1 Score", "abbreviation": "F1", "increase_better": True, "derivable": True},
                Score.AUROC : {"name": "Area under Roc Curve", "abbreviation": "AUC", "increase_better": True, "derivable": False},
                Score.NDCG : {"name": "Normalized Discounted Cumulative Gain", "abbreviation": "NDCG", "increase_better": True, "derivable": False},
                Score.CEL : {"name": "Cross-Entropy Loss", "abbreviation": "CEL", "increase_better": False, "derivable": False},
                Score.CEL_POS : {"name": "CEL among positive GT", "abbreviation": "CE-P", "increase_better": False, "derivable": False},
                Score.CEL_NEG : {"name": "CEL among negative GT", "abbreviation": "CE-N", "increase_better": False, "derivable": False}}

def get_score_from_arg(score_arg : str) -> Score:
    valid_score_args = [score.name.lower() for score in Score]
    if score_arg.lower() not in valid_score_args:
        raise ValueError(f"Invalid argument {score_arg} 'algorithm'. Possible values: {valid_score_args}.")
    return Score[score_arg.upper()]

def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if tn + fp != 0 else 0

def get_score(score : Score, y_true : np.ndarray, y_pred : np.ndarray, y_proba : np.ndarray) -> float:
    if score == Score.ACCURACY:
        return accuracy_score(y_true, y_pred)
    elif score == Score.RECALL:
        return recall_score(y_true, y_pred, zero_division = 0)
    elif score == Score.PRECISION:
        return precision_score(y_true, y_pred, zero_division = 0)
    elif score == Score.SPECIFICITY:
        return specificity_score(y_true, y_pred)
    elif score == Score.AUROC:
        return roc_auc_score(y_true, y_proba)
    elif score == Score.NDCG:
        return ndcg_score(y_true.reshape(1, -1), y_proba.reshape(1, -1))
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