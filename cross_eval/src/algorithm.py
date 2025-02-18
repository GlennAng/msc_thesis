from enum import Enum, auto
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, log_loss
from sklearn.svm import SVC
import numpy as np

"""
class Score(Enum):
    ACCURACY = auto()
    RECALL = auto()
    PRECISION = auto()
    SPECIFICITY = auto()
    BALANCED_ACCURACY = auto()
    F1_SCORE = auto()
    CEL = auto()
SCORES_NAMES_DICT = {Score.ACCURACY: "Accuracy", Score.RECALL: "Recall", Score.PRECISION: "Precision", Score.SPECIFICITY: "Specificity", 
                     Score.BALANCED_ACCURACY: "Balanced Accuracy", Score.F1_SCORE: "F1 Score", Score.CEL: "Cross-Entropy Loss"}
SCORES_ABBREVIATIONS_DICT = {Score.ACCURACY: "ACC", Score.RECALL: "REC", Score.PRECISION: "PRE", Score.SPECIFICITY: "SPE",
                             Score.BALANCED_ACCURACY: "BAL", Score.F1_SCORE: "F1", Score.CEL: "CEL"}
SCORES_INCREASE_BETTER_DICT = {Score.ACCURACY: True, Score.RECALL: True, Score.PRECISION: True, Score.SPECIFICITY: True,
                               Score.BALANCED_ACCURACY: True, Score.F1_SCORE: True, Score.CEL: False}

def get_score_from_arg(score_arg : str) -> Score:
    valid_score_args = [score.name.lower() for score in Score]
    if score_arg.lower() not in valid_score_args:
        raise ValueError(f"Invalid argument {score_arg} 'algorithm'. Possible values: {valid_score_args}.")
    return Score[score_arg.upper()]

def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if tn + fp != 0 else 0

def get_score(score : Score, y_true, y_pred, y_proba = None) -> float:
    if y_proba is not None:
        y_proba = np.array(y_proba)
    if score == Score.ACCURACY:
        return accuracy_score(y_true, y_pred)
    elif score == Score.RECALL:
        return recall_score(y_true, y_pred, zero_division = 0)
    elif score == Score.PRECISION:
        return precision_score(y_true, y_pred, zero_division = 0)
    elif score == Score.SPECIFICITY:
        return specificity_score(y_true, y_pred)
    elif score == Score.BALANCED_ACCURACY:
        return (get_score(Score.RECALL, y_true, y_pred) + get_score(Score.SPECIFICITY, y_true, y_pred)) / 2
    elif score == Score.F1_SCORE:
        recall_sc, precision_sc = get_score(Score.RECALL, y_true, y_pred), get_score(Score.PRECISION, y_true, y_pred)
        denominator = recall_sc + precision_sc
        return 2 * recall_sc * precision_sc / denominator if denominator != 0 else 0
    elif score == Score.CEL:
        return log_loss(y_true, y_proba)

"""
class Score(Enum):
    RECALL = auto()
    PRECISION = auto()
    SPECIFICITY = auto()
    BALANCED_ACCURACY = auto()
    CEL = auto()
    CEL_POS = auto()
    CEL_NEG = auto()
SCORES_NAMES_DICT = {Score.RECALL: "Recall", Score.PRECISION: "Precision", Score.SPECIFICITY: "Specificity", Score.BALANCED_ACCURACY: "Balanced Accuracy", 
                     Score.CEL: "Cross-Entropy Loss", Score.CEL_POS: "CEL among positive GT", Score.CEL_NEG: "CEL among negative GT"}
SCORES_ABBREVIATIONS_DICT = {Score.RECALL: "REC", Score.PRECISION: "PRE", Score.SPECIFICITY: "SPE", Score.BALANCED_ACCURACY: "BAL", 
                             Score.CEL: "CEL", Score.CEL_POS: "CEL_POS", Score.CEL_NEG: "CEL_NEG"}
SCORES_INCREASE_BETTER_DICT = {Score.RECALL: True, Score.PRECISION: True, Score.SPECIFICITY: True, Score.BALANCED_ACCURACY: True, 
                               Score.CEL: False, Score.CEL_POS: False, Score.CEL_NEG: False}

def get_score_from_arg(score_arg : str) -> Score:
    valid_score_args = [score.name.lower() for score in Score]
    if score_arg.lower() not in valid_score_args:
        raise ValueError(f"Invalid argument {score_arg} 'algorithm'. Possible values: {valid_score_args}.")
    return Score[score_arg.upper()]

def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if tn + fp != 0 else 0

def get_score(score : Score, y_true, y_pred, y_proba = None) -> float:
    if y_proba is not None:
        y_proba = np.array(y_proba)
    if score == Score.RECALL:
        return recall_score(y_true, y_pred, zero_division = 0)
    elif score == Score.PRECISION:
        return precision_score(y_true, y_pred, zero_division = 0)
    elif score == Score.SPECIFICITY:
        return specificity_score(y_true, y_pred)
    elif score == Score.BALANCED_ACCURACY:
        return (get_score(Score.RECALL, y_true, y_pred) + get_score(Score.SPECIFICITY, y_true, y_pred)) / 2
    elif score == Score.CEL:
        return log_loss(y_true, y_proba)
    elif score == Score.CEL_POS:
        pos_mask = y_true >= 0.5
        return -np.mean(np.log(y_proba[pos_mask])) 
    elif score == Score.CEL_NEG:
        neg_mask = y_true < 0.5
        return -np.mean(np.log(1 - y_proba[neg_mask]))
        
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