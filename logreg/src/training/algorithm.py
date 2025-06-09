import numpy as np, random
from enum import Enum, auto
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, log_loss, roc_auc_score, ndcg_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC
CONFIDENCE_THRESHOLD = 0.05

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
    CEL = auto()
    CEL_POS = auto()
    CEL_NEG = auto()
    AUROC_CLASSIFICATION = auto()
    NDCG= auto()
    MRR = auto()
    HIT_RATE_AT_1 = auto()
    INFO_NCE = auto()
    CONFIDENCE_ALL = auto()
    CONFIDENCE_TP = auto()
    CONFIDENCE_FP = auto()
    CONFIDENCE_TN = auto()
    CONFIDENCE_FN = auto()
    POSITIVE_GT_ABOVE_THRESHOLD = auto()
    SAMPLES_ABOVE_THRESHOLD = auto()
    SPECIFICITY_SAMPLES = auto()
    PRECISION_SAMPLES = auto()
    CONFIDENCE_ALL_SAMPLES = auto()
    CONFIDENCE_TOP_1_SAMPLES = auto()
    NDCG_SAMPLES = auto()
    NDCG_ALL = auto()
    MRR_SAMPLES = auto()
    MRR_ALL = auto()
    HIT_RATE_AT_1_SAMPLES = auto()
    HIT_RATE_AT_1_ALL = auto()
    INFO_NCE_SAMPLES = auto()
    INFO_NCE_ALL = auto()
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
                Score.CEL : {"name": "Cross-Entropy Loss", "abbreviation": "CEL", "increase_better": False, "derivable": False, "ranking": False},
                Score.CEL_POS : {"name": "CEL positive GT", "abbreviation": "CELP", "increase_better": False, "derivable": False, "ranking": False},
                Score.CEL_NEG : {"name": "CEL negative GT", "abbreviation": "CELN", "increase_better": False, "derivable": False, "ranking": False},
                Score.AUROC_CLASSIFICATION : {"name": "Area under Roc Curve Classification", "abbreviation": "AUC_C", "increase_better": True, "derivable": False, "ranking": False},
                Score.NDCG : {"name": "Normalized Discounted Cumulative Gain", "abbreviation": "NDCG", "increase_better": True, "derivable": False, "ranking": True},
                Score.MRR : {"name": "Mean Reciprocal Rank", "abbreviation": "MRR", "increase_better": True, "derivable": False, "ranking": True},
                Score.HIT_RATE_AT_1 : {"name": "Hit Rate @ 1", "abbreviation": "HR@1", "increase_better": True, "derivable": False, "ranking": True},
                Score.INFO_NCE : {"name": "InfoNCE", "abbreviation": "INCE", "increase_better": False, "derivable": False, "ranking": True},
                Score.CONFIDENCE_ALL : {"name": "Confidence All", "abbreviation": "C_ALL", "increase_better": True, "derivable": False, "ranking": False},
                Score.CONFIDENCE_TP : {"name": "Confidence True Positives", "abbreviation": "C_TP", "increase_better": True, "derivable": False, "ranking": False},
                Score.CONFIDENCE_FP : {"name": "Confidence False Positives", "abbreviation": "C_FP", "increase_better": False, "derivable": False, "ranking": False},
                Score.CONFIDENCE_TN : {"name": "Confidence True Negatives", "abbreviation": "C_TN", "increase_better": True, "derivable": False, "ranking": False},
                Score.CONFIDENCE_FN : {"name": "Confidence False Negatives", "abbreviation": "C_FN", "increase_better": False, "derivable": False, "ranking": False},
                Score.POSITIVE_GT_ABOVE_THRESHOLD : {"name": f"Positive Ground Truth above {CONFIDENCE_THRESHOLD}", "abbreviation": f"PGTAT\n{CONFIDENCE_THRESHOLD}", 
                                                        "increase_better": True, "derivable": False, "ranking": False},
                Score.SAMPLES_ABOVE_THRESHOLD : {"name": f"Samples above {CONFIDENCE_THRESHOLD}", "abbreviation": f"SAT\n{CONFIDENCE_THRESHOLD}", 
                                                    "increase_better": True, "derivable": False, "ranking": False},
                Score.SPECIFICITY_SAMPLES : {"name": "Specificity Samples", "abbreviation": "SPE\nSmpl", "increase_better": True, "derivable": False, "ranking": False},   
                Score.PRECISION_SAMPLES : {"name": "Precision Samples", "abbreviation": "PRE\nSmpl", "increase_better": True, "derivable": False, "ranking": False},
                Score.CONFIDENCE_ALL_SAMPLES : {"name": "Confidence All Samples", "abbreviation": "C_ALL\nSmpl", "increase_better": False, "derivable": False, "ranking": False},
                Score.CONFIDENCE_TOP_1_SAMPLES : {"name": "Confidence Top 1 Samples", "abbreviation": "C_T1\nSmpl", "increase_better": False, "derivable": False, "ranking": False},
                Score.NDCG_SAMPLES : {"name": "Normalized Discounted Cumulative Gain Samples", "abbreviation": "NDCG\nSmpl", "increase_better": True, "derivable": False, "ranking": True},
                Score.NDCG_ALL : {"name": "Normalized Discounted Cumulative Gain All", "abbreviation": "NDCG\nAll", "increase_better": True, "derivable": False, "ranking": True},
                Score.MRR_SAMPLES : {"name": "Mean Reciprocal Rank Samples", "abbreviation": "MRR\nSmpl", "increase_better": True, "derivable": False, "ranking": True},
                Score.MRR_ALL : {"name": "Mean Reciprocal Rank All", "abbreviation": "MRR\nAll", "increase_better": True, "derivable": False, "ranking": True},
                Score.HIT_RATE_AT_1_SAMPLES : {"name": "Hit Rate @ 1 Samples", "abbreviation": "HR@1\nSmpl", "increase_better": True, "derivable": False, "ranking": True},
                Score.HIT_RATE_AT_1_ALL : {"name": "Hit Rate @ 1 All", "abbreviation": "HR@1\nAll", "increase_better": True, "derivable": False, "ranking": True},
                Score.INFO_NCE_SAMPLES : {"name": "InfoNCE Samples", "abbreviation": "INCE\nSmpl", "increase_better": False, "derivable": False, "ranking": True},
                Score.INFO_NCE_ALL : {"name": "InfoNCE All", "abbreviation": "INCE\nAll", "increase_better": False, "derivable": False, "ranking": True}
                }
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
    elif score == Score.AUROC_CLASSIFICATION:
        return roc_auc_score(y_true, y_proba)
    elif score == Score.CONFIDENCE_ALL:
        return np.mean(y_proba)
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
    elif score == Score.POSITIVE_GT_ABOVE_THRESHOLD:
        positive_gt = (y_true == 1)
        if np.sum(positive_gt) == 0:
            return 0
        return np.mean(y_proba[positive_gt] > CONFIDENCE_THRESHOLD)
    elif score == Score.SAMPLES_ABOVE_THRESHOLD:
        return np.mean(y_negative_samples_proba > CONFIDENCE_THRESHOLD)
    elif score == Score.SPECIFICITY_SAMPLES:
        return np.mean(y_negative_samples_pred == 0)
    elif score == Score.PRECISION_SAMPLES:
        y = np.concatenate((y_true[y_true == 1], np.zeros(len(y_negative_samples_pred), dtype = y_true.dtype)))
        y_predi = np.concatenate((y_pred[y_true == 1], y_negative_samples_pred))
        return precision_score(y, y_predi, zero_division = 0)
    elif score == Score.CONFIDENCE_ALL_SAMPLES:
        return np.mean(y_negative_samples_proba)
    elif score == Score.CONFIDENCE_TOP_1_SAMPLES:
        return np.max(y_negative_samples_proba)

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

def get_ranking_scores(y_train_rated: np.ndarray, y_train_rated_logits: np.ndarray, y_val: np.ndarray, y_val_logits: np.ndarray, 
                       y_negrated_ranking_logits : np.ndarray, y_negative_samples_logits: np.ndarray, info_nce_temperature : float) -> dict:
    ranking_scores = {}
    y_train_pos_logits, y_val_pos_logits = y_train_rated_logits[y_train_rated == 1], y_val_logits[y_val == 1]
    y_train_pos_logits_n, y_val_pos_logits_n = len(y_train_pos_logits), len(y_val_pos_logits)
    y_pos_logits = np.concatenate((y_train_pos_logits, y_val_pos_logits))
    y_negative_all_logits = np.concatenate((y_negrated_ranking_logits, y_negative_samples_logits))
    for ranking_score in RANKING_SCORES:
        ranking_scores[f"train_{ranking_score.name.lower()}"] = np.zeros(max(1, y_train_pos_logits_n))
        ranking_scores[f"val_{ranking_score.name.lower()}"] = np.zeros(max(1, y_val_pos_logits_n))
    for i, y_pos_logit in enumerate(y_pos_logits):
        pos_rank = np.sum(y_negrated_ranking_logits >= y_pos_logit) + 1
        pos_rank_samples = np.sum(y_negative_samples_logits >= y_pos_logit) + 1
        pos_rank_all = np.sum(y_negative_all_logits >= y_pos_logit) + 1
        prefix, j = ("train", i) if i < y_train_pos_logits_n else ("val", i - y_train_pos_logits_n)
        for ranking_score in RANKING_SCORES:
            if ranking_score in [Score.INFO_NCE, Score.INFO_NCE_SAMPLES, Score.INFO_NCE_ALL]:
                ranking_scores[f"{prefix}_{ranking_score.name.lower()}"][j] = get_info_nce_score(ranking_score, y_pos_logit, y_negrated_ranking_logits, y_negative_samples_logits,
                                                                                                 y_negative_all_logits, info_nce_temperature)    
            else:
                ranking_scores[f"{prefix}_{ranking_score.name.lower()}"][j] = get_ranking_score(ranking_score, pos_rank, pos_rank_samples, pos_rank_all)
    for ranking_score in RANKING_SCORES:
        ranking_scores[f"train_{ranking_score.name.lower()}"] = np.mean(ranking_scores[f"train_{ranking_score.name.lower()}"])
        ranking_scores[f"val_{ranking_score.name.lower()}"] = np.mean(ranking_scores[f"val_{ranking_score.name.lower()}"])
    return ranking_scores

def get_ranking_score(ranking_score : Score, pos_rank : int, pos_rank_samples : int, pos_rank_all : int) -> float:
    ranking_score_split = ranking_score.name.lower().split("_")
    if ranking_score_split[-1] == "samples":
        pos_rank = pos_rank_samples
    elif ranking_score_split[-1] == "all":
        pos_rank = pos_rank_all
    if ranking_score in [Score.NDCG, Score.NDCG_SAMPLES, Score.NDCG_ALL]:
        return 1.0 / np.log2(pos_rank + 1)
    elif ranking_score in [Score.MRR, Score.MRR_SAMPLES, Score.MRR_ALL]:
        return 1.0 / pos_rank
    elif ranking_score in [Score.HIT_RATE_AT_1, Score.HIT_RATE_AT_1_SAMPLES, Score.HIT_RATE_AT_1_ALL]:
        return float(pos_rank == 1)

def get_info_nce_score(ranking_score : Score, y_pos_logit : float, y_negrated_ranking_logits : np.ndarray, y_negative_samples_logits : np.ndarray, y_negative_all_logits : np.ndarray,
                       info_nce_temperature : float) -> float:
    if ranking_score == Score.INFO_NCE:
        info_nce_array = np.concatenate((np.array([y_pos_logit]), y_negrated_ranking_logits))
    elif ranking_score == Score.INFO_NCE_SAMPLES:
        info_nce_array = np.concatenate((np.array([y_pos_logit]), y_negative_samples_logits))
    elif ranking_score == Score.INFO_NCE_ALL:
        info_nce_array = np.concatenate((np.array([y_pos_logit]), y_negative_all_logits))
    info_nce_array /= info_nce_temperature
    return -np.log(softmax(info_nce_array)[0] + 1e-10)
  
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
    SESSION_BASED = auto()

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