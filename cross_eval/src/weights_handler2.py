from enum import Enum, auto
import numpy as np

def load_hyperparameter_range(hyperparameter_range_raw : object) -> list:
    if hyperparameter_range_raw is None:
        raise ValueError("Hyperparameter Range is None.")
    if isinstance(hyperparameter_range_raw, list):
        hyperparameter_range = hyperparameter_range_raw
    elif isinstance(hyperparameter_range_raw, str):
        if hyperparameter_range_raw.startswith('linspace') or hyperparameter_range_raw.startswith('logspace'):
            hyperparameter_range = eval(f'np.{hyperparameter_range_raw}')
        elif hyperparameter_range_raw.startswith('np.linspace') or hyperparameter_range_raw.startswith('np.logspace'):
            hyperparameter_range = eval(hyperparameter_range_raw)
        else:
            hyperparameter_range = [hyperparameter_range_raw]
    else:
        hyperparameter_range = [hyperparameter_range_raw]
    return hyperparameter_range

class Weights_Group(Enum):
    GLOBAL = auto()
    POS = auto()
    NEG = auto()
    POSNEG = auto()

def get_weights_group_from_arg(weights_group_arg : str) -> Weights_Group:
    valid_weights_group_args = [weights_group.name.lower() for weights_group in Weights_Group]
    if weights_group_arg not in valid_weights_group_args:
        raise ValueError(f"Invalid Weights Group {weights_group_arg} in 'weights'. Possible values: {valid_weights_group_args}.")
    return Weights_Group[weights_group_arg.upper()]

class Weights_Scheme(Enum):
    UNWEIGHTED = auto()

    BOTH_CONSTANT_VW, BOTH_CONSTANT_HYP = auto(), auto()
    BOTH_CUBE_ROOT_VW, BOTH_CUBE_ROOT_HYP = auto(), auto()
    BOTH_LOGARITHM_VW, BOTH_LOGARITHM_HYP = auto(), auto()
    BOTH_SQUARE_ROOT_VW, BOTH_SQUARE_ROOT_HYP = auto(), auto()
    BOTH_LINEAR_VW, BOTH_LINEAR_HYP = auto(), auto()
    BOTH_SQUARE_VW, BOTH_SQUARE_HYP = auto(), auto()
    BOTH_EXPONENTIAL_VW, BOTH_EXPONENTIAL_HYP = auto(), auto()

    RATED_CONSTANT_VW, RATED_CONSTANT_HYP = auto(), auto()
    RATED_CUBE_ROOT_VW, RATED_CUBE_ROOT_HYP = auto(), auto()
    RATED_LOGARITHM_VW, RATED_LOGARITHM_HYP = auto(), auto()
    RATED_SQUARE_ROOT_VW, RATED_SQUARE_ROOT_HYP = auto(), auto()
    RATED_LINEAR_VW, RATED_LINEAR_HYP = auto(), auto()
    RATED_SQUARE_VW, RATED_SQUARE_HYP = auto(), auto()
    RATED_EXPONENTIAL_VW, RATED_EXPONENTIAL_HYP = auto(), auto()

TRANSFORMATION_WEIGHTS_VW = {Weights_Scheme.BOTH_CONSTANT_VW, Weights_Scheme.BOTH_CUBE_ROOT_VW, Weights_Scheme.BOTH_LOGARITHM_VW, Weights_Scheme.BOTH_SQUARE_ROOT_VW,
                             Weights_Scheme.BOTH_LINEAR_VW, Weights_Scheme.BOTH_SQUARE_VW, Weights_Scheme.BOTH_EXPONENTIAL_VW,
                             Weights_Scheme.RATED_CONSTANT_VW, Weights_Scheme.RATED_CUBE_ROOT_VW, Weights_Scheme.RATED_LOGARITHM_VW, Weights_Scheme.RATED_SQUARE_ROOT_VW,
                             Weights_Scheme.RATED_LINEAR_VW, Weights_Scheme.RATED_SQUARE_VW, Weights_Scheme.RATED_EXPONENTIAL_VW}
TRANSFORMATION_WEIGHTS_HYP = {Weights_Scheme.BOTH_CONSTANT_HYP, Weights_Scheme.BOTH_CUBE_ROOT_HYP, Weights_Scheme.BOTH_LOGARITHM_HYP, Weights_Scheme.BOTH_SQUARE_ROOT_HYP,
                              Weights_Scheme.BOTH_LINEAR_HYP, Weights_Scheme.BOTH_SQUARE_HYP, Weights_Scheme.BOTH_EXPONENTIAL_HYP,
                              Weights_Scheme.RATED_CONSTANT_HYP, Weights_Scheme.RATED_CUBE_ROOT_HYP, Weights_Scheme.RATED_LOGARITHM_HYP, Weights_Scheme.RATED_SQUARE_ROOT_HYP,
                              Weights_Scheme.RATED_LINEAR_HYP, Weights_Scheme.RATED_SQUARE_HYP, Weights_Scheme.RATED_EXPONENTIAL_HYP}

def get_weights_scheme_from_arg(weights_scheme_arg : str) -> Weights_Scheme:
    valid_weights_scheme_args = [weights_scheme.name.lower() for weights_scheme in Weights_Scheme]
    if weights_scheme_arg not in valid_weights_scheme_args:
        raise ValueError(f"Invalid Weights Scheme {weights_scheme_arg} in 'weights'. Possible values: {valid_weights_scheme_args}.")
    return Weights_Scheme[weights_scheme_arg.upper()]

class Weights_Handler():
    def __init__(self, config : dict) -> None:
        weights_arg = config["weights"]
        self.need_voting_weight = False
        self.set_weights_schemes(weights_arg)

    def set_weights_schemes(self, weights_arg : str) -> None:
        weights_schemes = self.parse_weights_arg(weights_arg)
        self.verify_weights_schemes(weights_schemes, weights_arg)
        if weights_schemes[Weights_Group.POSNEG] is not None:
            weights_schemes[Weights_Group.POS] = weights_schemes[Weights_Group.POSNEG]
            weights_schemes[Weights_Group.NEG] = weights_schemes[Weights_Group.POSNEG]
        self.global_weights_scheme = weights_schemes[Weights_Group.GLOBAL]
        self.pos_weights_scheme = weights_schemes[Weights_Group.POS]
        self.neg_weights_scheme = weights_schemes[Weights_Group.NEG]

    def parse_weights_arg(self, weights_arg : str) -> dict:
        weights_schemes = {weights_group : None for weights_group in Weights_Group}
        weights_arg_split = weights_arg.split(',')
        for weights_arg_split_element in weights_arg_split:
            weights_arg_split_element_split = weights_arg_split_element.split(':')
            weights_group_arg, weights_scheme_arg = weights_arg_split_element_split
            weights_group = get_weights_group_from_arg(weights_group_arg)
            weights_scheme = get_weights_scheme_from_arg(weights_scheme_arg)
            weights_schemes[weights_group] = weights_scheme
        return weights_schemes

    def verify_weights_schemes(self, weights_schemes : dict, weights_arg : str) -> None:
        if all(weights_scheme is None for weights_scheme in weights_schemes.values()):
            raise ValueError(f"Invalid Weights Command '{weights_arg}' in 'weights'.")
        
        if weights_schemes[Weights_Group.GLOBAL] is not None:
            if any([weights_schemes[Weights_Group.POS] is not None, weights_schemes[Weights_Group.NEG] is not None, 
                    weights_schemes[Weights_Group.POSNEG] is not None]):
                raise ValueError(f"Invalid Weights Command '{weights_arg}' in 'weights'. If 'global' is set, 'pos' and 'neg' must be None.")
        else:
            if weights_schemes[Weights_Group.POSNEG] is not None:
                if any([weights_schemes[Weights_Group.POS] is not None, weights_schemes[Weights_Group.NEG] is not None]):
                    raise ValueError(f"Invalid Weights Command '{weights_arg}' in 'weights'. If 'posneg' is set, 'pos' and 'neg' must be None.")
            else:
                if weights_schemes[Weights_Group.POS] is not None and weights_schemes[Weights_Group.NEG] is None:
                    raise ValueError(f"Invalid Weights Command '{weights_arg}' in 'weights'. If 'pos' is set, 'neg' must be set.")
                elif weights_schemes[Weights_Group.POS] is None and weights_schemes[Weights_Group.NEG] is not None:
                    raise ValueError(f"Invalid Weights Command '{weights_arg}' in 'weights'. If 'neg' is set, 'pos' must be set.")

    def load_weights_hyperparameters(self, config : dict) -> None:
        weights_hyperparameters_ranges = {}
        if self.global_weights_scheme is not None:
            weights_hyperparameters_ranges.update(self.load_weights_hyperparameters_global(self.global_weights_scheme, config))
        else:
            weights_hyperparameters_ranges.update(self.load_weights_hyperparameters_label(self.pos_weights_scheme, is_positive = True, config = config))
            weights_hyperparameters_ranges.update(self.load_weights_hyperparameters_label(self.neg_weights_scheme, is_positive = False, config = config))
        return weights_hyperparameters_ranges
        
    def load_weights_hyperparameters_global(self, weights_scheme : Weights_Scheme, config : dict) -> dict:
        return {}
    
    def load_weights_hyperparameters_label(self, weights_scheme : Weights_Scheme, is_positive : bool, config : dict) -> dict:
        if weights_scheme == Weights_Scheme.UNWEIGHTED:
            return {}
        elif weights_scheme in TRANSFORMATION_WEIGHTS_VW:
            if is_positive and config["include_base"]:
                self.need_voting_weight = True
            elif not is_positive and config["include_cache"]:
                self.need_voting_weight = True
            return {}
        elif weights_scheme in TRANSFORMATION_WEIGHTS_HYP:
            transformation_weights_hyp_str = f"weights_{'pos' if is_positive else 'neg'}rated_importance"
            return {transformation_weights_hyp_str : load_hyperparameter_range(config[transformation_weights_hyp_str])}
        

    def load_weights_for_user(self, hyperparameters : dict, hyperparameters_combination : tuple, voting_weight : float, 
                              train_posrated_n : int, train_negrated_n : int, base_n : int, cache_n : int) -> tuple:
        if self.global_weights_scheme is not None:
            w_p, w_n, w_b, w_c = self.load_weights_for_user_global(hyperparameters, hyperparameters_combination, voting_weight, train_posrated_n, train_negrated_n, base_n, cache_n)
        else:
            w_p, w_b = self.load_weights_for_user_label(hyperparameters, hyperparameters_combination, voting_weight,
                                                            train_posrated_n, train_negrated_n, base_n, cache_n, is_positive = True)
            w_n, w_c = self.load_weights_for_user_label(hyperparameters, hyperparameters_combination, voting_weight,
                                                            train_posrated_n, train_negrated_n, base_n, cache_n, is_positive = False)
        return w_p, w_n, w_b, w_c


    def load_weights_for_user_global(self, hyperparameters : dict, hyperparameters_combination : tuple, voting_weight : float,
                                     train_posrated_n : int, train_negrated_n : int, base_n : int, cache_n : int) -> tuple:
        pass

    def load_weights_for_user_label(self, hyperparameters : dict, hyperparameters_combination : tuple, voting_weight : float,
                                    train_posrated_n : int, train_negrated_n : int, base_n : int, cache_n : int, is_positive : bool) -> tuple:
        weights_scheme = self.pos_weights_scheme if is_positive else self.neg_weights_scheme
        if weights_scheme == Weights_Scheme.UNWEIGHTED:
            return 1.0, 1.0
        elif weights_scheme in TRANSFORMATION_WEIGHTS_VW or weights_scheme in TRANSFORMATION_WEIGHTS_HYP:
            if is_positive:
                subclass1_n, subclass2_n = train_posrated_n, base_n
                scalar = hyperparameters_combination[hyperparameters["weights_posrated_importance"]] if weights_scheme in TRANSFORMATION_WEIGHTS_HYP else voting_weight
            else:
                subclass1_n, subclass2_n = train_negrated_n, cache_n
                scalar = hyperparameters_combination[hyperparameters["weights_negrated_importance"]] if weights_scheme in TRANSFORMATION_WEIGHTS_HYP else voting_weight
            if subclass2_n > 0:
                subclass1_weight, subclass2_weight = self.load_transformation_weights_for_user_label(weights_scheme, scalar, subclass1_n, subclass2_n)
            else:
                subclass1_weight, subclass2_weight = 1 / subclass1_n, 0.0
            correction = train_posrated_n + train_negrated_n + base_n + cache_n
            return subclass1_weight * correction, subclass2_weight * correction
    
    def load_transformation_weights_for_user_label(self, weights_scheme : Weights_Scheme, scalar : float, subclass1_n : int, subclass2_n : int) -> tuple:
        subclass1_transformation, subclass2_transformation = self.get_transformation_functions(weights_scheme)
        denominator = scalar * subclass1_transformation(subclass1_n) + (1.0 - scalar) * subclass2_transformation(subclass2_n)
        subclass1_weight = scalar * subclass1_transformation(subclass1_n) / (denominator * subclass1_n) if subclass1_n > 0 else 0
        subclass2_weight = (1.0 - scalar) * subclass2_transformation(subclass2_n) / (denominator * subclass2_n) if subclass2_n > 0 else 0
        return subclass1_weight, subclass2_weight
    
    def get_transformation_functions(self, weights_scheme : Weights_Scheme) -> tuple:
        if weights_scheme in {Weights_Scheme.BOTH_CONSTANT_VW, Weights_Scheme.BOTH_CONSTANT_HYP}:
            return (lambda x : 1.0, lambda x : 1.0)
        elif weights_scheme in {Weights_Scheme.BOTH_CUBE_ROOT_VW, Weights_Scheme.BOTH_CUBE_ROOT_HYP}:
            return (lambda x : np.cbrt(x), lambda x : np.cbrt(x))
        elif weights_scheme in {Weights_Scheme.BOTH_LOGARITHM_VW, Weights_Scheme.BOTH_LOGARITHM_HYP}:
            return (lambda x : np.log(x + 1), lambda x : np.log(x + 1))
        elif weights_scheme in {Weights_Scheme.BOTH_SQUARE_ROOT_VW, Weights_Scheme.BOTH_SQUARE_ROOT_HYP}:
            return (lambda x : np.sqrt(x), lambda x : np.sqrt(x))
        elif weights_scheme in {Weights_Scheme.BOTH_LINEAR_VW, Weights_Scheme.BOTH_LINEAR_HYP}:
            return (lambda x : x, lambda x : x)
        elif weights_scheme in {Weights_Scheme.BOTH_SQUARE_VW, Weights_Scheme.BOTH_SQUARE_HYP}:
            return (lambda x : x ** 2, lambda x : x ** 2)
        elif weights_scheme in {Weights_Scheme.BOTH_EXPONENTIAL_VW, Weights_Scheme.BOTH_EXPONENTIAL_HYP}:
            return (lambda x : np.exp(x), lambda x : np.exp(x))
        else:
            func2 = lambda x : x
            if weights_scheme in {Weights_Scheme.RATED_CONSTANT_VW, Weights_Scheme.RATED_CONSTANT_HYP}:
                func1 = lambda x : 0.5
            elif weights_scheme in {Weights_Scheme.RATED_CUBE_ROOT_VW, Weights_Scheme.RATED_CUBE_ROOT_HYP}:
                func1 = lambda x : np.cbrt(x)
            elif weights_scheme in {Weights_Scheme.RATED_LOGARITHM_VW, Weights_Scheme.RATED_LOGARITHM_HYP}:
                func1 = lambda x : np.log(x + 1)
            elif weights_scheme in {Weights_Scheme.RATED_SQUARE_ROOT_VW, Weights_Scheme.RATED_SQUARE_ROOT_HYP}:
                func1 = lambda x : np.sqrt(x)
            elif weights_scheme in {Weights_Scheme.RATED_LINEAR_VW, Weights_Scheme.RATED_LINEAR_HYP}:
                func1 = lambda x : x
            elif weights_scheme in {Weights_Scheme.RATED_SQUARE_VW, Weights_Scheme.RATED_SQUARE_HYP}:
                func1 = lambda x : x ** 2
            elif weights_scheme in {Weights_Scheme.RATED_EXPONENTIAL_VW, Weights_Scheme.RATED_EXPONENTIAL_HYP}:
                func1 = lambda x : np.exp(x)
            return (func1, func2)
        
    def print_weights(self, hyperparameters : dict, weights_hyperparameters : list, hyperparameters_combination : tuple, voting_weight : float, 
                      n_posrated : int, n_negrated : int, n_base : int, n_cache : int, w_p : float, w_n : float, w_b : float, w_c : float) -> None:
        s = "<<< "
        for weights_hyperparameter in weights_hyperparameters:
            s += f"{weights_hyperparameter} = {hyperparameters_combination[hyperparameters[weights_hyperparameter]]:.3f}, "
        s += f"n_posrated = {n_posrated}, n_negrated = {n_negrated}, n_base = {n_base}, n_cache = {n_cache}, voting_weight = {voting_weight:.5f} >>>"
        s += f"   w_p : {w_p:.5f}, w_b : {w_b:.5f}, w_n : {w_n:.5f}, w_c : {w_c:.5f}."
        print(s)