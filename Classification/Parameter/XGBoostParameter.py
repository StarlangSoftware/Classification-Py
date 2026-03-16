"""
XGBoost Parameter Configuration
"""

from Classification.Parameter.Parameter import Parameter


class XGBoostParameter(Parameter):
    """
    Parameter class for XGBoost algorithm.
    
    This class encapsulates all hyperparameters used in the XGBoost gradient
    boosting implementation, including learning rate, tree structure parameters,
    regularization terms, and sampling ratios.
    
    Attributes:
        __learning_rate (float): Step size shrinkage to prevent overfitting (0 < eta <= 1)
        __n_estimators (int): Number of boosting rounds (trees)
        __max_depth (int): Maximum depth of trees
        __min_child_weight (float): Minimum sum of instance weight needed in a child
        __gamma (float): Minimum loss reduction required for split
        __subsample (float): Subsample ratio of training instances (0 < ratio <= 1)
        __colsample_bytree (float): Subsample ratio of columns when constructing each tree
        __reg_lambda (float): L2 regularization term on weights
        __reg_alpha (float): L1 regularization term on weights
        __early_stopping_rounds (int): Stop if no improvement for N rounds
    """
    
    def __init__(self, seed: int, 
                 learning_rate: float = 0.3,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 min_child_weight: float = 0.0,
                 gamma: float = 0.0,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 reg_lambda: float = 0.0,
                 reg_alpha: float = 0.0,
                 early_stopping_rounds: int = 10):
        """
        Initialize XGBoost parameters with validation.
        
        Args:
            seed (int): Random seed for reproducibility
            learning_rate (float, optional): Step size shrinkage to prevent overfitting. 
                Must be in (0, 1]. Defaults to 0.3
            n_estimators (int, optional): Number of boosting rounds (trees). 
                Must be at least 1. Defaults to 100
            max_depth (int, optional): Maximum depth of trees. 
                Must be at least 1. Defaults to 6
            min_child_weight (float, optional): Minimum sum of instance weight (hessian) 
                needed in a child. Must be non-negative. Defaults to 0.0
            gamma (float, optional): Minimum loss reduction required to make a split. 
                Must be non-negative. Defaults to 0.0
            subsample (float, optional): Subsample ratio of training instances. 
                Must be in (0, 1]. Defaults to 1.0
            colsample_bytree (float, optional): Subsample ratio of columns when 
                constructing each tree. Must be in (0, 1]. Defaults to 1.0
            reg_lambda (float, optional): L2 regularization term on weights. 
                Must be non-negative. Defaults to 0.0
            reg_alpha (float, optional): L1 regularization term on weights. 
                Must be non-negative. Defaults to 0.0
            early_stopping_rounds (int, optional): Number of rounds with no improvement 
                after which training will stop. Defaults to 10
        
        Raises:
            ValueError: If parameters are out of valid ranges
        """
        super().__init__(seed)
        
        # Validate parameters
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if min_child_weight < 0:
            raise ValueError("min_child_weight must be non-negative")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if not 0 < subsample <= 1:
            raise ValueError("subsample must be in (0, 1]")
        if not 0 < colsample_bytree <= 1:
            raise ValueError("colsample_bytree must be in (0, 1]")
        if reg_lambda < 0:
            raise ValueError("reg_lambda must be non-negative")
        if reg_alpha < 0:
            raise ValueError("reg_alpha must be non-negative")
        
        self.__learning_rate = learning_rate
        self.__n_estimators = n_estimators
        self.__max_depth = max_depth
        self.__min_child_weight = min_child_weight
        self.__gamma = gamma
        self.__subsample = subsample
        self.__colsample_bytree = colsample_bytree
        self.__reg_lambda = reg_lambda
        self.__reg_alpha = reg_alpha
        self.__early_stopping_rounds = early_stopping_rounds
    
    def getLearningRate(self) -> float:
        """
        Return the learning rate (eta).
        
        Returns:
            float: Learning rate value in (0, 1]
        """
        return self.__learning_rate
    
    def getNEstimators(self) -> int:
        """
        Return the number of boosting rounds (trees).
        
        Returns:
            int: Number of trees to build in the ensemble
        """
        return self.__n_estimators
    
    def getMaxDepth(self) -> int:
        """
        Return the maximum depth of trees.
        
        Returns:
            int: Maximum depth allowed for each tree
        """
        return self.__max_depth
    
    def getMinChildWeight(self) -> float:
        """
        Return the minimum sum of instance weight needed in a child.
        
        Returns:
            float: Minimum sum of hessian values required in a child node
        """
        return self.__min_child_weight
    
    def getGamma(self) -> float:
        """
        Return the minimum loss reduction required for split.
        
        Returns:
            float: Minimum gain required to make a split
        """
        return self.__gamma
    
    def getSubsample(self) -> float:
        """
        Return the subsample ratio of training instances.
        
        Returns:
            float: Proportion of instances to sample for each tree in (0, 1]
        """
        return self.__subsample
    
    def getColsampleByTree(self) -> float:
        """
        Return the subsample ratio of columns when constructing each tree.
        
        Returns:
            float: Proportion of features to sample for each tree in (0, 1]
        """
        return self.__colsample_bytree
    
    def getRegLambda(self) -> float:
        """
        Return the L2 regularization term on weights.
        
        Returns:
            float: L2 regularization parameter (ridge penalty)
        """
        return self.__reg_lambda
    
    def getRegAlpha(self) -> float:
        """
        Return the L1 regularization term on weights.
        
        Returns:
            float: L1 regularization parameter (lasso penalty)
        """
        return self.__reg_alpha
    
    def getEarlyStoppingRounds(self) -> int:
        """
        Return the number of rounds for early stopping.
        
        Returns:
            int: Number of consecutive rounds without improvement before stopping
        """
        return self.__early_stopping_rounds