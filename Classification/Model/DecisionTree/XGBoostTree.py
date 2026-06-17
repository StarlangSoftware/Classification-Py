"""
XGBoost Decision Tree
"""

import random
from typing import List
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Model.DecisionTree.XGBoostNode import XGBoostNode
from Classification.Parameter.XGBoostParameter import XGBoostParameter


class XGBoostTree(DecisionTree):
    """
    Single tree in the XGBoost ensemble.
    
    This class represents an individual decision tree used in the XGBoost
    gradient boosting ensemble. It extends the DecisionTree class with
    XGBoost-specific functionality including gradient-based splits and
    feature subsampling.
    
    Attributes:
        _root (XGBoostNode): Root node of the decision tree
    """
    
    def __init__(self, data: InstanceList, 
                 gradients: List[float], 
                 hessians: List[float],
                 instance_indices: List[int],
                 parameter: XGBoostParameter):
        """
        Initialize XGBoost tree with gradient information.
        
        Args:
            data (InstanceList): Training instances for building the tree
            gradients (List[float]): First-order gradient values for each instance
            hessians (List[float]): Second-order gradient (Hessian) values for each instance
            instance_indices (List[int]): Indices of instances to use for this tree
            parameter (XGBoostParameter): Hyperparameters controlling tree construction
                including max depth, regularization, and feature sampling
        """
        # Determine feature subset for this tree (colsample_bytree)
        _feature_subset = None
        if parameter and parameter.getColsampleByTree() < 1.0:
            n_features = data.get(0).attributeSize()
            n_sample = max(1, int(n_features * parameter.getColsampleByTree()))
            _feature_subset = random.sample(range(n_features), n_sample)
        
        _root = XGBoostNode(data, gradients, hessians, instance_indices, 
                           None, parameter, 0, _feature_subset)
        self._DecisionTree__root = _root
    
    def predictValue(self, instance: Instance) -> float:
        """
        Predict the raw value for gradient boosting.
        
        This method traverses the tree to find the leaf node corresponding
        to the given instance and returns its predicted value (weight).
        The returned value is used as an additive update in the gradient
        boosting process.
        
        Args:
            instance (Instance): Instance to predict the value for
            
        Returns:
            float: Raw predicted value (leaf weight) from this tree
        """
        return self._DecisionTree__root.predictLeafValue(instance)