"""
XGBoost Node for gradient boosting trees
"""

from typing import List, Optional
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Parameter.XGBoostParameter import XGBoostParameter


class XGBoostNode:
    """
    A node in the XGBoost decision tree.
    
    This class represents a node in a regression tree used for gradient boosting.
    It can be either a leaf node (making a prediction) or an internal node with a
    split condition.
    """
    
    def __init__(self, 
                 data: InstanceList,
                 gradients: List[float],
                 hessians: List[float],
                 instance_indices: List[int],
                 parent: Optional['XGBoostNode'],
                 parameter: XGBoostParameter,
                 depth: int = 0,
                 feature_subset: Optional[List[int]] = None):
        """
        Initialize an XGBoostNode.
        
        Args:
            data (InstanceList): Training instances for this node
            gradients (List[float]): First-order gradient values
            hessians (List[float]): Second-order gradient (Hessian) values
            instance_indices (List[int]): Indices of instances in this node
            parent (Optional[XGBoostNode]): Parent node
            parameter (XGBoostParameter): XGBoost hyperparameters
            depth (int): Current depth in the tree
            feature_subset (Optional[List[int]]): Subset of features to consider
        """
        self._data = data
        self._gradients = gradients
        self._hessians = hessians
        self._instance_indices = instance_indices
        self._parent = parent
        self._parameter = parameter
        self._depth = depth
        self._feature_subset = feature_subset
        
        self._children = []
        self._condition = None
        self._leaf = True
        self._leaf_value = 0.0
        
        # Calculate leaf value for this node
        self._leaf_value = self._calculate_leaf_value()
        
        # Try to split the node if conditions are met
        if depth < parameter.getMaxDepth() and len(instance_indices) >= parameter.getMinChildWeight():
            self._build_tree()
    
    def _calculate_leaf_value(self) -> float:
        """
        Calculate the leaf value (weight) for gradient boosting.
        
        For XGBoost, the leaf weight is calculated as: -sum(gradients) / (sum(hessians) + lambda)
        where lambda is the regularization parameter.
        
        Returns:
            float: The calculated leaf value
        """
        if not self._instance_indices:
            return 0.0
        
        sum_gradients = sum(self._gradients[i] for i in self._instance_indices)
        sum_hessians = sum(self._hessians[i] for i in self._instance_indices)
        
        # Add regularization (lambda)
        lambda_param = self._parameter.getRegLambda() if hasattr(self._parameter, 'getRegLambda') else 1.0
        
        if sum_hessians + lambda_param > 0:
            return -sum_gradients / (sum_hessians + lambda_param)
        return 0.0
    
    def _build_tree(self):
        """Build the tree by finding the best split."""
        best_gain = 0.0
        best_feature = -1
        best_threshold = None
        best_left_indices = None
        best_right_indices = None
        
        # Try each feature
        features_to_try = self._feature_subset if self._feature_subset else range(self._data.get(0).attributeSize())
        
        for feature_idx in features_to_try:
            # Find best split for this feature
            gain, threshold, left_indices, right_indices = self._find_best_split(feature_idx)
            
            if gain > best_gain and gain > 0:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
                best_left_indices = left_indices
                best_right_indices = right_indices
        
        # If we found a good split, create children
        if best_feature >= 0 and best_gain > 0:
            self._leaf = False
            
            # Create left child
            left_child = XGBoostNode(
                self._data, self._gradients, self._hessians,
                best_left_indices, self, self._parameter,
                self._depth + 1, self._feature_subset
            )
            self._children.append(left_child)
            
            # Create right child
            right_child = XGBoostNode(
                self._data, self._gradients, self._hessians,
                best_right_indices, self, self._parameter,
                self._depth + 1, self._feature_subset
            )
            self._children.append(right_child)
            
            self._condition = (best_feature, best_threshold)
    
    def _find_best_split(self, feature_idx: int):
        """
        Find the best split point for a given feature.
        
        Args:
            feature_idx (int): Index of the feature to split on
            
        Returns:
            tuple: (gain, threshold, left_indices, right_indices)
        """
        if not self._instance_indices:
            return 0.0, None, [], []
        
        # Get unique values for this feature
        attribute = self._data.get(self._instance_indices[0]).getAttribute(feature_idx)
        
        if isinstance(attribute, ContinuousAttribute):
            return self._find_best_continuous_split(feature_idx)
        else:
            return self._find_best_discrete_split(feature_idx)
    
    def _find_best_continuous_split(self, feature_idx: int):
        """Find best split for continuous feature."""
        values = []
        for idx in self._instance_indices:
            val = self._data.get(idx).getAttribute(feature_idx).getValue()
            values.append((val, idx))
        
        values.sort()
        
        best_gain = 0.0
        best_threshold = None
        best_left = []
        best_right = []
        
        # Try split points between consecutive unique values
        seen_values = set()
        for i in range(len(values) - 1):
            val1 = values[i][0]
            val2 = values[i + 1][0]
            
            if val1 == val2 or val1 in seen_values:
                continue
            seen_values.add(val1)
            
            threshold = (val1 + val2) / 2.0
            
            left_indices = [idx for val, idx in values if val <= threshold]
            right_indices = [idx for val, idx in values if val > threshold]
            
            if len(left_indices) < self._parameter.getMinChildWeight() or \
               len(right_indices) < self._parameter.getMinChildWeight():
                continue
            
            gain = self._calculate_split_gain(left_indices, right_indices)
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
                best_left = left_indices
                best_right = right_indices
        
        return best_gain, best_threshold, best_left, best_right
    
    def _find_best_discrete_split(self, feature_idx: int):
        """Find best split for discrete feature."""
        # Group instances by feature value
        groups = {}
        for idx in self._instance_indices:
            val = str(self._data.get(idx).getAttribute(feature_idx).getValue())
            if val not in groups:
                groups[val] = []
            groups[val].append(idx)
        
        best_gain = 0.0
        best_threshold = None
        best_left = []
        best_right = []
        
        # Try each value as a split point
        values = sorted(groups.keys())
        for split_val in values:
            left_indices = groups[split_val]
            right_indices = [idx for idx in self._instance_indices if idx not in left_indices]
            
            if len(left_indices) < self._parameter.getMinChildWeight() or \
               len(right_indices) < self._parameter.getMinChildWeight():
                continue
            
            gain = self._calculate_split_gain(left_indices, right_indices)
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = split_val
                best_left = left_indices
                best_right = right_indices
        
        return best_gain, best_threshold, best_left, best_right
    
    def _calculate_split_gain(self, left_indices: List[int], right_indices: List[int]) -> float:
        """
        Calculate the gain from a split.
        
        XGBoost gain formula:
        Gain = 0.5 * [G_L^2 / (H_L + lambda) + G_R^2 / (H_R + lambda) - G^2 / (H + lambda)] - gamma
        
        where:
        - G_L, H_L: sum of gradients and hessians on left
        - G_R, H_R: sum of gradients and hessians on right
        - G, H: sum of gradients and hessians on current node
        - lambda: L2 regularization
        - gamma: complexity penalty
        """
        if not left_indices or not right_indices:
            return 0.0
            
        sum_grad_left = sum(self._gradients[i] for i in left_indices)
        sum_hess_left = sum(self._hessians[i] for i in left_indices)
        
        sum_grad_right = sum(self._gradients[i] for i in right_indices)
        sum_hess_right = sum(self._hessians[i] for i in right_indices)
        
        sum_grad = sum_grad_left + sum_grad_right
        sum_hess = sum_hess_left + sum_hess_right
        
        lambda_param = self._parameter.getRegLambda() if hasattr(self._parameter, 'getRegLambda') else 1.0
        gamma = self._parameter.getGamma() if hasattr(self._parameter, 'getGamma') else 0.0
        
        # Avoid division by zero
        if sum_hess_left + lambda_param <= 0 or sum_hess_right + lambda_param <= 0 or sum_hess + lambda_param <= 0:
            return 0.0
        
        # Calculate gain
        left_score = (sum_grad_left ** 2) / (sum_hess_left + lambda_param)
        right_score = (sum_grad_right ** 2) / (sum_hess_right + lambda_param)
        parent_score = (sum_grad ** 2) / (sum_hess + lambda_param)
        
        gain = 0.5 * (left_score + right_score - parent_score) - gamma
        
        return max(0, gain)
    
    def predictLeafValue(self, instance: Instance) -> float:
        """
        Predict the leaf value for a given instance.
        
        Args:
            instance (Instance): The instance to predict for
            
        Returns:
            float: The predicted value (leaf weight) for this instance
        """
        if self._leaf:
            return self._leaf_value
        
        feature_idx, threshold = self._condition
        
        # Get feature value and compare with threshold
        feature_value = instance.getAttribute(feature_idx).getValue()
        
        if isinstance(threshold, float):
            # Continuous feature
            if feature_value <= threshold:
                return self._children[0].predictLeafValue(instance)
            else:
                return self._children[1].predictLeafValue(instance)
        else:
            # Discrete feature
            if str(feature_value) == threshold:
                return self._children[0].predictLeafValue(instance)
            else:
                return self._children[1].predictLeafValue(instance)
