"""
XGBoost Gradient Boosting Model
This module provides an enhanced XGBoost gradient boosting classifier with bug fixes,
performance optimizations, and additional features.
"""

from math import log, exp
import random
from typing import List, Dict, Optional
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DecisionTree.XGBoostTree import XGBoostTree
from Classification.Model.ValidatedModel import ValidatedModel
from Classification.Parameter.XGBoostParameter import XGBoostParameter


class XGBoostModel(ValidatedModel):
    """
    XGBoost Gradient Boosting Classifier.
    
    Attributes
    ----------
    __trees : List[XGBoostTree] or List[List[XGBoostTree]]
        Collection of decision trees. For binary classification, it's a flat list.
        For multiclass, it's a list of lists where each sublist contains trees for one class.
    __class_labels : List[str]
        Distinct class labels from the training set.
    __n_classes : int
        Number of distinct classes.
    __base_score : float
        Initial prediction score (log odds for binary classification).
    __parameter : XGBoostParameter
        Training parameters configuration.
    __feature_importance : Dict[int, float]
        Feature importance scores mapping feature index to importance value.
    __training_history : List[Dict]
        Training history containing validation metrics per iteration.
    """
    
    __trees: List
    __class_labels: List[str]
    __n_classes: int
    __base_score: float
    __parameter: Optional[XGBoostParameter]
    __feature_importance: Dict[int, float]
    __training_history: List[Dict]
    
    def __init__(self):
        """
        Initialize XGBoost classifier with empty state.
        
        Creates a new XGBoostModel instance with initialized but empty attributes
        for trees, class labels, and training metrics.
        """
        self.__trees = []
        self.__class_labels = []
        self.__n_classes = 0
        self.__base_score = 0.0
        self.__parameter = None
        self.__feature_importance = {}
        self.__training_history = []
    
    def __sigmoid(self, x: float) -> float:
        """
        Apply sigmoid function with numerical stability.
        
        Parameters
        ----------
        x : float
            Input value to transform.
        
        Returns
        -------
        float
            Sigmoid transformation of x, clamped between 0 and 1.
            Returns 1.0 for x > 20, 0.0 for x < -20 to prevent overflow.
        """
        if x > 20:
            return 1.0
        if x < -20:
            return 0.0
        return 1.0 / (1.0 + exp(-x))
    
    def __softmax(self, scores: List[float]) -> List[float]:
        """
        Apply softmax function with numerical stability.
        
        Parameters
        ----------
        scores : List[float]
            Raw scores for each class.
        
        Returns
        -------
        List[float]
            Normalized probability distribution over classes.
            Sum of all probabilities equals 1.0.
        """
        max_score = max(scores)
        exp_scores = [exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        return [e / sum_exp for e in exp_scores]
    
    def train(self, trainSet: InstanceList, parameters: XGBoostParameter,
              validationSet: Optional[InstanceList] = None) -> None:
        """
        Train the XGBoost classifier using gradient boosting.
        
        Parameters
        ----------
        trainSet : InstanceList
            Training dataset containing labeled instances.
        parameters : XGBoostParameter
            Configuration parameters for training (learning rate, max depth, etc.).
        validationSet : Optional[InstanceList], default=None
            Optional validation set for early stopping and performance monitoring.
        
        Returns
        -------
        None
            Model is trained in-place, modifying internal state.
        
        Notes
        -----
        - Automatically detects binary vs multiclass classification
        - Uses early stopping if validation set is provided
        - Supports instance subsampling for stochastic boosting
        """
        self.__parameter = parameters
        self.__class_labels = trainSet.getDistinctClassLabels()
        self.__n_classes = len(self.__class_labels)
        self.__training_history = []
        self.__trees = []
        
        random.seed(parameters.getSeed())
        
        if self.__n_classes == 2:
            self.__trainBinary(trainSet, parameters, validationSet)
        else:
            self.__trainMulticlass(trainSet, parameters, validationSet)
    
    def __trainBinary(self, trainSet: InstanceList, 
                     parameters: XGBoostParameter,
                     validationSet: Optional[InstanceList] = None) -> None:
        """
        Train for binary classification using logistic loss.
        
        Parameters
        ----------
        trainSet : InstanceList
            Training dataset with binary class labels.
        parameters : XGBoostParameter
            Training configuration parameters.
        validationSet : Optional[InstanceList], default=None
            Optional validation set for early stopping.
        
        Returns
        -------
        None
            Updates internal trees and base score.
        
        Notes
        -----
        - Initializes predictions with log odds of positive class
        - Uses gradient and hessian of logistic loss
        - Implements early stopping based on validation error
        """
        n_samples = trainSet.size()
        
        positive_count = sum(1 for i in range(n_samples) 
                            if trainSet.get(i).getClassLabel() == self.__class_labels[1])
        
        if positive_count == 0:
            self.__base_score = -5.0
        elif positive_count == n_samples:
            self.__base_score = 5.0
        else:
            self.__base_score = log(positive_count / (n_samples - positive_count))
        
        predictions = [self.__base_score] * n_samples
        
        best_val_error = float('inf')
        rounds_without_improvement = 0
        best_n_trees = 0
        
        for iteration in range(parameters.getNEstimators()):
            if parameters.getSubsample() < 1.0:
                n_subsample = max(1, int(n_samples * parameters.getSubsample()))
                sample_indices = random.sample(range(n_samples), n_subsample)
            else:
                sample_indices = list(range(n_samples))
            
            gradients = [0.0] * n_samples
            hessians = [0.0] * n_samples
            
            for i in range(n_samples):
                pred_prob = self.__sigmoid(predictions[i])
                true_label = 1.0 if trainSet.get(i).getClassLabel() == self.__class_labels[1] else 0.0
                
                gradients[i] = pred_prob - true_label
                hessians[i] = max(pred_prob * (1.0 - pred_prob), 1e-6)
            
            tree = XGBoostTree(trainSet, gradients, hessians, sample_indices, parameters)
            self.__trees.append(tree)
            
            learning_rate = parameters.getLearningRate()
            for i in range(n_samples):
                predictions[i] += learning_rate * tree.predictValue(trainSet.get(i))
            
            if validationSet is not None:
                val_error = self.__calculateError(validationSet)
                self.__training_history.append({
                    'iteration': iteration,
                    'validation_error': val_error
                })
                
                if val_error < best_val_error:
                    best_val_error = val_error
                    best_n_trees = iteration + 1
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                    
                    if rounds_without_improvement >= parameters.getEarlyStoppingRounds():
                        self.__trees = self.__trees[:best_n_trees]
                        break
    
    def __trainMulticlass(self, trainSet: InstanceList, 
                         parameters: XGBoostParameter,
                         validationSet: Optional[InstanceList] = None) -> None:
        """
        Train for multiclass classification using softmax loss.
        
        Parameters
        ----------
        trainSet : InstanceList
            Training dataset with multiple class labels.
        parameters : XGBoostParameter
            Training configuration parameters.
        validationSet : Optional[InstanceList], default=None
            Optional validation set for early stopping.
        
        Returns
        -------
        None
            Updates internal trees structure (one tree list per class).
        
        Notes
        -----
        - Uses one-vs-all approach with softmax probabilities
        - Trains separate tree ensemble for each class
        - Gradient and hessian computed from softmax derivatives
        """
        n_samples = trainSet.size()
        
        predictions = [[0.0 for _ in range(n_samples)] for _ in range(self.__n_classes)]
        
        self.__trees = [[] for _ in range(self.__n_classes)]
        
        best_val_error = float('inf')
        rounds_without_improvement = 0
        best_n_trees = 0
        
        for iteration in range(parameters.getNEstimators()):
            if parameters.getSubsample() < 1.0:
                n_subsample = max(1, int(n_samples * parameters.getSubsample()))
                sample_indices = random.sample(range(n_samples), n_subsample)
            else:
                sample_indices = list(range(n_samples))
            
            for class_idx in range(self.__n_classes):
                target_class = self.__class_labels[class_idx]
                
                gradients = [0.0] * n_samples
                hessians = [0.0] * n_samples
                
                for i in range(n_samples):
                    scores = [predictions[c][i] for c in range(self.__n_classes)]
                    probs = self.__softmax(scores)
                    
                    true_label = 1.0 if trainSet.get(i).getClassLabel() == target_class else 0.0
                    pred_prob = probs[class_idx]
                    
                    gradients[i] = pred_prob - true_label
                    hessians[i] = max(pred_prob * (1.0 - pred_prob), 1e-6)
                
                tree = XGBoostTree(trainSet, gradients, hessians, sample_indices, parameters)
                self.__trees[class_idx].append(tree)
                
                learning_rate = parameters.getLearningRate()
                for i in range(n_samples):
                    predictions[class_idx][i] += learning_rate * tree.predictValue(trainSet.get(i))
            
            if validationSet is not None:
                val_error = self.__calculateError(validationSet)
                self.__training_history.append({
                    'iteration': iteration,
                    'validation_error': val_error
                })
                
                if val_error < best_val_error:
                    best_val_error = val_error
                    best_n_trees = iteration + 1
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                    
                    if rounds_without_improvement >= parameters.getEarlyStoppingRounds():
                        for class_idx in range(self.__n_classes):
                            self.__trees[class_idx] = self.__trees[class_idx][:best_n_trees]
                        break
    
    def __calculateError(self, testSet: InstanceList) -> float:
        """
        Calculate classification error rate on a dataset.
        
        Parameters
        ----------
        testSet : InstanceList
            Dataset to evaluate predictions on.
        
        Returns
        -------
        float
            Error rate as fraction of misclassified instances (range: 0.0 to 1.0).
        """
        n_errors = 0
        for i in range(testSet.size()):
            instance = testSet.get(i)
            predicted = self.predict(instance)
            if predicted != instance.getClassLabel():
                n_errors += 1
        return n_errors / testSet.size() if testSet.size() > 0 else 0.0
    
    def predict(self, instance: Instance) -> str:
        """
        Predict the class label for a single instance.
        
        Parameters
        ----------
        instance : Instance
            Input instance to classify.
        
        Returns
        -------
        str
            Predicted class label.
        
        Notes
        -----
        - For binary: returns class with probability >= 0.5
        - For multiclass: returns class with highest score
        """
        if self.__trees and isinstance(self.__trees[0], list):
            scores = [0.0] * self.__n_classes
            learning_rate = self.__parameter.getLearningRate()
            
            for class_idx in range(self.__n_classes):
                for tree in self.__trees[class_idx]:
                    scores[class_idx] += learning_rate * tree.predictValue(instance)
            
            max_idx = scores.index(max(scores))
            return self.__class_labels[max_idx]
        else:
            score = self.__base_score
            learning_rate = self.__parameter.getLearningRate()
            
            for tree in self.__trees:
                score += learning_rate * tree.predictValue(instance)
            
            prob = self.__sigmoid(score)
            return self.__class_labels[1] if prob >= 0.5 else self.__class_labels[0]
    
    def predictProbability(self, instance: Instance) -> Dict[str, float]:
        """
        Predict probability distribution over all classes.
        
        Parameters
        ----------
        instance : Instance
            Input instance to get probability predictions for.
        
        Returns
        -------
        Dict[str, float]
            Dictionary mapping each class label to its predicted probability.
            Probabilities sum to 1.0.
        """
        if self.__trees and isinstance(self.__trees[0], list):
            scores = [0.0] * self.__n_classes
            learning_rate = self.__parameter.getLearningRate()
            
            for class_idx in range(self.__n_classes):
                for tree in self.__trees[class_idx]:
                    scores[class_idx] += learning_rate * tree.predictValue(instance)
            
            probs = self.__softmax(scores)
            return {self.__class_labels[i]: probs[i] for i in range(self.__n_classes)}
        else:
            score = self.__base_score
            learning_rate = self.__parameter.getLearningRate()
            
            for tree in self.__trees:
                score += learning_rate * tree.predictValue(instance)
            
            prob_positive = self.__sigmoid(score)
            return {
                self.__class_labels[0]: 1.0 - prob_positive,
                self.__class_labels[1]: prob_positive
            }
    
    def getTrainingHistory(self) -> List[Dict]:
        """
        Get the training history with validation metrics.
        
        Returns
        -------
        List[Dict]
            List of dictionaries containing iteration number and validation error.
            Empty list if no validation set was used during training.
        """
        return self.__training_history
    
    def getFeatureImportance(self) -> Dict[int, float]:
        """
        Get feature importance scores.
        
        Returns
        -------
        Dict[int, float]
            Dictionary mapping feature indices to their importance scores.
            Currently returns empty dict (feature not yet implemented).
        """
        return self.__feature_importance
    
    def loadModel(self, fileName: str) -> None:
        """
        Load a trained model from a file.
        
        Parameters
        ----------
        fileName : str
            Path to the file containing the serialized model.
        
        Returns
        -------
        None
            Model state is loaded in-place.
        
        Raises
        ------
        IOError
            If file cannot be read or model data is corrupted.
        """
        import pickle
        try:
            with open(fileName, 'rb') as f:
                model_data = pickle.load(f)
                self.__trees = model_data['trees']
                self.__class_labels = model_data['class_labels']
                self.__n_classes = model_data['n_classes']
                self.__base_score = model_data['base_score']
                self.__parameter = model_data['parameter']
        except Exception as e:
            raise IOError(f"Failed to load model from {fileName}: {str(e)}")
    
    def saveModel(self, fileName: str) -> None:
        """
        Save the trained model to a file.
        
        Parameters
        ----------
        fileName : str
            Path where the model should be saved.
        
        Returns
        -------
        None
            Model is serialized and written to disk.
        
        Raises
        ------
        IOError
            If file cannot be written or serialization fails.
        """
        import pickle
        try:
            model_data = {
                'trees': self.__trees,
                'class_labels': self.__class_labels,
                'n_classes': self.__n_classes,
                'base_score': self.__base_score,
                'parameter': self.__parameter
            }
            with open(fileName, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            raise IOError(f"Failed to save model to {fileName}: {str(e)}")