from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DecisionTree.DecisionNode import DecisionNode
from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Parameter.Parameter import Parameter


class C45Stump(Classifier):

    """
    Training algorithm for C4.5 Stump univariate decision tree classifier.

    PARAMETERS
    ----------
    trainSet : InstanceList
        Training data given to the algorithm.
    """
    def train(self, trainSet: InstanceList, parameters: Parameter):
        self.model = DecisionTree(DecisionNode(trainSet, None, None, True))
