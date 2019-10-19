from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DecisionTree.DecisionNode import DecisionNode
from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Parameter.C45Parameter import C45Parameter


class C45(Classifier):

    """
    Training algorithm for C4.5 univariate decision tree classifier. 20 percent of the data are left aside for pruning
    80 percent of the data is used for constructing the tree.

    PARAMETERS
    ----------
    trainSet : InstanceList
        Training data given to the algorithm.
    """
    def train(self, trainSet: InstanceList, parameters: C45Parameter):
        if parameters.isPrune():
            partition = trainSet.stratifiedPartition(parameters.getCrossValidationRatio(), parameters.getSeed())
            tree = DecisionTree(DecisionNode(partition.get(1)))
            tree.prune(partition.get(0))
        else:
            tree = DecisionTree(DecisionNode(trainSet))
        self.model = tree
