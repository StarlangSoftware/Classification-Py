from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.DecisionTree.DecisionNode import DecisionNode
from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Parameter.C45Parameter import C45Parameter


class C45(Classifier):

    def train(self, trainSet: InstanceList, parameters: C45Parameter):
        """
        Training algorithm for C4.5 univariate decision tree classifier. 20 percent of the data are left aside for
        pruning 80 percent of the data is used for constructing the tree.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm.
        parameters: C45Parameter
            Parameter of the C45 algorithm.
        """
        if parameters.isPrune():
            partition = Partition(trainSet, parameters.getCrossValidationRatio(), parameters.getSeed(), True)
            tree = DecisionTree(DecisionNode(partition.get(1)))
            tree.prune(partition.get(0))
        else:
            tree = DecisionTree(DecisionNode(trainSet))
        self.model = tree
