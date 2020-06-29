from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DecisionTree.DecisionNode import DecisionNode
from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Model.TreeEnsembleModel import TreeEnsembleModel
from Classification.Parameter.BaggingParameter import BaggingParameter
from Classification.Parameter.Parameter import Parameter


class Bagging(Classifier):

    def train(self, trainSet: InstanceList, parameters: BaggingParameter):
        """
        Bagging bootstrap ensemble method that creates individuals for its ensemble by training each classifier on a
        random redistribution of the training set.
        This training method is for a bagged decision tree classifier. 20 percent of the instances are left aside for
        pruning of the trees 80 percent of the instances are used for training the trees. The number of trees
        (forestSize) is a parameter, and basically the method will learn an ensemble of trees as a model.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm.
        parameters : Parameter
            Parameters of the bagging trees algorithm. ensembleSize returns the number of trees in the bagged forest.
        """
        forestSize = parameters.getEnsembleSize()
        forest = []
        for i in range(forestSize):
            bootstrap = trainSet.bootstrap(i)
            tree = DecisionTree(DecisionNode(InstanceList(bootstrap.getSample())))
            forest.append(tree)
        self.model = TreeEnsembleModel(forest)
